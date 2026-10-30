import base64
import hmac
import io
import os
from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from PIL import Image, UnidentifiedImageError
from pydantic import ValidationError
from starlette.exceptions import HTTPException

from . import errors, fallback, gpu, results
from .augmentation import Augmented
from .errors import BackendError
from .model import Model, Prediction
from .schemas import CONTRACT, LoadRequest, PredictRequest, PreviewRequest, TrainRequest, UnloadRequest
from .train import STANDARD_METRICS, Slots, TrainManager
from .transfer import SECRET_HEADER
from .weights import WeightCache

GB = 1 << 30
LOAD_WAIT_SECONDS = 2.0


def info(model: Model) -> dict:
    capabilities = ['predict', *model.capabilities]
    train = model.train_info()
    if train is not None and 'train' not in capabilities:
        capabilities.append('train')
    body = {
        'contract': CONTRACT, 'name': model.name, 'version': model.version, 'model_type': model.model_type,
        'tasks': list(model.tasks), 'capabilities': capabilities, 'concurrency': model.concurrency,
        'max_request_mb': model.max_request_mb, 'tasks_detail': model.tasks_detail(),
        'weights': {'roles': model.weight_roles, 'builtin': model.builtin},
    }
    if train is not None:
        # Run under the out-of-memory fallback wherever it runs, the cluster included.
        body['train'] = {**train, 'entrypoint': fallback.wrap(train['entrypoint']),
                         'metrics': [*STANDARD_METRICS, *train.get('metrics', [])]}
    found = gpu.detect()
    if found is not None:
        body['gpu'] = found
    return body


def _json(request_body: bytes | str, schema):
    try:
        return schema.model_validate_json(request_body)
    except ValidationError as error:
        raise errors.bad_request(f'invalid request: {error.errors(include_url=False)}') from error


async def _body(request: Request, schema):
    return _json(await request.body() or b'{}', schema)


def _decode(data: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(data))
    image.load()
    return image


def _png_data_url(image: Image.Image) -> str:
    out = io.BytesIO()
    image.save(out, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(out.getvalue()).decode()


def _predict_chunks(model: Model, loaded, task: str, images: list, params: dict, contexts: list) -> Prediction:
    """Calls predict() in batches of the model's size, halving it after running out of memory."""
    size = model.batch_size(task, params) or max(1, len(images))
    collected: list[dict] = []
    extra: dict = {}
    start = 0
    while start < len(images):
        end = start + size
        try:
            output = model.predict(loaded, task, images[start:end], params, contexts[start:end])
        except Exception as error:
            if errors.is_out_of_memory(error) and size > 1:
                size //= 2
                continue
            if errors.is_out_of_memory(error):
                raise errors.out_of_memory(str(error)) from error
            raise
        if isinstance(output, Prediction):
            extra |= output.extra
            output = output.results
        results.validate(task, output, len(images[start:end]))
        collected += output
        start = end
    return Prediction(collected, extra)


def create_app(model: Model, secret: str | None = None, cache_dir: Path | None = None,
               train_dir: Path | None = None) -> FastAPI:
    secret = secret if secret is not None else os.environ.get('MODEL_BACKEND_SECRET', '')
    if not secret:
        raise RuntimeError('MODEL_BACKEND_SECRET is not set')
    cache_root = cache_dir or Path(os.environ.get('MODEL_CACHE_DIR', '/cache'))
    cache = WeightCache(
        model, cache_root / 'weights', secret,
        max_bytes=int(float(os.environ.get('MODEL_CACHE_MAX_GB', '50')) * GB),
        max_loaded=int(os.environ.get('MODEL_MAX_LOADED', '2')),
        idle_unload_seconds=float(os.environ.get('MODEL_IDLE_UNLOAD_SECONDS', '0')),
        device=os.environ.get('MODEL_DEVICE', 'auto'))
    slots = Slots(model.concurrency)
    trainings = TrainManager(model, train_dir or cache_root / 'trainings', cache, secret, slots)
    app = FastAPI(title=f'{model.name} model backend', docs_url=None, redoc_url=None, openapi_url=None)
    app.state.cache = cache
    app.state.trainings = trainings
    max_bytes = model.max_request_mb * (1 << 20)

    def task_detail(task: str) -> dict:
        return model.tasks_detail().get(task, {})

    @app.middleware('http')
    async def authenticate(request: Request, call_next):
        if request.url.path.startswith('/v1/'):
            given = request.headers.get(SECRET_HEADER, '')
            if not hmac.compare_digest(given.encode(), secret.encode()):
                return JSONResponse(errors.error_body('unauthorized', 'missing or wrong model backend secret'),
                                    status_code=401)
        return await call_next(request)

    @app.exception_handler(BackendError)
    async def backend_error(request: Request, error: BackendError):
        headers = {'Retry-After': '1'} if error.status == 429 else None
        return JSONResponse(error.body(), status_code=error.status, headers=headers)

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, error: HTTPException):
        code = {404: 'not_found', 405: 'bad_request'}.get(error.status_code, 'bad_request')
        return JSONResponse(errors.error_body(code, str(error.detail)), status_code=error.status_code)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, error: RequestValidationError):
        return JSONResponse(errors.error_body('bad_request', str(error.errors())), status_code=400)

    @app.exception_handler(Exception)
    async def internal_error(request: Request, error: Exception):
        return JSONResponse(errors.error_body('internal', f'{type(error).__name__}: {error}'), status_code=500)

    @app.get('/health')
    def health():
        return {'status': 'ok', 'contract': CONTRACT, 'busy': slots.used, 'loaded': cache.loaded_versions()}

    @app.get('/v1/info')
    def get_info():
        return info(model)

    @app.post('/v1/weights/load')
    async def load_weights(request: Request):
        version = cache.load(await _body(request, LoadRequest))
        await run_in_threadpool(version.settled.wait, LOAD_WAIT_SECONDS)
        if version.state == 'error':
            raise version.error
        return JSONResponse(version.public(), status_code=200 if version.state == 'loaded' else 202)

    @app.get('/v1/weights/{sha256}')
    def weights_state(sha256: str):
        return cache.state(sha256)

    @app.post('/v1/weights/unload')
    async def unload_weights(request: Request):
        body = await _body(request, UnloadRequest)
        return {'unloaded': await run_in_threadpool(cache.unload, body.sha256)}

    @app.post('/v1/predict/{task}')
    async def predict(task: str, request: Request):
        if task not in model.tasks:
            raise errors.bad_request(f'this backend does not serve {task!r}; it serves {", ".join(model.tasks)}')
        if int(request.headers.get('content-length') or 0) > max_bytes:
            raise errors.too_large(f'the request is over {model.max_request_mb} MB')
        form = await request.form(max_files=100_000, max_fields=100_000, max_part_size=max_bytes)
        if 'request' not in form:
            raise errors.bad_request('the multipart body needs a part named request')
        part = form['request']
        body = _json(await part.read() if hasattr(part, 'read') else part, PredictRequest)
        limit = task_detail(task).get('batch', {}).get('max_items')
        if limit and len(body.items) > limit:
            raise errors.too_large(f'{len(body.items)} items is over the batch limit of {limit}')
        decoded: dict[str, Image.Image | dict] = {}
        for name in dict.fromkeys(item.image for item in body.items):
            upload = form.get(name)
            if upload is None or not hasattr(upload, 'read'):
                raise errors.bad_request(f'an item names image {name!r}, which is not in the request')
            try:
                decoded[name] = _decode(await upload.read())
            except (UnidentifiedImageError, OSError) as error:
                decoded[name] = {'code': 'bad_image', 'message': f'the image could not be read: {error}'}
        failed = {i: decoded[item.image] for i, item in enumerate(body.items) if isinstance(decoded[item.image], dict)}
        images = [decoded[item.image] for i, item in enumerate(body.items) if i not in failed]
        readable = [i for i in range(len(body.items)) if i not in failed]
        sha = body.weights.sha256 if body.weights else None
        slots.acquire()
        try:
            def run() -> Prediction:
                with cache.use(sha) as loaded:
                    return _predict_chunks(model, loaded, task, images, body.params,
                                           [body.items[i].context for i in readable])
            try:
                prediction = await run_in_threadpool(run)
            except results.InvalidResult as error:
                raise errors.internal(f'the model returned an invalid result: {error}') from error
        finally:
            slots.release()
        by_index = dict(zip(readable, prediction.results))
        out = [{'id': item.id, **(by_index[i] if i in by_index else {'error': failed[i]})}
               for i, item in enumerate(body.items)]
        return {'weights': {'sha256': sha} if sha else None, **prediction.extra, 'results': out}

    def trainable() -> None:
        if model.train_info() is None:
            raise BackendError(404, 'not_found', 'this backend does not train')

    @app.post('/v1/train')
    async def start_training(request: Request):
        trainable()
        training = trainings.start(await _body(request, TrainRequest))
        return JSONResponse({'train_id': training.id}, status_code=202)

    @app.get('/v1/train/tunables')
    def get_tunables():
        trainable()
        return {'tunables': model.tunables()}

    @app.get('/v1/train/augmentations')
    def get_augmentations():
        trainable()
        return model.augmentations()

    @app.post('/v1/train/augmentations/preview')
    async def preview_augmentations(request: Request):
        trainable()
        form = await request.form(max_files=1000, max_part_size=max_bytes)
        if 'request' not in form:
            raise errors.bad_request('the multipart body needs a part named request')
        part = form['request']
        body = _json(await part.read() if hasattr(part, 'read') else part, PreviewRequest)
        images = [_decode(await value.read()) for key, value in form.multi_items()
                  if key != 'request' and hasattr(value, 'read')]
        extra = {}
        for name in ('boxes', 'polygons'):
            given = getattr(body, name)
            if given is not None:
                if len(given) != len(images):
                    raise errors.bad_request(f'{name} are given for {len(given)} images, not the {len(images)} sent')
                extra[name] = given
        drawn = await run_in_threadpool(model.augmentation_preview, body.spec, images, body.seed, body.draws, **extra)
        shown = [item if isinstance(item, Augmented) else Augmented(item) for item in drawn]
        return {'images': [_png_data_url(item.image) for item in shown],
                **{name: [getattr(item, name) or [] for item in shown] for name in extra}}

    @app.get('/v1/train/{train_id}')
    def training_state(train_id: str):
        trainable()
        return trainings.get(train_id).public()

    @app.get('/v1/train/{train_id}/progress')
    def training_progress(train_id: str, start: int = Query(0, alias='from')):
        trainable()
        lines = trainings.get(train_id).progress_lines()[max(0, start):]
        return PlainTextResponse(''.join(line + '\n' for line in lines), media_type='application/x-ndjson')

    @app.post('/v1/train/{train_id}/cancel')
    def cancel_training(train_id: str):
        trainable()
        return trainings.cancel(trainings.get(train_id)).public()

    @app.get('/v1/train/{train_id}/artifacts')
    def training_artifacts(train_id: str):
        trainable()
        return trainings.artifacts(trainings.get(train_id))

    @app.get('/v1/train/{train_id}/artifacts/{name:path}')
    def training_artifact(train_id: str, name: str):
        trainable()
        path = trainings.artifact_path(trainings.get(train_id), name)
        if path is None:
            raise BackendError(404, 'not_found', f'no artifact {name}')
        return FileResponse(path, media_type='application/octet-stream')

    @app.delete('/v1/train/{train_id}')
    async def delete_training(train_id: str):
        trainable()
        await run_in_threadpool(trainings.delete, trainings.get(train_id))
        return {'deleted': train_id}

    return app


def serve(model: Model) -> None:
    import uvicorn
    uvicorn.run(create_app(model), host=os.environ.get('HOST', '0.0.0.0'), port=int(os.environ.get('PORT', '9100')))

