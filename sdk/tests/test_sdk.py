import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from deeplabel_backend import Loaded, Model, create_app
from deeplabel_backend.schemas import LoadRequest
from deeplabel_backend.train import legacy_env
from deeplabel_backend.weights import WeightCache
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'contract_tests'))
from fixture_server import SECRET_HEADER, FixtureServer

SECRET = 'unit-secret'


class OutOfMemoryError(RuntimeError):
    pass


class FakeModel(Model):
    name = 'fake'
    model_type = 'fake'
    tasks = ('ocr',)
    builtin = True
    weight_roles = {'weights': 'optional'}

    def __init__(self):
        self.calls: list[int] = []
        self.unloaded: list[list[str]] = []
        self.gate: threading.Event | None = None
        self.oom_above: int | None = None
        self.answer = None

    def tasks_detail(self):
        return {'ocr': {'batch': {'max_items': 64}, 'decodes': ['greedy']}}

    def load(self, files, device):
        return Loaded(handle=sorted(files), facts={'roles': sorted(files)})

    def unload(self, loaded):
        self.unloaded.append(loaded.handle)

    def predict(self, loaded, task, images, params, contexts):
        if self.gate is not None:
            self.gate.wait(5)
        if self.oom_above is not None and len(images) > self.oom_above:
            raise OutOfMemoryError('CUDA out of memory')
        self.calls.append(len(images))
        if self.answer is not None:
            return self.answer
        return [{'text': f'{image.width}', 'confidence': 1.0, 'conf_profile': []} for image in images]


def png() -> bytes:
    import io

    from PIL import Image
    out = io.BytesIO()
    Image.new('RGB', (12, 8), 'white').save(out, format='PNG')
    return out.getvalue()


def predict(client, count: int):
    body = {'items': [{'id': str(i), 'image': 'img'} for i in range(count)]}
    return client.post('/v1/predict/ocr', headers={SECRET_HEADER: SECRET},
                       files=[('request', ('r.json', json.dumps(body), 'application/json')),
                              ('img', ('img.png', png(), 'image/png'))])


@pytest.fixture
def model():
    return FakeModel()


@pytest.fixture
def client(model, tmp_path):
    return TestClient(create_app(model, secret=SECRET, cache_dir=tmp_path))


@pytest.fixture
def files():
    server = FixtureServer('127.0.0.1', SECRET)
    server.start()
    yield server
    server.stop()


def test_asking_for_the_builtin_weights_loads_them_and_reports_their_facts(client):
    response = client.get('/v1/weights/builtin', headers={SECRET_HEADER: SECRET})
    assert response.status_code == 200
    assert response.json() == {'state': 'loaded', 'facts': {'roles': []}}


def test_a_backend_without_builtin_weights_has_none_to_report(model, client):
    model.builtin = False
    assert client.get('/v1/weights/builtin', headers={SECRET_HEADER: SECRET}).json() == {'state': 'absent'}


def test_a_request_over_the_concurrency_gets_busy_with_retry_after(model, client):
    model.gate = threading.Event()
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(predict, client, 1)
        deadline = time.monotonic() + 5
        while client.get('/health').json()['busy'] == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        second = predict(client, 1)
        model.gate.set()
        assert first.result().status_code == 200
    assert second.status_code == 429
    assert second.headers['Retry-After'] == '1'
    assert second.json()['error'] == {'code': 'busy', 'message': 'the backend is at its concurrency limit',
                                      'retryable': True}


def test_running_out_of_memory_halves_the_batch_until_it_fits(model, client):
    model.oom_above = 2
    model.batch_size = lambda task, params: 8
    response = predict(client, 8)
    assert response.status_code == 200
    assert [r['id'] for r in response.json()['results']] == [str(i) for i in range(8)]
    assert model.calls == [2, 2, 2, 2]


def test_out_of_memory_on_a_single_item_is_a_retryable_503(model, client):
    model.oom_above = 0
    response = predict(client, 1)
    assert response.status_code == 503
    assert response.json()['error']['code'] == 'out_of_memory'
    assert response.json()['error']['retryable'] is True


def test_a_malformed_result_never_leaves_the_backend(model, client):
    model.answer = [{'text': 'x', 'confidence': 1.7}]
    response = predict(client, 1)
    assert response.status_code == 500
    assert response.json()['error']['code'] == 'internal'
    assert 'results[0].confidence: 1.7 is outside [0, 1]' in response.json()['error']['message']


def test_too_few_results_are_refused(model, client):
    model.answer = [{'text': 'x', 'confidence': 1.0}]
    response = predict(client, 2)
    assert response.status_code == 500
    assert 'predict returned 1 results for 2 items' in response.json()['error']['message']


def load(cache: WeightCache, files: FixtureServer, data: bytes, **extra) -> str:
    ref = files.add_file(data)
    version = cache.load(LoadRequest(sha256=ref['sha256'], files={'weights': ref, **extra}))
    assert version.settled.wait(10)
    assert version.state == 'loaded', version.error
    return ref['sha256']


def test_loading_past_max_loaded_unloads_the_least_recently_used(model, files, tmp_path):
    cache = WeightCache(model, tmp_path, SECRET, max_bytes=1 << 30, max_loaded=2)
    a, b = load(cache, files, b'a' * 10), load(cache, files, b'b' * 10)
    with cache.use(a):
        pass
    c = load(cache, files, b'c' * 10)
    assert sorted(cache.loaded_versions()) == sorted([a, c])
    assert cache.state(b) == {'state': 'cached'}
    assert model.unloaded == [['weights']]


def test_unloading_a_version_in_use_waits_for_its_request(model, files, tmp_path):
    cache = WeightCache(model, tmp_path, SECRET, max_bytes=1 << 30, max_loaded=2)
    a = load(cache, files, b'a' * 10)
    with cache.use(a):
        assert cache.unload(a) == [a]
        assert cache.state(a)['state'] == 'loaded'
    assert cache.state(a) == {'state': 'cached'}


def test_the_disk_budget_deletes_the_oldest_file_that_is_not_loaded(model, files, tmp_path):
    cache = WeightCache(model, tmp_path, SECRET, max_bytes=25, max_loaded=1)
    a = load(cache, files, b'a' * 10)
    os.utime(cache.file_path(a), (1, 1))
    b = load(cache, files, b'b' * 10)
    c = load(cache, files, b'c' * 10)
    assert not cache.file_path(a).exists()
    assert cache.file_path(b).exists() and cache.file_path(c).exists()


def test_two_loads_of_one_version_share_one_download(model, files, tmp_path):
    cache = WeightCache(model, tmp_path, SECRET, max_bytes=1 << 30, max_loaded=2)
    ref = files.add_file(os.urandom(1 << 20))
    request = LoadRequest(sha256=ref['sha256'], files={'weights': ref})
    first, second = cache.load(request), cache.load(request)
    assert first is second
    first.settled.wait(10)
    assert [path for path, _ in files.requests] == [f'/files/{ref["sha256"]}']


def test_a_restarted_backend_trusts_its_cached_files(model, files, tmp_path):
    data = b'w' * 100
    load(WeightCache(model, tmp_path, SECRET, max_bytes=1 << 30, max_loaded=2), files, data)
    files.requests.clear()
    load(WeightCache(model, tmp_path, SECRET, max_bytes=1 << 30, max_loaded=2), files, data)
    assert files.requests == []


def test_params_become_the_train_variables_our_train_scripts_read(tmp_path):
    (tmp_path / 'base').mkdir()
    (tmp_path / 'base' / 'weights').write_bytes(b'w')
    (tmp_path / 'params.json').write_text(json.dumps({
        'epochs': 30, 'batch_size': 8, 'augmentation': {'transform': {}},
        'hyperparams': {'learning_rate': 0.0002, 'freeze_backbone': True, 'warmup': 2}}))
    environ = {'TRAIN_PARAMS': str(tmp_path / 'params.json'), 'TRAIN_OUTPUT': str(tmp_path / 'output'),
               'TRAIN_BASE': str(tmp_path / 'base')}
    tunables = [{'key': 'learning_rate', 'env': ['TRAIN_BACKBONE_BASE_LR', 'TRAIN_DECODER_BASE_LR']},
                {'key': 'freeze_backbone', 'env': 'TRAIN_FREEZE_BACKBONE'}]
    assert legacy_env(environ, tunables) == {
        'TRAIN_EPOCHS': '30', 'TRAIN_BATCH': '8', 'TRAIN_AUGMENT': str(tmp_path / 'output' / 'augmentation.json'),
        'TRAIN_BASE_CHECKPOINT': str(tmp_path / 'base' / 'weights'),
        'TRAIN_BACKBONE_BASE_LR': '0.0002', 'TRAIN_DECODER_BASE_LR': '0.0002', 'TRAIN_FREEZE_BACKBONE': 'true',
        'TRAIN_WARMUP': '2'}
    assert json.loads((tmp_path / 'output' / 'augmentation.json').read_text()) == {'transform': {}}


def test_accumulation_and_a_learning_rate_scale_reach_the_tunables_with_those_roles(tmp_path):
    (tmp_path / 'base').mkdir()
    (tmp_path / 'params.json').write_text(json.dumps({
        'batch_size': 4, 'grad_accum_steps': 2, 'lr_scale': 2.0, 'hyperparams': {'encoder_lr': 0.5}}))
    environ = {'TRAIN_PARAMS': str(tmp_path / 'params.json'), 'TRAIN_OUTPUT': str(tmp_path / 'output'),
               'TRAIN_BASE': str(tmp_path / 'base')}
    tunables = [{'key': 'grad_accum_steps', 'role': 'accumulation', 'env': 'TRAIN_GRAD_ACCUM_STEPS'},
                {'key': 'learning_rate', 'role': 'learning_rate', 'env': 'TRAIN_LR'},
                {'key': 'encoder_lr', 'role': 'learning_rate', 'env': 'TRAIN_ENCODER_LR'},
                {'key': 'warmup', 'env': 'TRAIN_WARMUP'}]

    env = legacy_env(environ, tunables, lambda: {'learning_rate': 0.0001, 'encoder_lr': 0.00015, 'warmup': 3})

    assert env == {'TRAIN_BATCH': '4', 'TRAIN_GRAD_ACCUM_STEPS': '2', 'TRAIN_LR': '0.0002', 'TRAIN_ENCODER_LR': '0.5'}


def test_without_roles_or_defaults_the_scaling_is_left_to_the_trainer(tmp_path):
    (tmp_path / 'base').mkdir()
    (tmp_path / 'params.json').write_text(json.dumps({'batch_size': 4, 'grad_accum_steps': 2, 'lr_scale': 2.0}))
    environ = {'TRAIN_PARAMS': str(tmp_path / 'params.json'), 'TRAIN_OUTPUT': str(tmp_path / 'output'),
               'TRAIN_BASE': str(tmp_path / 'base')}

    assert legacy_env(environ, [{'key': 'learning_rate', 'role': 'learning_rate'}]) == {'TRAIN_BATCH': '4'}


OOM_ENTRYPOINT = """
import json, os, pathlib, sys
params = json.loads(pathlib.Path(os.environ['TRAIN_PARAMS']).read_text())
out = pathlib.Path(os.environ['TRAIN_OUTPUT'])
(out / f"tried-{params['batch_size']}").write_text('')
if params['batch_size'] > int(sys.argv[1]):
    print('torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB')
    sys.exit(1)
(out / 'model.ckpt').write_text(f"{params['batch_size']}x{params.get('grad_accum_steps')}")
"""


def fallback_run(tmp_path, monkeypatch, fits: int, batch: int = 8, accumulation: int | None = None):
    from deeplabel_backend import fallback
    (tmp_path / 'out').mkdir()
    (tmp_path / 'params.json').write_text(json.dumps({'batch_size': batch, 'grad_accum_steps': accumulation}))
    script = tmp_path / 'train.py'
    script.write_text(OOM_ENTRYPOINT)
    monkeypatch.setenv('TRAIN_PARAMS', str(tmp_path / 'params.json'))
    monkeypatch.setenv('TRAIN_OUTPUT', str(tmp_path / 'out'))
    return fallback.run([sys.executable, str(script), str(fits)], tmp_path / 'params.json', tmp_path / 'out')


def test_a_run_out_of_memory_halves_its_batch_and_doubles_accumulation_until_it_fits(tmp_path, monkeypatch):
    assert fallback_run(tmp_path, monkeypatch, fits=2) == 0

    out = tmp_path / 'out'
    assert (out / 'model.ckpt').read_text() == '2x4'
    assert sorted(p.name for p in out.iterdir()) == ['fallback.json', 'model.ckpt', 'tried-2']
    assert json.loads((out / 'fallback.json').read_text()) == {
        'asked': {'batch_size': 8, 'grad_accum_steps': 1}, 'trained': {'batch_size': 2, 'grad_accum_steps': 4}}


def test_a_run_that_fits_leaves_no_fallback_note(tmp_path, monkeypatch):
    assert fallback_run(tmp_path, monkeypatch, fits=8, accumulation=2) == 0

    assert (tmp_path / 'out' / 'model.ckpt').read_text() == '8x2'
    assert not (tmp_path / 'out' / 'fallback.json').exists()


def test_a_run_out_of_memory_at_batch_one_fails_as_it_would_have(tmp_path, monkeypatch):
    assert fallback_run(tmp_path, monkeypatch, fits=0, batch=2) == 1

    assert json.loads((tmp_path / 'params.json').read_text())['batch_size'] == 1
    assert not (tmp_path / 'out' / 'fallback.json').exists()


def test_a_failure_that_is_not_memory_is_not_retried(tmp_path):
    from deeplabel_backend import fallback
    (tmp_path / 'out').mkdir()
    (tmp_path / 'params.json').write_text(json.dumps({'batch_size': 8}))

    code = fallback.run([sys.executable, '-c', 'import sys; print("ValueError: bad snapshot"); sys.exit(3)'],
                        tmp_path / 'params.json', tmp_path / 'out')

    assert code == 3
    assert json.loads((tmp_path / 'params.json').read_text())['batch_size'] == 8


def test_the_card_is_named_by_the_id_batch_tables_use():
    from deeplabel_backend import gpu
    assert gpu.parse('NVIDIA GeForce RTX 5090, 32607\n') == {
        'name': 'NVIDIA GeForce RTX 5090', 'card': 'rtx5090', 'count': 1, 'memory_gb': 32}
    found = gpu.parse('NVIDIA H200, 143771\nNVIDIA H200, 143771\n')
    assert (found['card'], found['count'], found['memory_gb']) == ('h200', 2, 140)
    assert gpu.parse('NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887')['card'] == 'rtx6000bb'
    assert gpu.parse('Some Future Card, 1024')['card'] is None
    assert gpu.parse('') is None


def test_info_declares_the_entrypoint_under_the_fallback(model):
    model.train_info = lambda: {'tasks': ['ocr'], 'entrypoint': ['/srv/train', 'ocr']}
    from deeplabel_backend.app import info

    assert info(model)['train']['entrypoint'] == [sys.executable, '-m', 'deeplabel_backend.fallback', '/srv/train', 'ocr']


def test_a_training_running_when_the_backend_stopped_comes_back_as_an_error(model, tmp_path):
    run = tmp_path / 'trainings' / '7-1'
    run.mkdir(parents=True)
    (run / 'state.json').write_text(json.dumps({'id': '7-1', 'request': {}, 'state': 'running',
                                                'phase': 'training', 'error': None}))
    model.train_info = lambda: {'tasks': ['ocr'], 'entrypoint': ['true']}
    client = TestClient(create_app(model, secret=SECRET, cache_dir=tmp_path))
    state = client.get('/v1/train/7-1', headers={SECRET_HEADER: SECRET}).json()
    assert (state['state'], state['error']) == ('error', 'backend restarted')


def test_a_backend_without_a_secret_refuses_to_start(model, tmp_path, monkeypatch):
    monkeypatch.delenv('MODEL_BACKEND_SECRET', raising=False)
    with pytest.raises(RuntimeError, match='MODEL_BACKEND_SECRET is not set'):
        create_app(model, cache_dir=tmp_path)


def test_a_resumed_run_out_of_memory_keeps_its_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv('TRAIN_RESUME', str(tmp_path / 'out'))
    (tmp_path / 'out').mkdir()
    (tmp_path / 'out' / 'last.ckpt').write_text('epoch 40')
    (tmp_path / 'params.json').write_text(json.dumps({'batch_size': 4}))
    script = tmp_path / 'train.py'
    script.write_text(OOM_ENTRYPOINT)
    monkeypatch.setenv('TRAIN_PARAMS', str(tmp_path / 'params.json'))
    monkeypatch.setenv('TRAIN_OUTPUT', str(tmp_path / 'out'))
    from deeplabel_backend import fallback

    assert fallback.run([sys.executable, str(script), '2'], tmp_path / 'params.json', tmp_path / 'out') == 0
    assert (tmp_path / 'out' / 'last.ckpt').read_text() == 'epoch 40'


def test_a_training_runs_with_expandable_segments_unless_it_set_its_own(tmp_path, monkeypatch):
    from deeplabel_backend import fallback
    monkeypatch.delenv('PYTORCH_CUDA_ALLOC_CONF', raising=False)
    (tmp_path / 'out').mkdir()
    (tmp_path / 'params.json').write_text(json.dumps({'batch_size': 1}))
    show = [sys.executable, '-c', 'import os; open(os.environ["SEEN"], "w").write(os.environ["PYTORCH_CUDA_ALLOC_CONF"])']
    monkeypatch.setenv('SEEN', str(tmp_path / 'seen'))

    fallback.run(show, tmp_path / 'params.json', tmp_path / 'out')
    assert (tmp_path / 'seen').read_text() == 'expandable_segments:True'

    monkeypatch.setenv('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:512')
    fallback.run(show, tmp_path / 'params.json', tmp_path / 'out')
    assert (tmp_path / 'seen').read_text() == 'max_split_size_mb:512'


def test_a_base_is_laid_out_by_role_a_builtin_base_as_its_marker_and_none_left_empty(tmp_path):
    from deeplabel_backend.schemas import BaseWeights
    from deeplabel_backend.train import lay_out_base
    stored = tmp_path / 'stored'
    stored.write_bytes(b'w')
    ref = {'sha256': 'a' * 64, 'size': 1, 'url': 'http://x/a'}

    lay_out_base(tmp_path / 'files', BaseWeights(sha256='a' * 64, files={'weights': ref}), lambda r: stored)
    lay_out_base(tmp_path / 'builtin', BaseWeights(builtin=True), lambda r: stored)
    lay_out_base(tmp_path / 'scratch', None, lambda r: stored)

    assert (tmp_path / 'files' / 'weights').read_bytes() == b'w'
    assert [p.name for p in (tmp_path / 'builtin').iterdir()] == ['builtin']
    assert list((tmp_path / 'scratch').iterdir()) == []
