import hashlib
import json
import os
import shlex
import socket
import subprocess
import tempfile
import time
from pathlib import Path

import httpx
import pytest
from deeplabel_backend.schemas import ErrorBody
from fixture_server import SECRET_HEADER, FixtureServer
from fixtures import png, task_context, task_images, task_params

LOAD_TIMEOUT = 600
START_TIMEOUT = 60


def pytest_addoption(parser):
    group = parser.getgroup('model backend contract')
    group.addoption('--backend-url', help='base URL of the backend under test')
    group.addoption('--backend-cmd', help='start this command as the backend instead (PORT, '
                                          'MODEL_BACKEND_SECRET and MODEL_CACHE_DIR are set for it)')
    group.addoption('--backend-secret', default='contract-secret')
    group.addoption('--weights-file', help='a weights file the backend can load')
    group.addoption('--weights-file-lm', help='the lm file belonging to it')
    group.addoption('--weights-file-lm-dict', help='the lm_dict file belonging to it')
    group.addoption('--serve-host', default='127.0.0.1', help='address the backend reaches this suite at')
    group.addoption('--train', action='store_true', help='also run a training on a fixture snapshot')
    group.addoption('--train-image', help='also run the training entrypoint in this image with no network')
    group.addoption('--train-squashfs', help='also run it from this cluster squashfs (cluster-image.sh)')
    group.addoption('--train-timeout', type=float, default=600)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def _wait_healthy(url: str, process: subprocess.Popen | None) -> None:
    deadline = time.monotonic() + START_TIMEOUT
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise pytest.UsageError(f'the backend command exited with {process.returncode}')
        try:
            if httpx.get(f'{url}/health', timeout=2).status_code == 200:
                return
        except httpx.TransportError:
            pass
        time.sleep(0.2)
    raise pytest.UsageError(f'{url}/health did not answer within {START_TIMEOUT} s')


def pytest_configure(config):
    config.backend_process = None
    url, command = config.getoption('--backend-url'), config.getoption('--backend-cmd')
    if not url and not command:
        config.backend_url = None
        return
    if command:
        port = _free_port()
        url = f'http://127.0.0.1:{port}'
        env = os.environ | {'PORT': str(port), 'HOST': '127.0.0.1',
                            'MODEL_BACKEND_SECRET': config.getoption('--backend-secret'),
                            'MODEL_CACHE_DIR': tempfile.mkdtemp(prefix='contract-cache-')}
        config.backend_process = subprocess.Popen(shlex.split(command), env=env)
    config.backend_url = url.rstrip('/')
    _wait_healthy(config.backend_url, config.backend_process)
    response = httpx.get(f'{config.backend_url}/v1/info', timeout=10,
                         headers={SECRET_HEADER: config.getoption('--backend-secret')})
    config.backend_info = response.json() if response.status_code == 200 else {}


def pytest_unconfigure(config):
    if getattr(config, 'backend_process', None) is not None:
        config.backend_process.terminate()
        config.backend_process.wait(10)


def pytest_collection_modifyitems(config, items):
    if config.backend_url is None:
        skip = pytest.mark.skip(reason='needs --backend-url or --backend-cmd')
        for item in items:
            item.add_marker(skip)


def pytest_generate_tests(metafunc):
    if 'task' in metafunc.fixturenames:
        info = getattr(metafunc.config, 'backend_info', None) or {}
        metafunc.parametrize('task', info.get('tasks') or ['none'])


class Backend:
    def __init__(self, url: str, secret: str):
        self.url = url
        self.secret = secret
        self.http = httpx.Client(base_url=url, headers={SECRET_HEADER: secret}, timeout=120)

    def get(self, path: str, **kwargs) -> httpx.Response:
        return self.http.get(path, **kwargs)

    def post(self, path: str, **kwargs) -> httpx.Response:
        return self.http.post(path, **kwargs)

    def predict(self, task: str, body: dict, images: list[bytes]) -> httpx.Response:
        files = [('request', ('request.json', json.dumps(body), 'application/json'))]
        files += [(f'img{i}', (f'img{i}.png', data, 'image/png')) for i, data in enumerate(images)]
        return self.http.post(f'/v1/predict/{task}', files=files)

    def wait_weights(self, sha: str, timeout: float = LOAD_TIMEOUT) -> dict:
        deadline = time.monotonic() + timeout
        while True:
            state = self.get(f'/v1/weights/{sha}').json()
            if state['state'] in ('loaded', 'error') or time.monotonic() > deadline:
                return state
            time.sleep(0.5)


def assert_error(response: httpx.Response, status: int, code: str) -> dict:
    assert response.status_code == status, response.text
    body = ErrorBody.model_validate(response.json())
    assert body.error.code == code, response.text
    assert body.error.message
    return response.json()['error']


@pytest.fixture(scope='session')
def backend(pytestconfig) -> Backend:
    return Backend(pytestconfig.backend_url, pytestconfig.getoption('--backend-secret'))


@pytest.fixture(scope='session')
def info(pytestconfig) -> dict:
    return pytestconfig.backend_info


@pytest.fixture(scope='session')
def files(pytestconfig):
    server = FixtureServer(pytestconfig.getoption('--serve-host'), pytestconfig.getoption('--backend-secret'))
    server.start()
    yield server
    server.stop()


def _junk(size: int = 3 << 20) -> bytes:
    return os.urandom(size)


@pytest.fixture(scope='session')
def weights_files(pytestconfig, info) -> dict[str, bytes] | None:
    """The files of one loadable version: the ones given on the command line, or for a
    backend whose weights are optional, random bytes."""
    given = {role: pytestconfig.getoption(option) for role, option in
             (('weights', '--weights-file'), ('lm', '--weights-file-lm'), ('lm_dict', '--weights-file-lm-dict'))}
    if given['weights']:
        return {role: Path(path).read_bytes() for role, path in given.items() if path}
    if info['weights']['roles'].get('weights') != 'required':
        return {'weights': _junk()}
    return None


def load_request(files_server: FixtureServer, info: dict, files: dict[str, bytes], **refs) -> dict:
    references = {role: files_server.add_file(data) for role, data in files.items()} | refs
    return {'sha256': references['weights']['sha256'], 'model_type': info['model_type'], 'files': references}


@pytest.fixture(scope='session')
def loaded(backend, info, files, weights_files) -> dict | None:
    """The weights block predict requests carry: a version loaded once for the session,
    or None for a builtin backend tested without weights."""
    if weights_files is None:
        if info['weights']['builtin']:
            return None
        pytest.skip('needs --weights-file: this backend has no builtin weights')
    request = load_request(files, info, weights_files)
    response = backend.post('/v1/weights/load', json=request)
    assert response.status_code in (200, 202), response.text
    state = backend.wait_weights(request['sha256'])
    assert state['state'] == 'loaded', state
    return {'sha256': request['sha256'],
            'files': {role: ref['sha256'] for role, ref in request['files'].items() if role != 'weights'}}


def max_items(info: dict, task: str) -> int | None:
    return info['tasks_detail'].get(task, {}).get('batch', {}).get('max_items')


@pytest.fixture
def predict_body(info, loaded):
    def build(task: str, count: int = 2, capped: bool = True, **params) -> tuple[dict, list[bytes]]:
        if capped:
            count = min(count, max_items(info, task) or count)
        images = [png(image) for image in task_images(task)]
        items = [{'id': f'item-{count - i}', 'image': f'img{i % len(images)}', 'context': task_context(task, i)}
                 for i in range(count)]
        body = {'weights': loaded, 'params': task_params(task, info['tasks_detail'].get(task, {})) | params,
                'items': items}
        return body, images[:min(count, len(images))]
    return build


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
