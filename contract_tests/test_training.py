import json
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import pytest
from conftest import sha256
from fixtures import snapshot

FINISHED = ('done', 'error', 'cancelled')


@pytest.fixture
def trainable(pytestconfig, info):
    if not pytestconfig.getoption('--train'):
        pytest.skip('needs --train')
    if 'train' not in info['capabilities']:
        pytest.skip('the backend does not train')
    return info['train']


def default_batch(train: dict, target: str) -> int:
    """The batch DeepLabel sends when nobody chose one: the backend's own default."""
    return ((train.get('resources') or {}).get('default_batch') or {}).get(target) or 2


def start(backend, files, train: dict, name: str) -> str:
    snapshot_files, manifest = snapshot(train['snapshot'])
    run_id = uuid.uuid4().hex[:12]
    url = files.add_tree(run_id, snapshot_files, manifest)
    response = backend.post('/v1/train', json={
        'run': {'id': run_id, 'attempt': 1, 'name': name, 'owner': 'contract'},
        'task': train['tasks'][0], 'snapshot': {'manifest_url': url},
        'params': {'epochs': 1, 'batch_size': default_batch(train, 'local'), 'gpus': 1, 'hyperparams': {}}})
    assert response.status_code == 202, response.text
    return response.json()['train_id']


def wait(backend, train_id: str, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        state = backend.get(f'/v1/train/{train_id}').json()
        if state['state'] in FINISHED or time.monotonic() > deadline:
            return state
        time.sleep(0.5)


def test_a_fixture_snapshot_trains_to_done_with_verified_artifacts(backend, files, trainable, pytestconfig):
    train_id = start(backend, files, trainable, 'contract run')
    state = wait(backend, train_id, pytestconfig.getoption('--train-timeout'))
    assert state['state'] == 'done', state
    progress = [json.loads(line) for line in backend.get(f'/v1/train/{train_id}/progress').text.splitlines()]
    assert any('epoch' in point for point in progress)
    later = backend.get(f'/v1/train/{train_id}/progress', params={'from': len(progress) - 1}).text.splitlines()
    assert len(later) == 1
    listing = backend.get(f'/v1/train/{train_id}/artifacts').json()
    roles = {f['role'] for f in listing['files']}
    assert 'weights' in roles
    for entry in listing['files']:
        data = backend.get(f'/v1/train/{train_id}/artifacts/{entry["path"]}').content
        assert (len(data), sha256(data)) == (entry['size'], entry['sha256'])
    assert backend.http.delete(f'/v1/train/{train_id}').status_code == 200
    assert backend.get(f'/v1/train/{train_id}').status_code == 404


def test_starting_the_same_attempt_twice_is_one_training(backend, files, trainable):
    snapshot_files, manifest = snapshot(trainable['snapshot'])
    run_id = uuid.uuid4().hex[:12]
    body = {'run': {'id': run_id, 'attempt': 1, 'name': 'contract twice'}, 'task': trainable['tasks'][0],
            'snapshot': {'manifest_url': files.add_tree(run_id, snapshot_files, manifest)}}
    first, second = (backend.post('/v1/train', json=body) for _ in range(2))
    assert first.json()['train_id'] == second.json()['train_id']
    wait(backend, first.json()['train_id'], 120)
    backend.http.delete(f'/v1/train/{first.json()["train_id"]}')


def test_a_training_can_be_cancelled(backend, files, trainable, pytestconfig):
    train_id = start(backend, files, trainable, 'contract cancel [slow]')
    cancelled = backend.post(f'/v1/train/{train_id}/cancel')
    assert cancelled.status_code == 200
    state = wait(backend, train_id, 60)
    if state['state'] == 'done':
        pytest.skip('the training finished before the cancel reached it')
    assert state['state'] == 'cancelled', state
    backend.http.delete(f'/v1/train/{train_id}')


SQUASHFS_TOOLS = 'contract-squashfs-tools:1'
SQUASHFS_TOOLS_DOCKERFILE = ('FROM ubuntu:24.04\n'
                             'RUN apt-get update && apt-get install -y --no-install-recommends squashfs-tools\n')


def train_offline(image: str, trainable: dict, pytestconfig) -> None:
    """The entrypoint on a fixture snapshot inside `image`, as a cluster node runs it: no
    network, an unprivileged user, the snapshot, base and output mounted, the GPU where
    the machine has one."""
    run = Path(tempfile.mkdtemp(prefix='contract-offline-'))
    run.chmod(0o755)
    snapshot_files, manifest = snapshot(trainable['snapshot'])
    for path, data in snapshot_files.items():
        (run / 'snapshot' / path).parent.mkdir(parents=True, exist_ok=True)
        (run / 'snapshot' / path).write_bytes(data)
    listing = [{'path': p, 'sha256': sha256(d), 'size': len(d)} for p, d in snapshot_files.items()]
    (run / 'snapshot' / 'manifest.json').write_text(json.dumps({**manifest, 'files': listing}))
    (run / 'base').mkdir()
    (run / 'output').mkdir()
    (run / 'output').chmod(0o777)
    (run / 'params.json').write_text(json.dumps({'epochs': 1, 'batch_size': default_batch(trainable, 'cluster'), 'gpus': 1, 'hyperparams': {},
                                                 'task': trainable['tasks'][0], 'run': {'id': 'offline'}}))
    env = {'TRAIN_SNAPSHOT': '/run/snapshot', 'TRAIN_OUTPUT': '/run/output', 'TRAIN_BASE': '/run/base',
           'TRAIN_PARAMS': '/run/params.json', 'TRAIN_GPUS': '1'}
    command = ['docker', 'run', '--rm', '--network', 'none', '--user', '65534:65534', '--shm-size', '8g',
               # A cluster node has a GPU; so does this run, where the machine has one.
               *(['--gpus', 'all'] if shutil.which('nvidia-smi') else []),
               '-v', f'{run}:/run',
               *[arg for key, value in env.items() for arg in ('-e', f'{key}={value}')],
               '--entrypoint', trainable['entrypoint'][0], image, *trainable['entrypoint'][1:]]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=pytestconfig.getoption('--train-timeout'))
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert (run / 'output' / 'progress.jsonl').read_text().strip()
    artifacts = json.loads((run / 'output' / 'artifacts.json').read_text()) if (run / 'output' / 'artifacts.json').is_file() \
        else {'files': [{'role': 'weights', 'path': 'model.ckpt'}]}
    for entry in artifacts['files']:
        assert (run / 'output' / entry['path']).is_file(), entry


def test_the_entrypoint_trains_offline_in_its_image(pytestconfig, trainable):
    image = pytestconfig.getoption('--train-image')
    if not image:
        pytest.skip('needs --train-image')
    train_offline(image, trainable, pytestconfig)


def test_the_entrypoint_trains_offline_from_its_cluster_squashfs(pytestconfig, trainable):
    """The squashfs Slurm's Pyxis/enroot runs, imported back as a bare filesystem: no
    image metadata, so the environment is only what the file itself carries."""
    squashfs = pytestconfig.getoption('--train-squashfs')
    if not squashfs:
        pytest.skip('needs --train-squashfs')
    image = f'contract-squashfs:{uuid.uuid4().hex[:12]}'
    # Unpacked as root, so the files keep the owners and modes the cluster sees.
    subprocess.run(['docker', 'build', '-q', '-t', SQUASHFS_TOOLS, '-'], input=SQUASHFS_TOOLS_DOCKERFILE,
                   text=True, check=True, capture_output=True)
    squashfs = Path(squashfs).resolve()
    unpacked = subprocess.Popen(['docker', 'run', '--rm', '-v', f'{squashfs.parent}:/in:ro', SQUASHFS_TOOLS, 'sh', '-c',
                                 f'unsquashfs -q -d /root/fs /in/{squashfs.name} >/dev/null && tar -C /root/fs -c .'],
                                stdout=subprocess.PIPE)
    subprocess.run(['docker', 'import', '-', image], stdin=unpacked.stdout, check=True, capture_output=True)
    assert unpacked.wait() == 0
    try:
        train_offline(image, trainable, pytestconfig)
    finally:
        subprocess.run(['docker', 'rmi', image], capture_output=True)
