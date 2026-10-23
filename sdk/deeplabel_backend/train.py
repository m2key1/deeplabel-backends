import functools
import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import errors, fallback, transfer
from .model import Model
from .schemas import BaseWeights, FileRef, TrainRequest
from .weights import WeightCache

STATE_NAME = 'state.json'
LOG_NAME = 'train.log'
PROGRESS_NAME = 'progress.jsonl'
METRICS_NAME = 'metrics.json'
# In TRAIN_BASE, in place of files: the run starts from the backend's builtin weights.
BUILTIN_MARKER = 'builtin'
ARTIFACTS_NAME = 'artifacts.json'
DEFAULT_ARTIFACTS = {'files': [{'role': 'weights', 'path': 'model.ckpt'}]}
TERM_GRACE_SECONDS = 30
LOG_TAIL_LINES = 40
FINISHED = ('done', 'error', 'cancelled')

_EXCEPTION_LINE = re.compile(r'\b[A-Za-z_.]*(?:Error|Exception)\b[: ].*|CUDA out of memory.*')
_GENERIC_ERROR_LINE = re.compile(r'error: .*')


def extract_error(log_text: str) -> str | None:
    """The last log line naming an exception, else the last 'error:' line."""
    lines = list(reversed(log_text.splitlines()))
    for pattern in (_EXCEPTION_LINE, _GENERIC_ERROR_LINE):
        for line in lines:
            match = pattern.search(line)
            if match:
                return match.group(0).strip()[:300]
    return None


class Slots:
    """The backend's concurrency: predict requests and running trainings share it."""

    def __init__(self, limit: int):
        self.limit = limit
        self.used = 0
        self.lock = threading.Lock()

    def acquire(self) -> None:
        with self.lock:
            if self.used >= self.limit:
                raise errors.busy()
            self.used += 1

    def release(self) -> None:
        with self.lock:
            self.used -= 1


def entrypoint_env(run_dir: Path, gpus: int) -> dict[str, str]:
    """Section 6.6: the five variables the training entrypoint is started with."""
    return {'TRAIN_SNAPSHOT': str(run_dir / 'snapshot'), 'TRAIN_OUTPUT': str(run_dir / 'output'),
            'TRAIN_BASE': str(run_dir / 'base'), 'TRAIN_PARAMS': str(run_dir / 'params.json'),
            'TRAIN_GPUS': str(gpus)}


def loader_workers() -> int:
    """The data-loading processes a run's cores allow: every core it was given - Slurm's
    allocation, not the node's count - less one for the training process itself."""
    return max(1, len(os.sched_getaffinity(0)) - 1)


def run_hyperparams(params: dict, tunables: list[dict], defaults: Callable[[], dict] | None = None) -> dict:
    """The hyperparameters a run trains with: those it chose, plus grad_accum_steps on the
    tunable whose role is 'accumulation' and lr_scale on each tunable whose role is
    'learning_rate' - applied to its default, from `defaults` - wherever the run did not
    set that tunable itself, which then stands as given (6.9)."""
    chosen = dict(params.get('hyperparams') or {})
    found = functools.cache(defaults) if defaults is not None else None
    scaled = {}
    for knob in tunables:
        if knob['key'] in chosen:
            continue
        if knob.get('role') == 'accumulation' and params.get('grad_accum_steps'):
            scaled[knob['key']] = int(params['grad_accum_steps'])
        elif knob.get('role') == 'learning_rate' and params.get('lr_scale') and found is not None:
            base = found().get(knob['key'])
            if isinstance(base, (int, float)) and not isinstance(base, bool):
                scaled[knob['key']] = base * float(params['lr_scale'])
    return {**scaled, **chosen}


def lay_out_base(base: Path, weights: BaseWeights | None, fetch: Callable[[FileRef], Path]) -> None:
    """TRAIN_BASE as the entrypoint reads it (6.6): the base version's files by role, the
    builtin marker, or nothing for a run from scratch."""
    base.mkdir()
    if weights is not None and weights.builtin:
        (base / BUILTIN_MARKER).touch()
    for role, ref in (weights.files if weights is not None else {}).items():
        shutil.copyfile(fetch(ref), base / role)


def legacy_env(environ: dict[str, str], tunables: list[dict],
               defaults: Callable[[], dict] | None = None) -> dict[str, str]:
    """params.json translated into the TRAIN_* variables DeepLabel's own train scripts read
    (TRAIN_EPOCHS, TRAIN_BATCH, TRAIN_AUGMENT, TRAIN_BASE_CHECKPOINT, one per tunable, the
    run's scaling included - see run_hyperparams). A tunable names its variables in 'env';
    without it the key is upper-cased."""
    params = json.loads(Path(environ['TRAIN_PARAMS']).read_text(encoding='utf-8'))
    output = Path(environ['TRAIN_OUTPUT'])
    env = {}
    if params.get('epochs'):
        env['TRAIN_EPOCHS'] = str(params['epochs'])
    if params.get('batch_size'):
        env['TRAIN_BATCH'] = str(params['batch_size'])
    if params.get('augmentation'):
        augmentation = output / 'augmentation.json'
        output.mkdir(parents=True, exist_ok=True)
        augmentation.write_text(json.dumps(params['augmentation']), encoding='utf-8')
        env['TRAIN_AUGMENT'] = str(augmentation)
    base = Path(environ['TRAIN_BASE']) / 'weights'
    if base.is_file():
        env['TRAIN_BASE_CHECKPOINT'] = str(base)
    names = {t['key']: t.get('env') for t in tunables}
    for key, value in run_hyperparams(params, tunables, defaults).items():
        targets = names.get(key) or f'TRAIN_{key.upper()}'
        for name in [targets] if isinstance(targets, str) else targets:
            env[name] = json.dumps(value) if isinstance(value, (list, dict, bool)) else str(value)
    return env


class ProgressLog:
    """Appends progress events (format deeplabel-progress/1), flushed as they happen."""

    def __init__(self, output: Path):
        output.mkdir(parents=True, exist_ok=True)
        self.handle = open(output / PROGRESS_NAME, 'a', encoding='utf-8')

    def write(self, **point) -> None:
        """One point, stamped with the time it was written unless it carries its own; a
        `ts` of None leaves it unstamped."""
        stamped = {'ts': time.time(), **point}
        if stamped['ts'] is None:
            del stamped['ts']
        self.handle.write(json.dumps(stamped) + '\n')
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


# What every trainer reports beside its own metrics, as a train block's `metrics` declares
# them (6.9): the learning rate, the most GPU memory the run has held, and the time.
STANDARD_METRICS = [
    {'key': 'lr', 'label': 'learning rate', 'unit': 'number'},
    {'key': 'gpu/peak_memory_gb', 'label': 'peak GPU memory', 'unit': 'gb', 'better': 'lower'},
]


def system_metrics(optimizer=None) -> dict:
    """The standard metrics a trainer can read from inside its own process: the first
    parameter group's learning rate and the CUDA allocator's high-water mark."""
    point = {}
    if optimizer is not None and getattr(optimizer, 'param_groups', None):
        point['lr'] = float(optimizer.param_groups[0]['lr'])
    try:
        import torch
        if torch.cuda.is_available():
            point['gpu/peak_memory_gb'] = round(torch.cuda.max_memory_allocated() / (1 << 30), 2)
    except ImportError:
        pass
    return point


def write_outputs(output: Path, metrics: dict, files: list[dict], **extra) -> None:
    """metrics.json and artifacts.json, the two closing files of the entrypoint contract."""
    output.mkdir(parents=True, exist_ok=True)
    (output / METRICS_NAME).write_text(json.dumps(metrics), encoding='utf-8')
    (output / ARTIFACTS_NAME).write_text(json.dumps({'files': files, **extra}), encoding='utf-8')


@dataclass
class Training:
    id: str
    dir: Path
    request: dict
    state: str = 'queued'
    phase: str = ''
    error: str | None = None
    process: subprocess.Popen | None = None
    cancelled: threading.Event = field(default_factory=threading.Event)
    hashes: dict = field(default_factory=dict)

    @property
    def output(self) -> Path:
        return self.dir / 'output'

    def save(self) -> None:
        (self.dir / STATE_NAME).write_text(json.dumps(
            {'id': self.id, 'request': self.request, 'state': self.state, 'phase': self.phase,
             'error': self.error}), encoding='utf-8')

    def progress_lines(self) -> list[str]:
        path = self.output / PROGRESS_NAME
        if not path.is_file():
            return []
        return [line for line in path.read_text(encoding='utf-8', errors='replace').splitlines() if line.strip()]

    def log_text(self) -> str:
        path = self.dir / LOG_NAME
        return path.read_text(encoding='utf-8', errors='replace') if path.is_file() else ''

    def public(self) -> dict:
        tail = self.log_text().splitlines()[-LOG_TAIL_LINES:]
        return {'train_id': self.id, 'state': self.state, 'phase': self.phase, 'error': self.error,
                'progress_lines': len(self.progress_lines()), 'log_tail': '\n'.join(tail)}


class TrainManager:
    """Section 6.5: local trainings, each in a directory of its own that survives restarts.
    A training that was running when the backend stopped comes back as an error."""

    def __init__(self, model: Model, directory: Path, cache: WeightCache, secret: str, slots: Slots):
        self.model = model
        self.directory = directory
        self.cache = cache
        self.secret = secret
        self.slots = slots
        self.lock = threading.Lock()
        self.trainings: dict[str, Training] = {}
        directory.mkdir(parents=True, exist_ok=True)
        for state_file in directory.glob(f'*/{STATE_NAME}'):
            saved = json.loads(state_file.read_text(encoding='utf-8'))
            training = Training(saved['id'], state_file.parent, saved['request'], saved['state'],
                                saved['phase'], saved['error'])
            if training.state not in FINISHED:
                training.state, training.error = 'error', 'backend restarted'
                training.save()
            self.trainings[training.id] = training

    def get(self, train_id: str) -> Training:
        training = self.trainings.get(train_id)
        if training is None:
            raise errors.BackendError(404, 'not_found', f'no training {train_id}')
        return training

    def start(self, request: TrainRequest) -> Training:
        tasks = (self.model.train_info() or {}).get('tasks', [])
        if request.task not in tasks:
            raise errors.bad_request(f'this backend does not train {request.task!r}')
        train_id = re.sub(r'[^A-Za-z0-9_.-]', '_', f'{request.run.id}-{request.run.attempt}')
        with self.lock:
            if train_id in self.trainings:
                return self.trainings[train_id]
            self.slots.acquire()
            run_dir = self.directory / train_id
            shutil.rmtree(run_dir, ignore_errors=True)
            run_dir.mkdir(parents=True)
            training = Training(train_id, run_dir, request.model_dump(mode='json'))
            training.save()
            self.trainings[train_id] = training
        threading.Thread(target=self._run, args=(training, request), daemon=True).start()
        return training

    def _set(self, training: Training, **fields) -> None:
        for key, value in fields.items():
            setattr(training, key, value)
        training.save()

    def _run(self, training: Training, request: TrainRequest) -> None:
        try:
            self._set(training, state='fetching', phase='fetching snapshot')
            transfer.fetch_tree(request.snapshot.manifest_url, training.dir / 'snapshot', self.secret,
                                on_progress=lambda done, total: setattr(
                                    training, 'phase', f'fetching snapshot {done}/{total}'))
            if request.base_weights is not None and request.base_weights.files:
                self._set(training, phase='fetching base weights')
            lay_out_base(training.dir / 'base', request.base_weights, self.cache.fetch)
            params = {**request.params.model_dump(mode='json'), 'task': request.task,
                      'run': request.run.model_dump(mode='json')}
            (training.dir / 'params.json').write_text(json.dumps(params), encoding='utf-8')
            training.output.mkdir()
            with self.lock:
                if training.cancelled.is_set():
                    return
                self._set(training, state='running', phase='training')
                with open(training.dir / LOG_NAME, 'ab') as log:
                    training.process = subprocess.Popen(
                        fallback.wrap(self.model.train_command()), cwd=training.dir, stdout=log, stderr=subprocess.STDOUT,
                        env=os.environ | entrypoint_env(training.dir, request.params.gpus),
                        start_new_session=True)
            code = training.process.wait()
            if training.cancelled.is_set():
                return
            if code == 0:
                self._set(training, state='done', phase='')
            else:
                self._set(training, state='error', error=extract_error(training.log_text())
                          or f'the training entrypoint exited with code {code}')
        except Exception as error:
            if not training.cancelled.is_set():
                self._set(training, state='error', error=str(error) or type(error).__name__)
        finally:
            if training.cancelled.is_set():
                self._set(training, state='cancelled', phase='')
            self.slots.release()

    def cancel(self, training: Training) -> Training:
        with self.lock:
            if training.state in FINISHED:
                return training
            training.cancelled.set()
            process = training.process
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            threading.Thread(target=self._kill_later, args=(process,), daemon=True).start()
        return training

    @staticmethod
    def _kill_later(process: subprocess.Popen) -> None:
        try:
            process.wait(TERM_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)

    def artifacts(self, training: Training) -> dict:
        listing_file = training.output / ARTIFACTS_NAME
        listing = (json.loads(listing_file.read_text(encoding='utf-8')) if listing_file.is_file()
                   else DEFAULT_ARTIFACTS)
        files = []
        for entry in listing.get('files', []):
            path = self.artifact_path(training, entry['path'], listed_only=False)
            if path is None:
                continue
            stat = path.stat()
            key = (entry['path'], stat.st_size, stat.st_mtime_ns)
            if key not in training.hashes:
                training.hashes[key] = transfer.file_sha256(path)
            files.append({**entry, 'size': stat.st_size, 'sha256': training.hashes[key]})
        return {**listing, 'files': files}

    def artifact_path(self, training: Training, name: str, listed_only: bool = True) -> Path | None:
        if listed_only:
            listed = {f['path'] for f in self.artifacts(training)['files']} | {METRICS_NAME, PROGRESS_NAME,
                                                                           fallback.FALLBACK_NAME}
            if name not in listed:
                return None
        path = (training.output / name).resolve()
        if not path.is_relative_to(training.output.resolve()) or not path.is_file():
            return None
        return path

    def delete(self, training: Training) -> None:
        self.cancel(training)
        if training.process is not None:
            deadline = time.monotonic() + TERM_GRACE_SECONDS + 5
            while training.process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.1)
        with self.lock:
            self.trainings.pop(training.id, None)
        shutil.rmtree(training.dir, ignore_errors=True)
