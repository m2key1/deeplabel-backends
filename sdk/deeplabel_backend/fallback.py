"""python -m deeplabel_backend.fallback <entrypoint argv...>

Runs a training entrypoint and, when it dies of running out of GPU memory, runs it again
with half the batch and twice the gradient accumulation, so the effective batch stays what
was asked for (6.9). Down to a batch of 1; then the run fails as it would have. Every
retry starts the entrypoint afresh in an emptied TRAIN_OUTPUT - a resumed run keeps it, its
checkpoint is there - and what the run finally trained at is written to
TRAIN_OUTPUT/fallback.json for DeepLabel to record.

The backend runs every training through it, on the backend and inside its cluster image
alike, so material denser than what a card's batch was measured on does not end a run.
"""
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

FALLBACK_NAME = 'fallback.json'
OUT_OF_MEMORY = re.compile(r'CUDA out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED')


# Page and line batches vary in shape, which strands memory in the caching allocator's fixed
# segments: a 20 GB tensor found 18 GB free beside 24 GB reserved and unused. An entrypoint's
# own setting stands.
ALLOCATOR = {'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'}


def _run_once(command: list[str]) -> tuple[int, bool]:
    """The entrypoint's exit code, and whether its output named an out-of-memory error.
    Its output passes through to ours line by line, so the run's log is unchanged."""
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               env={**ALLOCATOR, **os.environ})
    out_of_memory = False
    for line in iter(process.stdout.readline, b''):
        sys.stdout.buffer.write(line)
        sys.stdout.buffer.flush()
        if OUT_OF_MEMORY.search(line.decode('utf-8', errors='replace')):
            out_of_memory = True
    return process.wait(), out_of_memory


def _empty(output: Path) -> None:
    for entry in output.iterdir():
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry)
        else:
            entry.unlink()


def run(command: list[str], params_path: Path, output: Path) -> int:
    params = json.loads(params_path.read_text(encoding='utf-8'))
    asked = {'batch_size': params.get('batch_size'), 'grad_accum_steps': params.get('grad_accum_steps') or 1}
    while True:
        code, out_of_memory = _run_once(command)
        batch = params.get('batch_size')
        if code == 0 or not out_of_memory or not batch or batch <= 1:
            break
        params['batch_size'] = batch // 2
        params['grad_accum_steps'] = (params.get('grad_accum_steps') or 1) * 2
        params_path.write_text(json.dumps(params), encoding='utf-8')
        print(f'deeplabel: out of GPU memory at batch {batch}; retrying at batch {params["batch_size"]} '
              f'with gradient accumulation {params["grad_accum_steps"]}', flush=True)
        # A resumed run's last checkpoint is in its output; it resumes from there again.
        if output.is_dir() and not os.environ.get('TRAIN_RESUME'):
            _empty(output)
    if code == 0 and params.get('batch_size') != asked['batch_size']:
        output.mkdir(parents=True, exist_ok=True)
        (output / FALLBACK_NAME).write_text(json.dumps(
            {'asked': asked, 'trained': {'batch_size': params['batch_size'],
                                         'grad_accum_steps': params['grad_accum_steps']}}), encoding='utf-8')
    return code


def wrap(entrypoint: list[str]) -> list[str]:
    """The entrypoint as the backend runs it: under this fallback, by this interpreter."""
    return [sys.executable, '-m', 'deeplabel_backend.fallback', *entrypoint]


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    return run(sys.argv[1:], Path(os.environ['TRAIN_PARAMS']), Path(os.environ['TRAIN_OUTPUT']))


if __name__ == '__main__':
    sys.exit(main())
