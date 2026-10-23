"""The training entrypoint of section 6.6: an ocr-lines/1 snapshot in, DeepOCR's own train.py
run over it, the weights and the figures it ended on out. The same command runs on the
backend itself and, in this backend's image, on the cluster."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Started by path, in the run's own directory: the wrappers are not on the path by themselves.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deeplabel_backend.train import BUILTIN_MARKER, legacy_env, write_outputs

from deepocr import BUILTIN_DIR, BUILTIN_FILES, MODEL_REPO, tunables

LM_BUILD_TIMEOUT = 3600


def native_snapshot(snapshot: Path, work: Path) -> Path:
    """The snapshot as train.py reads it: the crops where the contract put them, and as
    manifest.json the map of which dataset each crop came from - the contract manifest's
    `groups` - so validation reports a CER per dataset."""
    work.mkdir(parents=True, exist_ok=True)
    for split in ('train', 'valid'):
        link = work / split
        if not link.exists():
            link.symlink_to(snapshot / split, target_is_directory=True)
    manifest = json.loads((snapshot / 'manifest.json').read_text(encoding='utf-8'))
    (work / 'manifest.json').write_text(json.dumps(manifest.get('groups', {})), encoding='utf-8')
    return work


def run_default_env(params: dict) -> dict[str, str]:
    """What a run takes where nobody chose, for the knobs DeepLabel's runs have always set."""
    by_key = {knob['key']: knob for knob in tunables.KNOBS}
    env = {}
    for key, value in tunables.run_defaults(params.get('hyperparams') or {}).items():
        for name in by_key[key]['env']:
            env[name] = str(value)
    return env


def language_model(train_split: Path, output: Path) -> tuple[list[dict], str]:
    """A KenLM fitted to the run's own training text, listed by role, and how the build
    went (6.6). Best effort: a failed build leaves the run to finish without one."""
    out = output / 'lm' / 'domain'
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        done = subprocess.run([sys.executable, 'build_kenlm.py', '--input', str(train_split), '--out', str(out)],
                              cwd=MODEL_REPO, capture_output=True, text=True, timeout=LM_BUILD_TIMEOUT)
    except (OSError, subprocess.SubprocessError) as error:
        print(f'language model build failed to start ({error})', flush=True)
        return [], 'failed'
    if done.returncode or not (out.with_suffix('.bin').is_file() and out.with_suffix('.dict').is_file()):
        detail = (done.stderr or done.stdout or '').strip().splitlines()
        print(f'language model build failed ({detail[-1] if detail else "no output"})', flush=True)
        return [], 'failed'
    return [{'role': 'lm', 'path': 'lm/domain.bin'}, {'role': 'lm_dict', 'path': 'lm/domain.dict'}], 'built'


def base_env(base: Path) -> dict[str, str]:
    """A run on the builtin base (6.6) fine-tunes the weights built into the image."""
    if (base / BUILTIN_MARKER).exists():
        return {'TRAIN_BASE_CHECKPOINT': str(BUILTIN_DIR / BUILTIN_FILES['weights'])}
    return {}


def builtin_language_model(output: Path) -> list[dict]:
    """The builtin's KenLM, listed by role: a fine-tune keeps the builtin's byte vocabulary,
    so its LM stays index-parallel."""
    (output / 'lm').mkdir(parents=True, exist_ok=True)
    files = []
    for role, suffix in (('lm', '.bin'), ('lm_dict', '.dict')):
        path = f'lm/builtin{suffix}'
        shutil.copyfile(BUILTIN_DIR / BUILTIN_FILES[role], output / path)
        files.append({'role': role, 'path': path})
    return files


def main() -> int:
    params = json.loads(Path(os.environ['TRAIN_PARAMS']).read_text(encoding='utf-8'))
    output = Path(os.environ['TRAIN_OUTPUT'])
    work = native_snapshot(Path(os.environ['TRAIN_SNAPSHOT']), Path(tempfile.mkdtemp(prefix='native-snapshot-')))
    env = {**os.environ, 'TRAIN_SNAPSHOT': str(work)}
    env.update(run_default_env(params))
    env.update(legacy_env(env, tunables.KNOBS, lambda: {
        **{k['key']: k['default'] for k in tunables.declared()},
        **tunables.run_defaults(params.get('hyperparams') or {})}))
    # A cluster run resubmitted after its time limit (6.9) continues from the rolling checkpoint
    # train.py keeps in its run directory: weights, optimiser, schedule, epoch and best CER.
    env.update(base_env(Path(os.environ['TRAIN_BASE'])))
    if os.environ.get('TRAIN_RESUME'):
        env['OCR_RESUME'] = str(Path(os.environ['TRAIN_RESUME']) / 'run')
    gpus = int(os.environ.get('TRAIN_GPUS') or params.get('gpus') or 1)
    command = ([sys.executable, '-m', 'torch.distributed.run', f'--nproc_per_node={gpus}', 'train.py']
               if gpus > 1 else [sys.executable, 'train.py'])
    code = subprocess.call(command, cwd=MODEL_REPO, env=env)
    if code:
        return code
    found = json.loads((output / 'metrics.json').read_text(encoding='utf-8')) if (output / 'metrics.json').is_file() else {}
    cer = found.get('cer')
    files, lm = [{'role': 'weights', 'path': 'model.ckpt'}], {}
    if params.get('build_lm'):
        built, lm['lm_build'] = language_model(Path(os.environ['TRAIN_SNAPSHOT']) / 'train', output)
        files += built
    if not any(f['role'] == 'lm' for f in files) and base_env(Path(os.environ['TRAIN_BASE'])):
        files += builtin_language_model(output)
    write_outputs(output, {**found, 'val_metric': cer, 'val_metric_name': 'cer', 'higher_is_better': False,
                           'task': 'ocr'}, files, **lm)
    return 0


if __name__ == '__main__':
    sys.exit(main())
