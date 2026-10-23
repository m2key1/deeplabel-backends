import json
import os
import sys
import time
from pathlib import Path

from deeplabel_backend.train import ProgressLog, run_hyperparams, write_outputs

from deeplabel_stub.model import TRAIN_EPOCHS, train_tunables, weights_mode

SLOW_SECONDS = 15


def valid_groups(snapshot: Path) -> list[str]:
    manifest = json.loads((snapshot / 'manifest.json').read_text(encoding='utf-8'))
    groups = manifest.get('groups', {})
    stems = {Path(f['path']).stem for f in manifest.get('files', []) if f['path'].startswith('valid/')}
    return sorted({groups[stem] for stem in stems if stem in groups})


def main() -> None:
    """The e2e trainer: three epochs with loss, train/grad_norm and cer, a per-dataset cer
    when the valid split spans several datasets, and a weights file. Run names carry the
    markers specs use: [slow] waits first, [no-ts] leaves the timestamps out, [oom] runs out
    of memory above a batch of 2, as a dense page does on a small card, [lm-fails] fails the
    language model a run asks for."""
    params = json.loads(Path(os.environ['TRAIN_PARAMS']).read_text(encoding='utf-8'))
    output = Path(os.environ['TRAIN_OUTPUT'])
    name = params.get('run', {}).get('name', '')
    if '[slow]' in name:
        time.sleep(SLOW_SECONDS)
    if '[oom]' in name and (params.get('batch_size') or 0) > 2:
        print('torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 12.00 GiB', flush=True)
        sys.exit(1)
    tunables = train_tunables()
    trained_with = {'batch_size': params.get('batch_size'), 'gpus': params.get('gpus'),
                    'resumed': bool(os.environ.get('TRAIN_RESUME')),
                    **run_hyperparams(params, tunables, lambda: {t['key']: t.get('default') for t in tunables})}
    groups = valid_groups(Path(os.environ['TRAIN_SNAPSHOT']))
    progress = ProgressLog(output)
    stamp = {'ts': None} if '[no-ts]' in name else {}
    for epoch, (loss, cer) in enumerate(TRAIN_EPOCHS, start=1):
        progress.write(epoch=epoch, loss=loss, **stamp)
        progress.write(epoch=epoch, **{'train/grad_norm': round(3.0 / epoch, 3)}, **stamp)
        point = {'epoch': epoch, 'cer': cer, **stamp}
        if len(groups) > 1:
            point['cer_by_dataset'] = {group: round(cer + 0.01 * rank, 4) for rank, group in enumerate(groups)}
        progress.write(**point)
        time.sleep(0.1)
    progress.close()
    print(f'stub trainer: epoch {len(TRAIN_EPOCHS)}/{len(TRAIN_EPOCHS)} done', flush=True)
    (output / 'trained_with.json').write_text(json.dumps(trained_with), encoding='utf-8')
    # A model tuned from a slow or misreading base keeps its behaviour, so what it reads can
    # still be told apart from the truth.
    base = Path(os.environ.get('TRAIN_BASE') or output) / 'weights'
    inherited = weights_mode({'weights': base}) if base.is_file() else None
    (output / 'model.ckpt').write_text(inherited or f'stub-checkpoint {params.get("run", {}).get("id")}')
    files, lm = [{'role': 'weights', 'path': 'model.ckpt'}], {}
    if params.get('build_lm'):
        lm = {'lm_build': 'failed' if '[lm-fails]' in name else 'built'}
        if lm['lm_build'] == 'built':
            files += language_model(output, params.get('run', {}).get('id'))
    write_outputs(output, {'val_metric': 0.42, 'val_metric_name': 'cer', 'higher_is_better': False,
                           'task': params.get('task')}, files, **lm)


def language_model(output: Path, run_id) -> list[dict]:
    (output / 'lm').mkdir(exist_ok=True)
    (output / 'lm' / 'domain.bin').write_text(f'stub-lm {run_id}')
    (output / 'lm' / 'domain.dict').write_text(f'stub-lm-dict {run_id}')
    return [{'role': 'lm', 'path': 'lm/domain.bin'}, {'role': 'lm_dict', 'path': 'lm/domain.dict'}]


if __name__ == '__main__':
    main()
