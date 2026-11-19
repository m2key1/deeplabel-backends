import json
import shutil
from pathlib import Path

from deeplabel_backend.train import PROGRESS_NAME


def keep_best(output: Path) -> bool:
    """What train.py leaves when it ends by itself, made from where a run asked to finish was
    stopped: the best epoch's weights from its run directory as model.ckpt, and that epoch's
    CER in metrics.json. False when the run had not validated an epoch yet."""
    best = output / 'run' / 'best_model.pt'
    progress = output / PROGRESS_NAME
    cers = [point['cer'] for point in (json.loads(line) for line in progress.read_text(encoding='utf-8').splitlines()
                                        if line.strip()) if 'cer' in point] if progress.is_file() else []
    if not best.is_file() or not cers:
        return False
    shutil.copyfile(best, output / 'model.ckpt')
    (output / 'metrics.json').write_text(json.dumps({'cer': min(cers)}))
    return True
