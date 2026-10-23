import os
import sys
from pathlib import Path

import torch
from deeplabel_backend import Loaded, Model

from kraken_lines import tunables
from kraken_lines.segmenter import KrakenSegmenter

VERSION = '1.0.0'
# The image's interpreter, named outright: a cluster runtime need not carry the image's PATH.
ENTRYPOINT_PYTHON = '/srv/.venv/bin/python'
PRECISION = os.environ.get('KRAKEN_PRECISION', '32-true')


def device_of(name: str) -> str:
    return ('cuda' if torch.cuda.is_available() else 'cpu') if name == 'auto' else name


# What the trainer reports, for the run page (6.9): units, which way is better, the headline.
METRICS = [
    {'key': 'val/baseline_f1', 'label': 'baseline F1', 'unit': 'fraction', 'better': 'higher', 'headline': True},
    {'key': 'val/baseline_precision', 'label': 'baseline precision', 'unit': 'fraction', 'better': 'higher'},
    {'key': 'val/baseline_recall', 'label': 'baseline recall', 'unit': 'fraction', 'better': 'higher'},
    {'key': 'val/mean_iu', 'label': 'mean IoU', 'unit': 'fraction', 'better': 'higher'},
    {'key': 'val/freq_iu', 'label': 'frequency-weighted IoU', 'unit': 'fraction', 'better': 'higher', 'detail': True},
    {'key': 'val/mean_acc', 'label': 'mean accuracy', 'unit': 'fraction', 'better': 'higher', 'detail': True},
    {'key': 'val/accuracy', 'label': 'pixel accuracy', 'unit': 'fraction', 'better': 'higher', 'detail': True},
    {'key': 'train/loss', 'label': 'loss', 'unit': 'number', 'better': 'lower'},
]


class KrakenBackend(Model):
    """Kraken's baseline segmenter as a DeepLabel model backend (4.4): a page or an
    element's crop in, its text lines as boundary polygons out, as fractions of that image.
    The baselines Kraken finds stay here; DeepLabel's lines are polygons."""

    name = 'kraken'
    version = VERSION
    model_type = 'kraken-blla'
    tasks = ('lines',)
    weight_roles = {'weights': 'optional'}
    builtin = True
    max_request_mb = 128

    def tasks_detail(self) -> dict[str, dict]:
        return {'lines': {'batch': {'max_items': 24},
                          'params': {'scope': {'type': 'string', 'default': 'page', 'enum': ['page', 'crop']}}}}

    def load(self, files: dict[str, Path], device: str) -> Loaded:
        return Loaded(handle=KrakenSegmenter(files.get('weights'), device_of(device), PRECISION),
                      facts={'builtin': 'weights' not in files})

    def unload(self, loaded: Loaded) -> None:
        loaded.handle.close()
        loaded.handle = None
        torch.cuda.empty_cache()

    def predict(self, loaded, task, images, params, contexts):
        return [{'lines': [{'polygon': polygon, 'confidence': None} for polygon in lines]}
                for lines in loaded.handle.segment(images)]

    def train_info(self) -> dict | None:
        return {
            'tasks': ['lines'],
            'snapshot': 'page-lines/1',
            'line_shape': 'polygon',
            'entrypoint': [ENTRYPOINT_PYTHON, '/srv/kraken_lines/train_entrypoint.py'],
            'image': f'kraken-lines-backend:{VERSION}',
            'resources': {'gpus': {'min': 1, 'max': 1, 'multi_gpu': False},
                          # It peaks at 2 GB of the card, so an L40S holds it; 15 loader workers
                          # trained 12 pages/s there against 4.9 with 8 on an RTX 6000 (2026-10-20).
                          'cpus_per_gpu': 16, 'memory_per_gpu_gb': 64, 'preferred_gpu': 'l40s', 'default_batch': {'local': 1, 'cluster': 1}},
            'tunables': [{k: v for k, v in knob.items() if k != 'env'} for knob in tunables.declared()],
            'augmentation': None,
            'outputs': ['weights'],
            'resume': False,
            'metrics': METRICS,
        }

    def train_command(self) -> list[str]:
        # The image's interpreter and path on the cluster; here, this backend's own interpreter.
        return [sys.executable, str(Path(__file__).with_name('train_entrypoint.py'))]
