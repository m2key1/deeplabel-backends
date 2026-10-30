import os
import sys
import unicodedata
from pathlib import Path

import torch
from deeplabel_backend import Loaded, Model, Prediction, augmentation

import infer
from deepocr import BUILTIN_DIR, BUILTIN_FILES, tunables
from deepocr.engine import DECODES, DeepOCREngine

VERSION = '1.0.0'
# The image's interpreter, named outright: a cluster runtime need not carry the image's PATH.
ENTRYPOINT_PYTHON = '/srv/.venv/bin/python'
# The batch measured per cluster card (DeepLabel's trainer, 2026-09-16): the largest whose
# reserved memory stays under 90% of the card at the widest bucket, on the grouped model.
# Measured on the widest lines: the cluster cards on 2026-09-16/18, the 5090 on 2026-10-17
# (16 fills 90% of it and trains no faster than 8).
BATCH_BY_GPU = {'h200': 76, 'rtx6000bb': 50, 'l40s': 22, 'rtx5090': 8}
LOCAL_BATCH = 8
# How many line crops one forward pass takes; the engine halves it after an out-of-memory error.
FORWARD_BATCH = int(os.environ.get('DEEPOCR_BATCH', '32'))


def device_of(name: str) -> torch.device:
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


# What the trainer reports, for the run page (6.9): units, which way is better, the headline.
METRICS = [
    {'key': 'cer', 'label': 'CER', 'unit': 'fraction', 'better': 'lower', 'headline': True},
    {'key': 'best_cer', 'label': 'best CER', 'unit': 'fraction', 'better': 'lower'},
    {'key': 'loss', 'label': 'loss', 'unit': 'number', 'better': 'lower'},
]


class DeepOCRBackend(Model):
    """DeepOCR - CTC head, AR decoder, fused KenLM - as a DeepLabel model backend (4.5): a
    weights version is the checkpoint with the LM files built for it, and every read says
    which decode and which transcription conventions it wants."""

    name = 'deepocr'
    version = VERSION
    model_type = 'deepocr'
    tasks = ('ocr',)
    capabilities = ('lm',)
    weight_roles = {'weights': 'optional', 'lm': 'optional', 'lm_dict': 'optional'}
    builtin = True
    max_request_mb = 64

    def tasks_detail(self) -> dict[str, dict]:
        return {'ocr': {'batch': {'max_items': 512}, 'decodes': list(DECODES), 'default_decode': 'beam',
                        'params': {'beam_width': {'type': 'integer', 'default': infer.BEAM_WIDTH},
                                   'convention': {'type': 'object', 'default': {}}}}}

    def load(self, files: dict[str, Path], device: str) -> Loaded:
        builtin = 'weights' not in files
        if builtin:
            files = {role: BUILTIN_DIR / name for role, name in BUILTIN_FILES.items()}
        engine = DeepOCREngine(files['weights'], device_of(device), kenlm_path=files.get('lm'),
                               lm_dict_path=files.get('lm_dict'),
                               lm_weight=float(os.environ.get('DEEPOCR_LM_WEIGHT', infer.LM_WEIGHT)),
                               beam_width=infer.BEAM_WIDTH)
        return Loaded(handle=engine, facts={'target_height': engine.model.target_height,
                                            'max_width': engine.model.max_width or 0,
                                            'conventions': engine.convention_names, 'lm_loaded': engine.lm_on,
                                            'builtin': builtin})

    def unload(self, loaded: Loaded) -> None:
        loaded.handle.close()
        loaded.handle = None
        torch.cuda.empty_cache()

    def predict(self, loaded, task, images, params, contexts):
        engine: DeepOCREngine = loaded.handle
        convention = params.get('convention') or {}
        decode = engine.configure(params.get('decode') or 'beam', convention)
        engine.beam_width = int(params.get('beam_width') or infer.BEAM_WIDTH)
        texts, confs, profiles = engine.predict_batch(images, batch_size=FORWARD_BATCH)
        # The widest lines leave tens of gigabytes cached; another backend may share the card.
        torch.cuda.empty_cache()
        results = [{'text': unicodedata.normalize('NFC', text.strip()), 'confidence': max(0.0, min(1.0, conf)), 'conf_profile': profile}
                   for text, conf, profile in zip(texts, confs, profiles)]
        return Prediction(results, {'decode': decode, 'conventions': engine.honoured(convention)})

    def batch_size(self, task: str, params: dict) -> int | None:
        # The whole request: the engine orders it by width before batching, and halves a
        # batch that runs out of memory itself.
        return None

    def train_info(self) -> dict | None:
        return {
            'tasks': ['ocr'],
            'snapshot': 'ocr-lines/1',
            'entrypoint': [ENTRYPOINT_PYTHON, '/srv/deepocr/train_entrypoint.py'],
            'image': f'deepocr-backend:{VERSION}',
            'resources': {'gpus': {'min': 1, 'max': 8, 'multi_gpu': True}, 'cpus_per_gpu': 12,
                          'memory_per_gpu_gb': 128, 'shm_gb': 8, 'batch_by_gpu': BATCH_BY_GPU,
                          'default_batch': {'local': LOCAL_BATCH, 'cluster': min(v for k, v in BATCH_BY_GPU.items() if k != 'rtx5090')}},
            'tunables': self.tunables(),
            'augmentation': {'library': 'albumentations', 'targets': ['image']},
            'outputs': ['weights', 'lm', 'lm_dict'],
            'resume': True,
            'metrics': METRICS,
            # A run conditions on each dataset's convention flags and can teach its base new ones.
            'conventions': True,
        }

    def train_command(self) -> list[str]:
        # The image's `python` and path on the cluster; here, this backend's own interpreter.
        return [sys.executable, str(Path(__file__).with_name('train_entrypoint.py'))]

    def tunables(self) -> list[dict]:
        return [{k: v for k, v in knob.items() if k != 'env'} for knob in tunables.declared()]

    def augmentations(self) -> dict:
        return augmentation.catalog(tunables.default_augmentation())

    def augmentation_preview(self, spec, images, seed, draws, boxes=None):
        found = tunables.constants()
        return augmentation.preview(spec, images, seed, draws, height=found.get('TARGET_HEIGHT', 128),
                                    max_width=found.get('MAX_WIDTH', 3072))
