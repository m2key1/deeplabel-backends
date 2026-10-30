import os
import time
from pathlib import Path

from deeplabel_backend import Loaded, Model, Prediction, errors
from deeplabel_backend.augmentation import Augmented
from PIL import ImageOps

ELEMENT_SPECS = [
    ('Header', [0.1, 0.05, 0.9, 0.15]),
    ('Paragraph', [0.1, 0.2, 0.9, 0.85]),
]
LINES_PER_ELEMENT = 3
LAYOUT_CONFIDENCE = 0.9
# Distinct per class so scenarios can tell sorted orders apart; Paragraph lines 1-2
# sit under the default 0.85 review threshold, line 1 also under 0.7.
CONFIDENCES = {'Header': [0.97, 0.92, 0.9], 'Paragraph': [0.55, 0.82, 0.95]}
FALLBACK_CONFIDENCE = 0.9
PROFILE_BANDS = 20
OCR_FACTS = {'target_height': 256, 'max_width': 6144}
DECODES = ['beam', 'rescored', 'greedy']
CONVENTIONS = ['long_s', 'oblique_hyphen', 'typographic_punctuation', 'abbreviations']
MAP_CLASSES = [{'id': 1, 'name': 'Parcel', 'color': '#c0392b'}, {'id': 2, 'name': 'Road', 'color': '#7f8c8d'}]
EDGE_METHODS = [{'id': 'centreline', 'label': 'Ink centreline', 'needs_weights': False,
                 'params': {'block_size': {'type': 'number', 'default': 21},
                            'threshold_offset': {'type': 'number', 'default': 10}}}]
TUNABLES = [
    {'key': 'learning_rate', 'role': 'learning_rate', 'label': 'Learning rate', 'group': 'Schedule', 'kind': 'float',
     'help': 'Peak rate for every parameter group.', 'above': 0, 'default': 0.0001,
     'env': ['TRAIN_BACKBONE_BASE_LR', 'TRAIN_TRANSFORMER_BASE_LR', 'TRAIN_DECODER_BASE_LR']},
    {'key': 'warmup_epochs', 'label': 'Warmup epochs', 'group': 'Schedule', 'kind': 'float',
     'help': 'Linear warmup to the peak rate, in epochs.', 'min': 0, 'default': 1.0, 'env': 'TRAIN_WARMUP_EPOCHS'},
    {'key': 'grad_accum_steps', 'role': 'accumulation', 'label': 'Gradient accumulation', 'group': 'Memory',
     'kind': 'int', 'help': 'Batches summed before each optimiser step.', 'min': 1, 'default': 1,
     'env': 'TRAIN_GRAD_ACCUM_STEPS'},
    {'key': 'patience', 'label': 'Patience', 'group': 'Validation', 'kind': 'int',
     'help': 'Stop when the validation CER has not improved for this many epochs.', 'min': 1,
     'env': 'TRAIN_PATIENCE'},
    {'key': 'freeze_backbone', 'label': 'Freeze backbone', 'group': 'Fine-tuning', 'kind': 'bool',
     'help': 'Keep the CNN backbone fixed and train only the encoder and decoder.', 'default': False,
     'env': 'TRAIN_FREEZE_BACKBONE'},
]
# A knob with named settings, which a line detector has: its size.
LINE_TUNABLES = [
    {'key': 'variant', 'label': 'Detector variant', 'group': 'Model', 'kind': 'choice',
     'help': 'Detector size.', 'choices': ['tiny', 'small', 'medium'], 'env': 'TRAIN_VARIANT'},
]


def train_tunables() -> list[dict]:
    return TUNABLES + (LINE_TUNABLES if train_snapshot() == 'page-lines/1' else [])


AUGMENTATIONS = {
    'library': 'albumentations', 'version': 'stub',
    'transforms': [
        {'name': 'Affine', 'category': 'geometric', 'targets': ['image', 'mask', 'bboxes', 'keypoints'],
         'doc': 'Apply affine transformations: translation, rotation, scale, shear.', 'default_p': 0.5,
         'params': [
             {'name': 'rotate', 'kind': 'range', 'items': 'number', 'nullable': False, 'default': [0.0, 0.0],
              'doc': 'Rotation range in degrees.'},
             {'name': 'scale', 'kind': 'json', 'nullable': False, 'default': [1.0, 1.0], 'doc': 'Scaling factor.'},
             {'name': 'fill', 'kind': 'number', 'nullable': False, 'default': 0, 'doc': 'Fill value for new pixels.'},
             {'name': 'keep_ratio', 'kind': 'boolean', 'nullable': False, 'default': True, 'doc': None},
         ]},
        {'name': 'GaussNoise', 'category': 'pixel', 'targets': ['image'],
         'doc': 'Add Gaussian noise to the image.', 'default_p': 0.5,
         'params': [
             {'name': 'std_range', 'kind': 'range', 'items': 'number', 'nullable': False, 'default': [0.2, 0.44],
              'doc': 'Noise standard deviation as a fraction of the value range.'},
             {'name': 'per_channel', 'kind': 'boolean', 'nullable': False, 'default': False, 'doc': None},
         ]},
        {'name': 'Morphological', 'category': 'pixel', 'targets': ['image', 'mask'],
         'doc': 'Dilation or erosion with a structuring element.', 'default_p': 0.5,
         'params': [
             {'name': 'operation', 'kind': 'enum', 'options': ['erosion', 'dilation'], 'nullable': False,
              'default': 'dilation', 'doc': None},
             {'name': 'scale', 'kind': 'range', 'items': 'integer', 'nullable': False, 'default': [2, 3], 'doc': None},
         ]},
        {'name': 'InvertImg', 'category': 'pixel', 'targets': ['image'],
         'doc': 'Invert the input image by subtracting pixel values from 255.', 'default_p': 0.5, 'params': []},
        {'name': 'BBoxSafeRandomCrop', 'category': 'crops', 'targets': ['image', 'mask', 'bboxes'],
         'doc': 'Crop a region that keeps all bounding boxes.', 'default_p': 1.0, 'params': []},
        {'name': 'RandomCrop', 'category': 'crops', 'targets': ['image', 'mask', 'bboxes', 'keypoints'],
         'doc': 'Crop a random part of the input.', 'default_p': 1.0,
         'params': [
             {'name': 'height', 'kind': 'integer', 'nullable': False, 'default': None, 'doc': 'Crop height.'},
             {'name': 'width', 'kind': 'integer', 'nullable': False, 'default': None, 'doc': 'Crop width.'},
         ]},
    ],
    # The stack the stub trainer applies when a run names none, ending in the steps a
    # trainer adds itself.
    'default': {
        '__version__': 'stub',
        'transform': {
            '__class_fullname__': 'Compose', 'p': 1.0,
            'transforms': [
                {'__class_fullname__': 'Affine', 'p': 0.3, 'rotate': [-2.0, 2.0], 'scale': [0.95, 1.05],
                 'fill': 255, 'keep_ratio': True},
                {'__class_fullname__': 'GaussNoise', 'p': 0.3, 'std_range': [0.2, 0.44], 'per_channel': False},
                {'__class_fullname__': 'OneOf', 'p': 0.3, 'transforms': [
                    {'__class_fullname__': 'Morphological', 'p': 1.0, 'operation': 'erosion', 'scale': [3, 7]},
                    {'__class_fullname__': 'Morphological', 'p': 1.0, 'operation': 'dilation', 'scale': [3, 7]},
                ]},
                {'__class_fullname__': 'Normalize', 'p': 1.0},
                {'__class_fullname__': 'ToTensorV2', 'p': 1.0},
            ],
        },
    },
}
GROUPS = {'Compose', 'OneOf', 'SomeOf', 'Sequential'}
STEP_TARGETS = {t['name']: set(t['targets']) for t in AUGMENTATIONS['transforms']}
RANGES = {(t['name'], p['name']) for t in AUGMENTATIONS['transforms'] for p in t['params'] if p['kind'] == 'range'}
# What each snapshot kind's trainer hands its augmentation besides the image.
SNAPSHOT_TARGETS = {'ocr-lines/1': {'image'}, 'page-lines/1': {'image', 'mask'}, 'page-boxes/1': {'image', 'bboxes'}}


def stack_error(node: dict, supplied: set[str], path: str = '') -> str | None:
    """Why the library would refuse a stack, in the words it would use, or None."""
    name = node.get('__class_fullname__')
    label = f'{path}{name}'
    if name not in STEP_TARGETS and name not in GROUPS:
        return f'{label}: unknown transform {name}'
    if 'bboxes' in STEP_TARGETS.get(name, set()) and 'bboxes' not in supplied and name.startswith('BBox'):
        return f'{label}: applying it to an image failed: this step needs bounding boxes'
    for key, value in node.items():
        if (name, key) in RANGES and isinstance(value, list) and len(value) == 2 and value[0] > value[1]:
            return f'{label}: {key}: the low end must not exceed the high end'
    for index, child in enumerate(node.get('transforms', [])):
        error = stack_error(child, supplied, '' if name == 'Compose' else f'{label}[{index}].')
        if error:
            return error
    return None


# Behaviour a scenario opts into by importing a model whose weights file holds one of
# these words, since a backend never sees the dataset's name.
SLOW = 'stub-slow'
MISREAD = 'stub-misread'
SLOW_SECONDS = 15
# What a misreading model reads for the n-th line of a request, by n % 4, with the
# confidence it claims: exact, a case error, a digit error, and a lost space with a stray
# full stop - so each tab of the evaluation page has something exact to show.
MISREAD_CONFIDENCES = (0.95, 0.85, 0.75, 0.65)


def weights_mode(files: dict[str, Path]) -> str | None:
    weights = files.get('weights')
    if weights is None:
        return None
    word = weights.read_bytes()[:64].decode('utf-8', 'replace').strip()
    return word if word in (SLOW, MISREAD) else None


def misread(n: int, text: str) -> str:
    kind = n % 4
    if kind == 1 and text:
        return text[0].swapcase() + text[1:]
    if kind == 2 and text:
        last = text[-1]
        return text[:-1] + (str((int(last) + 1) % 10) if last.isdigit() else '')
    if kind == 3:
        return text.replace(' ', '', 1) + '.'
    return text


def line_quads(bbox: list) -> list:
    """Three lines inside a box, derived from the box alone, with a gap between rows."""
    x0, y0, x1, y1 = bbox
    step = (y1 - y0) / LINES_PER_ELEMENT
    quads = []
    for i in range(LINES_PER_ELEMENT):
        top = round(y0 + i * step + step * 0.15, 6)
        bottom = round(y0 + (i + 1) * step - step * 0.15, 6)
        quads.append([[x0, top], [x1, top], [x1, bottom], [x0, bottom]])
    return quads


def ocr_result(context: dict, honoured: list[str]) -> dict:
    class_name = context.get('element_class', 'Text')
    index = int(context.get('line_index', 0))
    per_class = CONFIDENCES.get(class_name, [FALLBACK_CONFIDENCE] * LINES_PER_ELEMENT)
    confidence = per_class[index % len(per_class)]
    profile = [{'x0': round(i / PROFILE_BANDS, 4), 'x1': round((i + 1) / PROFILE_BANDS, 4),
                'c': int(confidence * 100)} for i in range(PROFILE_BANDS)]
    tags = ''.join(f' [{name}]' for name in honoured)
    return {'text': f'Sample {class_name} line {index + 1}{tags}', 'confidence': confidence,
            'conf_profile': profile}


def edge_polyline(area: list) -> list:
    """One line across the selection, a quarter inset from its corners."""
    xs, ys = [p[0] for p in area], [p[1] for p in area]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    dx, dy = (x1 - x0) / 4, (y1 - y0) / 4
    return [[round(x0 + dx, 4), round(y0 + dy, 4)], [round(x0 + 2 * dx, 4), round(y0 + 2 * dy, 4)],
            [round(x1 - dx, 4), round(y1 - dy, 4)]]


def map_regions(class_ids: list[int]) -> list[dict]:
    """One square per class, side by side, the first with a hole."""
    regions = []
    for rank, class_id in enumerate(class_ids):
        x0 = round(0.1 + 0.3 * rank, 4)
        x1 = round(x0 + 0.2, 4)
        region = {'class_id': class_id, 'points': [[x0, 0.4], [x1, 0.4], [x1, 0.6], [x0, 0.6]], 'holes': []}
        if rank == 0:
            region['holes'] = [[[x0 + 0.05, 0.45], [x1 - 0.05, 0.45], [x1 - 0.05, 0.55], [x0 + 0.05, 0.55]]]
        regions.append(region)
    return regions


# One image serves every task, but a backend trains one snapshot kind; e2e runs a
# container per kind, each told which by STUB_TRAIN_SNAPSHOT.
SNAPSHOT_TASKS = {'ocr-lines/1': 'ocr', 'page-lines/1': 'lines', 'page-boxes/1': 'layout'}


def train_snapshot() -> str:
    snapshot = os.environ.get('STUB_TRAIN_SNAPSHOT') or 'ocr-lines/1'
    if snapshot not in SNAPSHOT_TASKS:
        raise ValueError(f'STUB_TRAIN_SNAPSHOT must be one of {sorted(SNAPSHOT_TASKS)}, not {snapshot!r}')
    return snapshot


# The loss and CER of each epoch the stub trainer runs.
TRAIN_EPOCHS = [(2.5, 0.5), (1.5, 0.3), (0.5, 0.1)]

# What the stub trainer reports, declared as a backend does (6.9).
METRICS = [
    {'key': 'cer', 'label': 'CER', 'unit': 'fraction', 'better': 'lower', 'headline': True},
    {'key': 'loss', 'label': 'loss', 'unit': 'number', 'better': 'lower'},
    {'key': 'train/grad_norm', 'label': 'gradient norm', 'unit': 'number', 'detail': True},
]


class StubModel(Model):
    """Every task, deterministic, no GPU. Builtin, so requests may omit weights; any file
    sent as weights loads too, which is what lets the stub exercise the transfer path."""

    name = 'stub'
    version = '1.0.0'
    model_type = 'stub'
    tasks = ('layout', 'lines', 'ocr', 'map_segmentation', 'edges')
    capabilities = ('lm',)
    concurrency = 4
    weight_roles = {'weights': 'optional', 'lm': 'optional', 'lm_dict': 'optional'}
    builtin = True

    def tasks_detail(self) -> dict[str, dict]:
        return {
            'layout': {'batch': {'max_items': 16}, 'labels': [name for name, _ in ELEMENT_SPECS], 'max_side': 3000,
                       'params': {'threshold': {'type': 'number', 'default': 0.3}}},
            'lines': {'batch': {'max_items': 64},
                      'params': {'scope': {'type': 'string', 'default': 'page', 'enum': ['page', 'crop']}}},
            'ocr': {'batch': {'max_items': 512}, 'decodes': DECODES, 'default_decode': 'beam',
                    'conventions': CONVENTIONS,
                    'params': {'lm_weight': {'type': 'number', 'default': 1.5},
                               'beam_width': {'type': 'integer', 'default': 10},
                               'convention': {'type': 'object', 'default': {}}}},
            'map_segmentation': {'batch': {'max_items': 8}, 'input_size': 512, 'classes': MAP_CLASSES,
                                 'params': {'class_ids': {'type': 'array'}, 'epsilon': {'type': 'number', 'default': 2.0},
                                            'min_area': {'type': 'integer', 'default': 12}}},
            'edges': {'batch': {'max_items': 1}, 'methods': EDGE_METHODS},
        }

    def load(self, files: dict[str, Path], device: str) -> Loaded:
        return Loaded(handle=weights_mode(files), facts={**OCR_FACTS, 'conventions': CONVENTIONS,
                                                   'lm_loaded': 'lm' in files,
                                                   'classes': len(MAP_CLASSES)})

    def predict(self, loaded, task, images, params, contexts):
        delay = SLOW_SECONDS if loaded.handle == SLOW else float(os.environ.get('STUB_PREDICT_DELAY') or 0)
        if delay:
            time.sleep(delay)
        if task == 'layout':
            return [{'detections': [{'label': name, 'bbox': bbox, 'confidence': LAYOUT_CONFIDENCE}
                                    for name, bbox in ELEMENT_SPECS]} for _ in images]
        if task == 'lines':
            boxes = [[0.0, 0.0, 1.0, 1.0]] if params.get('scope') == 'crop' else [b for _, b in ELEMENT_SPECS]
            return [{'lines': [{'polygon': quad, 'confidence': None} for box in boxes for quad in line_quads(box)]}
                    for _ in images]
        if task == 'ocr':
            decode = params.get('decode') or 'beam'
            if decode not in DECODES:
                decode = 'beam'
            asked = params.get('convention') or {}
            honoured = [name for name in CONVENTIONS if asked.get(name)]
            results = [ocr_result(context, honoured) for context in contexts]
            if loaded.handle == MISREAD:
                # Each decode starts the cycle one line further on, so their scores differ.
                shift = DECODES.index(decode)
                for n, result in enumerate(results, start=shift):
                    result.update(text=misread(n, result['text']), confidence=MISREAD_CONFIDENCES[n % 4])
            return Prediction(results, {'decode': decode, 'conventions': honoured})
        if task == 'map_segmentation':
            class_ids = params.get('class_ids') or [c['id'] for c in MAP_CLASSES]
            return [{'regions': map_regions(class_ids)} for _ in images]
        area = params.get('area') or [[0, 0], [1, 0], [1, 1], [0, 1]]
        if params.get('method', 'centreline') != 'centreline':
            return [{'error': {'code': 'unknown_method', 'message': f'unknown method {params["method"]!r}'}}
                    for _ in images]
        return [{'polylines': [edge_polyline(area)]} for _ in images]

    def train_info(self) -> dict:
        snapshot = train_snapshot()
        return {
            'tasks': [SNAPSHOT_TASKS[snapshot]],
            'snapshot': snapshot,
            'entrypoint': ['deeplabel-stub-train'],
            'image': 'deeplabel-model-stub',
            'resources': {'gpus': {'min': 1, 'max': 1, 'multi_gpu': False}, 'cpus_per_gpu': 2,
                          'memory_per_gpu_gb': 4, 'default_batch': {'local': 8, 'cluster': 22},
                          'batch_by_gpu': {'rtx5090': 8, 'rtx6000bb': 16, 'h200': 32}, 'effective_batch': 16},
            'tunables': [{k: v for k, v in t.items() if k != 'env'} for t in train_tunables()],
            'augmentation': {'library': 'albumentations', 'version': 'stub',
                             'targets': sorted(SNAPSHOT_TARGETS[snapshot])},
            'default_epochs': len(TRAIN_EPOCHS),
            'outputs': ['weights', 'lm', 'lm_dict'] if snapshot == 'ocr-lines/1' else ['weights'],
            'resume': True,
            'metrics': METRICS,
            **({'line_shape': 'box'} if snapshot == 'page-lines/1' else {}),
            **({'conventions': True} if snapshot == 'ocr-lines/1' else {}),
        }

    def tunables(self) -> list[dict]:
        return self.train_info()['tunables']

    def augmentations(self) -> dict:
        return AUGMENTATIONS

    def augmentation_preview(self, spec, images, seed, draws, boxes=None):
        error = stack_error(spec['transform'], SNAPSHOT_TARGETS[train_snapshot()] | ({'bboxes'} if boxes else set()))
        if error:
            raise errors.bad_request(error)
        drawn = [ImageOps.invert(image.convert('RGB')) for image in images]
        # The harshest of several draws is simply the more mangled one.
        drawn = drawn if draws <= 1 else [ImageOps.posterize(image, 1) for image in drawn]
        # A box comes back where it went in, so a scenario can tell it was carried through.
        return [Augmented(image, boxes[index] if boxes is not None else None) for index, image in enumerate(drawn)]
