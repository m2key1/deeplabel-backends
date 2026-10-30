"""What a trainer can augment with, for the training form's builder, and a stack applied
the way the trainer applies it (7.1): the albumentations catalog with every transform's
parameters classified for the builder, and a seeded preview - of the image alone, or of the
image and its boxes for a trainer that is handed boxes. albumentations is imported only
when asked for, so a backend that does not train needs none."""
import inspect
import re
import warnings
from dataclasses import dataclass

from pydantic.errors import PydanticInvalidForJsonSchema

from . import errors

# Need inputs a page or line cannot give (callables, reference images, spectrograms,
# volumes), or do what the trainer already does itself (normalisation, tensors).
EXCLUDED_MODULES = ('mixing', 'transforms3d', 'spectrogram', 'text')
EXCLUDED_NAMES = {'Lambda', 'NoOp', 'ToFloat', 'FromFloat', 'Normalize', 'ToTensorV2', 'ToTensor3D'}
TARGETS = ('image', 'mask', 'bboxes', 'keypoints')
GROUPS = [
    {'name': 'OneOf', 'doc': 'Applies exactly one of its steps, picked at random, with probability p.'},
    {'name': 'SomeOf', 'doc': 'Applies n of its steps, picked at random, with probability p.'},
    {'name': 'Sequential', 'doc': 'Applies all of its steps in order, with probability p.'},
]
ARG_LINE = re.compile(r'^\s*(\w+)\s*\(([^)]*)\):\s*(.*)$')


def jsonable(value):
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return str(value)


def is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def by_schema(schema: dict, nullable: bool) -> dict:
    kind = schema.get('type')
    if 'enum' in schema:
        return {'kind': 'enum', 'options': schema['enum'], 'nullable': nullable}
    if kind == 'boolean':
        return {'kind': 'boolean', 'nullable': nullable}
    if kind in ('integer', 'number'):
        return {'kind': kind, 'nullable': nullable,
                'min': schema.get('minimum', schema.get('exclusiveMinimum')),
                'max': schema.get('maximum', schema.get('exclusiveMaximum'))}
    if kind == 'array':
        items = schema.get('prefixItems') or ([schema['items']] if 'items' in schema else [])
        kinds = {i.get('type') for i in items}
        size = schema.get('minItems') if schema.get('minItems') == schema.get('maxItems') else None
        if kinds and kinds <= {'integer', 'number'} and size in (2, 3, 4):
            items_kind = 'integer' if kinds == {'integer'} else 'number'
            if size == 2:
                return {'kind': 'range', 'items': items_kind, 'nullable': nullable}
            return {'kind': 'tuple', 'items': items_kind, 'size': size, 'nullable': nullable}
    return {'kind': 'json', 'nullable': nullable}


def by_default(default, members: list[dict], nullable: bool) -> dict:
    """A union of shapes is rendered as whichever shape the default takes, which is
    what nearly every caller uses; anything else falls back to raw JSON."""
    types = {m.get('type') for m in members}
    if is_number(default) and types & {'number', 'integer'}:
        return {'kind': 'integer' if isinstance(default, int) else 'number', 'nullable': nullable}
    if isinstance(default, (list, tuple)) and len(default) == 2 and all(is_number(v) for v in default):
        arrays = [m for m in members if m.get('type') == 'array']
        if arrays:
            item_types = {i.get('type') for m in arrays for i in m.get('prefixItems', [])}
            return {'kind': 'range', 'items': 'integer' if item_types == {'integer'} else 'number',
                    'nullable': nullable}
    return {'kind': 'json', 'nullable': nullable}


def classify(prop: dict, default) -> dict:
    if 'anyOf' not in prop:
        return by_schema(prop, False)
    members = [m for m in prop['anyOf'] if m.get('type') != 'null']
    nullable = len(members) < len(prop['anyOf'])
    if len(members) == 1:
        return by_schema(members[0], nullable)
    return by_default(default, members, nullable)


def arg_docs(doc: str) -> dict:
    """The per-parameter lines of a Google-style Args section, continuation lines joined."""
    docs, current = {}, None
    in_args = False
    for line in (doc or '').splitlines():
        if line.strip() in ('Args:', 'Arguments:'):
            in_args = True
            continue
        if in_args and line.strip() and not line.startswith(' ' * 8):
            in_args = False
        if not in_args:
            continue
        match = ARG_LINE.match(line)
        if match and line.startswith(' ' * 8) and not line.startswith(' ' * 12):
            current = match.group(1)
            docs[current] = match.group(3).strip()
        elif current and line.strip():
            docs[current] += ' ' + line.strip()
    return docs


def summary(doc: str) -> str:
    first = (doc or '').strip().split('\n\n')[0]
    return re.sub(r'\s+', ' ', first).strip()


def describe_transform(cls) -> dict:
    schema = cls.InitSchema.model_json_schema()
    signature = inspect.signature(cls.__init__).parameters
    docs = arg_docs(cls.__doc__)
    params = []
    for name, prop in schema['properties'].items():
        if name in ('p', 'strict'):
            continue
        parameter = signature.get(name)
        default = None if parameter is None or parameter.default is inspect.Parameter.empty else parameter.default
        params.append({'name': name, 'default': jsonable(default), 'doc': docs.get(name),
                       **classify(prop, default)})
    p_default = signature['p'].default if 'p' in signature and signature['p'].default is not inspect.Parameter.empty else 0.5
    return {
        'name': cls.__name__,
        'category': cls.__module__.split('.')[2] if cls.__module__.startswith('albumentations.augmentations.') else 'other',
        'targets': [t.name.lower() for t in getattr(cls, '_targets', ()) if t.name.lower() in TARGETS],
        'doc': summary(cls.__doc__),
        'default_p': p_default,
        'params': params,
    }


def catalog(default: dict | None = None) -> dict:
    """Every transform the builder may offer, and as `default` the stack the trainer applies
    when a run names none, in the library's own dict form."""
    import albumentations
    from albumentations.core.transforms_interface import BasicTransform
    transforms = []
    for name, cls in inspect.getmembers(albumentations, inspect.isclass):
        if not issubclass(cls, BasicTransform) or cls is BasicTransform or name != cls.__name__:
            continue
        module = cls.__module__
        if module.startswith(('albumentations.core', 'albumentations.pytorch')):
            continue
        if any(f'.{excluded}' in module for excluded in EXCLUDED_MODULES) or name in EXCLUDED_NAMES:
            continue
        if not hasattr(cls, 'InitSchema'):
            continue
        try:
            transforms.append(describe_transform(cls))
        except PydanticInvalidForJsonSchema:
            # A parameter JSON cannot describe (HEStain's ndarray) cannot be set in the form.
            continue
    transforms.sort(key=lambda t: t['name'])
    found = {'library': 'albumentations', 'version': albumentations.__version__, 'transforms': transforms,
             'groups': GROUPS}
    return {**found, 'default': default} if default is not None else found


def to_spec(steps: list) -> dict:
    """A stack of built transforms in the dict form a spec travels in, as JSON has it."""
    import albumentations as A
    return jsonable(A.to_dict(A.Compose(steps)))


GROUP_NAMES = ('Compose', 'OneOf', 'SomeOf', 'Sequential')


VALIDATION_HEADER = re.compile(r'^\d+ validation errors? for ')


def describe(exc: Exception, building: bool = False) -> str:
    """One line a person can act on. Albumentations re-raises pydantic's report as a
    plain ValueError, so its "field / message [type=...]" block is unpacked by hand into
    "field: message" - the raw text names a private schema class and a docs URL.

    A KeyError only means an unknown transform while a step is being built. Applying
    one raises KeyError for a target it was not handed, and calling that an unknown
    transform named "bboxes" sends the reader looking for something that does not
    exist."""
    if isinstance(exc, KeyError):
        if building:
            return f'unknown transform {exc.args[0]}'
        return f'it asks for {exc.args[0]}, which this task does not supply'
    lines = [line for line in str(exc).strip().splitlines() if line.strip()]
    if not lines:
        return type(exc).__name__
    if not VALIDATION_HEADER.match(lines[0]):
        return lines[0]
    problems = []
    for field, detail in zip(lines[1::2], lines[2::2]):
        message = detail.strip().split(' [type=')[0].removeprefix('Value error, ')
        problems.append(f'{field.strip()}: {message}')
    return '; '.join(problems) or lines[0]


def leaves(node: dict, path: str = ''):
    name = node.get('__class_fullname__', '?')
    label = f'{path}{name}'
    if name in GROUP_NAMES:
        for index, child in enumerate(node.get('transforms', [])):
            yield from leaves(child, f'{label}[{index}].')
    else:
        yield label, node


def build(spec: dict):
    import albumentations as A
    with warnings.catch_warnings():
        # An unknown keyword is only a warning to albumentations; in a spec it is a mistake.
        warnings.filterwarnings('error', message=r'Argument\(s\).*not valid')
        # Leaves first, one at a time, so an error names the step it belongs to.
        for label, node in leaves(spec['transform']):
            try:
                A.from_dict({'transform': node})
            except Exception as exc:
                raise ValueError(f'{label}: {describe(exc, building=True)}') from exc
        return A.from_dict(spec)


def resize(image, height: int, max_width: int):
    width = max(1, int(image.width * (height / image.height)))
    if max_width and width > max_width:
        width = max_width
    return image.resize((width, height), resample=1)


def fit(image, max_side: int):
    """The image no larger than max_side on its longer edge, and the factor it was scaled by."""
    scale = min(1.0, max_side / max(image.width, image.height))
    if scale == 1.0:
        return image, 1.0
    return image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), resample=1), scale


def deviation(before, after) -> float:
    """How far a draw strayed, 0 to 1. A step that changes the shape - a crop - has
    strayed as far as it can, so it sorts above any amount of recolouring."""
    import numpy as np
    if before.shape != after.shape:
        return 1.0
    return float(np.abs(before.astype('float32') - after.astype('float32')).mean() / 255.0)


def to_image(array):
    import numpy as np
    from PIL import Image
    if array.dtype != np.uint8:
        # albumentations keeps float images in 0..1
        array = (np.clip(array, 0, 1) * 255).astype(np.uint8)
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    return Image.fromarray(array)


@dataclass
class Augmented:
    """One preview image, and for a stack applied with boxes, where each box went: in the
    order they were given, in this image's pixels, None for one the draw cut away."""
    image: object
    boxes: list | None = None


def with_boxes(pipeline):
    import albumentations as A
    return A.Compose(pipeline.transforms, p=pipeline.p, bbox_params=A.BboxParams(
        format='pascal_voc', label_fields=['indices'], clip=True, min_visibility=0.0))


def preview(spec: dict, images: list, seed: int, draws: int, *, height: int | None = None, max_width: int = 0,
            max_side: int | None = None, boxes: list | None = None) -> list[Augmented]:
    """Each image as the trainer is handed it - a line at the height the model reads, a page
    no larger than max_side - put through the stack once, or over several draws the draw
    that strayed furthest. Given boxes (per image, [x0, y0, x1, y1] in its pixels), they go
    through the stack with it. A stack that cannot be built or applied is refused with the
    step named."""
    import numpy as np
    try:
        pipeline = build(spec)
    except Exception as exc:
        raise errors.bad_request(str(exc) if isinstance(exc, ValueError) else describe(exc)) from exc
    if boxes is not None:
        pipeline = with_boxes(pipeline)
    pipeline.set_random_seed(seed)
    shown = []
    for index, image in enumerate(images):
        image = image.convert('RGB')
        scale = 1.0
        if height:
            image = resize(image, height, max_width)
        elif max_side:
            image, scale = fit(image, max_side)
        source = np.array(image)
        given = None
        if boxes is not None:
            given = [[max(0.0, min(v * scale, limit)) for v, limit in zip(box, (image.width, image.height) * 2)]
                     for box in boxes[index]]
        kept, worst = None, -1.0
        for _ in range(max(1, draws)):
            try:
                drawn = (pipeline(image=source) if given is None
                         else pipeline(image=source, bboxes=np.array(given, dtype='float32').reshape(-1, 4),
                                       indices=list(range(len(given)))))
            except Exception as exc:
                raise errors.bad_request(f'image {index}: {describe(exc)}') from exc
            score = deviation(source, drawn['image'])
            if score > worst:
                kept, worst = drawn, score
        moved = None
        if given is not None:
            moved = [None] * len(given)
            for box, at in zip(kept['bboxes'], kept['indices']):
                moved[int(at)] = [round(float(v), 2) for v in box]
        shown.append(Augmented(to_image(kept['image']), moved))
    return shown
