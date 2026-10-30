"""An RF-DETR trainer's augmentation in the albumentations form DeepLabel's builder edits
(7.1), and back. RF-DETR trains with its stack as `aug_config`, a dict of {name: params} in
order, so it takes each step once; it builds the stack step by step, skips a step it cannot
build with only a log line, moves boxes and masks only through the steps it lists as
geometric, and runs a OneOf or Sequential group every time whatever its p. A stack is
checked against all of these here, so a run fails at the start rather than training on
something else than was chosen."""
from . import augmentation

CONTAINERS = ('OneOf', 'SomeOf', 'Sequential')
ALWAYS_RUN = ('OneOf', 'Sequential')


def geometric_steps() -> frozenset[str]:
    from rfdetr.datasets.transforms import GEOMETRIC_TRANSFORMS
    return frozenset(GEOMETRIC_TRANSFORMS)


def entries(aug_config) -> list[tuple[str, dict]]:
    found = aug_config if isinstance(aug_config, list) else [{k: v} for k, v in aug_config.items()]
    return [next(iter(entry.items())) for entry in found]


def built(name: str, params):
    import albumentations as A
    if name in CONTAINERS:
        nested = params if isinstance(params, list) else params.get('transforms', [])
        rest = {} if isinstance(params, list) else {k: v for k, v in params.items() if k != 'transforms'}
        if name in ALWAYS_RUN:
            rest['p'] = 1.0
        return getattr(A, name)([built(n, p) for n, p in entries(nested)], **rest)
    return getattr(A, name)(**params)


def to_spec(aug_config) -> dict:
    """An aug_config as the stack RF-DETR builds from it."""
    return augmentation.to_spec([built(name, params) for name, params in entries(aug_config)])


def moves_the_page(cls) -> bool:
    return cls.__module__.startswith(('albumentations.augmentations.geometric', 'albumentations.augmentations.crops'))


def problem(node: dict, geometric: frozenset[str], carried: str, path: str = '') -> str | None:
    import albumentations as A
    name = node.get('__class_fullname__', '?')
    label = f'{path}{name}'
    if name in ALWAYS_RUN and node.get('p', 1.0) != 1.0:
        return (f"{label}: RF-DETR runs a {name} group every time, so its p must be 1 - "
                "set how often through its steps' p")
    if name in CONTAINERS:
        for index, child in enumerate(node.get('transforms', [])):
            found = problem(child, geometric, carried, f'{label}[{index}].')
            if found:
                return found
        return None
    if not hasattr(A, name):
        return f"{label}: RF-DETR builds only albumentations' own steps"
    if moves_the_page(getattr(A, name)) and name not in geometric:
        return f'{label}: RF-DETR would move the page but not its {carried} with it'
    return None


def to_aug_config(spec: dict, carried: str, geometric: frozenset[str] | None = None) -> dict:
    """A spec as RF-DETR's aug_config, or ValueError naming the step it cannot take.
    `carried` names what the page's annotations are to the reader: boxes, lines."""
    augmentation.build(spec)
    config = {}
    for index, node in enumerate(spec['transform'].get('transforms', [])):
        label = f'Compose[{index}].'
        found = problem(node, geometric_steps() if geometric is None else geometric, carried, label)
        if found:
            raise ValueError(found)
        [(name, params)] = entry(node).items()
        if name in config:
            raise ValueError(f'{label}{name}: RF-DETR takes each step once - put the copies in a group')
        config[name] = params
    return config


def entry(node: dict) -> dict:
    name = node['__class_fullname__']
    params = {k: v for k, v in node.items() if k not in ('__class_fullname__', 'transforms')}
    if name in CONTAINERS:
        params['transforms'] = [entry(child) for child in node.get('transforms', [])]
    return {name: params}
