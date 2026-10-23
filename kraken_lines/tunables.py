"""Every knob a Kraken segmentation run may tune, as the training form shows it (6.2), with
its default read from Kraken's own training configuration, so the default shown is what an
untuned run uses."""
from kraken.configs import BLLASegmentationTrainingConfig, BLLASegmentationTrainingDataConfig

KNOBS = [
    {'key': 'lrate', 'role': 'learning_rate', 'label': 'Learning rate', 'group': 'Schedule', 'kind': 'float', 'above': 0,
     'help': 'Peak rate of the optimiser.', 'source': 'model'},
    {'key': 'schedule', 'label': 'Rate schedule', 'group': 'Schedule', 'kind': 'choice',
     'choices': ['constant', 'cosine', 'exponential', 'step', 'reduceonplateau'],
     'help': 'How the rate moves over the run.', 'source': 'model'},
    {'key': 'cos_min_lr', 'label': 'Final cosine rate', 'group': 'Schedule', 'kind': 'float', 'min': 0,
     'help': 'Where the cosine schedule ends.', 'source': 'model'},
    {'key': 'warmup', 'label': 'Warm-up steps', 'group': 'Schedule', 'kind': 'int', 'min': 0,
     'help': 'Steps over which the rate ramps up to its peak.', 'source': 'model'},
    {'key': 'weight_decay', 'label': 'Weight decay', 'group': 'Schedule', 'kind': 'float', 'min': 0,
     'help': 'AdamW weight decay.', 'source': 'model'},
    {'key': 'accumulate_grad_batches', 'role': 'accumulation', 'label': 'Gradient accumulation', 'group': 'Memory', 'kind': 'int', 'min': 1,
     'help': 'Pages summed before each optimiser step.', 'source': 'model'},
    {'key': 'dice_weight', 'label': 'Dice loss weight', 'group': 'Model', 'kind': 'float', 'min': 0,
     'help': 'Weight of the Dice term beside the per-pixel loss; 0 turns it off.', 'source': 'model'},
    {'key': 'line_width', 'label': 'Target line width', 'group': 'Model', 'kind': 'int', 'min': 1,
     'help': 'Thickness of each centre line in the training target, in pixels after scaling.', 'source': 'data'},
    {'key': 'augment', 'label': 'Augmentation', 'group': 'Data', 'kind': 'bool',
     'help': "Kraken's own image augmentation.", 'source': 'data'},
    {'key': 'precision', 'label': 'Precision', 'group': 'Memory', 'kind': 'choice',
     'choices': ['32-true', 'bf16-mixed'],
     'help': 'bfloat16 mixed precision, as Kraken recommends, or float32 throughout. Measured on the 5090 '
             'the same F1 (0.678 against 0.672 after ten epochs), 28% less memory, no faster - '
             'the run waits on its data, not the card.',
     'source': 'trainer'},
]
# What the Lightning trainer itself is given, beside Kraken's two configurations.
TRAINER_DEFAULTS = {'precision': 'bf16-mixed'}


def declared() -> list[dict]:
    """The train block's tunables, each with Kraken's default."""
    defaults = {'model': BLLASegmentationTrainingConfig(), 'data': BLLASegmentationTrainingDataConfig()}
    return [{**{k: v for k, v in knob.items() if k != 'source'},
             'default': TRAINER_DEFAULTS[knob['key']] if knob['source'] == 'trainer'
             else getattr(defaults[knob['source']], knob['key'])}
            for knob in KNOBS]
