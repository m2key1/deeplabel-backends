"""Every knob a DeepOCR training run may tune, as the training form shows it (6.2): the
variables train.py reads for it, and its default - read from train.py itself, by importing
it the way a training does, so the default shown is what an untuned run uses."""
import functools
import json
import subprocess
import sys

from deepocr import MODEL_REPO

PROBE_TIMEOUT = 120

KNOBS = [{'key': 'learning_rate', 'role': 'learning_rate',
  'label': 'Learning rate',
  'group': 'Schedule',
  'kind': 'float',
  'help': 'Peak rate for every parameter group, scaled by the square root of the effective batch over 60. A '
          'per-group rate below wins over it.',
  'constants': ['BACKBONE_BASE_LR', 'TRANSFORMER_BASE_LR', 'DECODER_BASE_LR'],
  'env': ['TRAIN_LR'],
  'above': 0},
 {'key': 'backbone_lr', 'role': 'learning_rate',
  'label': 'Backbone rate',
  'group': 'Schedule',
  'kind': 'float',
  'help': 'Peak rate for the CNN backbone only.',
  'constants': ['BACKBONE_BASE_LR'],
  'env': ['TRAIN_BACKBONE_BASE_LR'],
  'above': 0},
 {'key': 'encoder_lr', 'role': 'learning_rate',
  'label': 'Encoder rate',
  'group': 'Schedule',
  'kind': 'float',
  'help': 'Peak rate for the MoE transformer encoder only.',
  'constants': ['TRANSFORMER_BASE_LR'],
  'env': ['TRAIN_TRANSFORMER_BASE_LR'],
  'above': 0},
 {'key': 'decoder_lr', 'role': 'learning_rate',
  'label': 'Decoder rate',
  'group': 'Schedule',
  'kind': 'float',
  'help': 'Peak rate for the character decoder only.',
  'constants': ['DECODER_BASE_LR'],
  'env': ['TRAIN_DECODER_BASE_LR'],
  'above': 0},
 {'key': 'warmup_epochs',
  'label': 'Warmup epochs',
  'group': 'Schedule',
  'kind': 'float',
  'help': 'Linear warmup to the peak rate, in epochs; fractions allowed.',
  'constants': ['WARMUP_EPOCHS'],
  'env': ['TRAIN_WARMUP_EPOCHS'],
  'min': 0},
 {'key': 'min_lr',
  'label': 'Minimum rate',
  'group': 'Schedule',
  'kind': 'float',
  'help': 'Floor the cosine decay reaches at the end of the run.',
  'constants': ['MIN_LR'],
  'env': ['TRAIN_MIN_LR'],
  'min': 0},
 {'key': 'weight_decay',
  'label': 'Weight decay',
  'group': 'Schedule',
  'kind': 'float',
  'help': 'AdamW weight decay for every group.',
  'constants': ['WEIGHT_DECAY'],
  'env': ['TRAIN_WEIGHT_DECAY'],
  'min': 0},
 {'key': 'ema_decay',
  'label': 'EMA decay',
  'group': 'Schedule',
  'kind': 'float',
  'help': 'Decay of the weight average that is validated and saved; closer to 1 averages over more steps.',
  'constants': ['EMA_DECAY'],
  'env': ['TRAIN_EMA_DECAY'],
  'min': 0,
  'max': 1},
 {'key': 'ctc_loss_weight',
  'label': 'CTC weight',
  'group': 'Loss',
  'kind': 'float',
  'help': 'Weight of the CTC head in the total loss.',
  'constants': ['CTC_LOSS_WEIGHT'],
  'env': ['TRAIN_CTC_LOSS_WEIGHT'],
  'min': 0},
 {'key': 'decoder_loss_weight',
  'label': 'Decoder weight',
  'group': 'Loss',
  'kind': 'float',
  'help': 'Weight of the autoregressive decoder head in the total loss.',
  'constants': ['DECODER_LOSS_WEIGHT'],
  'env': ['TRAIN_DECODER_LOSS_WEIGHT'],
  'min': 0},
 {'key': 'ctc_entropy_weight',
  'label': 'CTC entropy weight',
  'group': 'Loss',
  'kind': 'float',
  'help': 'Entropy smoothing on the CTC output, against overconfident blanks.',
  'constants': ['CTC_ENTROPY_WEIGHT'],
  'env': ['TRAIN_CTC_ENTROPY_WEIGHT'],
  'min': 0},
 {'key': 'decoder_label_smoothing',
  'label': 'Label smoothing',
  'group': 'Loss',
  'kind': 'float',
  'help': 'Label smoothing of the decoder cross-entropy.',
  'constants': ['DECODER_LABEL_SMOOTHING'],
  'env': ['TRAIN_DECODER_LABEL_SMOOTHING'],
  'min': 0,
  'max': 1},
 {'key': 'decoder_eos_weight',
  'label': 'End-of-sequence weight',
  'group': 'Loss',
  'kind': 'float',
  'help': 'Extra weight on the end token, so the decoder learns to stop.',
  'constants': ['DECODER_EOS_WEIGHT'],
  'env': ['TRAIN_DECODER_EOS_WEIGHT'],
  'min': 0},
 {'key': 'moe_penalty',
  'label': 'MoE balance penalty',
  'group': 'Loss',
  'kind': 'float',
  'help': 'Switch load-balance loss coefficient across the experts.',
  'constants': ['MOE_PENALTY'],
  'env': ['TRAIN_MOE_PENALTY'],
  'min': 0},
 {'key': 'router_z_loss',
  'label': 'Router z-loss',
  'group': 'Loss',
  'kind': 'float',
  'help': 'Keeps router logits small for stable expert routing.',
  'constants': ['ROUTER_Z_LOSS_COEF'],
  'env': ['TRAIN_ROUTER_Z_LOSS_COEF'],
  'min': 0},
 {'key': 'drop_path',
  'label': 'Drop path',
  'group': 'Regularisation',
  'kind': 'float',
  'help': 'Stochastic depth rate in the encoder layers.',
  'constants': ['DROP_PATH'],
  'env': ['TRAIN_DROP_PATH'],
  'min': 0,
  'max': 1},
 {'key': 'decoder_dropout',
  'label': 'Decoder dropout',
  'group': 'Regularisation',
  'kind': 'float',
  'help': 'Dropout inside the decoder.',
  'constants': ['DECODER_DROPOUT'],
  'env': ['TRAIN_DECODER_DROPOUT'],
  'min': 0,
  'max': 1},
 {'key': 'mask_ratio',
  'label': 'Mask ratio',
  'group': 'Regularisation',
  'kind': 'float',
  'help': 'Share of feature columns masked at the start of the run; annealed to the final ratio.',
  'constants': ['MASK_RATIO'],
  'env': ['TRAIN_MASK_RATIO'],
  'min': 0,
  'max': 1},
 {'key': 'mask_ratio_final',
  'label': 'Final mask ratio',
  'group': 'Regularisation',
  'kind': 'float',
  'help': 'Share of feature columns still masked at the end of the run.',
  'constants': ['MASK_RATIO_FINAL'],
  'env': ['TRAIN_MASK_RATIO_FINAL'],
  'min': 0,
  'max': 1},
 {'key': 'disable_aug_after_epoch',
  'label': 'Augmentation off after epoch',
  'group': 'Regularisation',
  'kind': 'int',
  'help': 'Train the remaining epochs on clean images. Unset keeps augmentation on throughout.',
  'constants': ['DISABLE_AUG_AFTER_EPOCH'],
  'env': ['TRAIN_DISABLE_AUG_AFTER_EPOCH'],
  'min': 0},
 {'key': 'patience',
  'label': 'Patience',
  'group': 'Validation',
  'kind': 'int',
  'help': 'Stop when the validation CER has not improved for this many epochs.',
  'constants': [],
  'env': ['TRAIN_PATIENCE'],
  'min': 1},
 {'key': 'valid_limit',
  'label': 'Validation limit',
  'group': 'Validation',
  'kind': 'int',
  'help': 'Cap on the lines scored per validation pass; a seeded, representative subset.',
  'constants': ['VALID_LIMIT'],
  'env': ['TRAIN_VALID_LIMIT'],
  'min': 1},
 {'key': 'val_frequency',
  'label': 'Validate every',
  'group': 'Validation',
  'kind': 'int',
  'help': 'Epochs between validation passes.',
  'constants': ['VAL_FREQUENCY'],
  'env': ['TRAIN_VAL_FREQUENCY'],
  'min': 1},
 {'key': 'beam_val_frequency',
  'label': 'Beam validation every',
  'group': 'Validation',
  'kind': 'int',
  'help': 'Epochs between the expensive KenLM beam-search validations.',
  'constants': ['BEAM_VAL_FREQUENCY'],
  'env': ['TRAIN_BEAM_VAL_FREQUENCY'],
  'min': 1},
 {'key': 'decoder_val_frequency',
  'label': 'Decoder validation every',
  'group': 'Validation',
  'kind': 'int',
  'help': 'Epochs between decoder-rescored validations. Every epoch while the checkpoint is selected on the '
          'decoder, whatever this says.',
  'constants': ['DECODER_VAL_FREQUENCY'],
  'env': ['TRAIN_DECODER_VAL_FREQUENCY'],
  'min': 1},
 {'key': 'beam_width',
  'label': 'Beam width',
  'group': 'Validation',
  'kind': 'int',
  'help': 'Beam width of the beam-search validation.',
  'constants': ['BEAM_WIDTH'],
  'env': ['TRAIN_BEAM_WIDTH'],
  'min': 1},
 {'key': 'select_on_decoder',
  'label': 'Select on decoder CER',
  'group': 'Validation',
  'kind': 'bool',
  'help': 'Keep the checkpoint with the best decoder CER instead of the best CTC CER.',
  'constants': ['SELECT_ON_DECODER'],
  'env': ['TRAIN_SELECT_ON_DECODER']},
 {'key': 'save_best_of_heads',
  'label': 'Save best of either head',
  'group': 'Validation',
  'kind': 'bool',
  'help': 'Keep the checkpoint whenever either head improves.',
  'constants': ['SAVE_BEST_OF_HEADS'],
  'env': ['TRAIN_SAVE_BEST_OF_HEADS']},
 {'key': 'freeze_backbone',
  'label': 'Freeze backbone',
  'group': 'Fine-tuning',
  'kind': 'bool',
  'help': 'Keep the CNN backbone fixed and train only the encoder and decoder.',
  'constants': ['FREEZE_BACKBONE'],
  'env': ['TRAIN_FREEZE_BACKBONE']},
 {'key': 'reset_experts',
  'label': 'Reset experts',
  'group': 'Fine-tuning',
  'kind': 'bool',
  'help': 'Re-initialise the expert weights of the base checkpoint.',
  'constants': ['RESET_EXPERTS'],
  'env': ['TRAIN_RESET_EXPERTS']},
 {'key': 'reset_moe_router',
  'label': 'Reset routers',
  'group': 'Fine-tuning',
  'kind': 'bool',
  'help': 'Re-initialise the expert routers of the base checkpoint.',
  'constants': ['RESET_MOE_ROUTER'],
  'env': ['TRAIN_RESET_MOE_ROUTER']},
 {'key': 'warmup_router_epochs',
  'label': 'Router warmup epochs',
  'group': 'Fine-tuning',
  'kind': 'int',
  'help': 'Epochs over which expert routing is ramped in from an even mix.',
  'constants': ['WARMUP_ROUTER_EPOCHS'],
  'env': ['TRAIN_WARMUP_ROUTER_EPOCHS'],
  'min': 0},
 {'key': 'decoder_warmup_epochs',
  'label': 'Decoder warmup epochs',
  'group': 'Fine-tuning',
  'kind': 'int',
  'help': 'Epochs over which the decoder loss weight ramps up to its full value.',
  'constants': ['DECODER_WARMUP_EPOCHS'],
  'env': ['TRAIN_DECODER_WARMUP_EPOCHS'],
  'min': 0},
 {'key': 'sampler_seed',
  'label': 'Sampler seed',
  'group': 'Data',
  'kind': 'int',
  'help': 'Seed of the batch order; the same seed reproduces the same order.',
  'constants': ['SAMPLER_SEED'],
  'env': ['TRAIN_SAMPLER_SEED'],
  'min': 0},
 {'key': 'num_workers',
  'label': 'Loader workers',
  'group': 'Data',
  'kind': 'int',
  'help': 'Data loader processes per GPU. Host memory grows with GPUs times workers.',
  'constants': ['NUM_WORKERS'],
  'env': ['TRAIN_NUM_WORKERS'],
  'min': 0}]

# What a run started from DeepLabel takes where nobody chose: train.py scores both heads every
# epoch by default, which on a cluster run spent half the wall clock validating the decoder.
RUN_DEFAULTS = {'decoder_val_frequency': 5}
# Knobs that make the decoder's error rate pick the checkpoint; it is then scored every round.
SELECTS_ON_DECODER = ('select_on_decoder', 'save_best_of_heads')


# Read along, not tuned: the input geometry the augmentation preview has to match.
GEOMETRY = ('TARGET_HEIGHT', 'MAX_WIDTH', 'EPOCHS', 'BATCH_SIZE')

PROBE = """
import importlib, json, sys
sys.path.insert(0, '.')
module = importlib.import_module('train')
names = sys.argv[1].split(',')
print(json.dumps({n: getattr(module, n) for n in names if isinstance(getattr(module, n, None), (bool, int, float, str))}))
"""


@functools.cache
def constants() -> dict:
    names = sorted({c for knob in KNOBS for c in knob['constants']} | set(GEOMETRY))
    done = subprocess.run([sys.executable, '-c', PROBE, ','.join(names)], cwd=MODEL_REPO, capture_output=True,
                          text=True, timeout=PROBE_TIMEOUT)
    if done.returncode:
        raise RuntimeError(f'train.py could not be read for its defaults: {done.stderr.strip()[-300:]}')
    return json.loads(done.stdout.strip().splitlines()[-1])


STACK_PROBE = """
import importlib, json, os, sys
sys.path.insert(0, '.')
os.environ.pop('TRAIN_AUGMENT', None)
import albumentations as A
print(json.dumps(A.to_dict(importlib.import_module('train').get_transforms()[0])))
"""


@functools.cache
def default_augmentation() -> dict:
    """The stack train.py applies when a run names none, as get_transforms() builds it."""
    done = subprocess.run([sys.executable, '-c', STACK_PROBE], cwd=MODEL_REPO, capture_output=True, text=True,
                          timeout=PROBE_TIMEOUT)
    if done.returncode:
        raise RuntimeError(f'train.py could not be read for its augmentation: {done.stderr.strip()[-300:]}')
    return json.loads(done.stdout.strip().splitlines()[-1])


def default(knob: dict, found: dict):
    if knob['key'] in RUN_DEFAULTS:
        return RUN_DEFAULTS[knob['key']]
    values = {json.dumps(found.get(name)) for name in knob['constants']}
    return found.get(knob['constants'][0]) if knob['constants'] and len(values) == 1 else None


def declared() -> list[dict]:
    """The train block's tunables: each knob with its default, and the variables it sets."""
    found = constants()
    return [{**{k: v for k, v in knob.items() if k != 'constants'}, 'default': default(knob, found)}
            for knob in KNOBS]


def run_defaults(chosen: dict) -> dict:
    """The knobs a run takes for itself, minus any its own choices rule out."""
    defaults = dict(RUN_DEFAULTS)
    if any(chosen.get(knob) for knob in SELECTS_ON_DECODER):
        defaults.pop('decoder_val_frequency', None)
    return {key: value for key, value in defaults.items() if key not in chosen}
