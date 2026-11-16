import json

from deepocr import tunables
from deeplabel_backend.train import legacy_env


def test_a_runs_gradient_accumulation_reaches_train_py(tmp_path):
    params = tmp_path / 'params.json'
    params.write_text(json.dumps({'batch_size': 25, 'grad_accum_steps': 2, 'hyperparams': {}}))
    environ = {'TRAIN_PARAMS': str(params), 'TRAIN_OUTPUT': str(tmp_path / 'output'),
               'TRAIN_BASE': str(tmp_path / 'base')}

    env = legacy_env(environ, tunables.KNOBS)

    assert env['TRAIN_BATCH'] == '25'
    assert env['TRAIN_GRAD_ACCUM_STEPS'] == '2'

