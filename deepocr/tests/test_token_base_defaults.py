import pytest

torch = pytest.importorskip('torch')
import train_entrypoint  # noqa: E402
from deepocr import tunables  # noqa: E402
from deeplabel_backend.train import BUILTIN_MARKER  # noqa: E402

TOKEN_BASE = {'token_decoder_lr': 1e-4, 'select_on_decoder': True, 'decoder_label_smoothing': 0.0}


def test_a_token_decoder_base_fine_tunes_as_the_token_decoder_was_trained():
    assert tunables.run_defaults({}, 'token') == TOKEN_BASE


def test_a_character_decoder_base_keeps_the_runs_defaults():
    assert tunables.run_defaults({}, 'char') == {'decoder_val_frequency': 5}
    assert tunables.run_defaults({}, None) == {'decoder_val_frequency': 5}


def test_a_value_the_run_chose_wins_over_the_bases_default():
    chosen = {'token_decoder_lr': 5e-5, 'select_on_decoder': False}
    assert tunables.run_defaults(chosen, 'token') == {'decoder_label_smoothing': 0.0, 'decoder_val_frequency': 5}


def test_the_cross_attention_rate_is_a_knob_with_its_own_variable():
    knob = next(k for k in tunables.KNOBS if k['key'] == 'token_decoder_lr')
    assert knob['env'] == ['TRAIN_TOKEN_DECODER_NEW_LR']
    assert knob['constants'] == ['TOKEN_DECODER_NEW_LR']
    assert knob['defaults_by_decoder'] == {'token': 1e-4}


def test_the_form_is_told_each_token_base_default():
    declared = {k['key']: k.get('defaults_by_decoder') for k in tunables.KNOBS}
    assert {key: declared[key] for key in TOKEN_BASE} == {key: {'token': value} for key, value in TOKEN_BASE.items()}


@pytest.mark.parametrize('decoder', ['token', 'char'])
def test_the_base_checkpoint_names_its_decoder(tmp_path, decoder):
    base = tmp_path / 'base'
    base.mkdir()
    torch.save({'model_state_dict': {}, 'decoder_type': decoder}, base / 'weights')
    assert train_entrypoint.base_decoder(base) == decoder


def test_an_older_checkpoint_without_the_field_has_a_character_decoder(tmp_path):
    base = tmp_path / 'base'
    base.mkdir()
    torch.save({'model_state_dict': {}}, base / 'weights')
    assert train_entrypoint.base_decoder(base) == 'char'


def test_a_run_from_scratch_has_no_base_decoder(tmp_path):
    base = tmp_path / 'base'
    base.mkdir()
    assert train_entrypoint.base_decoder(base) is None


def test_the_builtin_base_is_read_from_the_image(tmp_path, monkeypatch):
    builtin = tmp_path / 'builtin'
    builtin.mkdir()
    torch.save({'model_state_dict': {}, 'decoder_type': 'char'}, builtin / 'deepocr-v3.pt')
    monkeypatch.setattr(train_entrypoint, 'BUILTIN_DIR', builtin)
    base = tmp_path / 'base'
    base.mkdir()
    (base / BUILTIN_MARKER).touch()
    assert train_entrypoint.base_decoder(base) == 'char'


def test_a_token_base_run_gets_the_variables_train_py_reads():
    env = train_entrypoint.run_default_env({'hyperparams': {}}, 'token')
    assert env == {'TRAIN_TOKEN_DECODER_NEW_LR': '0.0001', 'TRAIN_SELECT_ON_DECODER': 'True',
                   'TRAIN_DECODER_LABEL_SMOOTHING': '0.0'}
