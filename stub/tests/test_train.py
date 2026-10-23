import json

import pytest

from deeplabel_stub import train


@pytest.fixture
def run(tmp_path, monkeypatch):
    return start(tmp_path, monkeypatch, {})


def start(tmp_path, monkeypatch, extra: dict):
    snapshot, output, base = tmp_path / 'snapshot', tmp_path / 'output', tmp_path / 'base'
    for folder in (snapshot, output, base):
        folder.mkdir()
    (snapshot / 'manifest.json').write_text(json.dumps({'files': []}))
    (tmp_path / 'params.json').write_text(json.dumps({'task': 'ocr', 'run': {'id': 5, 'name': 'tuned'}, **extra}))
    monkeypatch.setattr(train.time, 'sleep', lambda seconds: None)
    for name, path in (('SNAPSHOT', snapshot), ('OUTPUT', output), ('BASE', base),
                       ('PARAMS', tmp_path / 'params.json')):
        monkeypatch.setenv(f'TRAIN_{name}', str(path))
    return base, output


def test_a_model_tuned_from_a_misreading_base_misreads_too(run):
    base, output = run
    (base / 'weights').write_text('stub-misread')
    train.main()
    assert (output / 'model.ckpt').read_text() == 'stub-misread'


def test_a_model_tuned_from_any_other_base_is_a_checkpoint_of_its_run(run):
    base, output = run
    (base / 'weights').write_text('some checkpoint')
    train.main()
    assert (output / 'model.ckpt').read_text() == 'stub-checkpoint 5'


def listed(output) -> dict:
    return json.loads((output / 'artifacts.json').read_text())


def test_a_run_asking_for_a_language_model_gets_one(tmp_path, monkeypatch):
    _base, output = start(tmp_path, monkeypatch, {'build_lm': True})
    train.main()
    artifacts = listed(output)
    assert artifacts['lm_build'] == 'built'
    assert [(f['role'], (output / f['path']).read_text()) for f in artifacts['files'][1:]] == [
        ('lm', 'stub-lm 5'), ('lm_dict', 'stub-lm-dict 5')]


def test_a_language_model_build_marked_to_fail_fails_and_leaves_the_weights(tmp_path, monkeypatch):
    _base, output = start(tmp_path, monkeypatch, {'build_lm': True})
    params = output.parent / 'params.json'
    params.write_text(json.dumps({**json.loads(params.read_text()), 'run': {'id': 5, 'name': 'tuned [lm-fails]'}}))
    train.main()
    artifacts = listed(output)
    assert (artifacts['lm_build'], [f['role'] for f in artifacts['files']]) == ('failed', ['weights'])


def test_a_run_not_asking_for_one_gets_none(run):
    _base, output = run
    train.main()
    assert 'lm_build' not in listed(output)
