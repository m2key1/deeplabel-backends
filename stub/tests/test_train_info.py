import pytest

from deeplabel_stub.model import StubModel


@pytest.mark.parametrize('snapshot, task', [('ocr-lines/1', 'ocr'), ('page-lines/1', 'lines'),
                                             ('page-boxes/1', 'layout')])
def test_the_snapshot_kind_comes_from_the_environment(monkeypatch, snapshot, task):
    monkeypatch.setenv('STUB_TRAIN_SNAPSHOT', snapshot)
    train = StubModel().train_info()
    assert train['snapshot'] == snapshot
    assert train['tasks'] == [task]


def test_without_the_environment_the_stub_trains_ocr(monkeypatch):
    monkeypatch.delenv('STUB_TRAIN_SNAPSHOT', raising=False)
    train = StubModel().train_info()
    assert (train['snapshot'], train['tasks']) == ('ocr-lines/1', ['ocr'])


def test_an_unknown_snapshot_kind_is_refused(monkeypatch):
    monkeypatch.setenv('STUB_TRAIN_SNAPSHOT', 'map-masks/1')
    with pytest.raises(ValueError, match='STUB_TRAIN_SNAPSHOT'):
        StubModel().train_info()


def test_only_the_line_trainer_offers_a_detector_size(monkeypatch):
    monkeypatch.setenv('STUB_TRAIN_SNAPSHOT', 'page-lines/1')
    variant = next(t for t in StubModel().train_info()['tunables'] if t['key'] == 'variant')
    assert variant['choices'] == ['tiny', 'small', 'medium']
    monkeypatch.setenv('STUB_TRAIN_SNAPSHOT', 'ocr-lines/1')
    assert 'variant' not in [t['key'] for t in StubModel().train_info()['tunables']]


def test_only_the_line_trainer_declares_the_shape_its_lines_train_as(monkeypatch):
    """Neither DeepLabel's own default nor its fallback, so a form showing it shows the backend's."""
    monkeypatch.setenv('STUB_TRAIN_SNAPSHOT', 'page-lines/1')
    assert StubModel().train_info()['line_shape'] == 'box'
    monkeypatch.setenv('STUB_TRAIN_SNAPSHOT', 'ocr-lines/1')
    assert 'line_shape' not in StubModel().train_info()


def test_only_the_ocr_trainer_builds_a_language_model(monkeypatch):
    monkeypatch.setenv('STUB_TRAIN_SNAPSHOT', 'ocr-lines/1')
    assert StubModel().train_info()['outputs'] == ['weights', 'lm', 'lm_dict']
    monkeypatch.setenv('STUB_TRAIN_SNAPSHOT', 'page-lines/1')
    assert StubModel().train_info()['outputs'] == ['weights']


def test_an_ocr_stub_says_it_learns_convention_flags(monkeypatch):
    monkeypatch.setenv('STUB_TRAIN_SNAPSHOT', 'ocr-lines/1')
    assert StubModel().train_info()['conventions'] is True


def test_a_lines_stub_does_not(monkeypatch):
    monkeypatch.setenv('STUB_TRAIN_SNAPSHOT', 'page-lines/1')
    assert 'conventions' not in StubModel().train_info()

