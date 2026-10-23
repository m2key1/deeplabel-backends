from pathlib import Path

import pytest
from PIL import Image

from deeplabel_stub import model as stub
from deeplabel_stub.model import StubModel


@pytest.fixture
def sleeps(monkeypatch):
    calls = []
    monkeypatch.setattr(stub.time, 'sleep', calls.append)
    return calls


def loaded_with(tmp_path: Path, content: str | None):
    files = {}
    if content is not None:
        weights = tmp_path / 'weights'
        weights.write_text(content)
        files['weights'] = weights
    return StubModel().load(files, 'cpu')


def read(loaded, count: int) -> list[dict]:
    images = [Image.new('RGB', (40, 10), 'white')] * count
    contexts = [{'element_class': 'Header', 'line_index': 0}] * count
    return StubModel().predict(loaded, 'ocr', images, {}, contexts).results


def test_slow_weights_wait_before_every_prediction(tmp_path, sleeps):
    StubModel().predict(loaded_with(tmp_path, 'stub-slow'), 'layout', [Image.new('RGB', (8, 8))], {}, [{}])
    assert sleeps == [stub.SLOW_SECONDS]


def test_the_predict_delay_applies_to_any_weights(tmp_path, sleeps, monkeypatch):
    monkeypatch.setenv('STUB_PREDICT_DELAY', '0.3')
    StubModel().predict(loaded_with(tmp_path, None), 'layout', [Image.new('RGB', (8, 8))], {}, [{}])
    assert sleeps == [0.3]


def test_plain_weights_read_every_line_exactly(tmp_path, sleeps):
    results = read(loaded_with(tmp_path, 'any checkpoint bytes'), 4)
    assert [r['text'] for r in results] == ['Sample Header line 1'] * 4
    assert sleeps == []


def test_misread_weights_cycle_through_four_errors(tmp_path, sleeps):
    results = read(loaded_with(tmp_path, 'stub-misread'), 5)
    assert [r['text'] for r in results] == ['Sample Header line 1', 'sample Header line 1',
                                            'Sample Header line 2', 'SampleHeader line 1.',
                                            'Sample Header line 1']
    assert [r['confidence'] for r in results] == [0.95, 0.85, 0.75, 0.65, 0.95]


def test_a_misreading_model_errs_on_other_lines_in_another_decode(tmp_path, sleeps):
    loaded = loaded_with(tmp_path, 'stub-misread')
    images = [Image.new('RGB', (40, 10), 'white')] * 2
    contexts = [{'element_class': 'Header', 'line_index': 0}] * 2
    greedy = StubModel().predict(loaded, 'ocr', images, {'decode': 'greedy'}, contexts).results
    assert [r['text'] for r in greedy] == ['Sample Header line 2', 'SampleHeader line 1.']
