"""A run that names no epochs trains for train.py's own EPOCHS, and the training form is told
how many (MODEL-BACKENDS.md 6.2)."""
import pytest

pytest.importorskip('torch')
from deepocr import tunables
from deepocr.model import DeepOCRBackend


def test_the_form_is_told_the_epochs_train_py_runs_by_default(monkeypatch):
    probed = {**tunables.constants(), 'EPOCHS': 37}
    monkeypatch.setattr(tunables, 'constants', lambda: probed)

    assert DeepOCRBackend().train_info()['default_epochs'] == 37
