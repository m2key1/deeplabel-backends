"""A run that names no epochs trains for 50, and the training form is told so
(MODEL-BACKENDS.md 6.2)."""
import pytest

pytest.importorskip('torch')
from kraken_lines.model import KrakenBackend


def test_the_form_is_told_the_epochs_a_run_trains_for_when_it_names_none():
    assert KrakenBackend().train_info()['default_epochs'] == 50
