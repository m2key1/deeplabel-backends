"""Kraken trains on a centre line derived from each line's outline, so it takes the outline
as drawn (MODEL-BACKENDS.md 6.3)."""
from kraken_lines.model import KrakenBackend


def test_kraken_trains_on_the_polygons_as_drawn():
    assert KrakenBackend().train_info()['line_shape'] == 'polygon'
