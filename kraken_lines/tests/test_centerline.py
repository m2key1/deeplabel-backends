import math

import pytest

from kraken_lines.centerline import centerline


def test_an_upright_box_has_its_middle_as_centre_line():
    line = centerline([[10, 20], [210, 20], [210, 60], [10, 60]])
    assert all(y == pytest.approx(40) for _, y in line)
    assert line[0][0] < line[-1][0]
    assert 10 < line[0][0] < 20 and 200 < line[-1][0] < 210


def test_a_slanted_line_keeps_its_slant_and_direction():
    angle = math.radians(10)
    box = [[0, 0], [300, 0], [300, 30], [0, 30]]
    turned = [[x * math.cos(angle) - y * math.sin(angle), x * math.sin(angle) + y * math.cos(angle)] for x, y in box]
    line = centerline(turned)
    (x0, y0), (x1, y1) = line[0], line[-1]
    assert math.degrees(math.atan2(y1 - y0, x1 - x0)) == pytest.approx(10, abs=0.5)
    assert x0 < x1


def test_a_degenerate_outline_has_none():
    assert centerline([[0, 0], [10, 0], [20, 0]]) is None
