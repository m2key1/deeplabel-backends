from dataclasses import dataclass
from itertools import product

OVERLAP_FRACTION = 4


@dataclass(frozen=True)
class Window:
    x: int
    y: int
    keep_left: int
    keep_top: int
    keep_right: int
    keep_bottom: int


def axis_spans(length: int, window: int) -> list[tuple[int, int, int]]:
    """(start, keep_from, keep_to) per window along one side, keep_* in tile pixels. A side
    no longer than the window gets one window at 0 (the runner pads the tile out to it)."""
    if length <= window:
        return [(0, 0, length)]
    stride = window - window // OVERLAP_FRACTION
    starts = list(range(0, length - window + 1, stride))
    if starts[-1] != length - window:
        starts.append(length - window)
    cuts = [(nxt + start + window) // 2 for start, nxt in zip(starts, starts[1:])]
    bounds = [0, *cuts, length]
    return [(start, bounds[i], bounds[i + 1]) for i, start in enumerate(starts)]


def windows(width: int, height: int, window: int) -> list[Window]:
    """Every window over a width x height tile, row by row."""
    return [Window(x, y, left, top, right, bottom)
            for (y, top, bottom), (x, left, right)
            in product(axis_spans(height, window), axis_spans(width, window))]
