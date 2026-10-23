import numpy as np
import shapely.geometry as geom
from shapely import affinity

SAMPLES = 16


def centerline(polygon: list[list[float]]) -> list[list[float]] | None:
    """The middle of a line's outline, from its start to its end: the outline turned so its
    long side lies along x, cut across at even steps, the midpoint of each cut, turned back.
    Follows a slant or a gentle curve; None for an outline too small to have one."""
    ring = geom.Polygon(polygon)
    if not ring.is_valid:
        ring = ring.buffer(0)
    if ring.is_empty or ring.area <= 0:
        return None
    corners = np.asarray(ring.minimum_rotated_rectangle.exterior.coords)[:4]
    sides = [corners[1] - corners[0], corners[2] - corners[1]]
    along = max(sides, key=lambda side: np.hypot(*side))
    angle = np.degrees(np.arctan2(along[1], along[0]))
    if abs(angle) > 90:
        angle -= 180 * np.sign(angle)
    centre = ring.centroid
    level = affinity.rotate(ring, -angle, origin=centre)
    x0, y0, x1, y1 = level.bounds
    inset = (x1 - x0) / (2 * SAMPLES)
    points = []
    for x in np.linspace(x0 + inset, x1 - inset, SAMPLES):
        cut = level.intersection(geom.LineString([(x, y0 - 1), (x, y1 + 1)]))
        if cut.is_empty:
            continue
        _, low, _, high = cut.bounds
        points.append((x, (low + high) / 2))
    if len(points) < 2:
        return None
    turned = affinity.rotate(geom.LineString(points), angle, origin=centre)
    return [list(point) for point in turned.coords]
