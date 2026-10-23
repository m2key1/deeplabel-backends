from collections.abc import Callable

TOLERANCE = 1e-6


class InvalidResult(ValueError):
    pass


def _fraction(value, where: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise InvalidResult(f'{where}: {value!r} is not a number')
    if not -TOLERANCE <= value <= 1 + TOLERANCE:
        raise InvalidResult(f'{where}: {value} is outside [0, 1]')


def _point(point, where: str) -> None:
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        raise InvalidResult(f'{where}: a point is [x, y]')
    _fraction(point[0], f'{where}.x')
    _fraction(point[1], f'{where}.y')


def _ring(points, where: str, minimum: int) -> None:
    if not isinstance(points, list) or len(points) < minimum:
        raise InvalidResult(f'{where}: needs at least {minimum} points')
    for index, point in enumerate(points):
        _point(point, f'{where}[{index}]')


def _items(result: dict, key: str, where: str) -> list:
    value = result.get(key)
    if not isinstance(value, list):
        raise InvalidResult(f'{where}.{key} must be a list')
    return value


def _confidence(value, where: str, optional: bool = False) -> None:
    if value is None and optional:
        return
    _fraction(value, where)


def _layout(result: dict, where: str) -> None:
    for index, detection in enumerate(_items(result, 'detections', where)):
        at = f'{where}.detections[{index}]'
        if not isinstance(detection.get('label'), str) or not detection['label']:
            raise InvalidResult(f'{at}.label must be a string')
        bbox = detection.get('bbox')
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise InvalidResult(f'{at}.bbox is [x0, y0, x1, y1]')
        for value in bbox:
            _fraction(value, f'{at}.bbox')
        if bbox[0] > bbox[2] or bbox[1] > bbox[3]:
            raise InvalidResult(f'{at}.bbox has x0 > x1 or y0 > y1')
        _confidence(detection.get('confidence'), f'{at}.confidence')


def _lines(result: dict, where: str) -> None:
    for index, line in enumerate(_items(result, 'lines', where)):
        at = f'{where}.lines[{index}]'
        _ring(line.get('polygon'), f'{at}.polygon', 3)
        _confidence(line.get('confidence'), f'{at}.confidence', optional=True)


def _ocr(result: dict, where: str) -> None:
    if not isinstance(result.get('text'), str):
        raise InvalidResult(f'{where}.text must be a string')
    _confidence(result.get('confidence'), f'{where}.confidence', optional=True)
    previous = 0.0
    for index, band in enumerate(result.get('conf_profile') or []):
        at = f'{where}.conf_profile[{index}]'
        _fraction(band.get('x0'), f'{at}.x0')
        _fraction(band.get('x1'), f'{at}.x1')
        if band['x0'] > band['x1'] + TOLERANCE or band['x0'] < previous - TOLERANCE:
            raise InvalidResult(f'{at}: bands must be ordered left to right')
        previous = band['x1']
        c = band.get('c')
        if not isinstance(c, int) or isinstance(c, bool) or not 0 <= c <= 100:
            raise InvalidResult(f'{at}.c must be an integer 0 to 100')


def _map_segmentation(result: dict, where: str) -> None:
    for index, region in enumerate(_items(result, 'regions', where)):
        at = f'{where}.regions[{index}]'
        if not isinstance(region.get('class_id'), int):
            raise InvalidResult(f'{at}.class_id must be an integer')
        _ring(region.get('points'), f'{at}.points', 3)
        for hole_index, hole in enumerate(region.get('holes') or []):
            _ring(hole, f'{at}.holes[{hole_index}]', 3)


def _edges(result: dict, where: str) -> None:
    for index, polyline in enumerate(_items(result, 'polylines', where)):
        _ring(polyline, f'{where}.polylines[{index}]', 2)


VALIDATORS: dict[str, Callable[[dict, str], None]] = {
    'layout': _layout,
    'lines': _lines,
    'ocr': _ocr,
    'map_segmentation': _map_segmentation,
    'edges': _edges,
}


def validate(task: str, results: list, expected: int) -> None:
    """Raises InvalidResult unless results hold one well-formed result per item."""
    if not isinstance(results, list) or len(results) != expected:
        count = len(results) if isinstance(results, list) else 'no list'
        raise InvalidResult(f'predict returned {count} results for {expected} items')
    for index, result in enumerate(results):
        where = f'results[{index}]'
        if not isinstance(result, dict):
            raise InvalidResult(f'{where} must be an object')
        if 'error' in result:
            error = result['error']
            if not isinstance(error, dict) or not {'code', 'message'} <= error.keys():
                raise InvalidResult(f'{where}.error needs code and message')
            continue
        VALIDATORS[task](result, where)
