import httpx
import pytest
from conftest import assert_error
from fixture_server import SECRET_HEADER


def routes(info: dict) -> list[tuple[str, str]]:
    listed = [('GET', '/v1/info'), ('POST', '/v1/weights/load'), ('GET', '/v1/weights/' + '0' * 64),
              ('POST', '/v1/weights/unload')]
    listed += [('POST', f'/v1/predict/{task}') for task in info['tasks']]
    if 'train' in info['capabilities']:
        listed += [('POST', '/v1/train'), ('GET', '/v1/train/tunables'), ('GET', '/v1/train/augmentations'),
                   ('GET', '/v1/train/x'), ('GET', '/v1/train/x/progress'), ('POST', '/v1/train/x/cancel'),
                   ('GET', '/v1/train/x/artifacts'), ('DELETE', '/v1/train/x')]
    return listed


@pytest.mark.parametrize('header', [None, 'wrong-secret'], ids=['missing', 'wrong'])
def test_every_v1_route_refuses_a_missing_or_wrong_secret(backend, info, header):
    headers = {SECRET_HEADER: header} if header else {}
    checked = routes(info)
    for method, path in checked:
        response = httpx.request(method, backend.url + path, headers=headers, json={}, timeout=10)
        assert_error(response, 401, 'unauthorized')
    assert len(checked) >= 4 + len(info['tasks'])
