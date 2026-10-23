import os

import pytest
from conftest import assert_error, load_request, sha256


@pytest.fixture
def need_weights(weights_files):
    if weights_files is None:
        pytest.skip('needs --weights-file')
    return weights_files


def test_a_version_loads_and_loading_it_again_is_instant(backend, info, files, need_weights, loaded):
    request = load_request(files, info, need_weights)
    state = backend.wait_weights(request['sha256'])
    assert state['state'] == 'loaded'
    assert isinstance(state['facts'], dict)
    again = backend.post('/v1/weights/load', json=request, timeout=5)
    assert again.status_code == 200, again.text
    assert again.json()['state'] == 'loaded'


def test_a_dropped_download_resumes_with_a_range_request(backend, info, files):
    data = os.urandom(4 << 20)
    weights = files.add_file(data, drop_once=True)
    request = {'sha256': weights['sha256'], 'model_type': info['model_type'], 'files': {'weights': weights}}
    response = backend.post('/v1/weights/load', json=request)
    assert response.status_code in (200, 202, 500), response.text
    state = backend.wait_weights(weights['sha256'])
    # Random bytes need not load as a model; what matters is that the transfer completed.
    if state['state'] == 'error':
        assert state['error']['code'] not in ('weights_unavailable', 'weights_checksum_mismatch'), state
    else:
        assert state['state'] == 'loaded'
    assert files.ranged_requests(weights['sha256']), 'the second attempt asked for the rest with Range'


def test_a_corrupted_download_is_a_checksum_mismatch(backend, info, files):
    data = os.urandom(1 << 20)
    claimed = sha256(data + b'!')
    weights = files.add_file(data, claimed_sha=claimed)
    response = backend.post('/v1/weights/load', json={'sha256': claimed, 'model_type': info['model_type'],
                                                      'files': {'weights': weights}})
    if response.status_code >= 400:
        assert_error(response, 422, 'weights_checksum_mismatch')
    state = backend.wait_weights(claimed)
    assert state['state'] == 'error'
    assert state['error']['code'] == 'weights_checksum_mismatch'
    assert state['error']['retryable'] is False


def test_a_file_deeplabel_cannot_serve_is_unavailable(backend, info, files):
    missing = sha256(os.urandom(32))
    response = backend.post('/v1/weights/load', json={'sha256': missing, 'model_type': info['model_type'],
                                                      'files': {'weights': files.missing_file(missing)}})
    if response.status_code >= 400:
        assert_error(response, 424, 'weights_unavailable')
    state = backend.wait_weights(missing)
    assert state['state'] == 'error'
    assert state['error']['code'] == 'weights_unavailable'


def test_weights_of_another_family_are_refused(backend, info, files):
    if not info['model_type']:
        pytest.skip('the backend declares no model_type')
    weights = files.add_file(os.urandom(1024))
    response = backend.post('/v1/weights/load', json={'sha256': weights['sha256'], 'model_type': 'not-' + info['model_type'],
                                                      'files': {'weights': weights}})
    assert_error(response, 409, 'weights_incompatible')


def test_an_unknown_version_reports_absent(backend):
    assert backend.get('/v1/weights/' + sha256(os.urandom(32))).json() == {'state': 'absent'}


def test_predict_with_a_version_never_loaded_is_refused(backend, info, predict_body):
    task = info['tasks'][0]
    body, images = predict_body(task, count=1)
    body['weights'] = {'sha256': sha256(os.urandom(32)), 'files': {}}
    assert_error(backend.predict(task, body, images), 409, 'weights_not_loaded')
