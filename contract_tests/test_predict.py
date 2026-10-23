import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from conftest import assert_error, max_items
from deeplabel_backend import results


def results_of(response) -> list[dict]:
    assert response.status_code == 200, response.text
    return response.json()['results']


def test_results_are_valid_and_follow_the_request_order(backend, task, predict_body):
    body, images = predict_body(task, count=3)
    out = results_of(backend.predict(task, body, images))
    assert [r['id'] for r in out] == [item['id'] for item in body['items']]
    assert [r for r in out if 'error' in r] == []
    results.validate(task, [{k: v for k, v in r.items() if k != 'id'} for r in out], len(body['items']))


def test_the_same_request_gives_the_same_answer(backend, task, predict_body):
    body, images = predict_body(task)
    assert results_of(backend.predict(task, body, images)) == results_of(backend.predict(task, body, images))


def test_a_batch_over_max_items_is_too_large(backend, info, task, predict_body):
    limit = max_items(info, task)
    if not limit:
        pytest.skip(f'{task} declares no batch.max_items')
    body, images = predict_body(task, count=limit + 1, capped=False)
    assert_error(backend.predict(task, body, images), 413, 'too_large')


def test_an_undeclared_task_is_a_bad_request(backend, info, predict_body):
    body, images = predict_body(info['tasks'][0], count=1)
    assert_error(backend.predict('no-such-task', body, images), 400, 'bad_request')


def test_an_unreadable_image_fails_only_its_own_item(backend, info, task, predict_body):
    if (max_items(info, task) or 2) < 2:
        pytest.skip(f'{task} takes one item per request')
    body, images = predict_body(task, count=2)
    body['items'][1]['image'] = 'broken'
    response = backend.http.post(f'/v1/predict/{task}', files=[
        ('request', ('request.json', json.dumps(body), 'application/json')),
        ('img0', ('img0.png', images[0], 'image/png')),
        ('broken', ('broken.png', b'not an image', 'image/png'))])
    first, second = results_of(response)
    assert 'error' not in first
    assert set(second['error']) >= {'code', 'message'}


def test_ocr_reports_a_declared_decode_and_ordered_profiles(backend, info, predict_body):
    if 'ocr' not in info['tasks']:
        pytest.skip('the backend does not serve ocr')
    declared = info['tasks_detail']['ocr'].get('decodes') or []
    assert declared, 'an ocr backend declares its decodes'
    for decode in declared:
        body, images = predict_body('ocr', decode=decode)
        response = backend.predict('ocr', body, images)
        assert response.json()['decode'] in declared
        for result in results_of(response):
            bands = result.get('conf_profile') or []
            assert [b['x0'] for b in bands] == sorted(b['x0'] for b in bands)


def known_conventions(backend, info, loaded) -> list[str]:
    if loaded is not None:
        return backend.wait_weights(loaded['sha256']).get('facts', {}).get('conventions') or []
    return info['tasks_detail']['ocr'].get('conventions') or []


def test_ocr_honours_the_conventions_it_knows_and_ignores_the_rest(backend, info, loaded, predict_body):
    if 'ocr' not in info['tasks']:
        pytest.skip('the backend does not serve ocr')
    known = known_conventions(backend, info, loaded)
    asked = {name: True for name in reversed(known)} | {'no_such_flag': True}

    body, images = predict_body('ocr', convention=asked)
    response = backend.predict('ocr', body, images)
    assert response.status_code == 200, response.text
    assert response.json()['conventions'] == known

    body, images = predict_body('ocr', convention={})
    assert backend.predict('ocr', body, images).json()['conventions'] == []


def test_requests_over_the_concurrency_are_turned_away_with_retry_after(backend, info, predict_body):
    task = info['tasks'][0]
    body, images = predict_body(task, count=2)
    with ThreadPoolExecutor(info['concurrency'] * 4) as pool:
        responses = list(pool.map(lambda _: backend.predict(task, body, images), range(info['concurrency'] * 4)))
    assert {r.status_code for r in responses} <= {200, 429}
    for response in responses:
        if response.status_code == 429:
            assert_error(response, 429, 'busy')
            assert response.headers.get('Retry-After')
    assert any(r.status_code == 200 for r in responses)
