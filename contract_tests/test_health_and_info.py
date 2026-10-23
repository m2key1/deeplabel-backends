import httpx
from deeplabel_backend.schemas import TASKS, Health, Info


def test_health_needs_no_secret_and_names_the_contract(backend):
    response = httpx.get(f'{backend.url}/health', timeout=10)
    assert response.status_code == 200
    health = Health.model_validate(response.json())
    assert health.status == 'ok'
    assert health.contract.split('.')[0] == '1'


def test_info_declares_contract_one_and_details_every_task(info):
    parsed = Info.model_validate(info)
    assert parsed.contract.split('.')[0] == '1'
    assert 'predict' in parsed.capabilities
    assert set(parsed.tasks) <= set(TASKS)
    assert set(parsed.tasks) <= set(parsed.tasks_detail)
    assert parsed.weights['roles'], 'weights.roles lists the files a version consists of'
    assert isinstance(parsed.weights['builtin'], bool)


def test_a_trainable_backend_declares_its_train_block(info):
    if 'train' not in info['capabilities']:
        assert info.get('train') is None
        return
    train = info['train']
    assert train['tasks'] and set(train['tasks']) <= set(TASKS)
    assert train['snapshot'] in ('ocr-lines/1', 'page-lines/1', 'page-boxes/1', 'map-masks/1')
    assert train['entrypoint'] and all(isinstance(part, str) for part in train['entrypoint'])
    for tunable in train.get('tunables', []):
        assert {'key', 'label', 'kind'} <= tunable.keys()


def test_a_trainable_backends_batches_and_scaling_roles_are_well_formed(info):
    """6.9: a batch per card is per GPU; the effective batch is the one its learning rates
    were tuned at; at most one tunable carries the accumulation."""
    train = info.get('train')
    if train is None:
        return
    resources = train.get('resources') or {}
    for card, batch in (resources.get('batch_by_gpu') or {}).items():
        assert isinstance(card, str) and isinstance(batch, int) and batch >= 1
    effective = resources.get('effective_batch')
    assert effective is None or (isinstance(effective, int) and effective >= 1)
    roles = [t.get('role') for t in train.get('tunables', []) if t.get('role')]
    assert set(roles) <= {'accumulation', 'learning_rate'}
    assert roles.count('accumulation') <= 1


def test_the_card_a_backend_reports_is_described_when_it_has_one(info):
    found = info.get('gpu')
    if found is None:
        return
    assert isinstance(found['name'], str) and found['count'] >= 1
    assert found['card'] is None or isinstance(found['card'], str)


def test_a_trainable_backend_declares_its_metrics_for_the_run_page(info):
    """6.9: each metric's unit and better direction; `*` in a key matches many (one per
    class); exactly one headline; the standard ones the SDK adds are there."""
    train = info.get('train')
    if train is None:
        return
    metrics = train.get('metrics') or []
    for metric in metrics:
        assert isinstance(metric['key'], str) and metric['key']
        assert metric.get('unit', 'number') in ('fraction', 'number', 'gb', 'seconds', 'per_second')
        assert metric.get('better') in (None, 'higher', 'lower')
    assert {'lr', 'gpu/peak_memory_gb'} <= {m['key'] for m in metrics}
    assert sum(1 for m in metrics if m.get('headline')) == 1
