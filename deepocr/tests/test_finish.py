import json

from deepocr.finish import keep_best


def test_a_run_stopped_early_leaves_what_train_py_would_have(tmp_path):
    (tmp_path / 'run').mkdir()
    (tmp_path / 'run' / 'best_model.pt').write_bytes(b'epoch-1 weights')
    (tmp_path / 'progress.jsonl').write_text('\n'.join(json.dumps(p) for p in [
        {'epoch': 1, 'loss': 0.56}, {'epoch': 1, 'cer': 0.0222, 'best_cer': 0.0222},
        {'epoch': 2, 'loss': 0.51}, {'epoch': 2, 'cer': 0.0235, 'best_cer': 0.0222}]) + '\n')

    assert keep_best(tmp_path) is True

    assert (tmp_path / 'model.ckpt').read_bytes() == b'epoch-1 weights'
    assert json.loads((tmp_path / 'metrics.json').read_text()) == {'cer': 0.0222}


def test_a_run_stopped_before_its_first_validation_has_nothing_to_keep(tmp_path):
    (tmp_path / 'progress.jsonl').write_text(json.dumps({'epoch': 1, 'loss': 0.56}) + '\n')

    assert keep_best(tmp_path) is False

    assert not (tmp_path / 'model.ckpt').exists()
    assert not (tmp_path / 'metrics.json').exists()
