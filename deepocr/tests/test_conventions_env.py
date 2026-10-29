import json

import train_entrypoint


def snapshot(tmp_path, **manifest):
    root = tmp_path / 'snapshot'
    root.mkdir()
    (root / 'manifest.json').write_text(json.dumps({'kind': 'ocr-lines/1', 'groups': {'line_1': 'job1'}, **manifest}))
    return root


def test_the_flags_of_each_dataset_reach_train_py_as_its_conventions_file(tmp_path):
    root = snapshot(tmp_path, conventions={'names': ['long_s', 'keep_hyphens'],
                                           'by_dataset': {'job1': [1.0, 0.0]}})
    work = tmp_path / 'work'
    work.mkdir()

    env = train_entrypoint.conventions_env(root, work)

    assert env == {'TRAIN_CONVENTIONS': str(work / 'conventions.json')}
    assert json.loads((work / 'conventions.json').read_text()) == {
        'names': ['long_s', 'keep_hyphens'], 'by_dataset': {'job1': [1.0, 0.0]}}


def test_a_snapshot_without_flags_trains_as_before(tmp_path):
    work = tmp_path / 'work'
    work.mkdir()
    assert train_entrypoint.conventions_env(snapshot(tmp_path), work) == {}
    assert not (work / 'conventions.json').exists()


def test_a_snapshot_naming_no_flags_trains_as_before_too(tmp_path):
    work = tmp_path / 'work'
    work.mkdir()
    assert train_entrypoint.conventions_env(snapshot(tmp_path, conventions={'names': []}), work) == {}
