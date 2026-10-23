import json

import pytest
from deeplabel_backend import Loaded
from deeplabel_backend.evaluate import add_to_metrics, final_evaluation, layout_scores, line_scores
from PIL import Image


def test_layout_is_scored_per_class_and_averaged():
    truth = [[('Header', [0, 0, .5, .1]), ('Paragraph', [0, .2, 1, .5])]]
    found = [[('Header', [0, 0, .5, .1], .9), ('Paragraph', [.6, .6, .9, .9], .8)]]

    scores = layout_scores(truth, found)

    assert scores['by_class'] == {'Header': {'AP50': pytest.approx(1.0), 'instances': 1},
                                  'Paragraph': {'AP50': 0.0, 'instances': 1}}
    assert scores['mAP50'] == pytest.approx(0.5)


def test_lines_are_matched_one_to_one():
    truth = [[('Line', [0, 0, 1, .1]), ('Line', [0, .2, 1, .3])]]
    found = [[('Line', [0, 0, 1, .1], None), ('Line', [0, 0, 1, .1], None), ('Line', [0, .8, 1, .9], None)]]

    scores = line_scores(truth, found)

    assert (scores['precision'], scores['recall']) == (pytest.approx(1 / 3), 0.5)
    assert scores['F1'] == pytest.approx(0.4)


class HalfRightLines:
    """Finds the first line of every page and nothing else."""

    def __init__(self):
        self.unloaded = False

    def load(self, files, device):
        assert files == {'weights': 'trained.ckpt'}
        return Loaded(handle='model')

    def unload(self, loaded):
        self.unloaded = True

    def predict(self, loaded, task, images, params, contexts):
        return [{'lines': [{'polygon': [[0, 0], [1, 0], [1, .1], [0, .1]], 'confidence': .9}]} for _ in images]


def lines_snapshot(root):
    for split in ('train', 'valid'):
        (root / split / 'images').mkdir(parents=True)
        (root / split / 'lines').mkdir()
    for stem, dataset in (('a_page1', 'job-a'), ('b_page1', 'job-b')):
        Image.new('RGB', (100, 100)).save(root / 'valid' / 'images' / f'{stem}.png')
        lines = [[[0, 0], [100, 0], [100, 10], [0, 10]]] + ([[[0, 50], [100, 50], [100, 60], [0, 60]]]
                                                           if dataset == 'job-b' else [])
        (root / 'valid' / 'lines' / f'{stem}.json').write_text(json.dumps(
            {'width': 100, 'height': 100, 'lines': [{'id': i, 'polygon': p} for i, p in enumerate(lines)]}))
    (root / 'manifest.json').write_text(json.dumps({'groups': {'a_page1': 'job-a', 'b_page1': 'job-b'}}))
    return root


def test_a_lines_run_is_scored_overall_and_per_dataset_through_the_model(tmp_path):
    model = HalfRightLines()

    evaluation = final_evaluation(model, {'weights': 'trained.ckpt'}, lines_snapshot(tmp_path), 'lines')

    assert evaluation['metric'] == 'F1'
    assert evaluation['overall']['F1'] == pytest.approx(0.8)
    assert evaluation['by_dataset']['job-a']['F1'] == pytest.approx(1.0)
    assert evaluation['by_dataset']['job-b']['recall'] == pytest.approx(0.5)
    assert model.unloaded


def test_the_evaluation_joins_the_metrics_with_its_headline(tmp_path):
    (tmp_path / 'metrics.json').write_text(json.dumps({'val_metric': 0.7}))

    add_to_metrics(tmp_path, {'metric': 'F1', 'overall': {'F1': 0.8}, 'by_dataset': {}})

    metrics = json.loads((tmp_path / 'metrics.json').read_text())
    assert metrics['final/F1'] == 0.8 and metrics['val_metric'] == 0.7
    assert metrics['evaluation']['metric'] == 'F1'


class FirstLineOf:
    """A model whose every weights file finds the first line of a page, and whose base
    (any other file) finds nothing - told apart by the files it was loaded with."""
    weight_roles = {'weights': 'required'}

    def __init__(self, trained):
        self.trained = trained
        self.loaded_with = []

    def load(self, files, device):
        self.loaded_with.append(files)
        return Loaded(handle=files.get('weights') == self.trained)

    def unload(self, loaded):
        pass

    def predict(self, loaded, task, images, params, contexts):
        line = [{'polygon': [[0, 0], [1, 0], [1, .1], [0, .1]], 'confidence': .9}]
        return [{'lines': line if loaded.handle else []} for _ in images]


def with_test_split(root):
    """lines_snapshot, plus its valid pages again as a test split."""
    lines_snapshot(root)
    for part in ('images', 'lines'):
        (root / 'test' / part).mkdir(parents=True)
        for path in (root / 'valid' / part).iterdir():
            (root / 'test' / part / path.name).write_bytes(path.read_bytes())
    return root


def test_the_test_split_is_scored_with_the_trained_model_and_its_base(tmp_path, monkeypatch):
    from deeplabel_backend.evaluate import after_training
    snapshot = with_test_split(tmp_path / 'snap')
    output, base = tmp_path / 'out', tmp_path / 'base'
    output.mkdir(), base.mkdir()
    (base / 'weights').write_bytes(b'base')
    monkeypatch.setenv('TRAIN_SNAPSHOT', str(snapshot))
    monkeypatch.setenv('TRAIN_BASE', str(base))
    model = FirstLineOf(output / 'model.ckpt')

    after_training(model, output, 'model.ckpt', 'lines')

    test = json.loads((output / 'metrics.json').read_text())['test_evaluation']
    assert test['trained']['overall']['recall'] == pytest.approx(2 / 3)
    assert test['trained']['by_dataset']['job-b']['recall'] == 0.5
    assert test['base']['overall']['recall'] == 0.0
    assert model.loaded_with[-1] == {'weights': base / 'weights'}


def test_a_run_without_test_pages_or_a_base_scores_only_what_it_has(tmp_path, monkeypatch):
    from deeplabel_backend.evaluate import after_training
    output = tmp_path / 'out'
    output.mkdir()
    monkeypatch.setenv('TRAIN_SNAPSHOT', str(lines_snapshot(tmp_path / 'plain')))
    monkeypatch.setenv('TRAIN_BASE', str(tmp_path / 'no-base'))
    after_training(FirstLineOf(output / 'model.ckpt'), output, 'model.ckpt', 'lines')
    assert 'test_evaluation' not in json.loads((output / 'metrics.json').read_text())

    monkeypatch.setenv('TRAIN_SNAPSHOT', str(with_test_split(tmp_path / 'tested')))
    after_training(FirstLineOf(output / 'model.ckpt'), output, 'model.ckpt', 'lines')
    test = json.loads((output / 'metrics.json').read_text())['test_evaluation']
    assert 'trained' in test and 'base' not in test


def test_a_run_on_a_builtin_base_scores_the_builtin_and_a_run_from_scratch_no_base(tmp_path, monkeypatch):
    from deeplabel_backend.evaluate import after_training
    output, base = tmp_path / 'out', tmp_path / 'base'
    output.mkdir(), base.mkdir()
    monkeypatch.setenv('TRAIN_SNAPSHOT', str(with_test_split(tmp_path / 'snap')))
    monkeypatch.setenv('TRAIN_BASE', str(base))
    model = FirstLineOf(output / 'model.ckpt')
    model.weight_roles = {'weights': 'optional'}

    after_training(model, output, 'model.ckpt', 'lines')
    assert 'base' not in json.loads((output / 'metrics.json').read_text())['test_evaluation']

    (base / 'builtin').touch()
    after_training(model, output, 'model.ckpt', 'lines')
    assert json.loads((output / 'metrics.json').read_text())['test_evaluation']['base']['overall']['recall'] == 0.0
    assert model.loaded_with[-1] == {}
