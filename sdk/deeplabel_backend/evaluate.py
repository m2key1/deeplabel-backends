"""The final evaluation of a layout or lines run: the trained model reads the
snapshot's validation pages through the backend's own predict path, and its answers are
scored against the snapshot, overall, per dataset (the manifest's groups) and per class.

Layout is scored as COCO-style AP at IoU 0.5 per class, averaged into mAP50. Lines are
scored as precision, recall and F1 of lines matched one to one at IoU 0.5, a line's
outline taken as its bounding box - what decides whether a reading finds the line at all.
The result goes into metrics.json as `evaluation`, which DeepLabel shows as a table. A
snapshot with a test split - the run's test datasets, never trained on - is
scored as well, by the trained model and by the base it started from, as `test_evaluation`.
"""
import json
from itertools import accumulate
from pathlib import Path
from statistics import fmean

from PIL import Image

from .train import BUILTIN_MARKER

MATCH_IOU = 0.5
PAGE_BATCH = 4


def iou(a, b) -> float:
    inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _matches(truth: list, found: list, label: str | None = None) -> tuple[list[bool], int]:
    """Greedy one-to-one matching at MATCH_IOU, the most confident answer first: whether
    each answer found a line or box of its class, and how many were there to find."""
    wanted = sum(1 for page in truth for name, _ in page if label is None or name == label)
    candidates = sorted(((conf if conf is not None else 1.0, -order, p, box)
                         for p, page in enumerate(found)
                         for order, (name, box, conf) in enumerate(page) if label is None or name == label),
                        reverse=True)
    taken = [set() for _ in truth]
    hits = []
    for _, _, p, box in candidates:
        best, best_iou = None, MATCH_IOU
        for t, (name, gt) in enumerate(truth[p]):
            if (label is None or name == label) and t not in taken[p] and (overlap := iou(box, gt)) >= best_iou:
                best, best_iou = t, overlap
        if best is not None:
            taken[p].add(best)
        hits.append(best is not None)
    return hits, wanted


def average_precision(truth: list, found: list, label: str) -> float:
    """COCO's 101-point interpolated AP at IoU 0.5 for one class."""
    hits, wanted = _matches(truth, found, label)
    if not hits or not wanted:
        return 0.0
    tp = list(accumulate(int(hit) for hit in hits))
    recall = [t / wanted for t in tp]
    precision = [t / n for n, t in enumerate(tp, 1)]
    envelope = list(accumulate(reversed(precision), max))[::-1]
    return fmean(max((e for e, r in zip(envelope, recall) if r >= level), default=0.0)
                 for level in (i / 100 for i in range(101)))


def layout_scores(truth: list, found: list) -> dict:
    classes = sorted({name for page in truth for name, _ in page})
    by_class = {name: {'AP50': average_precision(truth, found, name),
                       'instances': sum(1 for page in truth for n, _ in page if n == name)} for name in classes}
    return {'mAP50': fmean(c['AP50'] for c in by_class.values()) if by_class else 0.0,
            'pages': len(truth), 'instances': sum(len(page) for page in truth), 'by_class': by_class}


def line_scores(truth: list, found: list) -> dict:
    hits, wanted = _matches(truth, found)
    correct = sum(hits)
    precision = correct / len(hits) if hits else 0.0
    recall = correct / wanted if wanted else 0.0
    return {'F1': 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            'precision': precision, 'recall': recall, 'pages': len(truth), 'instances': wanted}


def _box(points) -> list[float]:
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def layout_truth(split: Path) -> list[tuple[str, Path, list]]:
    """page-boxes/1: (stem, image, [(class, box as page fractions)]) per page."""
    coco = json.loads((split / '_annotations.coco.json').read_text(encoding='utf-8'))
    names = {c['id']: c['name'] for c in coco['categories']}
    pages = []
    for image in coco['images']:
        w, h = image['width'], image['height']
        boxes = [(names[a['category_id']], [a['bbox'][0] / w, a['bbox'][1] / h, (a['bbox'][0] + a['bbox'][2]) / w,
                                             (a['bbox'][1] + a['bbox'][3]) / h])
                 for a in coco['annotations'] if a['image_id'] == image['id']]
        pages.append((Path(image['file_name']).stem, split / 'images' / image['file_name'], boxes))
    return pages


def lines_truth(split: Path) -> list[tuple[str, Path, list]]:
    """page-lines/1: (stem, image, [('Line', box as page fractions)]) per page."""
    pages = []
    for described in sorted((split / 'lines').glob('*.json')):
        page = json.loads(described.read_text(encoding='utf-8'))
        w, h = page['width'], page['height']
        boxes = [('Line', [v / s for v, s in zip(_box(line['polygon']), (w, h, w, h))]) for line in page['lines']]
        pages.append((described.stem, next((split / 'images').glob(f'{described.stem}.*')), boxes))
    return pages


def answers(task: str, result: dict) -> list:
    """A predict result as (class, box as page fractions, confidence) per answer."""
    if task == 'layout':
        return [(d['label'], d['bbox'], d.get('confidence')) for d in result.get('detections', [])]
    return [('Line', _box(line['polygon']), line.get('confidence')) for line in result.get('lines', [])]


SPLIT_NAMES = {'valid': 'validation', 'test': 'test'}


def final_evaluation(model, files: dict[str, Path], snapshot: Path, task: str, device: str = 'auto',
                     params: dict | None = None, split: str = 'valid') -> dict:
    """Loads the weights into the backend's own model and scores its reading of one split -
    validation unless told otherwise - overall and per dataset; layout also per class."""
    pages = (layout_truth if task == 'layout' else lines_truth)(snapshot / split)
    if not pages:
        return {'error': f'no {SPLIT_NAMES[split]} pages to evaluate on'}
    manifest = snapshot / 'manifest.json'
    groups = json.loads(manifest.read_text(encoding='utf-8')).get('groups', {}) if manifest.is_file() else {}
    loaded = model.load(files, device)
    found = []
    try:
        for start in range(0, len(pages), PAGE_BATCH):
            batch = pages[start:start + PAGE_BATCH]
            images = []
            for _, path, _ in batch:
                with Image.open(path) as image:
                    images.append(image.convert('RGB'))
            results = model.predict(loaded, task, images, params or {}, [None] * len(images))
            found.extend(answers(task, result) for result in results)
    finally:
        model.unload(loaded)
    score = layout_scores if task == 'layout' else line_scores
    truth = [boxes for _, _, boxes in pages]
    by_dataset = {}
    for dataset in sorted({groups.get(stem, '') for stem, _, _ in pages}):
        chosen = [i for i, (stem, _, _) in enumerate(pages) if groups.get(stem, '') == dataset]
        by_dataset[dataset or 'unknown'] = {key: value for key, value in
                                            score([truth[i] for i in chosen], [found[i] for i in chosen]).items()
                                            if key != 'by_class'}
    return {'metric': 'mAP50' if task == 'layout' else 'F1', 'overall': score(truth, found),
            'by_dataset': by_dataset}


def add_to_metrics(output: Path, evaluation: dict, test: dict | None = None) -> None:
    """Merges the evaluation into metrics.json; the run's headline becomes its overall score."""
    path = output / 'metrics.json'
    metrics = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}
    metrics['evaluation'] = evaluation
    if test:
        metrics['test_evaluation'] = test
    if 'overall' in evaluation:
        metrics[f"final/{evaluation['metric']}"] = evaluation['overall'][evaluation['metric']]
    path.write_text(json.dumps(metrics), encoding='utf-8')


def _guarded(evaluate) -> dict:
    """An evaluation, or what stopped it: a failure is recorded, and the run stands."""
    import traceback
    try:
        return evaluate()
    except Exception as error:
        traceback.print_exc()
        return {'error': f'{type(error).__name__}: {error}'[:300]}


def base_files(model, base: Path | None) -> dict[str, Path] | None:
    """The files the run's base model loads from: its weights where the run brought them,
    none for the backend's builtin; None for a run from scratch, which has no base to score."""
    if base is not None and (base / 'weights').is_file():
        return {'weights': base / 'weights'}
    return {} if base is not None and (base / BUILTIN_MARKER).exists() else None


def test_evaluation(model, trained: dict[str, Path], snapshot: Path, base: Path | None, task: str) -> dict | None:
    """The test split read by the trained model and by its base, so "did it help?" has an
    answer on material the run never saw; None when the run has no test datasets."""
    if not (snapshot / 'test').is_dir():
        return None
    result = {'trained': _guarded(lambda: final_evaluation(model, trained, snapshot, task, split='test'))}
    files = base_files(model, base)
    if files is not None:
        result['base'] = _guarded(lambda: final_evaluation(model, files, snapshot, task, split='test'))
    return result


def after_training(model, output: Path, weights: str, task: str) -> None:
    """The entrypoint's last step: the evaluation of the weights it just wrote, on the
    snapshot it trained from - its validation split, and its test split where it has one."""
    import os
    snapshot = Path(os.environ['TRAIN_SNAPSHOT'])
    trained = {'weights': output / weights}
    evaluation = _guarded(lambda: final_evaluation(model, trained, snapshot, task))
    base = os.environ.get('TRAIN_BASE')
    test = test_evaluation(model, trained, snapshot, Path(base) if base else None, task)
    add_to_metrics(output, evaluation, test)
    print(f"deeplabel: final evaluation {json.dumps(evaluation.get('overall', evaluation))[:300]}", flush=True)
    if test:
        print(f"deeplabel: test evaluation {json.dumps({k: v.get('overall', v) for k, v in test.items()})[:300]}",
              flush=True)
