"""The training entrypoint of section 6.6: a page-lines/1 snapshot in, written as PAGE XML
Kraken trains from - each line its outline, and as its centre line the middle of that
outline, since DeepLabel keeps no baselines - and Kraken's segmentation weights out. Without
a base the run fine-tunes Kraken's own default model. The same command runs on the backend
and, in this backend's image, on the cluster."""
import json
import os
import sys
import tempfile
from collections import OrderedDict
from importlib import resources
from pathlib import Path
from xml.sax.saxutils import quoteattr

# Started by path, in the run's own directory: the wrappers are not on the path by themselves.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from deeplabel_backend import evaluate
from deeplabel_backend.train import ProgressLog, loader_workers, run_hyperparams, system_metrics, write_outputs
from kraken.configs import BLLASegmentationTrainingConfig, BLLASegmentationTrainingDataConfig
from kraken.models.convert import convert_models
from kraken.train import BLLASegmentationDataModule, BLLASegmentationModel, KrakenTrainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint

from kraken_lines import tunables
from kraken_lines.centerline import centerline
from kraken_lines.model import KrakenBackend

LINE_TYPE = 'default'
WEIGHTS_NAME = 'model.safetensors'


def _points(points) -> str:
    return ' '.join(f'{round(x)},{round(y)}' for x, y in points)


def page_xml(page: dict, image_name: str) -> str | None:
    lines = []
    for number, line in enumerate(page['lines']):
        middle = centerline(line['polygon'])
        if middle:
            lines.append(f'<TextLine id="l{number}"><Coords points="{_points(line["polygon"])}"/>'
                         f'<Baseline points="{_points(middle)}"/></TextLine>')
    if not lines:
        return None
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            '<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">'
            f'<Page imageFilename={quoteattr(image_name)} imageWidth="{page["width"]}" imageHeight="{page["height"]}">'
            f'<TextRegion id="r0"><Coords points="0,0 {page["width"]},0 {page["width"]},{page["height"]} 0,{page["height"]}"/>'
            + ''.join(lines) + '</TextRegion></Page></PcGts>\n')


def page_files(snapshot: Path, work: Path, split: str) -> list[str]:
    """One PAGE file per page with any line, beside a link to its image."""
    target = work / split
    target.mkdir(parents=True, exist_ok=True)
    written = []
    for described in sorted((snapshot / split / 'lines').glob('*.json')):
        image = next((snapshot / split / 'images').glob(f'{described.stem}.*'))
        xml = page_xml(json.loads(described.read_text(encoding='utf-8')), image.name)
        if xml is None:
            continue
        (target / image.name).symlink_to(image)
        (target / f'{described.stem}.xml').write_text(xml, encoding='utf-8')
        written.append(str(target / f'{described.stem}.xml'))
    return written


# Kraken's metric names, as the train block's METRICS declares them; the per-step loss and
# the duplicate of the checkpoint's metric are left out.
REPORTED = {'train_loss_epoch': 'train/loss', 'val_bl_f1': 'val/baseline_f1',
            'val_bl_precision': 'val/baseline_precision', 'val_bl_recall': 'val/baseline_recall',
            'val_mean_iu': 'val/mean_iu', 'val_freq_iu': 'val/freq_iu', 'val_mean_acc': 'val/mean_acc',
            'val_accuracy': 'val/accuracy'}


class ReportEpochs(Callback):
    """Each validated epoch into progress.jsonl, as DeepLabel's run page reads it."""

    def __init__(self, progress: ProgressLog):
        self.progress = progress

    def on_validation_epoch_end(self, trainer, module):
        metrics = {REPORTED[key]: float(value) for key, value in trainer.callback_metrics.items()
                   if key in REPORTED and torch.is_tensor(value) and value.numel() == 1}
        self.progress.write(epoch=trainer.current_epoch + 1, **metrics,
                            **system_metrics(trainer.optimizers[0] if trainer.optimizers else None))


def main() -> int:
    params = json.loads(Path(os.environ['TRAIN_PARAMS']).read_text(encoding='utf-8'))
    output = Path(os.environ['TRAIN_OUTPUT'])
    output.mkdir(parents=True, exist_ok=True)
    snapshot = Path(os.environ['TRAIN_SNAPSHOT'])
    work = Path(tempfile.mkdtemp(prefix='kraken-'))
    chosen = run_hyperparams(params, tunables.KNOBS, lambda: {k['key']: k['default'] for k in tunables.declared()})
    by_source = {source: {k['key']: chosen[k['key']] for k in tunables.KNOBS if k['source'] == source and k['key'] in chosen}
                 for source in ('model', 'data')}
    data = BLLASegmentationTrainingDataConfig(
        training_data=page_files(snapshot, work, 'train'), evaluation_data=page_files(snapshot, work, 'valid'),
        format_type='page', partition=1, topline=None, region_class_mapping={},
        line_class_mapping=OrderedDict([(LINE_TYPE, 2)]), num_workers=loader_workers(),
        batch_size=params.get('batch_size') or 1, **by_source['data'])
    config = BLLASegmentationTrainingConfig(
        epochs=params.get('epochs') or 50, quit='fixed', checkpoint_path=str(work / 'checkpoints'),
        resize='new', **by_source['model'])
    best = ModelCheckpoint(dirpath=config.checkpoint_path, save_top_k=1, monitor='val_metric', mode='max',
                           filename='best')
    trainer = KrakenTrainer(accelerator='gpu' if torch.cuda.is_available() else 'cpu', devices=1,
                            precision=chosen.get('precision', tunables.TRAINER_DEFAULTS['precision']),
                            max_epochs=config.epochs, enable_progress_bar=False,
                            enable_model_summary=False, accumulate_grad_batches=config.accumulate_grad_batches,
                            callbacks=[best, ReportEpochs(ProgressLog(output))],
                            gradient_clip_val=config.gradient_clip_val, num_sanity_val_steps=0,
                            use_distributed_sampler=False, val_check_interval=1.0)
    base = Path(os.environ.get('TRAIN_BASE', '')) / 'weights'
    start = base if base.is_file() else resources.files('kraken').joinpath('blla.mlmodel')
    with trainer.init_module(empty_init=False):
        model = BLLASegmentationModel.load_from_weights(str(start), config=config)
    trainer.fit(model, BLLASegmentationDataModule(data))
    convert_models([best.best_model_path], output / WEIGHTS_NAME, weights_format='safetensors')
    score = best.best_model_score.item() if best.best_model_score is not None else None
    metrics = {key: float(value) for key, value in trainer.callback_metrics.items()
               if torch.is_tensor(value) and value.numel() == 1}
    write_outputs(output, {**metrics, 'val_metric': score, 'val_metric_name': 'mean IoU', 'higher_is_better': True,
                           'task': 'lines'},
                  [{'role': 'weights', 'path': WEIGHTS_NAME}])
    evaluate.after_training(KrakenBackend(), output, WEIGHTS_NAME, 'lines')
    return 0


if __name__ == '__main__':
    sys.exit(main())
