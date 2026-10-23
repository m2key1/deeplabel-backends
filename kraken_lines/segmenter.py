import io
import multiprocessing
import os
from pathlib import Path

import torch
from PIL import Image

# Kraken spends nine tenths of a page on one CPU core turning the network's output into
# lines, so the pages of a request are read side by side, each worker process with its own
# copy of the small model, taking turns on the GPU: 0.64 pages/s with six on the 5090
# against 0.16 in one process. 0 reads in this process, one page after another.
WORKERS = int(os.environ.get('KRAKEN_WORKERS', str(max(1, min(6, (os.cpu_count() or 2) // 2)))))

_reader = None


def _taking_turns(compute, card):
    """The network pass of one page, holding the card alone: its maps are moved off the GPU
    and the memory released before the next worker's pass, so the workers' vectorising
    overlaps and their memory does not add up."""
    def run(image):
        with card:
            maps = compute(image)
            maps = {key: value.cpu() if torch.is_tensor(value) else value for key, value in maps.items()}
            torch.cuda.empty_cache()
        return maps
    return run


def _open(weights: str | None, device: str, precision: str, card=None) -> None:
    global _reader
    from kraken.configs import SegmentationInferenceConfig
    from kraken.tasks import SegmentationTaskModel

    model = SegmentationTaskModel.load_model(weights)
    if card is not None:
        for net in model.seg_models:
            net._compute_segmentation_map = _taking_turns(net._compute_segmentation_map, card)
    config = SegmentationInferenceConfig(accelerator='gpu' if device.startswith('cuda') else 'cpu',
                                         device='auto', precision=precision)
    _reader = (model, config)


def _segment(png: bytes) -> list[list[list[float]]]:
    """One page's lines as boundary polygons, in fractions of the page."""
    model, config = _reader
    image = Image.open(io.BytesIO(png))
    found = model.predict(image, config)
    width, height = image.size
    return [[[min(1.0, max(0.0, x / width)), min(1.0, max(0.0, y / height))] for x, y in line.boundary]
            for line in found.lines if line.boundary and len(line.boundary) >= 3]


def _png(image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return buffer.getvalue()


class KrakenSegmenter:
    """Kraken's baseline segmenter: lines found as baselines, each returned with the
    boundary polygon Kraken computes around it. Without weights, Kraken's own default
    model (blla.mlmodel, shipped in the package)."""

    def __init__(self, weights: Path | None, device: str, precision: str):
        settings = (str(weights) if weights else None, device, precision)
        if WORKERS:
            spawn = multiprocessing.get_context('spawn')
            self.pool = spawn.Pool(WORKERS, _open, (*settings, spawn.Lock()))
        else:
            self.pool = None
            _open(*settings)

    def segment(self, images) -> list[list[list[list[float]]]]:
        pages = [_png(image) for image in images]
        return self.pool.map(_segment, pages, chunksize=1) if self.pool else [_segment(page) for page in pages]

    def close(self) -> None:
        if self.pool:
            self.pool.terminate()
