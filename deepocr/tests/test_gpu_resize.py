import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip('torch')
from deepocr.gpu_resize import resize_lines  # noqa: E402


def test_lines_are_resized_as_pil_lanczos_resizes_them():
    rng = np.random.default_rng(0)
    # Enlarged (short crops) and shrunk (a tall one), as line crops come.
    sizes = [(40, 300), (97, 1210), (310, 2000)]
    target = 128
    crops = [rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for h, w in sizes]
    out_widths = [max(1, int(w * target / h)) for h, w in sizes]
    h_max, w_max = max(h for h, _ in sizes), max(w for _, w in sizes)
    native = torch.zeros(len(crops), h_max, w_max, 3, dtype=torch.uint8)
    for i, crop in enumerate(crops):
        native[i, :crop.shape[0], :crop.shape[1]] = torch.from_numpy(crop)

    out = resize_lines(native, torch.tensor([h for h, _ in sizes]), torch.tensor([w for _, w in sizes]),
                       target, torch.tensor(out_widths), max(out_widths))

    for i, crop in enumerate(crops):
        expected = np.asarray(Image.fromarray(crop).resize((out_widths[i], target), Image.Resampling.LANCZOS))
        got = out[i, :, :, :out_widths[i]].permute(1, 2, 0).numpy().astype(np.int16)
        diff = np.abs(got - expected.astype(np.int16))
        assert diff.max() <= 2
        assert (diff == 0).mean() > 0.999
