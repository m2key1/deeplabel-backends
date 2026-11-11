import types

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip('torch')
from deepocr.engine import DeepOCREngine, resolve_decode, spread_over_bytes  # noqa: E402

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def test_the_decoder_decode_runs_only_on_a_model_with_a_decoder():
    assert resolve_decode('decoder', lm_on=True, use_decoder=True) == 'decoder'
    assert resolve_decode('decoder', lm_on=True, use_decoder=False) == 'beam'
    assert resolve_decode('decoder', lm_on=False, use_decoder=False) == 'greedy'


def test_a_subword_probability_goes_to_every_byte_it_spells_and_end_tokens_to_none():
    token_probs = torch.tensor([[0.9, 0.5, 0.7, 0.2], [0.6, 0.8, 0.3, 0.1]])
    byte_counts = torch.tensor([[3, 1, 0, 0], [2, 2, 1, 0]])
    first, second = spread_over_bytes(token_probs, byte_counts)
    assert first == pytest.approx([0.9, 0.9, 0.9, 0.5])
    assert second == pytest.approx([0.6, 0.6, 0.8, 0.8, 0.3])


def test_a_narrow_line_is_padded_white_to_the_widest_line_of_its_batch():
    engine = DeepOCREngine.__new__(DeepOCREngine)
    engine.device = torch.device('cpu')
    engine.model = types.SimpleNamespace(target_height=32, patch_size=32, patch_size_w=8, max_width=0,
                                         use_2d_structure=False)
    engine._mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    engine._std = torch.tensor(STD).view(1, 3, 1, 1)
    engine._white = (1.0 - engine._mean) / engine._std
    black = [Image.fromarray(np.zeros((32, width, 3), dtype=np.uint8)) for width in (40, 160)]

    x, masks, in_lens = engine._prepare_batch(black)

    pixels = x * engine._std + engine._mean
    assert x.shape == (2, 3, 32, 160)
    assert torch.allclose(pixels[0, :, :, 40:], torch.ones(3, 32, 120), atol=1e-6)
    assert torch.allclose(pixels[0, :, :, :40], torch.zeros(3, 32, 40), atol=1e-6)
    assert torch.allclose(pixels[1], torch.zeros(3, 32, 160), atol=1e-6)
    assert in_lens.tolist() == [5, 20]
