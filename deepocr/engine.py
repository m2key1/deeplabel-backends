import copy
import math
import multiprocessing
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.functional as AF
from PIL import Image

import infer
from deepocr.conf_profile import character_profile, uniform_profile
from deepocr.gpu_resize import resize_lines

DECODES = ('beam', 'rescored', 'greedy')
# The KenLM beam search runs on the CPU, one line after another; the lines of a batch are
# split over this many worker processes, each with its own decoder: 39.6 lines/s with eight
# on the 5090 against 12.5 in this process, the same text. 0 decodes in this process.
DECODE_WORKERS = int(os.environ.get('DEEPOCR_DECODE_WORKERS', '8'))

_tokenizer = None


def _open_decoder(tokenizer) -> None:
    global _tokenizer
    torch.set_num_threads(1)
    _tokenizer = tokenizer


def _decoder_pool(tokenizer):
    """Processes that only decode on the CPU: each gets the tokenizer with its tensors on
    the CPU and sees no GPU, or every one would open a CUDA context of half a gigabyte."""
    travelling = copy.copy(tokenizer)
    travelling._decoder_cache = {}
    for name, value in vars(travelling).items():
        if torch.is_tensor(value):
            setattr(travelling, name, value.cpu())
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    try:
        return multiprocessing.get_context('spawn').Pool(DECODE_WORKERS, _open_decoder, (travelling,))
    finally:
        if visible is None:
            os.environ.pop('CUDA_VISIBLE_DEVICES')
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = visible


def _beam(log_probs: np.ndarray, lengths: np.ndarray, beam_width: int) -> list[str]:
    return _tokenizer.batch_decode_beam_search(torch.from_numpy(log_probs).float(), torch.from_numpy(lengths),
                                               beam_width=beam_width)


def resolve_decode(decode: str, lm_on: bool, use_decoder: bool) -> str:
    """The mode that will actually run, given what this version carries: one whose machinery
    is missing degrades rather than failing (4.5)."""
    if decode == 'beam' and not lm_on:
        return 'rescored' if use_decoder else 'greedy'
    if decode == 'rescored' and not use_decoder:
        return 'beam' if lm_on else 'greedy'
    return decode if decode in DECODES else 'greedy'


class DeepOCREngine:
    """DeepOCR (CTC head + AR decoder) from the DeepOCR repo, decoded per its
    inference doc. Default decode is CTC beam search with the fused union char-KenLM
    (DeepOCR's shipped best setup); without the LM artifact it falls back to CTC beam
    n-best reranked by the AR decoder. Batched. Confidence is the AR decoder's
    per-character probability of the produced text (a decode-path-independent signal)."""

    BEAM_WIDTH = 4
    NBEST = 8
    ALPHA = 0.3
    BETA = 0.0

    def __init__(self, checkpoint: str, device, kenlm_path=None, lm_dict_path=None, lm_weight=1.0,
                 beam_width=10):
        self._infer = infer
        self.device = device
        self.beam_width = beam_width

        self.lm_on = bool(kenlm_path) and Path(kenlm_path).exists() and bool(lm_dict_path) and Path(lm_dict_path).exists()
        # load_model reads these module globals when building the tokenizer's beam decoder.
        self._infer.KENLM_PATH = Path(kenlm_path) if self.lm_on else None
        self._infer.LM_DICT_PATH = Path(lm_dict_path) if self.lm_on else None
        self._infer.LM_WEIGHT = lm_weight

        self.model, self.tokenizer = self._infer.load_model(str(checkpoint), device)
        # check_kenlm_artifacts may refuse an LM that is not index-parallel to this vocab.
        self.lm_on = self.lm_on and bool(getattr(self.tokenizer, 'kenlm_path', None))
        self.decode = 'beam'
        # Older checkpoints are not conditioned on a convention and their _encode takes none.
        self.convention_names = list(getattr(self.model, 'convention_names', None) or [])
        self._convention_row = None
        # _encode calls set_grad_enabled(not freeze_backbone), which overrides the
        # ambient no_grad and builds autograd graphs for the backbone at inference.
        self.model.freeze_backbone = True
        # encode() canonicalises before tokenising, so the character sequence the tokens
        # correspond to is the canonical one, not the text as produced.
        self.canonicalize = getattr(sys.modules[type(self.tokenizer).__module__],
                                    'canonicalize', lambda text: text)
        normalize = self._infer._NORMALIZE.transforms[0]
        self._mean = torch.tensor(normalize.mean, device=device).view(1, 3, 1, 1)
        self._std = torch.tensor(normalize.std, device=device).view(1, 3, 1, 1)
        self.decoders = None
        if self.lm_on and DECODE_WORKERS:
            self.decoders = _decoder_pool(self.tokenizer)

    def close(self) -> None:
        if self.decoders:
            self.decoders.terminate()

    def _beam_search(self, log_probs, in_lens) -> list[str]:
        """DeepOCR's own batched beam search, its lines split over the decoder workers."""
        if not self.decoders:
            return self.tokenizer.batch_decode_beam_search(log_probs, in_lens, beam_width=self.beam_width)
        # numpy, not tensors: a pool sends tensors as shared-memory file handles, and many
        # requests run out of them. fp16 halves the transfer without changing the text.
        log_probs, in_lens = log_probs.half().cpu().numpy(), in_lens.cpu().numpy()
        parts = [part for part in np.array_split(np.arange(log_probs.shape[1]), DECODE_WORKERS) if len(part)]
        texts = self.decoders.starmap(_beam, [(np.array(log_probs[:, part], order='C'), in_lens[part], self.beam_width)
                                              for part in parts])
        return [text for part in texts for text in part]

    def _prepare_batch(self, pil_images):
        """The crops as the model reads them: resized on the GPU exactly as PIL's LANCZOS
        does in training, normalised, and padded to one width."""
        target_h, ps_w = self.model.target_height, self.model.patch_size_w
        crops = [np.asarray(img.convert('RGB')) for img in pil_images]
        heights = [crop.shape[0] for crop in crops]
        widths = [crop.shape[1] for crop in crops]
        new_ws = [max(1, int(w * (target_h / h))) for h, w in zip(heights, widths)]
        if self.model.max_width:
            new_ws = [min(w, self.model.max_width) for w in new_ws]
        target_ws = [math.ceil(w / ps_w) * ps_w for w in new_ws]
        batch_w = max(target_ws)

        native = np.zeros((len(crops), max(heights), max(widths), 3), dtype=np.uint8)
        for i, crop in enumerate(crops):
            native[i, :crop.shape[0], :crop.shape[1]] = crop
        new_w = torch.tensor(new_ws, device=self.device)
        x = resize_lines(torch.from_numpy(native).to(self.device), torch.tensor(heights, device=self.device),
                         torch.tensor(widths, device=self.device), target_h, new_w, batch_w)
        x = (x / 255.0 - self._mean) / self._std
        x = x.masked_fill(torch.arange(batch_w, device=self.device)[None, None, None, :] >= new_w[:, None, None, None], 1.0)

        num_ph, num_pw = target_h // self.model.patch_size, batch_w // ps_w
        real = torch.tensor(target_ws) // ps_w
        padded = torch.arange(num_pw)[None, :] >= real[:, None]
        if self.model.use_2d_structure:
            masks = padded[:, :, None].expand(-1, num_pw, num_ph).reshape(len(crops), -1)
            in_lens = real * num_ph
        else:
            masks, in_lens = padded, real
        return x, masks.to(self.device), in_lens.long()

    def _align_tokens(self, log_probs, in_lens, texts):
        """Frame span and score of every target token, by CTC forced alignment of the produced
        text against the CTC head. None for a line whose hypothesis cannot be aligned - more
        tokens than frames, or an alignment that did not consume every token."""
        batch = log_probs.permute(1, 0, 2).float().contiguous()
        blank = self.tokenizer.blank_token_id
        aligned = []
        for i, text in enumerate(texts):
            ids = self.tokenizer.encode(text)
            frames = int(in_lens[i].item())
            if not ids or len(ids) > frames:
                aligned.append(None)
                continue
            targets = torch.tensor([ids], dtype=torch.int32, device=batch.device)
            labels, scores = AF.forced_align(batch[i:i + 1, :frames], targets, blank=blank)
            spans = AF.merge_tokens(labels[0], scores[0], blank=blank)
            aligned.append([(span.start, span.end, float(math.exp(span.score))) for span in spans]
                           if len(spans) == len(ids) else None)
        return aligned

    def _char_token_counts(self, text):
        canonical = self.canonicalize(text)
        if getattr(self.tokenizer, 'use_bytes', False):
            return [len(char.encode('utf-8')) for char in canonical]
        return [1] * len(canonical)

    def _profiles(self, texts, aligned, token_probs, in_lens):
        """One band per character, placed where the alignment says its ink is. Falls back to
        the legacy even spacing whenever the alignment is unavailable."""
        profiles = []
        for i, text in enumerate(texts):
            probs = token_probs[i]
            spans = aligned[i]
            profile = None
            if spans is not None and len(spans) == len(probs):
                spans = [(start, end, prob) for (start, end, _), prob in zip(spans, probs)]
                profile = character_profile(self._char_token_counts(text), spans,
                                            int(in_lens[i].item()))
            profiles.append(profile if profile is not None else uniform_profile(probs))
        return profiles

    def _ctc_token_probs(self, aligned):
        """Greedy fallback: the alignment's own per-token probability. The CTC head is
        entropy-regularized at train time so these read low even on correct lines."""
        return [[prob for _, _, prob in spans] if spans else [] for spans in aligned]

    def _decoder_token_probs(self, memory, mem_mask, texts):
        """Per-token probability the AR decoder - the path that produces the rescored text -
        assigns to each token it emitted. One teacher-forced pass over the winning hypotheses."""
        device = memory.device
        encoded = [self.tokenizer.encode(t) for t in texts]
        target_lengths = torch.tensor([len(s) for s in encoded], dtype=torch.long, device=device)
        targets_concat = torch.cat([
            torch.tensor(s, dtype=torch.long, device=device) if s
            else torch.empty(0, dtype=torch.long, device=device)
            for s in encoded
        ])
        decoder_input, decoder_target, tgt_pad = self.model._build_decoder_io(targets_concat, target_lengths)
        logits = self.model._decode_teacher_forced(memory, mem_mask, decoder_input, tgt_pad)
        probs = F.softmax(logits.float(), dim=-1).permute(1, 0, 2)    # [N, L, V]
        token_probs = probs.gather(2, decoder_target.unsqueeze(-1)).squeeze(-1)    # [N, L]
        # The EOS position is excluded: it is not a character the user can look at.
        return [token_probs[i, :int(target_lengths[i].item())].tolist() for i in range(len(texts))]

    def honoured(self, convention: dict[str, bool]) -> list[str]:
        """The flags a read asks the model for that it knows, in the model's own order; a
        flag it does not know is ignored, not an error (4.5)."""
        return [name for name in self.convention_names if convention.get(name)]

    def configure(self, decode: str, convention: dict[str, bool]) -> str:
        """Set one request's decode and convention; returns the decode that will run."""
        self.decode = resolve_decode(decode, self.lm_on, bool(self.model.use_decoder))
        self._convention_row = torch.tensor(
            [[1.0 if convention.get(name) else 0.0 for name in self.convention_names]], device=self.device)
        return self.decode

    def _convention_args(self, batch_size: int) -> dict:
        if not self.convention_names:
            return {}
        return {'conventions': self._convention_row.expand(batch_size, -1)}

    def _forward_chunk(self, chunk):
        x, masks, in_lens = self._prepare_batch(chunk)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                            enabled=self.device.type == 'cuda'):
            memory, mem_mask, _, _ = self.model._encode(
                x, attention_masks=masks, **self._convention_args(len(chunk)))
        memory = memory.float()
        log_probs = F.log_softmax(self.model.output_layer(memory), dim=-1)

        if self.decode == 'greedy':
            preds = log_probs.argmax(dim=2).permute(1, 0).cpu()
            texts = [self.tokenizer.ctc_decode(preds[i, :in_lens[i].item()].tolist(),
                                               skip_special_tokens=True)
                     for i in range(len(chunk))]
        elif self.decode == 'beam':
            # CTC beam search with the model's own KenLM fused in.
            texts = self._beam_search(log_probs, in_lens)
        else:
            # CTC beam n-best, reranked by the AR decoder head.
            nbest = self.tokenizer.batch_beam_search_nbest(
                log_probs, in_lens, beam_width=self.BEAM_WIDTH, nbest=self.NBEST)
            texts = self.model.rescore_hypotheses(
                memory, mem_mask, nbest, alpha=self.ALPHA, beta=self.BETA)

        aligned = self._align_tokens(log_probs, in_lens, texts)
        if self.model.use_decoder and self.decode != 'greedy':
            token_probs = self._decoder_token_probs(memory, mem_mask, texts)
        else:
            token_probs = self._ctc_token_probs(aligned)
        profiles = self._profiles(texts, aligned, token_probs, in_lens)
        confs = [sum(e['c'] for e in p) / len(p) / 100 if p else 0.0 for p in profiles]
        return texts, confs, profiles

    def _run_chunk(self, chunk):
        """Halve the chunk and retry on CUDA OOM; a single image that still OOMs raises.
        The retry happens outside the except block so the exception traceback does not
        keep the failed batch's tensors alive."""
        try:
            return self._forward_chunk(chunk)
        except torch.OutOfMemoryError:
            if len(chunk) == 1:
                raise
        torch.cuda.empty_cache()
        mid = len(chunk) // 2
        t1, c1, p1 = self._run_chunk(chunk[:mid])
        t2, c2, p2 = self._run_chunk(chunk[mid:])
        return t1 + t2, c1 + c2, p1 + p2

    def predict_batch(self, images, batch_size=16, contexts=None):
        """Reads the lines in batches of similar width and answers in the order given. A
        narrow line padded out to a wide one's width reads worse: on DeepOCR's validation
        lines, unsorted batches scored 2.83% CER against 2.32% sorted."""
        order = sorted(range(len(images)), key=lambda i: images[i].width / max(1, images[i].height))
        texts, confs, profiles = [None] * len(images), [None] * len(images), [None] * len(images)
        with torch.no_grad():
            for start in range(0, len(order), batch_size):
                chunk = order[start:start + batch_size]
                chunk_texts, chunk_confs, chunk_profiles = self._run_chunk([images[i] for i in chunk])
                for i, text, conf, profile in zip(chunk, chunk_texts, chunk_confs, chunk_profiles):
                    texts[i], confs[i], profiles[i] = text, conf, profile
        return texts, confs, profiles
