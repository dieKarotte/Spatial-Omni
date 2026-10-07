"""Spatial-aware processor for Phi-4-multimodal.

Wraps the official ``Phi4MMProcessor`` (composition, no vendored-code edits)
and adds an independent ``<|spatial|>`` placeholder modality:

1. Registers ``<|spatial|>`` as a tokenizer special token (one id per token).
2. Expands each single ``<|spatial|>`` in the text into ``T_i`` copies, where
   ``T_i`` derives from the FOA clip duration at the LLM-side spatial rate
   (SO-Encoder native 10 Hz / projector shuffle 4 = 2.5 Hz), using
   round-half-to-even exactly like the Qwen SO processor.
3. Delegates text tokenization + native audio (fbank) processing to the
   official processor, which itself expands ``<|audio_1|>`` placeholders at
   token-id level.
4. Packs the FOA waveform into fixed-length tensors for the SO branch.

The model wrapper later replaces the embeddings at ``<|spatial|>`` id
positions with projected SO-Encoder outputs, so the placeholder count in
``input_ids`` MUST equal ``spatial_token_lengths`` (verified here and again
in the model via placeholder-count alignment).
"""

from __future__ import annotations

from fractions import Fraction
from typing import List, Optional, Sequence, Union

import numpy as np
import torch

SPATIAL_TOKEN = "<|spatial|>"
SAMPLE_RATE = 16000
MAX_AUDIO_SECONDS = 20.0
MAX_AUDIO_SAMPLES = int(SAMPLE_RATE * MAX_AUDIO_SECONDS)
SO_ENCODER_TOKEN_RATE = 10.0
SO_PROJECTOR_SHUFFLE_FACTOR = 4
SO_LLM_TOKEN_RATE = SO_ENCODER_TOKEN_RATE / SO_PROJECTOR_SHUFFLE_FACTOR  # 2.5 Hz


class SoPhi4Processor:
    """Official Phi4MMProcessor + ``<|spatial|>`` expansion + FOA packing."""

    def __init__(self, base_processor) -> None:
        self.base = base_processor
        self.tokenizer = base_processor.tokenizer
        self.spatial_token_id = self._register_spatial_token()

    def _register_spatial_token(self) -> int:
        vocab = self.tokenizer.get_vocab()
        if SPATIAL_TOKEN not in vocab:
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": [SPATIAL_TOKEN]}
            )
        token_id = int(self.tokenizer.convert_tokens_to_ids(SPATIAL_TOKEN))
        if token_id is None or token_id < 0:
            raise RuntimeError(f"Failed to register {SPATIAL_TOKEN} on tokenizer")
        return token_id

    # ------------------------------------------------------------------
    # duration -> spatial token count (round-half-to-even, same as Qwen SO)
    # ------------------------------------------------------------------
    @staticmethod
    def spatial_token_count(num_samples: int) -> int:
        rate = Fraction(str(SO_LLM_TOKEN_RATE)).limit_denominator(1000)
        numerator = int(num_samples) * int(rate.numerator)
        denominator = int(SAMPLE_RATE) * int(rate.denominator)
        quotient, remainder = divmod(numerator, denominator)
        twice = remainder * 2
        if twice > denominator or (twice == denominator and quotient % 2 == 1):
            quotient += 1
        return max(1, quotient)

    # ------------------------------------------------------------------
    def __call__(
        self,
        text: Union[str, List[str]],
        audios: Optional[Sequence] = None,
        spatial_audio: Optional[Sequence[Union[np.ndarray, torch.Tensor]]] = None,
        padding=False,
        return_tensors: str = "pt",
    ):
        """Build model inputs with spatial placeholders expanded.

        Args:
            text: list of strings; each must contain exactly one
                ``<|spatial|>`` placeholder when ``spatial_audio`` is given.
            audios: official-format audio list ``[(waveform, sample_rate)]``
                for the native speech path (mono, 16 kHz).
            spatial_audio: list of FOA arrays ``[T_i, 4]`` (float, 16 kHz),
                one per sample, already capped at ``MAX_AUDIO_SAMPLES``.
        """
        if isinstance(text, str):
            text = [text]

        spatial_token_lengths = None
        if spatial_audio is not None:
            if len(spatial_audio) != len(text):
                raise ValueError(
                    f"spatial_audio batch {len(spatial_audio)} != text batch {len(text)}"
                )
            lengths = [min(int(w.shape[0]), MAX_AUDIO_SAMPLES) for w in spatial_audio]
            spatial_token_lengths = [self.spatial_token_count(n) for n in lengths]
            expanded = []
            for t, count in zip(text, spatial_token_lengths):
                if t.count(SPATIAL_TOKEN) != 1:
                    raise ValueError(
                        f"Each text must contain exactly one {SPATIAL_TOKEN}, got {t.count(SPATIAL_TOKEN)}"
                    )
                expanded.append(t.replace(SPATIAL_TOKEN, SPATIAL_TOKEN * count))
            text = expanded

        out = self.base(
            text=text,
            audios=audios,
            padding=padding,
            return_tensors=return_tensors,
        )
        # The official model checks `input_image_embeds is not None`, so the
        # empty tensors the processor emits for text/audio-only batches would
        # wrongly activate the vision tower. Normalize them to None.
        for key in ("input_image_embeds", "image_sizes", "image_attention_mask"):
            v = out.get(key)
            if torch.is_tensor(v) and v.numel() == 0:
                out[key] = None

        if spatial_audio is not None:
            batch = len(text)
            spa = torch.zeros(batch, MAX_AUDIO_SAMPLES, 4, dtype=torch.float32)
            spa_mask = torch.zeros(batch, MAX_AUDIO_SAMPLES, dtype=torch.float32)
            for i, (wav, length) in enumerate(zip(spatial_audio, lengths)):
                w = torch.as_tensor(np.asarray(wav), dtype=torch.float32)
                if w.ndim != 2 or w.shape[1] != 4:
                    raise ValueError(
                        f"spatial_audio[{i}] must be [T,4], got {tuple(w.shape)}"
                    )
                spa[i, :length] = w[:length]
                spa_mask[i, :length] = 1.0
            out["spatial_audio"] = spa
            out["spatial_audio_attention_mask"] = spa_mask
            out["spatial_audio_lengths"] = torch.tensor(lengths, dtype=torch.long)
            out["spatial_token_lengths"] = torch.tensor(
                spatial_token_lengths, dtype=torch.long
            )

            # Hard verification: placeholder ids in input_ids must equal the
            # per-sample token counts, otherwise the model-side index_put
            # would misalign (this is the "spatial token is real embeddings,
            # not plain text" invariant).
            input_ids = out["input_ids"]
            if isinstance(input_ids, torch.Tensor):
                id_rows = input_ids
            else:
                id_rows = [torch.as_tensor(row) for row in input_ids]
            for i, row in enumerate(id_rows):
                got = int((row == self.spatial_token_id).sum().item())
                want = spatial_token_lengths[i]
                if got != want:
                    raise RuntimeError(
                        f"spatial placeholder mismatch on sample {i}: "
                        f"input_ids has {got}, expected {want}"
                    )
        return out

    # convenience passthroughs ------------------------------------------
    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)
