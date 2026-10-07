"""QA collator for SO-Phi4 training on SO-Dataset (FOA).

Builds one training batch per record:
    prompt text  : "<|user|><|audio_1|><|spatial|>\n{prompt}<|end|><|assistant|>"
    answer text  : "{answer}<|end|><|endoftext|>"
    labels       : -100 everywhere except the answer span (suffix included)

Audio routing per sample:
    * mono W channel (FOA channel 0) → official audio path (fbank → conformer),
      keeping Phi-4's native audio understanding in the loop;
    * full 4-channel FOA → SO-Encoder path (``spatial_audio`` tensors).

The processor expands ``<|spatial|>`` at string level and ``<|audio_1|>`` at
token-id level, and always LEFT-pads input_ids to the batch max — the answer
span therefore sits at the right edge, which the label builder relies on
(and asserts).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import soundfile as sf
import torch

from .processing import MAX_AUDIO_SAMPLES, SAMPLE_RATE, SPATIAL_TOKEN

IGNORE_INDEX = -100
ANSWER_SUFFIX = "<|end|><|endoftext|>"
_PROMPT_TEMPLATE = (
    "<|user|><|audio_1|>" + SPATIAL_TOKEN + "\n{prompt}<|end|><|assistant|>"
)


def _load_foa(path: str) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != SAMPLE_RATE:
        import scipy.signal

        gcd = np.gcd(sr, SAMPLE_RATE)
        wav = scipy.signal.resample_poly(wav, SAMPLE_RATE // gcd, sr // gcd, axis=0)
    if wav.shape[1] != 4:
        raise ValueError(f"Expected 4-channel FOA wav, got {wav.shape[1]}ch: {path}")
    return wav[:MAX_AUDIO_SAMPLES]


def _load_mono(path: str) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != SAMPLE_RATE:
        import scipy.signal

        gcd = np.gcd(sr, SAMPLE_RATE)
        wav = scipy.signal.resample_poly(wav, SAMPLE_RATE // gcd, sr // gcd, axis=0)
    mono = wav.mean(axis=1) if wav.shape[1] > 1 else wav[:, 0]
    return mono[:MAX_AUDIO_SAMPLES].astype(np.float32)


class SoPhi4QACollator:
    def __init__(self, processor, verify_labels: bool = True,
                 max_prompt_tokens: int = 512, max_answer_tokens: int = 256) -> None:
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.verify_labels = verify_labels
        self.max_prompt_tokens = max_prompt_tokens
        self.max_answer_tokens = max_answer_tokens

    def _truncate_text(self, text: str, max_tokens: int) -> str:
        """Token-level truncation (decode round-trip keeps BPE boundaries clean)."""
        ids = self.tokenizer(text).input_ids
        if len(ids) <= max_tokens:
            return text
        return self.tokenizer.decode(ids[:max_tokens])

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        wavs, mono_audios, full_texts, answer_texts = [], [], [], []
        has_spatial_list: List[bool] = []
        for feat in features:
            is_spatial = bool(feat.get("_replay_has_spatial", feat.get("has_spatial", True)))
            if is_spatial:
                wav = _load_foa(feat["audio_path"])
                mono = wav[:, 0].copy()  # W channel for the native path
            else:
                mono = _load_mono(feat["audio_path"])
                wav = np.zeros((mono.shape[0], 4), dtype=np.float32)
                wav[:, 0] = mono  # W-only FOA; X/Y/Z stay zero
            wavs.append(wav)
            mono_audios.append((mono.copy(), SAMPLE_RATE))
            has_spatial_list.append(is_spatial)
            prompt = self._truncate_text(feat["prompt"].rstrip(), self.max_prompt_tokens)
            answer = self._truncate_text(feat["answer"].strip(), self.max_answer_tokens)
            full_texts.append(
                _PROMPT_TEMPLATE.format(prompt=prompt) + answer + ANSWER_SUFFIX
            )
            answer_texts.append(answer + ANSWER_SUFFIX)

        out = self.processor(
            text=full_texts,
            audios=mono_audios,
            spatial_audio=wavs,
            return_tensors="pt",
        )
        input_ids = out["input_ids"]  # [B, L] left-padded
        batch, length = input_ids.shape

        # --- labels: only the answer span (right edge) is supervised -------
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        for i, ans in enumerate(answer_texts):
            ans_ids = self.tokenizer(ans).input_ids
            tail = input_ids[i, length - len(ans_ids) :]
            if self.verify_labels and not torch.equal(
                tail, torch.as_tensor(ans_ids, dtype=tail.dtype)
            ):
                raise RuntimeError(
                    f"answer-span mismatch on sample {i}: tokenization of the "
                    "full text does not end with the standalone answer ids"
                )
            labels[i, length - len(ans_ids) :] = tail

        out["labels"] = labels
        if not all(has_spatial_list):
            out["has_spatial"] = torch.as_tensor(has_spatial_list, dtype=torch.bool)
        return out
