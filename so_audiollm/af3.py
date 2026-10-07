"""SO integration for NVIDIA Audio Flamingo 3 (AF3).

AF3's NVILA checkpoint splits into three standard components that load with
plain transformers classes (no NVILA code needed):
    llm/                 Qwen2ForCausalLM (Qwen2.5-7B, vocab 151672)
    sound_tower/         bare WhisperEncoder state dict (whisper-large-v3)
    sound_mm_projector/  MLP 1280 -> 3584 -> 3584 (keys layers.0/2)

Native modality convention (mirrors AF3's llava_arch scatter):
    a single ``<sound>`` (id 151669) in text expands to
    L = round(((mel_valid - 1)//2 + 1) / 10) * 10 ids (50 Hz frames, 20 s ->
    1000); those rows are replaced by projector(encoder(mel))[ :L ].

SO modality: ``<|spatial|>`` is added to the tokenizer (id 151672, embeddings
resized; the new row is never read because injection replaces it), expanded
to 2.5 Hz counts, replaced by SpatialBranch outputs.

Prompt (qwen2 chat template, same as AF3 training):
    <|im_start|>user\n<sound>...<|spatial|>...\n{q}<|im_end|>\n
    <|im_start|>assistant\n{a}<|im_end|>
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import AutoTokenizer, WhisperConfig, WhisperFeatureExtractor
from transformers.models.whisper.modeling_whisper import WhisperEncoder
from transformers import Qwen2ForCausalLM

from .common import (
    MAX_AUDIO_SAMPLES,
    SAMPLE_RATE,
    SPATIAL_TOKEN,
    SpatialBranch,
    load_foa,
    load_mono,
    pack_spatial_batch,
    spatial_token_count,
)

SOUND_TOKEN = "<sound>"
IGNORE_INDEX = -100


def af3_native_token_count(num_samples: int) -> int:
    """AF3 llava_arch convention: mel frames -> encoder frames -> round to 10."""
    mel_valid = min(num_samples, MAX_AUDIO_SAMPLES) // 160  # hop 160
    enc = (mel_valid - 1) // 2 + 1
    return max(10, int(round(enc / 10.0) * 10))


class SoAF3Model(nn.Module):
    """AF3 components + SO spatial branch with dual placeholder injection."""

    def __init__(self, model_dir: str, beats_checkpoint: str, beats_repo: str,
                 attn_impl: str = "sdpa", enable_replay: bool = False,
                 null_alignment_weight: float = 0.05) -> None:
        self.null_alignment_weight = float(null_alignment_weight)
        super().__init__()
        self.llm = Qwen2ForCausalLM.from_pretrained(
            os.path.join(model_dir, "llm"),
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
        )
        tower_dir = os.path.join(model_dir, "sound_tower")
        enc_cfg = WhisperConfig.from_pretrained(tower_dir)
        self.sound_tower = WhisperEncoder(enc_cfg)
        state = load_file(os.path.join(tower_dir, "model.safetensors"))
        self.sound_tower.load_state_dict(state, strict=True)
        self.sound_tower.to(torch.bfloat16)
        self.sound_projector = nn.Sequential(
            nn.Linear(enc_cfg.d_model, self.llm.config.hidden_size),
            nn.GELU(),
            nn.Linear(self.llm.config.hidden_size, self.llm.config.hidden_size),
        )
        proj_state = load_file(os.path.join(model_dir, "sound_mm_projector", "model.safetensors"))
        self.sound_projector.load_state_dict(
            {k.replace("layers.", ""): v for k, v in proj_state.items()})
        self.sound_projector.to(torch.bfloat16)

        self.spatial = SpatialBranch(beats_checkpoint, beats_repo,
                                     llm_hidden=self.llm.config.hidden_size,
                                     enable_replay=enable_replay)
        self.spatial.build()
        self.spatial.reset_projector()

        self.sound_token_id: Optional[int] = None
        self.spatial_token_id: Optional[int] = None

    def set_token_ids(self, sound_id: int, spatial_id: int) -> None:
        self.sound_token_id = int(sound_id)
        self.spatial_token_id = int(spatial_id)

    # ------------------------------------------------------------------
    def encode_native(self, mel: torch.Tensor, native_lengths: torch.LongTensor):
        """mel [B,128,3000] -> flattened projector rows for <sound> positions."""
        feats = self.sound_tower(mel.to(self.sound_tower.conv1.weight.dtype)).last_hidden_state
        feats = self.sound_projector(feats)  # [B,1500,H]
        rows = [feats[i, : int(n)] for i, n in enumerate(native_lengths.tolist()) if int(n) > 0]
        if not rows:
            return feats.new_zeros((0, feats.shape[-1]))
        return torch.cat(rows, dim=0)

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                sound_mel=None, sound_lengths=None,
                spatial_audio=None, spatial_audio_attention_mask=None,
                spatial_audio_lengths=None, has_spatial=None, **kwargs):
        assert self.sound_token_id is not None and self.spatial_token_id is not None
        embeds = self.llm.get_input_embeddings()(input_ids)

        if sound_mel is not None:
            native_flat = self.encode_native(sound_mel, sound_lengths)
            pos = torch.nonzero(input_ids == self.sound_token_id, as_tuple=True)
            if native_flat.shape[0] != pos[0].shape[0]:
                raise RuntimeError(
                    f"<sound> injection mismatch: {native_flat.shape[0]} rows vs "
                    f"{pos[0].shape[0]} placeholders")
            embeds = embeds.index_put(pos, native_flat.to(embeds.dtype))

        loss_null = None
        if spatial_audio is not None:
            flat, counts, loss_null = self.spatial(
                spatial_audio, spatial_audio_attention_mask, spatial_audio_lengths,
                input_ids, self.spatial_token_id, has_spatial=has_spatial)
            pos = torch.nonzero(input_ids == self.spatial_token_id, as_tuple=True)
            if flat.shape[0] != pos[0].shape[0]:
                raise RuntimeError(
                    f"spatial injection mismatch: {flat.shape[0]} rows vs "
                    f"{pos[0].shape[0]} placeholders")
            embeds = embeds.index_put(pos, flat.to(embeds.dtype))

        out = self.llm(inputs_embeds=embeds, attention_mask=attention_mask,
                       labels=labels, use_cache=False)
        if loss_null is not None and out.loss is not None:
            out.loss = out.loss + self.null_alignment_weight * loss_null.to(out.loss.device)
        return out


def build_af3(model_dir: str, beats_checkpoint: str, beats_repo: str,
              attn_impl: str = "sdpa", enable_replay: bool = False,
              null_alignment_weight: float = 0.05):
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, "llm"))
    if SPATIAL_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [SPATIAL_TOKEN]})
    model = SoAF3Model(model_dir, beats_checkpoint, beats_repo, attn_impl,
                       enable_replay=enable_replay,
                       null_alignment_weight=null_alignment_weight)
    if len(tokenizer) > model.llm.get_input_embeddings().weight.shape[0]:
        model.llm.resize_token_embeddings(len(tokenizer))
    model.set_token_ids(tokenizer.convert_tokens_to_ids(SOUND_TOKEN),
                        tokenizer.convert_tokens_to_ids(SPATIAL_TOKEN))
    return model, tokenizer


class SoAF3Collator:
    """SO-Dataset QA records -> AF3 batch (native mono W + spatial FOA)."""

    def __init__(self, tokenizer, max_answer_tokens: int = 256,
                 max_question_tokens: int = 512) -> None:
        self.tokenizer = tokenizer
        self.max_answer_tokens = max_answer_tokens
        self.max_question_tokens = max_question_tokens
        self.mel = WhisperFeatureExtractor(
            feature_size=128, sampling_rate=SAMPLE_RATE, hop_length=160,
            chunk_length=30, n_fft=400)

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        tok = self.tokenizer
        wavs, monos, texts, answers = [], [], [], []
        native_counts, expected_spatial, has_spatial_list = [], [], []
        for feat in features:
            is_spatial = bool(feat.get("_replay_has_spatial", feat.get("has_spatial", True)))
            has_spatial_list.append(is_spatial)
            if is_spatial:
                wav = load_foa(feat["audio_path"])
                mono = wav[:, 0].copy()
            else:
                mono = load_mono(feat["audio_path"])
                wav = np.zeros((mono.shape[0], 4), dtype=np.float32)
                wav[:, 0] = mono  # W-only FOA; X/Y/Z stay zero
            wavs.append(wav)
            monos.append(mono)
            n_native = af3_native_token_count(mono.shape[0])
            t_sp = spatial_token_count(min(wav.shape[0], MAX_AUDIO_SAMPLES))
            native_counts.append(n_native)
            expected_spatial.append(t_sp)
            ans_ids = tok(str(feat["answer"]).strip(), add_special_tokens=False).input_ids
            ans = tok.decode(ans_ids[: self.max_answer_tokens])
            q_ids = tok(str(feat["prompt"]).rstrip(), add_special_tokens=False).input_ids
            prompt = tok.decode(q_ids[: self.max_question_tokens])
            texts.append(
                "<|im_start|>user\n" + SOUND_TOKEN * n_native + SPATIAL_TOKEN * t_sp
                + f"\n{prompt}<|im_end|>\n<|im_start|>assistant\n" + ans + "<|im_end|>")
            answers.append(ans + "<|im_end|>")

        enc = tok(texts, return_tensors="pt", padding=True, padding_side="right",
                  add_special_tokens=False)
        input_ids, attention_mask = enc.input_ids, enc.attention_mask

        sound_id = tok.convert_tokens_to_ids(SOUND_TOKEN)
        spatial_id = tok.convert_tokens_to_ids(SPATIAL_TOKEN)
        for i in range(len(features)):
            got_native = int((input_ids[i] == sound_id).sum())
            got_spatial = int((input_ids[i] == spatial_id).sum())
            if got_native != native_counts[i] or got_spatial != expected_spatial[i]:
                raise RuntimeError(
                    f"placeholder mismatch sample {i}: native {got_native}!={native_counts[i]} "
                    f"or spatial {got_spatial}!={expected_spatial[i]}")

        labels = torch.full_like(input_ids, IGNORE_INDEX)
        for i, ans in enumerate(answers):
            ans_ids = tok(ans, add_special_tokens=False).input_ids
            seq_len = int(attention_mask[i].sum())
            start = seq_len - len(ans_ids)
            if not torch.equal(input_ids[i, start:seq_len],
                               torch.as_tensor(ans_ids, dtype=input_ids.dtype)):
                raise RuntimeError(f"answer span mismatch on sample {i}")
            labels[i, start:seq_len] = input_ids[i, start:seq_len]

        mel_out = self.mel([m for m in monos], sampling_rate=SAMPLE_RATE,
                           return_tensors="pt", padding="max_length")
        spa, spa_mask, spa_lens, _ = pack_spatial_batch(wavs)
        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "sound_mel": mel_out.input_features,
            "sound_lengths": torch.tensor(native_counts, dtype=torch.long),
            "spatial_audio": spa,
            "spatial_audio_attention_mask": spa_mask,
            "spatial_audio_lengths": spa_lens,
        }
        if not all(has_spatial_list):
            out["has_spatial"] = torch.as_tensor(has_spatial_list, dtype=torch.bool)
        return out
