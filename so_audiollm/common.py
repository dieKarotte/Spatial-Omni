"""Spatial audio helpers for the Audio Flamingo 3 integration.

Spatial-token accounting matches the Qwen / Phi-4 SO integrations exactly:
SO-Encoder native 10 Hz, pixel-shuffle factor 4 -> 2.5 Hz LLM-side rate,
sample count -> token count via round-half-to-even.
"""

from __future__ import annotations

from fractions import Fraction
from typing import List, Optional

import numpy as np
import torch

from spatial_omni.modules.so_encoder import SOEncoder
from spatial_omni.modules.so_token_projector import build_so_token_projector
import soundfile as sf

SPATIAL_TOKEN = "<|spatial|>"
SAMPLE_RATE = 16000
MAX_AUDIO_SECONDS = 20.0
MAX_AUDIO_SAMPLES = int(SAMPLE_RATE * MAX_AUDIO_SECONDS)
SO_ENCODER_TOKEN_RATE = 10.0
SO_PROJECTOR_SHUFFLE_FACTOR = 4
SO_LLM_TOKEN_RATE = SO_ENCODER_TOKEN_RATE / SO_PROJECTOR_SHUFFLE_FACTOR  # 2.5 Hz

def load_foa(path: str) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != SAMPLE_RATE:
        import scipy.signal

        gcd = np.gcd(sr, SAMPLE_RATE)
        wav = scipy.signal.resample_poly(wav, SAMPLE_RATE // gcd, sr // gcd, axis=0)
    if wav.shape[1] != 4:
        raise ValueError(f"Expected 4-channel FOA wav, got {wav.shape[1]}ch: {path}")
    return wav[:MAX_AUDIO_SAMPLES]

def load_mono(path: str) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != SAMPLE_RATE:
        import scipy.signal

        gcd = np.gcd(sr, SAMPLE_RATE)
        wav = scipy.signal.resample_poly(wav, SAMPLE_RATE // gcd, sr // gcd, axis=0)
    mono = wav.mean(axis=1) if wav.shape[1] > 1 else wav[:, 0]
    return mono[:MAX_AUDIO_SAMPLES].astype(np.float32)


def spatial_token_count(num_samples: int) -> int:
    rate = Fraction(str(SO_LLM_TOKEN_RATE)).limit_denominator(1000)
    numerator = int(num_samples) * int(rate.numerator)
    denominator = int(SAMPLE_RATE) * int(rate.denominator)
    quotient, remainder = divmod(numerator, denominator)
    twice = remainder * 2
    if twice > denominator or (twice == denominator and quotient % 2 == 1):
        quotient += 1
    return max(1, quotient)


def pack_spatial_batch(wavs: List[np.ndarray]):
    """FOA list [T_i,4] -> padded tensors (audio, mask, lengths, token_lengths)."""
    batch = len(wavs)
    lengths = [min(int(w.shape[0]), MAX_AUDIO_SAMPLES) for w in wavs]
    spa = torch.zeros(batch, MAX_AUDIO_SAMPLES, 4, dtype=torch.float32)
    mask = torch.zeros(batch, MAX_AUDIO_SAMPLES, dtype=torch.float32)
    for i, (wav, length) in enumerate(zip(wavs, lengths)):
        w = torch.as_tensor(np.asarray(wav), dtype=torch.float32)
        if w.ndim != 2 or w.shape[1] != 4:
            raise ValueError(f"spatial wav[{i}] must be [T,4], got {tuple(w.shape)}")
        spa[i, :length] = w[:length]
        mask[i, :length] = 1.0
    token_lengths = torch.tensor([spatial_token_count(n) for n in lengths], dtype=torch.long)
    return spa, mask, torch.tensor(lengths, dtype=torch.long), token_lengths


class SpatialBranch(torch.nn.Module):
    """SO-Encoder + pixel-shuffle projector + placeholder alignment.

    The spatial branch maps encoder 10 Hz tokens -> shuffle-4 projector ->
    per-sample truncate/edge-pad so counts equal the placeholder counts in
    ``input_ids``, then flatten valid rows for index_put injection.
    """

    def __init__(self, beats_checkpoint: str, beats_repo: Optional[str],
                 llm_hidden: int, freeze_backbone: bool = True,
                 projector_hidden: int = 768, enable_replay: bool = False) -> None:
        super().__init__()
        self.so_encoder = SOEncoder(
            checkpoint_path=beats_checkpoint,
            beats_repo_path=beats_repo or None,
            freeze_backbone=freeze_backbone,
            max_audio_seconds=MAX_AUDIO_SECONDS,
            encoder_token_rate=SO_ENCODER_TOKEN_RATE,
        )
        self.so_projector = build_so_token_projector(
            projector_type="pixel_shuffle",
            input_dim=self.so_encoder.encoder_dim,
            hidden_dim=projector_hidden,
            output_dim=llm_hidden,
            shuffle_factor=SO_PROJECTOR_SHUFFLE_FACTOR,
        )
        # Mono-replay null bank: fills <|spatial|> rows of
        # replay samples; the encoder's W-only output is MSE-aligned to it so
        # mono input maps to the null state.
        if enable_replay:
            n_null = int(round(MAX_AUDIO_SECONDS * SO_LLM_TOKEN_RATE))
            self.spatial_null = torch.nn.Parameter(
                torch.randn(n_null, llm_hidden) * 0.02)
        else:
            self.spatial_null = None

    def build(self) -> None:
        self.so_encoder._build_model()

    def reset_projector(self) -> None:
        for m in self.so_projector.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        for n, p in self.so_projector.named_parameters():
            assert not p.isnan().any(), f"so_projector.{n} NaN after reset"
        if self.spatial_null is not None:
            torch.nn.init.normal_(self.spatial_null, mean=0.0, std=0.02)

    def forward(self, spatial_audio, spatial_mask, spatial_lengths,
                input_ids: torch.LongTensor, spatial_token_id: int,
                has_spatial: Optional[torch.Tensor] = None):
        """Return flattened projected rows matching placeholder positions.

        Output:
            flat  – [sum_i C_i, D_llm] rows in batch order
            counts – [B] placeholder counts per sample (asserted > alignment)
            loss_null – scalar MSE(W-only encoder out, spatial_null) or None
        """
        enc = self.so_encoder(spatial_audio, spatial_mask, spatial_lengths)
        projected = self.so_projector(enc.spatial_tokens)
        k = int(getattr(self.so_projector, "shuffle_factor", 1))
        lens = enc.spatial_token_lengths
        if k > 1:
            lens = torch.div(lens, k, rounding_mode="floor")
            lens = torch.clamp(lens, min=0, max=int(projected.shape[1]))
            lens = torch.where((enc.spatial_token_lengths > 0) & (lens == 0),
                               torch.ones_like(lens), lens)
        counts = (input_ids == spatial_token_id).sum(dim=1).to(lens.device)
        batch, _, hidden = projected.shape

        loss_null = None
        if has_spatial is not None:
            if self.spatial_null is None:
                raise ValueError("has_spatial given but enable_replay=False")
            has_spatial = has_spatial.to(device=projected.device, dtype=torch.bool)
            null_bank = self.spatial_null.to(projected.dtype)
            replay_idx = (~has_spatial).nonzero(as_tuple=True)[0]
            if replay_idx.numel() > 0:
                w_rows, t_rows = [], []
                for i in replay_idx.tolist():
                    src_len = int(lens[i].item())
                    if src_len <= 0:
                        continue
                    w_rows.append(projected[i, :src_len])
                    t_rows.append(null_bank[:src_len].detach())
                if w_rows:
                    loss_null = torch.nn.functional.mse_loss(
                        torch.cat(w_rows, dim=0), torch.cat(t_rows, dim=0))

        rows = []
        for i, (src_len, tgt_len) in enumerate(zip(lens.tolist(), counts.tolist())):
            if tgt_len == 0:
                continue
            if has_spatial is not None and not bool(has_spatial[i]):
                # replay row: inject the null bank (encoder output only feeds MSE)
                null_bank = self.spatial_null.to(projected.dtype)
                rows.append(null_bank[:tgt_len])
                continue
            copy_len = min(src_len, tgt_len)
            row = projected[i, :copy_len]
            if tgt_len > copy_len:  # edge-pad with last valid frame
                pad = projected[i, copy_len - 1 : copy_len].expand(tgt_len - copy_len, hidden) \
                    if copy_len > 0 else projected.new_zeros((tgt_len, hidden))
                row = torch.cat([row, pad], dim=0)
            rows.append(row)
        if not rows:
            return projected.new_zeros((0, hidden)), counts, loss_null
        return torch.cat(rows, dim=0), counts, loss_null
