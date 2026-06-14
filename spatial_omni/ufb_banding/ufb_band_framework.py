"""
UFB banding / SSCV framework skeleton.

This mirrors the paper (2411.03172) and original SPUR-paper description:
  1) Banded covariance (mel-band aggregation)
  2) One-pole smoothing (learnable alpha)
  3) Real-valued vectorization to M^2 + log/normalize

All forward paths are intentionally not implemented yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import math

import torch
from torch import nn


@dataclass
class BandingConfig:
    sample_rate: int = 16000
    frame_length: int = 1024
    hop_length: int = 512
    n_fft: int = 1024
    n_bands: int = 48
    fmin: float = 0.0
    fmax: Optional[float] = None
    center: bool = False


class BandingFrontend(nn.Module):
    """
    Banded covariance frontend.

    Inputs:
      pcm: (B, T_samples, M)
    Outputs:
      cov: (B, T_frames, B_bands, M, M) complex Hermitian
    """

    def __init__(self, config: BandingConfig) -> None:
        super().__init__()
        self.config = config

    @staticmethod
    def _resolve_stft_real_dtype(dtype: torch.dtype) -> torch.dtype:
        # cuFFT does not support bfloat16 for STFT; keep frontend robust by
        # promoting reduced-precision inputs to fp32 for spectral ops.
        if dtype in (torch.float16, torch.bfloat16):
            return torch.float32
        return dtype

    def forward(self, pcm: torch.Tensor) -> torch.Tensor:
        if pcm.ndim != 3:
            raise ValueError(f"pcm must be (B, T_samples, M), got {pcm.shape}")
        if pcm.shape[-1] < 1:
            raise ValueError("pcm must have at least 1 channel.")

        bsz, n_samples, n_ch = pcm.shape
        device = pcm.device
        stft_real_dtype = self._resolve_stft_real_dtype(pcm.dtype)

        window = torch.hann_window(self.config.frame_length, device=device, dtype=stft_real_dtype)
        pcm_2d = pcm.to(dtype=stft_real_dtype).transpose(1, 2).reshape(bsz * n_ch, n_samples)
        stft = torch.stft(
            pcm_2d,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.frame_length,
            window=window,
            center=self.config.center,
            return_complex=True,
        )
        n_freqs, n_frames = stft.shape[-2], stft.shape[-1]
        stft = stft.reshape(bsz, n_ch, n_freqs, n_frames).permute(0, 3, 2, 1)  # (B, T_frames, F_bins, M)

        weights = self._mel_filterbank(
            n_fft=self.config.n_fft,
            sample_rate=self.config.sample_rate,
            n_bands=self.config.n_bands,
            fmin=self.config.fmin,
            fmax=self.config.fmax,
            device=device,
            dtype=stft.real.dtype,
        )
        weights = weights / (weights.sum(dim=-1, keepdim=True).clamp_min(1e-8))
        # Einsum mixes real mel weights and complex STFT; align dtypes explicitly.
        weights = weights.to(dtype=stft.dtype)

        cov = torch.einsum("kf,btfm,btfn->btkmn", weights, stft, stft.conj())
        return cov

    @staticmethod
    def _mel_filterbank(
        n_fft: int,
        sample_rate: int,
        n_bands: int,
        fmin: float,
        fmax: Optional[float],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        n_freqs = n_fft // 2 + 1
        if fmax is None:
            fmax = float(sample_rate) / 2.0

        mel_min = 2595.0 * torch.log10(torch.tensor(1.0 + fmin / 700.0, device=device, dtype=dtype))
        mel_max = 2595.0 * torch.log10(torch.tensor(1.0 + fmax / 700.0, device=device, dtype=dtype))
        mels = torch.linspace(mel_min, mel_max, n_bands + 2, device=device, dtype=dtype)
        hz = 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
        bins = torch.floor((n_fft + 1) * hz / sample_rate).long().clamp(0, n_freqs - 1)

        fb = torch.zeros((n_bands, n_freqs), device=device, dtype=dtype)
        for i in range(n_bands):
            left, center, right = bins[i], bins[i + 1], bins[i + 2]
            if center == left:
                center = left + 1
            if right == center:
                right = center + 1
            fb[i, left:center] = torch.linspace(0.0, 1.0, center - left, device=device, dtype=dtype)
            fb[i, center:right] = torch.linspace(1.0, 0.0, right - center, device=device, dtype=dtype)

        return fb

    @staticmethod
    def infer_output_shape(
        batch: int, frames: int, bands: int, channels: int
    ) -> Tuple[int, int, int, int, int]:
        return (batch, frames, bands, channels, channels)


class OnePoleSmoother(nn.Module):
    """
    One-pole temporal smoothing.

    Inputs:
      cov: (B, T_frames, B_bands, M, M)
    Outputs:
      cov_smooth: (B, T_frames, B_bands, M, M)
    """

    def __init__(self, alpha_init: float = 0.7, learnable: bool = True) -> None:
        super().__init__()
        alpha_init = float(alpha_init)
        alpha_init = min(max(alpha_init, 1e-6), 1 - 1e-6)
        logit = math.log(alpha_init / (1.0 - alpha_init))
        alpha = torch.tensor(logit)
        if learnable:
            self.alpha = nn.Parameter(alpha)
        else:
            self.register_buffer("alpha", alpha)
        self.learnable = learnable

    def forward(self, cov: torch.Tensor) -> torch.Tensor:
        if cov.ndim != 5:
            raise ValueError(f"cov must be (B, T, B_bands, M, M), got {cov.shape}")
        alpha = torch.sigmoid(self.alpha)
        alpha = alpha.clamp(0.0, 0.999)

        outputs = []
        prev = cov[:, 0]
        outputs.append(prev)
        for t in range(1, cov.shape[1]):
            cur = (1.0 - alpha) * cov[:, t] + alpha * prev
            outputs.append(cur)
            prev = cur
        return torch.stack(outputs, dim=1)


class SSCVVectorizer(nn.Module):
    """
    Hermitian covariance -> real-valued SSCV (M^2).

    Inputs:
      cov_smooth: (B, T_frames, B_bands, M, M)
    Outputs:
      sscv: (B, T_frames, B_bands, M^2) real
    """

    def __init__(self, channels: int, diag_transform: str = "real_dft") -> None:
        super().__init__()
        self.channels = channels
        self.sscv_dim = channels * channels
        self.diag_transform = diag_transform
        self.register_buffer("diag_dft", self._build_real_dft(channels), persistent=False)

    def forward(self, cov_smooth: torch.Tensor) -> torch.Tensor:
        if cov_smooth.ndim != 5:
            raise ValueError(f"cov_smooth must be (B, T, B_bands, M, M), got {cov_smooth.shape}")
        m = cov_smooth.shape[-1]
        if m != self.channels:
            raise ValueError(f"cov_smooth last dim {m} != channels {self.channels}")

        diag = torch.diagonal(cov_smooth, dim1=-2, dim2=-1).real
        if self.diag_transform == "real_dft":
            diag = torch.matmul(diag, self.diag_dft.T)
        elif self.diag_transform != "identity":
            raise ValueError(f"Unknown diag_transform: {self.diag_transform}")

        off_parts = []
        scale = torch.sqrt(torch.tensor(2.0, device=cov_smooth.device, dtype=diag.dtype))
        for i in range(m):
            for j in range(i + 1, m):
                cij = cov_smooth[..., i, j]
                off_parts.append(scale * cij.real)
                off_parts.append(scale * cij.imag)
        if off_parts:
            off = torch.stack(off_parts, dim=-1)
            r = torch.cat([diag, off], dim=-1)
        else:
            r = diag

        r0 = r[..., :1].clamp_min(1e-10)
        r_norm = r / r0
        return torch.cat([torch.log(r0), r_norm[..., 1:]], dim=-1)

    def infer_output_shape(
        self, batch: int, frames: int, bands: int
    ) -> Tuple[int, int, int, int]:
        return (batch, frames, bands, self.sscv_dim)

    @staticmethod
    def _build_real_dft(channels: int) -> torch.Tensor:
        eye = torch.eye(channels, dtype=torch.float32)
        dft = torch.fft.fft(eye)
        return dft.real


class SSCVExtractor(nn.Module):
    """
    Full SSCV pipeline:
      BandingFrontend -> OnePoleSmoother -> SSCVVectorizer

    Inputs:
      pcm: (B, T_samples, M)
    Outputs:
      sscv: (B, T_frames, B_bands, M^2)
    """

    def __init__(
        self,
        banding: BandingFrontend,
        smoother: OnePoleSmoother,
        vectorizer: SSCVVectorizer,
    ) -> None:
        super().__init__()
        self.banding = banding
        self.smoother = smoother
        self.vectorizer = vectorizer

    def forward(self, pcm: torch.Tensor) -> torch.Tensor:
        cov = self.banding(pcm)
        cov_smooth = self.smoother(cov)
        sscv = self.vectorizer(cov_smooth)
        # Return as (B, T, M^2, B_bands) to match removed legacy spatial_features layout.
        return sscv.permute(0, 1, 3, 2)

    def infer_output_shape(
        self, batch: int, frames: int, bands: int
    ) -> Tuple[int, int, int, int]:
        return self.vectorizer.infer_output_shape(batch, frames, bands)
