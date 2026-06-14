"""Online feature-bridge scaffold for the SELD task-233 spatial branch."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import os
from typing import Optional

import torch
from torch import nn

from ..utils.spatial_seld_utils import (
    attention_mask_to_lengths,
    build_1d_attention_mask,
    clamp_lengths,
    samples_to_feature_frames,
)


@dataclass
class SeldFeatureBridgeOutput:
    """Output container for baseline-compatible SELD features.

    Attributes:
        features:
            Baseline-compatible FOA features with shape `[B, 7, T_feat_max, 64]`.
            Channel semantics are:
            - channels `0..3`: log-mel features for `W, X, Y, Z`
            - channels `4..6`: FOA intensity-vector features
        feature_attention_mask:
            Boolean mask of shape `[B, T_feat_max]` marking valid feature frames.
        feature_lengths:
            Valid feature lengths in frames, shape `[B]`.
    """

    features: torch.FloatTensor
    feature_attention_mask: torch.BoolTensor
    feature_lengths: torch.LongTensor


class SeldFeatureBridge(nn.Module):
    """Convert padded FOA waveforms into task-233 feature tensors.

    Input:
        `spatial_audio`:
            Tensor of shape `[B, T_audio, 4]`.
        `spatial_audio_attention_mask`:
            Optional mask of shape `[B, T_audio]`.
        `spatial_audio_lengths`:
            Optional valid waveform lengths, shape `[B]`.

    Processing:
        1. Validate that the input is `16 kHz`, `4-channel FOA`.
        2. Convert sample-level masks into valid waveform lengths.
        3. Convert waveform lengths into expected baseline feature lengths.
        4. Delegate the actual STFT/mel/intensity-vector extraction to an
           intentionally unimplemented private hook.

    Output:
        [`SeldFeatureBridgeOutput`]
            - `features`: `[B, 7, T_feat_max, 64]`
            - `feature_attention_mask`: `[B, T_feat_max]`
            - `feature_lengths`: `[B]`
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        max_audio_seconds: float = 20.0,
        num_feature_channels: int = 7,
        num_mel_bins: int = 64,
        hop_length: int = 320,
        baseline_repo_path: Optional[str] = None,
        task_id: str = "233",
        feature_stats_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.baseline_repo_path = baseline_repo_path
        self.task_id = str(task_id)
        self.feature_stats_dir = feature_stats_dir
        self._cached_task_params = None

        frontend_params = self._resolve_frontend_params(
            sample_rate=sample_rate,
            max_audio_seconds=max_audio_seconds,
            num_feature_channels=num_feature_channels,
            num_mel_bins=num_mel_bins,
            hop_length=hop_length,
        )
        self.sample_rate = int(frontend_params["sample_rate"])
        self.max_audio_seconds = float(frontend_params["max_audio_seconds"])
        self.num_feature_channels = int(frontend_params["num_feature_channels"])
        self.num_mel_bins = int(frontend_params["num_mel_bins"])
        self.hop_length = int(frontend_params["hop_length"])
        self.max_audio_samples = int(round(self.sample_rate * self.max_audio_seconds))
        self.max_feature_frames = self.max_audio_samples // self.hop_length
        self.win_length = int(frontend_params["win_length"])
        self.n_fft = int(frontend_params["n_fft"])
        self.mel_fmin = float(frontend_params["mel_fmin"])
        self.mel_fmax = frontend_params["mel_fmax"]
        self.db_amin = 1e-10
        self.db_top = 80.0
        self.eps = 1e-8

        self.register_buffer("mel_wts", torch.empty(0), persistent=False)
        self.register_buffer("norm_mean", torch.empty(0), persistent=False)
        self.register_buffer("norm_scale", torch.empty(0), persistent=False)
        self.register_buffer("stft_window", torch.hann_window(self.win_length), persistent=False)

    def forward(
        self,
        spatial_audio: torch.Tensor,
        spatial_audio_attention_mask: Optional[torch.Tensor] = None,
        spatial_audio_lengths: Optional[torch.LongTensor] = None,
    ) -> SeldFeatureBridgeOutput:
        """Build baseline-compatible SELD features from padded FOA waveforms.

        Args:
            spatial_audio:
                Tensor of shape `[B, T_audio, 4]`.
            spatial_audio_attention_mask:
                Optional mask of shape `[B, T_audio]`.
            spatial_audio_lengths:
                Optional valid lengths, shape `[B]`.

        Returns:
            [`SeldFeatureBridgeOutput`].

        Raises:
            NotImplementedError:
                Always, until `_extract_online_features` is implemented.
        """

        if spatial_audio.ndim != 3:
            raise ValueError(
                f"spatial_audio must have shape [B, T_audio, 4], got {tuple(spatial_audio.shape)}"
            )
        if spatial_audio.shape[-1] != 4:
            raise ValueError(
                f"spatial_audio must contain exactly 4 FOA channels, got {spatial_audio.shape[-1]}"
            )

        mask_lengths = attention_mask_to_lengths(
            spatial_audio_attention_mask,
            max_length=spatial_audio.shape[1],
        )
        if spatial_audio_lengths is None:
            if mask_lengths is None:
                spatial_audio_lengths = spatial_audio.new_full(
                    (spatial_audio.shape[0],),
                    fill_value=spatial_audio.shape[1],
                    dtype=torch.long,
                )
            else:
                spatial_audio_lengths = mask_lengths
        elif mask_lengths is not None and not torch.equal(spatial_audio_lengths.cpu(), mask_lengths.cpu()):
            raise ValueError(
                "spatial_audio_lengths and spatial_audio_attention_mask disagree on valid waveform lengths"
            )

        spatial_audio_lengths = clamp_lengths(
            spatial_audio_lengths.to(device=spatial_audio.device, dtype=torch.long),
            max_length=min(spatial_audio.shape[1], self.max_audio_samples),
        )
        feature_lengths = samples_to_feature_frames(
            spatial_audio_lengths,
            hop_length=self.hop_length,
        )
        feature_attention_mask = build_1d_attention_mask(
            feature_lengths,
            max_length=self.max_feature_frames,
        )
        return self._extract_online_features(
            spatial_audio=spatial_audio,
            spatial_audio_lengths=spatial_audio_lengths,
            feature_lengths=feature_lengths,
            feature_attention_mask=feature_attention_mask,
        )

    def _extract_online_features(
        self,
        spatial_audio: torch.Tensor,
        spatial_audio_lengths: torch.LongTensor,
        feature_lengths: torch.LongTensor,
        feature_attention_mask: torch.BoolTensor,
    ) -> SeldFeatureBridgeOutput:
        """Extract online baseline-compatible SELD features.

        Args:
            spatial_audio:
                Padded FOA waveform batch, shape `[B, T_audio, 4]`.
            spatial_audio_lengths:
                Valid waveform lengths, shape `[B]`.
            feature_lengths:
                Expected baseline feature lengths, shape `[B]`.
            feature_attention_mask:
                Boolean feature-frame mask, shape `[B, T_feat_max]`.

        Returns:
            [`SeldFeatureBridgeOutput`].

        Notes:
            Reproduces the DCASE task-233 feature pipeline:
            `4ch log-mel + 3ch FOA intensity vectors -> [B, 7, T_feat, 64]`,
            followed by the baseline `foa_wts` normalization.
        """

        self._ensure_frontend_assets()

        batch_size, max_audio_steps, num_channels = spatial_audio.shape
        if max_audio_steps != self.max_audio_samples:
            max_frames = min(feature_attention_mask.shape[1], self.max_feature_frames)
        else:
            max_frames = self.max_feature_frames

        audio_channels_first = spatial_audio.transpose(1, 2).to(dtype=torch.float32)
        stft_input = audio_channels_first.reshape(batch_size * num_channels, max_audio_steps)
        # The STFT + mel + intensity pipeline has no trainable parameters (only
        # frozen buffers). It's run in fp32 for numerical stability. We do NOT
        # wrap in torch.no_grad() here: that would sever the autograd chain
        # from spatial_audio into the downstream adapter, and for the IV path
        # the in-place assignment pattern
        #     spatial_tokens[idx, :n] = adapter(...)
        # in iv_spatial_adapters.py empirically produces all-NaN grads on the
        # adapter weights when the LHS has requires_grad=False. Autograd can
        # safely skip the (frozen) bridge internals; it just needs the graph
        # to remain traceable from spatial_audio forward.
        #
        # cuFFT plan exhaustion on long runs (the original Run 2 crash at
        # step 2586) is handled below by catching the RuntimeError and
        # falling back to a one-shot CPU STFT.
        try:
            stft = torch.stft(
                stft_input,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.stft_window.to(device=spatial_audio.device, dtype=torch.float32),
                center=True,
                pad_mode="constant",
                normalized=False,
                onesided=True,
                return_complex=True,
            )
        except RuntimeError as exc:
            # `cuFFT_INTERNAL_ERROR` typically means the plan cache is
            # exhausted / corrupted (e.g. after long runs on a shared GPU).
            # Clear the plan cache and retry once on CPU so the run can
            # continue instead of hard-crashing.
            if "cuFFT" not in str(exc) and "CUFFT" not in str(exc):
                raise
            try:
                if stft_input.is_cuda:
                    torch.cuda.empty_cache()
                    try:
                        # torch >= 2.0: per-device plan cache clear
                        from torch.backends import cuda as _torch_cuda_backend  # noqa: WPS433
                        cache = _torch_cuda_backend.cufft_plan_cache[stft_input.device.index]
                        cache.clear()
                    except Exception:  # noqa: BLE001
                        pass
                stft_cpu = torch.stft(
                    stft_input.detach().cpu(),
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    win_length=self.win_length,
                    window=self.stft_window.to(device="cpu", dtype=torch.float32),
                    center=True,
                    pad_mode="constant",
                    normalized=False,
                    onesided=True,
                    return_complex=True,
                )
                stft = stft_cpu.to(device=stft_input.device)
            except Exception:
                # Re-raise the original cuFFT error; the outer training
                # loop will emit a NaN/Inf skip and continue (see
                # `_sync_skip_flag` in train_spatial_iv_qa.py).
                raise exc
        max_frames = min(max_frames, stft.shape[-1])
        stft = stft[..., :max_frames]
        linear_spectra = stft.reshape(batch_size, num_channels, stft.shape[-2], stft.shape[-1]).permute(0, 3, 2, 1)

        mel_wts = self.mel_wts.to(device=spatial_audio.device, dtype=torch.float32)
        norm_mean = self.norm_mean.to(device=spatial_audio.device, dtype=torch.float32)
        norm_scale = self.norm_scale.to(device=spatial_audio.device, dtype=torch.float32).clamp_min(self.eps)

        mel_features = self._build_log_mel_features(linear_spectra, mel_wts)
        intensity_features = self._build_intensity_vector_features(linear_spectra, mel_wts)
        flattened_features = torch.cat((mel_features, intensity_features), dim=-1)
        normalized_features = (flattened_features - norm_mean.view(1, 1, -1)) / norm_scale.view(1, 1, -1)

        valid_feature_mask = feature_attention_mask[:, :max_frames].unsqueeze(-1).to(normalized_features.dtype)
        normalized_features = normalized_features * valid_feature_mask
        features = normalized_features.reshape(
            batch_size,
            max_frames,
            self.num_feature_channels,
            self.num_mel_bins,
        ).permute(0, 2, 1, 3)
        return SeldFeatureBridgeOutput(
            features=features,
            feature_attention_mask=feature_attention_mask[:, :max_frames],
            feature_lengths=clamp_lengths(feature_lengths, max_length=max_frames),
        )

    @staticmethod
    def _next_greater_power_of_2(value: int) -> int:
        if value <= 0:
            raise ValueError(f"value must be > 0, got {value}")
        return 1 << (value - 1).bit_length()

    def _ensure_frontend_assets(self) -> None:
        if self.mel_wts.numel() == 0:
            import librosa

            mel_wts = librosa.filters.mel(
                sr=self.sample_rate,
                n_fft=self.n_fft,
                n_mels=self.num_mel_bins,
                fmin=self.mel_fmin,
                fmax=self.mel_fmax,
            ).T
            self.mel_wts = torch.as_tensor(mel_wts, dtype=torch.float32)

        if self.norm_mean.numel() == 0 or self.norm_scale.numel() == 0:
            import joblib

            scaler = joblib.load(self._resolve_normalization_weights_path())
            self.norm_mean = torch.as_tensor(scaler.mean_, dtype=torch.float32)
            self.norm_scale = torch.as_tensor(scaler.scale_, dtype=torch.float32)

    def _resolve_normalization_weights_path(self) -> str:
        if self.feature_stats_dir:
            if os.path.isdir(self.feature_stats_dir):
                candidate = os.path.join(self.feature_stats_dir, "foa_wts")
            else:
                candidate = self.feature_stats_dir
            if os.path.exists(candidate):
                return candidate
            raise FileNotFoundError(f"Normalization weights not found: {candidate}")

        params = self._load_task_params()
        candidate = os.path.join(params["feat_label_dir"], "foa_wts")
        if not os.path.exists(candidate):
            raise FileNotFoundError(f"Normalization weights not found: {candidate}")
        return candidate

    def _load_task_params(self) -> dict:
        if self._cached_task_params is not None:
            return self._cached_task_params
        if not self.baseline_repo_path:
            raise ValueError(
                "baseline_repo_path is required to derive task-233 normalization weights when "
                "feature_stats_dir is not provided."
            )

        params_path = os.path.join(self.baseline_repo_path, "parameters.py")
        spec = importlib.util.spec_from_file_location("seld_parameters_for_bridge", params_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load baseline parameters module from {params_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._cached_task_params = module.get_params(self.task_id)
        return self._cached_task_params

    def _resolve_frontend_params(
        self,
        sample_rate: int,
        max_audio_seconds: float,
        num_feature_channels: int,
        num_mel_bins: int,
        hop_length: int,
    ) -> dict:
        params = None
        if self.baseline_repo_path:
            params = self._load_task_params()

        sample_rate = int(params.get("fs", sample_rate)) if params is not None else int(sample_rate)
        max_audio_seconds = float(max_audio_seconds)
        num_mel_bins = int(params.get("nb_mel_bins", num_mel_bins)) if params is not None else int(num_mel_bins)
        hop_length = int(params.get("hop_len", hop_length)) if params is not None else int(hop_length)
        win_length = int(params.get("win_len", 2 * hop_length)) if params is not None else int(2 * hop_length)
        n_fft = int(params.get("n_fft", self._next_greater_power_of_2(win_length))) if params is not None else int(self._next_greater_power_of_2(win_length))
        mel_fmin = float(params.get("mel_fmin", 0.0)) if params is not None else 0.0
        mel_fmax = params.get("mel_fmax", None) if params is not None else None
        if mel_fmax is not None:
            mel_fmax = float(mel_fmax)

        return {
            "sample_rate": sample_rate,
            "max_audio_seconds": max_audio_seconds,
            "num_feature_channels": int(num_feature_channels),
            "num_mel_bins": num_mel_bins,
            "hop_length": hop_length,
            "win_length": win_length,
            "n_fft": n_fft,
            "mel_fmin": mel_fmin,
            "mel_fmax": mel_fmax,
        }

    def _build_log_mel_features(
        self,
        linear_spectra: torch.Tensor,
        mel_wts: torch.Tensor,
    ) -> torch.Tensor:
        power_spectra = torch.abs(linear_spectra) ** 2
        mel_spectra = torch.einsum("btfc,fm->btmc", power_spectra, mel_wts)
        log_mel_spectra = 10.0 * torch.log10(torch.clamp(mel_spectra, min=self.db_amin))
        peak = log_mel_spectra.amax(dim=(1, 2), keepdim=True)
        log_mel_spectra = torch.maximum(log_mel_spectra, peak - self.db_top)
        return log_mel_spectra.permute(0, 1, 3, 2).reshape(
            linear_spectra.shape[0],
            linear_spectra.shape[1],
            self.num_mel_bins * linear_spectra.shape[-1],
        )

    def _build_intensity_vector_features(
        self,
        linear_spectra: torch.Tensor,
        mel_wts: torch.Tensor,
    ) -> torch.Tensor:
        w_channel = linear_spectra[..., 0]
        xyz_channels = linear_spectra[..., 1:]
        intensity = torch.real(torch.conj(w_channel).unsqueeze(-1) * xyz_channels)
        energy = self.eps + (
            torch.abs(w_channel) ** 2 + (torch.abs(xyz_channels) ** 2).sum(dim=-1) / 3.0
        )
        normalized_intensity = intensity / energy.unsqueeze(-1)
        intensity_mel = torch.einsum("btfc,fm->btmc", normalized_intensity, mel_wts)
        return intensity_mel.permute(0, 1, 3, 2).reshape(
            linear_spectra.shape[0],
            linear_spectra.shape[1],
            self.num_mel_bins * xyz_channels.shape[-1],
        )
