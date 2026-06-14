"""IV 和 Neural-IV spatial encoder baselines.

从 DCASE2024_seld_baseline（legacy-qwen-2.5-omni/qwen2_5_omni_legacy/modules/
simple_spatial_adapters.py）迁移而来。两个 baseline 都依赖 `SeldFeatureBridge`
产出的 7 通道特征（前 4 通道 log-mel，后 3 通道为归一化 intensity vector）。

返回签名:
    (spatial_tokens [B, T_s, token_dim],
     spatial_lengths [B])
与 `SOEncoderOutput` 结构对齐，保证上游 `masked_scatter` 注入逻辑可复用。

Output rate (2.5 Hz spatial token):
  T_feat → (feature_to_seld_ratio=5) → T_seld → (downsample_factor=4) → T_spat
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from ..utils.spatial_seld_utils import (
    build_1d_attention_mask,
    feature_frames_to_seld_frames,
    seld_frames_to_spatial_tokens,
)
from .seld_feature_bridge import (
    SeldFeatureBridge,
    SeldFeatureBridgeOutput,
)


@dataclass
class IVSpatialAdapterOutput:
    """与 `SOEncoderOutput` 签名对齐的容器。"""

    spatial_tokens: torch.FloatTensor       # [B, T_s, token_dim]
    spatial_token_lengths: torch.LongTensor # [B]


class _BaseSimpleSpatialAdapter(nn.Module):
    """Common feature resolution and length bookkeeping for simple baselines."""

    def __init__(
        self,
        feature_bridge: SeldFeatureBridge,
        token_dim: int,
        feature_to_seld_ratio: int,
        downsample_factor: int,
        output_scale: float = 0.02,
    ) -> None:
        super().__init__()
        self.feature_bridge = feature_bridge
        self.token_dim = int(token_dim)
        self.feature_to_seld_ratio = int(feature_to_seld_ratio)
        self.downsample_factor = int(downsample_factor)
        self.output_scale = float(output_scale)
        self.feature_clip_value = 10.0

    def _resolve_feature_output(
        self,
        spatial_audio: Optional[torch.Tensor],
        spatial_audio_attention_mask: Optional[torch.Tensor],
        spatial_audio_lengths: Optional[torch.LongTensor],
        seld_features: Optional[torch.Tensor],
        seld_feature_attention_mask: Optional[torch.Tensor],
        seld_feature_lengths: Optional[torch.LongTensor],
    ) -> SeldFeatureBridgeOutput:
        """Return `SeldFeatureBridgeOutput` either from cached features or
        on-the-fly from raw FOA audio.

        Note: The feature bridge performs IV = Re(conj(W) * [X,Y,Z]) / (|W|² +
        (|XYZ|²)/3 + eps). In bf16 this energy denominator can underflow on
        near-silent frames, producing Inf/NaN. We force fp32 for this step.

        fp32 scope (IV path):
            feature_bridge (STFT + log-mel + intensity vector + foa_wts
            normalization) is a *pure operator*. It holds only non-trainable
            buffers (``mel_wts``, ``norm_mean``, ``norm_scale``, ``stft_window``)
            and has no trainable parameters in any training stage. We therefore
            force **fp32 + no_grad** around it so that:

              1. `torch.stft` does not build a differentiable cuFFT plan (saves
                 a complex spectrogram's worth of activation memory per batch
                 and avoids `cuFFT_INTERNAL_ERROR` on long runs when the plan
                 workspace starts competing with the LLM's attention buffers);
              2. the intensity-vector energy normalization cannot underflow to
                 NaN/Inf in bf16 near-silent frames;
              3. no backward pass is attempted through the frozen
                 feature_bridge.
        """
        if seld_features is not None:
            if seld_features.ndim != 4:
                raise ValueError(
                    "seld_features must have shape [B, C, T_feat, M], "
                    f"got {tuple(seld_features.shape)}"
                )
            if seld_feature_lengths is None:
                if seld_feature_attention_mask is None:
                    seld_feature_lengths = seld_features.new_full(
                        (seld_features.shape[0],),
                        fill_value=seld_features.shape[2],
                        dtype=torch.long,
                    )
                else:
                    seld_feature_lengths = seld_feature_attention_mask.to(torch.long).sum(dim=1)
            if seld_feature_attention_mask is None:
                seld_feature_attention_mask = build_1d_attention_mask(
                    seld_feature_lengths.to(dtype=torch.long),
                    max_length=seld_features.shape[2],
                )
            return SeldFeatureBridgeOutput(
                features=seld_features,
                feature_attention_mask=seld_feature_attention_mask,
                feature_lengths=seld_feature_lengths.to(dtype=torch.long),
            )

        if spatial_audio is None:
            raise ValueError(
                "IV/Neural-IV spatial baselines require either `spatial_audio` "
                "or pre-computed `seld_features`."
            )
        # Force fp32 for the feature bridge. We used to wrap in torch.no_grad()
        # here to save memory + avoid cuFFT plan pressure, but that breaks the
        # autograd chain into the per-sample adapter loop below:
        #
        #   iv_features.requires_grad = False (under no_grad)
        #   spatial_tokens = iv_features.new_zeros(...)  # also requires_grad=False
        #   spatial_tokens[idx, :n] = self.token_head(...)  # ← in-place assign of
        #     a grad-requiring RHS into a non-grad LHS. In the outer DDP +
        #     find_unused_parameters=True + gradient_checkpointing path this
        #     empirically produces all-NaN grads on the adapter weights at
        #     step 1 (skip_g ≈ 50-100%). The original (Run 2) configuration
        #     without no_grad ran clean for 2586 steps before the separate
        #     cuFFT crash — so the memory-saving no_grad is a net loss.
        #
        # The SELD233 bridge itself still has its own autograd-safety guards
        # inside `_extract_online_features` (and that call also runs fp32).
        with torch.autocast(device_type=spatial_audio.device.type, enabled=False):
            fb_out = self.feature_bridge(
                spatial_audio=spatial_audio.to(dtype=torch.float32),
                spatial_audio_attention_mask=spatial_audio_attention_mask,
                spatial_audio_lengths=spatial_audio_lengths,
            )
        # Sanitize features to guard against any remaining extreme values.
        clean_features = torch.nan_to_num(
            fb_out.features, nan=0.0, posinf=self.feature_clip_value, neginf=-self.feature_clip_value
        ).clamp(min=-self.feature_clip_value, max=self.feature_clip_value)
        return SeldFeatureBridgeOutput(
            features=clean_features,
            feature_attention_mask=fb_out.feature_attention_mask,
            feature_lengths=fb_out.feature_lengths,
        )

    def _compute_spatial_lengths(self, feature_lengths: torch.LongTensor) -> torch.LongTensor:
        seld_lengths = feature_frames_to_seld_frames(
            feature_lengths.to(dtype=torch.long),
            feature_to_seld_ratio=self.feature_to_seld_ratio,
        )
        return seld_frames_to_spatial_tokens(
            seld_lengths,
            downsample_factor=self.downsample_factor,
        )

    def _pool_time_axis(self, features: torch.Tensor, target_length: int) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError(f"features must have shape [T, D], got {tuple(features.shape)}")
        pooled = F.adaptive_avg_pool1d(features.transpose(0, 1).unsqueeze(0), output_size=target_length)
        return pooled.squeeze(0).transpose(0, 1)

    def _sanitize_tensor(self, tensor: torch.Tensor, clip_value: Optional[float] = None) -> torch.Tensor:
        clip = self.feature_clip_value if clip_value is None else float(clip_value)
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=clip, neginf=-clip)
        return tensor.clamp(min=-clip, max=clip)

    @staticmethod
    def _init_linear_stack(module: nn.Module, final_std: float = 1e-3) -> None:
        linear_layers = [child for child in module.modules() if isinstance(child, nn.Linear)]
        for index, layer in enumerate(linear_layers):
            if index == len(linear_layers) - 1:
                nn.init.normal_(layer.weight, mean=0.0, std=final_std)
            else:
                nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


class IVSpatialAdapter(_BaseSimpleSpatialAdapter):
    """Use handcrafted FOA intensity vectors directly as spatial tokens.

    Pipeline:
        FOA audio [B, T, 4] → feature_bridge → [B, 7, T_feat, M]
          → take channels 4..6 (IV x,y,z) → [B, 3, T_feat, M]
          → (optional band_pool to reduce M) → reshape → [T_feat, 3M]
          → adaptive_avg_pool1d to T_spat → LayerNorm → MLP(dim → token_dim)
          → output_scale * ... → [B, T_spat, token_dim]
    """

    def __init__(
        self,
        feature_bridge: SeldFeatureBridge,
        token_dim: int = 256,
        feature_to_seld_ratio: int = 5,
        downsample_factor: int = 4,
        band_pool: int = 0,
        num_mel_bins: int = 64,
        output_scale: float = 0.02,
    ) -> None:
        super().__init__(
            feature_bridge=feature_bridge,
            token_dim=token_dim,
            feature_to_seld_ratio=feature_to_seld_ratio,
            downsample_factor=downsample_factor,
            output_scale=output_scale,
        )
        self.band_pool = int(band_pool)
        self.num_mel_bins = int(num_mel_bins)
        # 3 xyz directions × (band_pool if active else num_mel_bins) mel bins
        self.input_dim = 3 * (self.band_pool if self.band_pool > 0 else self.num_mel_bins)
        self.token_norm = nn.LayerNorm(self.input_dim)
        self.token_head = nn.Sequential(
            nn.Linear(self.input_dim, self.token_dim),
            nn.GELU(),
            nn.Linear(self.token_dim, self.token_dim),
        )
        self._init_linear_stack(self.token_head)

    def forward(
        self,
        spatial_audio: Optional[torch.Tensor] = None,
        spatial_audio_attention_mask: Optional[torch.Tensor] = None,
        spatial_audio_lengths: Optional[torch.LongTensor] = None,
        seld_features: Optional[torch.Tensor] = None,
        seld_feature_attention_mask: Optional[torch.Tensor] = None,
        seld_feature_lengths: Optional[torch.LongTensor] = None,
    ) -> IVSpatialAdapterOutput:
        feature_output = self._resolve_feature_output(
            spatial_audio=spatial_audio,
            spatial_audio_attention_mask=spatial_audio_attention_mask,
            spatial_audio_lengths=spatial_audio_lengths,
            seld_features=seld_features,
            seld_feature_attention_mask=seld_feature_attention_mask,
            seld_feature_lengths=seld_feature_lengths,
        )
        # [B, 7, T_feat, M] → IV channels are last 3: [B, 3, T_feat, M]
        iv_features = feature_output.features[:, 4:7]
        spatial_lengths = self._compute_spatial_lengths(feature_output.feature_lengths)
        max_tokens = int(spatial_lengths.max().item()) if spatial_lengths.numel() else 0
        max_tokens = max(max_tokens, 1)
        target_dtype = self.token_head[0].weight.dtype
        spatial_tokens = iv_features.new_zeros(
            (iv_features.shape[0], max_tokens, self.token_dim),
            dtype=target_dtype,
        )

        for index in range(iv_features.shape[0]):
            current_feature_length = int(feature_output.feature_lengths[index].item())
            current_token_length = int(spatial_lengths[index].item())
            if current_token_length <= 0 or current_feature_length <= 0:
                continue
            # Keep batch dim for adaptive_avg_pool2d; compute on fp32 through token_head.
            current_iv = iv_features[index : index + 1, :, :current_feature_length, :].to(dtype=target_dtype)
            current_iv = self._sanitize_tensor(current_iv)
            if self.band_pool > 0 and self.band_pool != current_iv.shape[-1]:
                current_iv = F.adaptive_avg_pool2d(
                    current_iv, output_size=(current_feature_length, self.band_pool)
                )
            # [1, 3, T_feat, M] → [T_feat, 3, M] → [T_feat, 3*M]
            current_iv = current_iv.squeeze(0).permute(1, 0, 2).reshape(current_feature_length, -1)
            current_iv = self._pool_time_axis(current_iv, target_length=current_token_length)
            current_iv = self.token_norm(self._sanitize_tensor(current_iv))
            current_iv = self.output_scale * self.token_head(current_iv)
            current_iv = self._sanitize_tensor(current_iv, clip_value=1.0)
            spatial_tokens[index, :current_token_length] = current_iv

        return IVSpatialAdapterOutput(
            spatial_tokens=spatial_tokens,
            spatial_token_lengths=spatial_lengths,
        )


class NeuralIVSpatialAdapter(_BaseSimpleSpatialAdapter):
    """Learn spatial tokens from IV features with a small CNN + MLP.

    Pipeline:
        FOA audio → feature_bridge → [B, 7, T_feat, M] → IV [B, 3, T_feat, M]
          → Conv2d(3→mid→hidden_channels) + GELU → [B, C, T_feat, M]
          → mean over mel (dim=-1) → [T_feat, C]
          → adaptive_avg_pool1d to T_spat → LayerNorm → MLP(C → token_dim)
          → output_scale * ... → [B, T_spat, token_dim]
    """

    def __init__(
        self,
        feature_bridge: SeldFeatureBridge,
        token_dim: int = 256,
        feature_to_seld_ratio: int = 5,
        downsample_factor: int = 4,
        hidden_channels: int = 64,
        output_scale: float = 0.02,
    ) -> None:
        super().__init__(
            feature_bridge=feature_bridge,
            token_dim=token_dim,
            feature_to_seld_ratio=feature_to_seld_ratio,
            downsample_factor=downsample_factor,
            output_scale=output_scale,
        )
        self.hidden_channels = int(hidden_channels)
        mid_channels = max(16, self.hidden_channels // 2)
        self.conv_encoder = nn.Sequential(
            nn.Conv2d(3, mid_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(mid_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.token_norm = nn.LayerNorm(self.hidden_channels)
        self.token_head = nn.Sequential(
            nn.Linear(self.hidden_channels, self.token_dim),
            nn.GELU(),
            nn.Linear(self.token_dim, self.token_dim),
        )
        self._init_conv_encoder()
        self._init_linear_stack(self.token_head)

    def _init_conv_encoder(self) -> None:
        for module in self.conv_encoder.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight, a=0.0, mode="fan_in", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        spatial_audio: Optional[torch.Tensor] = None,
        spatial_audio_attention_mask: Optional[torch.Tensor] = None,
        spatial_audio_lengths: Optional[torch.LongTensor] = None,
        seld_features: Optional[torch.Tensor] = None,
        seld_feature_attention_mask: Optional[torch.Tensor] = None,
        seld_feature_lengths: Optional[torch.LongTensor] = None,
    ) -> IVSpatialAdapterOutput:
        feature_output = self._resolve_feature_output(
            spatial_audio=spatial_audio,
            spatial_audio_attention_mask=spatial_audio_attention_mask,
            spatial_audio_lengths=spatial_audio_lengths,
            seld_features=seld_features,
            seld_feature_attention_mask=seld_feature_attention_mask,
            seld_feature_lengths=seld_feature_lengths,
        )
        iv_features = feature_output.features[:, 4:7]
        spatial_lengths = self._compute_spatial_lengths(feature_output.feature_lengths)
        max_tokens = int(spatial_lengths.max().item()) if spatial_lengths.numel() else 0
        max_tokens = max(max_tokens, 1)
        target_dtype = self.conv_encoder[0].weight.dtype
        spatial_tokens = iv_features.new_zeros(
            (iv_features.shape[0], max_tokens, self.token_dim),
            dtype=target_dtype,
        )

        for index in range(iv_features.shape[0]):
            current_feature_length = int(feature_output.feature_lengths[index].item())
            current_token_length = int(spatial_lengths[index].item())
            if current_token_length <= 0 or current_feature_length <= 0:
                continue
            current_iv = iv_features[index : index + 1, :, :current_feature_length, :].to(dtype=target_dtype)
            current_iv = self._sanitize_tensor(current_iv)
            current_hidden = self.conv_encoder(current_iv)           # [1, C, T_feat, M]
            current_hidden = self._sanitize_tensor(current_hidden)
            # Average over mel dim, remove batch axis, transpose to [T_feat, C]
            current_hidden = current_hidden.mean(dim=-1).squeeze(0).transpose(0, 1)
            current_hidden = self._pool_time_axis(current_hidden, target_length=current_token_length)
            current_hidden = self.token_norm(self._sanitize_tensor(current_hidden))
            current_hidden = self.output_scale * self.token_head(current_hidden)
            current_hidden = self._sanitize_tensor(current_hidden, clip_value=1.0)
            spatial_tokens[index, :current_token_length] = current_hidden

        return IVSpatialAdapterOutput(
            spatial_tokens=spatial_tokens,
            spatial_token_lengths=spatial_lengths,
        )
