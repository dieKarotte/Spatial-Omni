"""Spatial token adapter scaffold built on top of SELD task-233 hidden states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from ..utils.spatial_seld_utils import build_1d_attention_mask, seld_frames_to_spatial_tokens
from .seld_backbone import SeldBackbone, SeldBackboneOutput
from .seld_feature_bridge import SeldFeatureBridge


@dataclass
class SeldSpatialAdapterOutput:
    """Output container for low-rate spatial tokens.

    Attributes:
        spatial_tokens:
            Low-rate spatial token sequence, shape `[B, T_spat_max, D_spat]`.
        spatial_token_attention_mask:
            Boolean token mask of shape `[B, T_spat_max]`.
        spatial_token_lengths:
            Valid spatial token counts, shape `[B]`.
        seld_hidden_states:
            Backbone hidden sequence before token downsampling, shape
            `[B, T_seld_max, D_seld]`.
        seld_hidden_attention_mask:
            Boolean SELD hidden mask, shape `[B, T_seld_max]`.
        seld_hidden_lengths:
            Valid SELD hidden lengths, shape `[B]`.
    """

    spatial_tokens: torch.FloatTensor
    spatial_token_attention_mask: torch.BoolTensor
    spatial_token_lengths: torch.LongTensor
    seld_hidden_states: torch.FloatTensor
    seld_hidden_attention_mask: torch.BoolTensor
    seld_hidden_lengths: torch.LongTensor


class SeldSpatialAdapter(nn.Module):
    """Convert SELD hidden states into low-rate spatial tokens.

    Input modes:
        1. Raw FOA waveform path:
            - `spatial_audio`: `[B, T_audio, 4]`
            - optional waveform mask/lengths
        2. Precomputed feature path:
            - `seld_features`: `[B, 7, T_feat_max, 64]`
            - optional feature mask/lengths
        3. Precomputed SELD hidden path:
            - `seld_hidden_states`: `[B, T_seld_max, D_seld]`
            - optional hidden mask/lengths

    Processing:
        `audio -> feature_bridge -> backbone -> token downsample -> token MLP`

    Output:
        [`SeldSpatialAdapterOutput`]
            - `spatial_tokens`: `[B, T_spat_max, D_spat]`
            - `spatial_token_attention_mask`: `[B, T_spat_max]`
            - `spatial_token_lengths`: `[B]`
    """

    def __init__(
        self,
        feature_bridge: Optional[SeldFeatureBridge] = None,
        backbone: Optional[SeldBackbone] = None,
        hidden_dim: int = 128,
        token_dim: int = 256,
        downsample_factor: int = 4,
        # Alias: input_dim is accepted as a synonym for hidden_dim
        input_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if input_dim is not None:
            hidden_dim = input_dim
        self.feature_bridge = feature_bridge
        self.backbone = backbone
        self.hidden_dim = int(hidden_dim)
        self.token_dim = int(token_dim)
        self.downsample_factor = int(downsample_factor)

        self.token_norm = nn.LayerNorm(self.hidden_dim)
        self.token_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.token_dim),
            nn.GELU(),
            nn.Linear(self.token_dim, self.token_dim),
        )

    def forward(
        self,
        spatial_audio: Optional[torch.Tensor] = None,
        spatial_audio_attention_mask: Optional[torch.Tensor] = None,
        spatial_audio_lengths: Optional[torch.LongTensor] = None,
        seld_features: Optional[torch.Tensor] = None,
        seld_feature_attention_mask: Optional[torch.Tensor] = None,
        seld_feature_lengths: Optional[torch.LongTensor] = None,
        seld_hidden_states: Optional[torch.Tensor] = None,
        seld_hidden_attention_mask: Optional[torch.Tensor] = None,
        seld_hidden_lengths: Optional[torch.LongTensor] = None,
    ) -> SeldSpatialAdapterOutput:
        """Build low-rate spatial tokens from one of the supported input modes.

        Returns:
            [`SeldSpatialAdapterOutput`].
        """
        backbone_output = self._resolve_backbone_output(
            spatial_audio=spatial_audio,
            spatial_audio_attention_mask=spatial_audio_attention_mask,
            spatial_audio_lengths=spatial_audio_lengths,
            seld_features=seld_features,
            seld_feature_attention_mask=seld_feature_attention_mask,
            seld_feature_lengths=seld_feature_lengths,
            seld_hidden_states=seld_hidden_states,
            seld_hidden_attention_mask=seld_hidden_attention_mask,
            seld_hidden_lengths=seld_hidden_lengths,
        )
        spatial_tokens, spatial_token_lengths = self._downsample_hidden_states(
            backbone_output.hidden_states,
            backbone_output.hidden_lengths,
        )
        target_dtype = self.token_norm.weight.dtype
        if spatial_tokens.dtype != target_dtype:
            spatial_tokens = spatial_tokens.to(dtype=target_dtype)
        spatial_tokens = self.token_head(self.token_norm(spatial_tokens))
        spatial_token_attention_mask = build_1d_attention_mask(
            spatial_token_lengths,
            max_length=spatial_tokens.shape[1],
        )
        return SeldSpatialAdapterOutput(
            spatial_tokens=spatial_tokens,
            spatial_token_attention_mask=spatial_token_attention_mask,
            spatial_token_lengths=spatial_token_lengths,
            seld_hidden_states=backbone_output.hidden_states,
            seld_hidden_attention_mask=backbone_output.hidden_attention_mask,
            seld_hidden_lengths=backbone_output.hidden_lengths,
        )

    def _resolve_backbone_output(
        self,
        spatial_audio: Optional[torch.Tensor],
        spatial_audio_attention_mask: Optional[torch.Tensor],
        spatial_audio_lengths: Optional[torch.LongTensor],
        seld_features: Optional[torch.Tensor],
        seld_feature_attention_mask: Optional[torch.Tensor],
        seld_feature_lengths: Optional[torch.LongTensor],
        seld_hidden_states: Optional[torch.Tensor],
        seld_hidden_attention_mask: Optional[torch.Tensor],
        seld_hidden_lengths: Optional[torch.LongTensor],
    ) -> SeldBackboneOutput:
        """Resolve the backbone output from raw audio, features, or hidden states."""

        if seld_hidden_states is not None:
            if seld_hidden_states.ndim != 3:
                raise ValueError(
                    "seld_hidden_states must have shape [B, T_seld_max, D_seld], "
                    f"got {tuple(seld_hidden_states.shape)}"
                )
            if seld_hidden_states.shape[-1] != self.hidden_dim:
                raise ValueError(
                    f"Expected hidden dim {self.hidden_dim}, got {seld_hidden_states.shape[-1]}"
                )
            if seld_hidden_lengths is None:
                if seld_hidden_attention_mask is None:
                    seld_hidden_lengths = seld_hidden_states.new_full(
                        (seld_hidden_states.shape[0],),
                        fill_value=seld_hidden_states.shape[1],
                        dtype=torch.long,
                    )
                else:
                    seld_hidden_lengths = seld_hidden_attention_mask.to(torch.long).sum(dim=1)
            if seld_hidden_attention_mask is None:
                seld_hidden_attention_mask = build_1d_attention_mask(
                    seld_hidden_lengths,
                    max_length=seld_hidden_states.shape[1],
                )
            return SeldBackboneOutput(
                hidden_states=seld_hidden_states,
                hidden_attention_mask=seld_hidden_attention_mask,
                hidden_lengths=seld_hidden_lengths,
            )

        if seld_features is not None:
            return self.backbone(
                seld_features=seld_features,
                seld_feature_attention_mask=seld_feature_attention_mask,
                seld_feature_lengths=seld_feature_lengths,
            )

        if spatial_audio is not None:
            feature_output = self.feature_bridge(
                spatial_audio=spatial_audio,
                spatial_audio_attention_mask=spatial_audio_attention_mask,
                spatial_audio_lengths=spatial_audio_lengths,
            )
            return self.backbone(
                seld_features=feature_output.features,
                seld_feature_attention_mask=feature_output.feature_attention_mask,
                seld_feature_lengths=feature_output.feature_lengths,
            )

        raise ValueError(
            "SeldSpatialAdapter requires one of: "
            "`spatial_audio`, `seld_features`, or `seld_hidden_states`."
        )

    def _downsample_hidden_states(
        self,
        hidden_states: torch.Tensor,
        hidden_lengths: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.LongTensor]:
        """Reduce `10 Hz` SELD hidden states to low-rate spatial tokens.

        Args:
            hidden_states:
                Tensor of shape `[B, T_seld_max, D_seld]`.
            hidden_lengths:
                Valid SELD hidden lengths, shape `[B]`.

        Returns:
            Tuple of:
            - pooled hidden states `[B, T_spat_max, D_seld]`
            - spatial token lengths `[B]`
        """

        if hidden_states.ndim != 3:
            raise ValueError(
                f"hidden_states must have shape [B, T_seld_max, D_seld], got {tuple(hidden_states.shape)}"
            )
        if hidden_lengths.ndim != 1:
            raise ValueError(f"hidden_lengths must be 1D, got {tuple(hidden_lengths.shape)}")

        batch_size, max_seld_steps, hidden_dim = hidden_states.shape
        if hidden_dim != self.hidden_dim:
            raise ValueError(f"Expected hidden dim {self.hidden_dim}, got {hidden_dim}")

        target_lengths = seld_frames_to_spatial_tokens(
            hidden_lengths,
            downsample_factor=self.downsample_factor,
        )
        padded_steps = int(target_lengths.max().item()) * self.downsample_factor

        if padded_steps > max_seld_steps:
            pad_size = padded_steps - max_seld_steps
            hidden_states = torch.cat(
                [
                    hidden_states,
                    hidden_states.new_zeros(batch_size, pad_size, hidden_dim),
                ],
                dim=1,
            )

        hidden_mask = build_1d_attention_mask(hidden_lengths, max_length=hidden_states.shape[1])
        hidden_states = hidden_states[:, :padded_steps, :]
        hidden_mask = hidden_mask[:, :padded_steps]

        grouped_states = hidden_states.reshape(
            batch_size,
            padded_steps // self.downsample_factor,
            self.downsample_factor,
            hidden_dim,
        )
        grouped_mask = hidden_mask.reshape(
            batch_size,
            padded_steps // self.downsample_factor,
            self.downsample_factor,
        ).to(dtype=hidden_states.dtype)
        valid_counts = grouped_mask.sum(dim=2).clamp_min(1.0).unsqueeze(-1)
        pooled_states = (grouped_states * grouped_mask.unsqueeze(-1)).sum(dim=2) / valid_counts
        return pooled_states, target_lengths
