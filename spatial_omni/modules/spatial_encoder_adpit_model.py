from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class SpatialEncoderADPITModel(nn.Module):
    def __init__(
        self,
        spatial_feature_extractor: nn.Module,
        spatial_encoder: nn.Module,
        hidden_dim: int,
        num_classes: int,
        num_tracks: int = 3,
        head_hidden_dim: int = 256,
        predict_distance: bool = True,
    ) -> None:
        super().__init__()
        self.spatial_feature_extractor = spatial_feature_extractor
        self.spatial_encoder = spatial_encoder
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.num_tracks = int(num_tracks)
        self.head_hidden_dim = int(head_hidden_dim)
        self.predict_distance = bool(predict_distance)

        self.sed_head = self._build_head(self.num_classes * self.num_tracks)
        self.accdoa_head = self._build_head(self.num_classes * self.num_tracks * 3)
        self.dist_head = self._build_head(self.num_classes * self.num_tracks) if self.predict_distance else None

    def _build_head(self, output_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(self.hidden_dim, self.head_hidden_dim),
            nn.GELU(),
            nn.Linear(self.head_hidden_dim, int(output_dim)),
        )

    def forward(self, spatial_audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        spatial_features = self.spatial_feature_extractor(spatial_audio)
        if spatial_features.dim() != 4:
            raise ValueError(f"Expected 4D spatial features, got shape {tuple(spatial_features.shape)}")

        # Feature extractor returns (B, T, C_spat, F_band). Spatial encoder expects (B, C, F, T).
        spatial_features = spatial_features.permute(0, 2, 3, 1).contiguous()
        h = self.spatial_encoder(spatial_features)
        if h.dim() != 3:
            raise ValueError(f"Expected encoder output (B, T, D), got shape {tuple(h.shape)}")

        batch_size, time_steps, _ = h.shape
        sed_logits = self.sed_head(h).view(batch_size, time_steps, self.num_classes, self.num_tracks)
        v_hat = self.accdoa_head(h).view(batch_size, time_steps, self.num_classes, self.num_tracks, 3)

        outputs: Dict[str, torch.Tensor] = {
            "h": h,
            "sed_logits": sed_logits,
            "V_hat": v_hat,
        }
        if self.dist_head is not None:
            outputs["logr_hat"] = self.dist_head(h).view(batch_size, time_steps, self.num_classes, self.num_tracks)
        else:
            outputs["logr_hat"] = h.new_zeros(batch_size, time_steps, self.num_classes, self.num_tracks)
        return outputs
