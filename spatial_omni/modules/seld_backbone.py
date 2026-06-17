"""SELDNet-233 backbone scaffold used by the independent spatial modality."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.util
import os
from typing import Optional

import torch
from torch import nn

from ..utils.spatial_seld_utils import (
    attention_mask_to_lengths,
    build_1d_attention_mask,
    clamp_lengths,
    feature_frames_to_seld_frames,
)


@dataclass
class SeldBackboneOutput:
    """Output container for the SELD backbone hidden sequence.

    Attributes:
        hidden_states:
            Final shared temporal representation after the last MHSA block,
            shape `[B, T_seld_max, 128]`.
        hidden_attention_mask:
            Boolean mask of shape `[B, T_seld_max]`.
        hidden_lengths:
            Valid hidden lengths in SELD time steps, shape `[B]`.
    """

    hidden_states: torch.FloatTensor
    hidden_attention_mask: torch.BoolTensor
    hidden_lengths: torch.LongTensor


class SeldBackbone(nn.Module):
    """Wrap the task-233 SELDNet encoder and expose the MHSA hidden sequence.

    Input:
        `seld_features`:
            Tensor of shape `[B, 7, T_feat_max, 64]`.
        `seld_feature_attention_mask`:
            Optional feature-frame mask of shape `[B, T_feat_max]`.
        `seld_feature_lengths`:
            Optional valid feature-frame lengths, shape `[B]`.

    Processing:
        1. Validate the feature tensor shape expected by task `233`.
        2. Convert feature lengths into expected `10 Hz` SELD hidden lengths.
        3. Delegate model loading, checkpoint restore, and MHSA extraction to
           explicit scaffold hooks.

    Output:
        [`SeldBackboneOutput`]
            - `hidden_states`: `[B, T_seld_max, 128]`
            - `hidden_attention_mask`: `[B, T_seld_max]`
            - `hidden_lengths`: `[B]`
    """

    def __init__(
        self,
        baseline_repo_path: str,
        checkpoint_path: str,
        task_id: str = "233",
        num_feature_channels: int = 7,
        num_mel_bins: int = 64,
        hidden_dim: int = 128,
        feature_to_seld_ratio: int = 5,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.baseline_repo_path = baseline_repo_path
        self.checkpoint_path = checkpoint_path
        self.task_id = str(task_id)
        self.num_feature_channels = int(num_feature_channels)
        self.num_mel_bins = int(num_mel_bins)
        self.hidden_dim = int(hidden_dim)
        self.feature_to_seld_ratio = int(feature_to_seld_ratio)
        self.freeze_backbone = bool(freeze_backbone)
        self._baseline_model = None
        self._baseline_params = None
        self._captured_hidden = None
        self._hook_handle = None

    def forward(
        self,
        seld_features: torch.Tensor,
        seld_feature_attention_mask: Optional[torch.Tensor] = None,
        seld_feature_lengths: Optional[torch.LongTensor] = None,
    ) -> SeldBackboneOutput:
        """Run the SELD backbone on baseline-compatible features.

        Args:
            seld_features:
                Tensor of shape `[B, 7, T_feat_max, 64]`.
            seld_feature_attention_mask:
                Optional mask of shape `[B, T_feat_max]`.
            seld_feature_lengths:
                Optional valid feature lengths, shape `[B]`.

        Returns:
            [`SeldBackboneOutput`].

        Raises:
            NotImplementedError:
                Always, until `_run_seldnet_backbone` is implemented.
        """

        if seld_features.ndim != 4:
            raise ValueError(
                "seld_features must have shape [B, 7, T_feat_max, 64], "
                f"got {tuple(seld_features.shape)}"
            )
        if seld_features.shape[1] != self.num_feature_channels:
            raise ValueError(
                f"Expected {self.num_feature_channels} feature channels, got {seld_features.shape[1]}"
            )
        if seld_features.shape[-1] != self.num_mel_bins:
            raise ValueError(
                f"Expected {self.num_mel_bins} mel bins, got {seld_features.shape[-1]}"
            )

        mask_lengths = attention_mask_to_lengths(
            seld_feature_attention_mask,
            max_length=seld_features.shape[2],
        )
        if seld_feature_lengths is None:
            if mask_lengths is None:
                seld_feature_lengths = seld_features.new_full(
                    (seld_features.shape[0],),
                    fill_value=seld_features.shape[2],
                    dtype=torch.long,
                )
            else:
                seld_feature_lengths = mask_lengths
        elif mask_lengths is not None and not torch.equal(seld_feature_lengths.cpu(), mask_lengths.cpu()):
            raise ValueError(
                "seld_feature_lengths and seld_feature_attention_mask disagree on valid frame counts"
            )

        seld_feature_lengths = clamp_lengths(
            seld_feature_lengths.to(device=seld_features.device, dtype=torch.long),
            max_length=seld_features.shape[2],
        )
        hidden_lengths = feature_frames_to_seld_frames(
            seld_feature_lengths,
            feature_to_seld_ratio=self.feature_to_seld_ratio,
        )
        hidden_attention_mask = build_1d_attention_mask(
            hidden_lengths,
            max_length=max(1, seld_features.shape[2] // self.feature_to_seld_ratio),
        )
        return self._run_seldnet_backbone(
            seld_features=seld_features,
            seld_feature_lengths=seld_feature_lengths,
            hidden_lengths=hidden_lengths,
            hidden_attention_mask=hidden_attention_mask,
        )

    def _run_seldnet_backbone(
        self,
        seld_features: torch.Tensor,
        seld_feature_lengths: torch.LongTensor,
        hidden_lengths: torch.LongTensor,
        hidden_attention_mask: torch.BoolTensor,
    ) -> SeldBackboneOutput:
        """Execute task-233 SELDNet and capture the last MHSA hidden sequence.

        Args:
            seld_features:
                Feature tensor `[B, 7, T_feat_max, 64]`.
            seld_feature_lengths:
                Valid feature lengths `[B]`.
            hidden_lengths:
                Expected hidden sequence lengths `[B]`.
            hidden_attention_mask:
                Boolean mask `[B, T_seld_max]`.

        Returns:
            [`SeldBackboneOutput`].

        Notes:
            Dynamically loads baseline task `233`, restores the configured
            checkpoint, and captures the tensor after the final MHSA residual +
            LayerNorm.
        """

        model = self._get_or_create_model(device=seld_features.device)
        self._captured_hidden = None
        if self.freeze_backbone:
            with torch.no_grad():
                _ = model(seld_features.to(device=seld_features.device, dtype=torch.float32))
        else:
            _ = model(seld_features.to(device=seld_features.device, dtype=torch.float32))

        if self._captured_hidden is None:
            raise RuntimeError("Failed to capture SELD hidden states from the final MHSA LayerNorm.")

        hidden_states = self._captured_hidden
        max_hidden_steps = hidden_attention_mask.shape[1]
        if hidden_states.shape[1] < max_hidden_steps:
            pad_steps = max_hidden_steps - hidden_states.shape[1]
            hidden_states = torch.cat(
                (
                    hidden_states,
                    hidden_states.new_zeros(hidden_states.shape[0], pad_steps, hidden_states.shape[2]),
                ),
                dim=1,
            )
        elif hidden_states.shape[1] > max_hidden_steps:
            hidden_states = hidden_states[:, :max_hidden_steps, :]

        hidden_states = hidden_states * hidden_attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        return SeldBackboneOutput(
            hidden_states=hidden_states,
            hidden_attention_mask=hidden_attention_mask,
            hidden_lengths=hidden_lengths,
        )

    def _get_or_create_model(self, device: torch.device) -> nn.Module:
        if self._baseline_model is None:
            params_module = self._load_baseline_module("parameters.py", "seld_parameters_for_backbone")
            model_module = self._load_baseline_module("seldnet_model.py", "seld_model_for_backbone")

            params = params_module.get_params(self.task_id)
            feat_shape = (
                1,
                self.num_feature_channels,
                int(params["feature_sequence_length"]),
                self.num_mel_bins,
            )
            if params["multi_accdoa"]:
                output_width = int(params["unique_classes"]) * 3 * 4
            else:
                output_width = int(params["unique_classes"]) * 4
            out_shape = (1, int(params["label_sequence_length"]), output_width)

            model = model_module.SeldModel(feat_shape, out_shape, params)
            checkpoint_state = self._load_checkpoint_state(self._resolve_checkpoint_path())
            model_state = model.state_dict()
            compatible_state = {
                key: value
                for key, value in checkpoint_state.items()
                if key in model_state and model_state[key].shape == value.shape
            }
            model.load_state_dict(compatible_state, strict=False)

            if self.freeze_backbone:
                for parameter in model.parameters():
                    parameter.requires_grad = False
                model.eval()

            self._hook_handle = model.layer_norm_list[-1].register_forward_hook(self._capture_hidden_hook)
            self._baseline_model = model
            self._baseline_params = params

        if next(self._baseline_model.parameters()).device != device:
            self._baseline_model = self._baseline_model.to(device=device)
        if self.freeze_backbone:
            self._baseline_model.eval()
        return self._baseline_model

    def _load_baseline_module(self, file_name: str, module_name: str):
        # First try the vendored copy bundled under spatial_omni.encoders.seldnet
        # (zero external deps — works without --baseline-repo-path / DCASE_BASELINE_REPO).
        module_basename = os.path.splitext(file_name)[0]
        try:
            return importlib.import_module(
                f"spatial_omni.encoders.seldnet.{module_basename}"
            )
        except ImportError:
            pass
        # Fall back to loading from an external SELD baseline checkout
        # (legacy path; required pre-vendoring). Loaded directly from the file
        # path so it never mutates sys.path.
        if not self.baseline_repo_path:
            raise ImportError(
                f"Could not import vendored seldnet.{module_basename} and no "
                f"external baseline_repo_path is set to load {file_name}."
            )
        module_path = os.path.join(self.baseline_repo_path, file_name)
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load baseline module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _resolve_checkpoint_path(self) -> str:
        candidate = self.checkpoint_path
        if not os.path.isabs(candidate):
            candidate = os.path.join(self.baseline_repo_path, candidate)
        if not os.path.exists(candidate):
            raise FileNotFoundError(f"SELD checkpoint not found: {candidate}")
        return candidate

    @staticmethod
    def _load_checkpoint_state(checkpoint_path: str) -> dict:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict):
            for key in ("model_state_dict", "state_dict", "model"):
                value = checkpoint.get(key)
                if isinstance(value, dict):
                    return value
        if not isinstance(checkpoint, dict):
            raise TypeError(f"Unsupported checkpoint format at {checkpoint_path}: {type(checkpoint)}")
        return checkpoint

    def _capture_hidden_hook(self, module, inputs, output) -> None:
        self._captured_hidden = output
