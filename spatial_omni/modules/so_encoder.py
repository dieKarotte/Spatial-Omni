# """Wrapper around Spatial-BEATs encoder for integration into the Spatial-Omni pipeline.

# This module provides ``SOEncoder`` which loads a trained
# Spatial-BEATs checkpoint from the beats repo and exposes a simple
# ``forward()`` API that accepts FOA waveforms and returns spatial token
# embeddings ready for projection into the LLM hidden space.
# """

# from __future__ import annotations

# import os
# import sys
# from dataclasses import dataclass
# from typing import Optional

# import torch
# import torch.nn as nn


# @dataclass
# class SOEncoderOutput:
#     """Output container for the Spatial-BEATs encoder wrapper.

#     Attributes:
#         spatial_tokens:
#             ``[B, T_s_max, 768]`` encoder spatial embeddings.
#         spatial_token_lengths:
#             ``[B]`` valid temporal token counts per sample.
#     """

#     spatial_tokens: torch.FloatTensor
#     spatial_token_lengths: torch.LongTensor


# class SOEncoder(nn.Module):
#     """Loads a frozen Spatial-BEATs encoder and exposes a Spatial-Omni-compatible API.

#     Parameters:
#         checkpoint_path:
#             Path to the Spatial-BEATs ``.pt`` checkpoint file.
#         beats_repo_path:
#             Path to the beats repo root. If ``None``, defaults to
#             ``../../unilm/beats`` relative to this source file.
#         freeze_backbone:
#             If ``True``, freeze all parameters and keep the model in eval mode.
#         max_audio_seconds:
#             Maximum audio duration in seconds (used to clamp durations).
#     """

#     def __init__(
#         self,
#         checkpoint_path: str,
#         beats_repo_path: Optional[str] = None,
#         freeze_backbone: bool = True,
#         max_audio_seconds: float = 20.0,
#     ) -> None:
#         super().__init__()

#         # --- resolve beats repo path and import SOBackbone ---
#         if beats_repo_path is None:
#             beats_repo_path = os.path.normpath(
#                 os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "unilm", "beats")
#             )
#         if beats_repo_path not in sys.path:
#             sys.path.insert(0, beats_repo_path)

#         from so_backbone import SOBackbone, SOBackboneConfig  # type: ignore

#         # --- load checkpoint ---
#         checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

#         # Support two checkpoint formats:
#         # 1. Original BEATs-style: {"cfg": dict, "model": state_dict}
#         # 2. train_so_pretrain.py style: {"train_cfg": {"model": SOBackboneConfig, ...},
#         #                                   "model_state_dict": state_dict}
#         if "cfg" in checkpoint:
#             cfg = SOBackboneConfig(checkpoint["cfg"])
#             state_dict = checkpoint["model"]
#         elif "train_cfg" in checkpoint:
#             cfg = checkpoint["train_cfg"]["model"]
#             if not isinstance(cfg, SOBackboneConfig):
#                 cfg = SOBackboneConfig(cfg if isinstance(cfg, dict) else cfg.__dict__)
#             state_dict = checkpoint["model_state_dict"]
#         else:
#             raise KeyError(
#                 f"Checkpoint at {checkpoint_path} has unrecognized format. "
#                 f"Expected keys \'cfg\'+'model' or \'train_cfg\'+'model_state_dict', "
#                 f"got: {list(checkpoint.keys())}"
#             )

#         # self.model = SOBackbone(cfg)
#         with torch.device("cpu"):                                                                                                                            
#               self.model = SOBackbone(cfg)

#         # Load state dict with strict=False to handle head mismatches etc.
#         missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
#         if missing:
#             print(f"[SOEncoder] Missing keys: {missing}")
#         if unexpected:
#             print(f"[SOEncoder] Unexpected keys: {unexpected}")

#         # --- store useful config values ---
#         self.target_token_rate: float = float(cfg.target_token_rate)
#         self.sample_rate: int = int(cfg.sample_rate)
#         self.encoder_dim: int = int(cfg.encoder_embed_dim)
#         self.max_audio_seconds: float = float(max_audio_seconds)
#         self._freeze_backbone = freeze_backbone

#         # --- optionally freeze ---
#         if freeze_backbone:
#             for param in self.model.parameters():
#                 param.requires_grad = False
#             self.model.eval()

#     def train(self, mode: bool = True) -> "SOEncoder":
#         """Override train() to keep the model in eval when frozen."""
#         if self._freeze_backbone:
#             # Keep wrapper in training mode but model always in eval
#             super().train(mode)
#             self.model.eval()
#             return self
#         return super().train(mode)

#     def _compute_clip_durations(
#         self,
#         spatial_audio: torch.Tensor,
#         spatial_audio_lengths: Optional[torch.LongTensor] = None,
#     ) -> torch.FloatTensor:
#         """Compute per-sample durations in seconds.

#         Args:
#             spatial_audio:
#                 ``[B, 4, T]`` waveform tensor (channel-first).
#             spatial_audio_lengths:
#                 Optional ``[B]`` waveform sample counts.

#         Returns:
#             ``[B]`` float tensor of durations clamped to ``max_audio_seconds``.
#         """
#         B = spatial_audio.shape[0]
#         if spatial_audio_lengths is not None:
#             durations = spatial_audio_lengths.float() / float(self.sample_rate)
#         else:
#             T = spatial_audio.shape[-1]
#             durations = torch.full(
#                 (B,), float(T) / float(self.sample_rate),
#                 device=spatial_audio.device, dtype=torch.float32,
#             )
#         return durations.clamp(max=self.max_audio_seconds)

#     def forward(
#         self,
#         spatial_audio: torch.Tensor,
#         spatial_audio_attention_mask: Optional[torch.Tensor] = None,
#         spatial_audio_lengths: Optional[torch.LongTensor] = None,
#     ) -> SOEncoderOutput:
#         """Run the Spatial-BEATs encoder on FOA audio.

#         Args:
#             spatial_audio:
#                 ``[B, T_audio, 4]`` time-last FOA waveform batch.
#             spatial_audio_attention_mask:
#                 Optional ``[B, T_audio]`` mask (currently unused; kept for API
#                 compatibility with the SELD233 path).
#             spatial_audio_lengths:
#                 Optional ``[B]`` waveform sample counts per item.

#         Returns:
#             ``SOEncoderOutput`` with:
#             - ``spatial_tokens``: ``[B, T_s_max, 768]``
#             - ``spatial_token_lengths``: ``[B]``
#         """
#         # [B, T_audio, 4] -> [B, 4, T] for BEATs
#         waveform = spatial_audio.transpose(1, 2).contiguous()

#         clip_durations = self._compute_clip_durations(waveform, spatial_audio_lengths)

#         # Run the encoder
#         spatial_embeddings, temporal_padding_mask = self.model.extract_features(
#             waveform=waveform,
#             clip_duration_seconds=clip_durations,
#         )

#         # Compute valid lengths
#         target_num_steps = self.model.compute_target_num_steps(
#             waveform=waveform,
#             clip_duration_seconds=clip_durations,
#         )

#         return SOEncoderOutput(
#             spatial_tokens=spatial_embeddings,
#             spatial_token_lengths=target_num_steps,
#         )
"""Lazy-init wrapper around Spatial-BEATs encoder.

Why lazy init?
from_pretrained(low_cpu_mem_usage=True) runs __init__ inside
init_empty_weights(), which makes standard nn.init.* a no-op so layers
become meta-device placeholders.  BEATs' init_bert_params bypasses that
by calling data.copy_(data.cpu().normal_(...)) directly, crashing with
'NotImplementedError: Cannot copy out of meta tensor'.
Solution: defer SOBackbone() to _build_model(), called AFTER
from_pretrained() returns when no meta-device context is active.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


@dataclass
class SOEncoderOutput:
    spatial_tokens: torch.FloatTensor        # [B, T_s_max, D]
    spatial_token_lengths: torch.LongTensor  # [B]


class SOEncoder(nn.Module):

    def __init__(
        self,
        checkpoint_path: str,
        beats_repo_path: Optional[str] = None,
        freeze_backbone: bool = True,
        max_audio_seconds: float = 20.0,
        encoder_token_rate: float = 10.0,
    ) -> None:
        super().__init__()

        if beats_repo_path is None:
            beats_repo_path = os.path.normpath(
                os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "unilm", "beats")
            )
        self._beats_repo_path: str = beats_repo_path
        self._checkpoint_path: str = checkpoint_path
        self._freeze_backbone: bool = freeze_backbone
        self.max_audio_seconds: float = float(max_audio_seconds)

        # Encoder native token rate (set by caller; SO-Encoder = 10 Hz). The cfg's
        # target_token_rate field is the framework-level "goal" rate and does
        # not necessarily match the native fused-token rate a given ckpt emits,
        # so we don't read it here.
        self.encoder_token_rate: float = float(encoder_token_rate)
        self.sample_rate: int = 16000
        self.encoder_dim: int = 768

        self._beats_cfg = None
        self._beats_state_dict = None
        self.model = None  # type: ignore[assignment]

        self._preload_checkpoint()

    @staticmethod
    def _config_to_dict(cfg: Any) -> Dict[str, Any]:
        if cfg is None:
            return {}
        if isinstance(cfg, dict):
            return dict(cfg)
        if hasattr(cfg, "__dict__"):
            return {
                key: value
                for key, value in vars(cfg).items()
                if not key.startswith("_")
            }
        raise TypeError(f"Unsupported SOBackbone config type: {type(cfg)!r}")

    @classmethod
    def _materialize_config(cls, config_cls, raw_cfg: Any):
        # Start from the current upstream defaults, then overlay checkpoint
        # values so newly added config fields keep sane defaults.
        cfg = config_cls()
        cfg.update(cls._config_to_dict(raw_cfg))
        return cfg

    def _import_so_backbone(self):
        """Resolve SOBackbone / SOBackboneConfig from either the bundled
        ``spatial_omni.encoders.beats`` or a user-supplied external repo.

        Order of preference:
          1. Internal package ``spatial_omni.encoders.beats.so_backbone``.
          2. ``$SO_BEATS_REPO`` / ``--beats-repo``: a copy with a ``so_backbone``
             module (renamed open-source layout).
          3. Same external repo with the legacy ``spatial_beats`` module name
             (Spatial-BEATs upstream); we shim it back into our class names.
        """
        try:
            from spatial_omni.encoders.beats.so_backbone import (  # type: ignore
                SOBackbone,
                SOBackboneConfig,
            )
            return SOBackbone, SOBackboneConfig
        except Exception:
            pass

        if self._beats_repo_path and self._beats_repo_path not in sys.path:
            sys.path.insert(0, self._beats_repo_path)

        try:
            from so_backbone import SOBackbone, SOBackboneConfig  # type: ignore
            return SOBackbone, SOBackboneConfig
        except Exception:
            pass

        # Legacy external-repo layout (unilm/beats fork before rename).
        from spatial_beats import SpatialBEATs as SOBackbone  # type: ignore
        from spatial_beats import SpatialBEATsConfig as SOBackboneConfig  # type: ignore
        return SOBackbone, SOBackboneConfig

    def _install_legacy_pickle_shim(self):
        """Make pickled checkpoints that reference the legacy
        ``spatial_beats.SpatialBEATs(Config)`` symbols loadable against the
        renamed ``so_backbone.SOBackbone(Config)`` classes.
        """
        try:
            from spatial_omni.encoders.beats import so_backbone as _so_pkg  # type: ignore
        except Exception:
            return
        # Bind legacy class names onto our module so pickle's find_class works.
        if not hasattr(_so_pkg, "SpatialBEATs"):
            _so_pkg.SpatialBEATs = _so_pkg.SOBackbone  # type: ignore[attr-defined]
        if not hasattr(_so_pkg, "SpatialBEATsConfig"):
            _so_pkg.SpatialBEATsConfig = _so_pkg.SOBackboneConfig  # type: ignore[attr-defined]
        # Expose the package under the legacy name on sys.modules.
        sys.modules.setdefault("spatial_beats", _so_pkg)
        sys.modules.setdefault("so_backbone", _so_pkg)

    def _preload_checkpoint(self) -> None:
        _, SOBackboneConfig = self._import_so_backbone()
        self._install_legacy_pickle_shim()
        ckpt = torch.load(self._checkpoint_path, map_location="cpu", weights_only=False)
        if "cfg" in ckpt:
            raw_cfg = ckpt["cfg"]
            state_dict = ckpt["model"]
        elif "train_cfg" in ckpt:
            raw_cfg = ckpt["train_cfg"]["model"]
            state_dict = ckpt["model_state_dict"]
        else:
            raise KeyError(
                f"Unrecognised checkpoint format in {self._checkpoint_path}. "
                f"Got keys: {list(ckpt.keys())}"
            )
        cfg = self._materialize_config(SOBackboneConfig, raw_cfg)
        self._beats_cfg = cfg
        self._beats_state_dict = state_dict
        self.sample_rate = int(cfg.sample_rate)
        self.encoder_dim = int(cfg.encoder_embed_dim)
        print(
            f"[BEATsWrapper] Loaded ckpt: readout_scheme={getattr(cfg, 'readout_scheme', None)!r} "
            f"patch_adapter_version={getattr(cfg, 'patch_adapter_version', None)!r} "
            f"use_trunk_spatial_adapters={getattr(cfg, 'use_trunk_spatial_adapters', None)} "
            f"cfg.target_token_rate={getattr(cfg, 'target_token_rate', None)} "
            f"wrapper.encoder_token_rate={self.encoder_token_rate}"
        )

    def _build_model(self) -> None:
        """Build SOBackbone on CPU. Call AFTER from_pretrained() returns."""
        if self.model is not None:
            return
        SOBackbone, _ = self._import_so_backbone()
        beats_model = SOBackbone(self._beats_cfg)
        missing, unexpected = beats_model.load_state_dict(self._beats_state_dict, strict=False)
        if missing:
            print(f"[BEATsWrapper] Missing keys ({len(missing)}): {missing[:5]}")
        if unexpected:
            print(f"[BEATsWrapper] Unexpected keys ({len(unexpected)}): {unexpected[:5]}")
        if self._freeze_backbone:
            for p in beats_model.parameters():
                p.requires_grad = False
            beats_model.eval()
        self.model = beats_model
        self._beats_state_dict = None

    def train(self, mode: bool = True) -> "SOEncoder":
        super().train(mode)
        if self._freeze_backbone and self.model is not None:
            self.model.eval()
        return self

    def _compute_clip_durations(self, waveform, spatial_audio_lengths=None):
        B = waveform.shape[0]
        if spatial_audio_lengths is not None:
            d = spatial_audio_lengths.float() / float(self.sample_rate)
        else:
            T = waveform.shape[-1]
            d = torch.full((B,), float(T) / float(self.sample_rate),
                            device=waveform.device, dtype=torch.float32)
        return d.clamp(max=self.max_audio_seconds)

    def forward(self, spatial_audio, spatial_audio_attention_mask=None,
                spatial_audio_lengths=None):
        if self.model is None:
            raise RuntimeError(
                "model is None -- call _build_model() after from_pretrained()."
            )
        waveform = spatial_audio.transpose(1, 2).contiguous()  # [B,T,4]->[B,4,T]
        durations = self._compute_clip_durations(waveform, spatial_audio_lengths)
        embeddings, _ = self.model.extract_features(
            waveform=waveform, clip_duration_seconds=durations)
        steps = self.model.compute_target_num_steps(
            waveform=waveform, clip_duration_seconds=durations)
        return SOEncoderOutput(
            spatial_tokens=embeddings, spatial_token_lengths=steps)
