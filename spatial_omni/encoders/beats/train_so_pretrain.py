"""Training skeleton for the simplified Spatial-BEATs pipeline.

This file defines the stage-1 encoder-only training interfaces and the expected
hand-off between dataset, model, and loss modules. Actual optimization and
training logic is intentionally left unimplemented.
"""

import argparse
import contextlib
from dataclasses import asdict, dataclass, field
import copy
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW, Optimizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from .so_backbone import LOCAL_SPATIAL_FRAME_SCHEMES, SOBackbone, SOBackboneConfig, SOBackboneOutput
from .so_dataset import (
    SpatialBatch,
    SpatialDataset,
    SpatialDatasetConfig,
    collate_spatial_batch,
    load_source_vocabulary,
)
from .so_loss import (
    accumulate_frame_track_seld,
    accumulate_mono_ast_seld,
    build_frame_accdoa_validation_examples,
    build_frame_slot_validation_examples,
    build_frame_track_validation_examples,
    build_mono_ast_validation_examples,
    build_pretrunk_ast_validation_examples,
    build_primary_source_window_mask,
    collect_frame_track_csv_rows,
    build_validation_examples,
    OfficialDCASEMetricsAccumulator,
    SELDMetricsAccumulator,
    SpatialLossConfig,
    SpatialLossOutput,
    build_framewise_validation_examples,
    compute_frame_accdoa_losses,
    compute_frame_accdoa_validation_metrics,
    compute_frame_slot_losses,
    compute_frame_slot_validation_metrics,
    compute_frame_track_losses,
    compute_frame_track_validation_metrics,
    compute_framewise_losses,
    compute_framewise_validation_metrics,
    compute_mono_ast_losses,
    compute_mono_ast_validation_metrics,
    compute_pretrunk_ast_losses,
    compute_pretrunk_ast_validation_metrics,
    compute_spatial_validation_metrics,
    compute_spatial_losses,
    match_fixed_slots,
)

FRAME_SUPERVISION_MODES: Tuple[str, ...] = (
    "local_spatial_slot",
    "local_spatial_track",
    "local_spatial_accdoa",
)


DEFAULT_OV1_MANIFEST = ""
DEFAULT_OV2_MANIFEST = ""
DEFAULT_OV3_MANIFEST = ""


def _is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def _get_rank() -> int:
    return dist.get_rank() if _is_dist_initialized() else 0


def _get_world_size() -> int:
    return dist.get_world_size() if _is_dist_initialized() else 1


def _is_main_process() -> bool:
    return _get_rank() == 0


def _log(message: str) -> None:
    if _is_main_process():
        tqdm.write(message)


def _format_metrics(metrics: Dict[str, float], supervision_mode: str) -> str:
    """Format epoch metrics into a compact, human-readable string.

    Only shows metrics that are meaningful for the given supervision_mode.
    Zero-valued fields that are irrelevant (e.g. activity metrics in mono_ast)
    are suppressed to reduce noise.
    """
    is_mono = supervision_mode in ("mono_ast", "pretrunk_ast")

    # Always show these
    parts = [f"loss={metrics.get('loss_total', 0):.4f}"]

    if is_mono:
        # mono_ast: show individual loss components that can be nonzero
        cls_l = metrics.get("loss_cls_aux", 0)
        dir_l = metrics.get("loss_direction", 0)
        dist_l = metrics.get("loss_dist", 0)
        sem_l = metrics.get("loss_temp", 0)   # semantic anchor reuses loss_temp slot
        parts.append(f"cls_loss={cls_l:.4f}")
        parts.append(f"dir_loss={dir_l:.4f}")
        parts.append(f"dist_loss={dist_l:.4f}")
        if sem_l > 1e-6:
            parts.append(f"anchor_loss={sem_l:.4f}")
    else:
        # slot / track / accdoa: show frame-level losses
        for key in ("loss_activity", "loss_cls_aux", "loss_direction", "loss_dist"):
            v = metrics.get(key, 0)
            if abs(v) > 1e-8:
                parts.append(f"{key.replace('loss_', '')}={v:.4f}")
        if abs(metrics.get("loss_temp", 0)) > 1e-8:
            parts.append(f"aux={metrics.get('loss_temp', 0):.4f}")

    # Evaluation metrics
    is_frame_track = supervision_mode == "local_spatial_track"

    if is_frame_track:
        # Activity progress-bar proxy (no threshold, purely for tqdm readability):
        # activity_precision ≈ mean prob on (b, t) frames that HAVE GT source(s)
        #                     for the top-num_active_gt predicted tracks.
        # activity_recall    ≈ mean prob on "supposed-inactive" (b, k, t) cells.
        # activity_acc       = separation between the two.
        act_active = metrics.get("activity_precision", 0)
        act_inactive = metrics.get("activity_recall", 0)
        act_sep = metrics.get("activity_acc", 0)
        parts.append(f"act↑={act_active:.3f}")
        parts.append(f"act↓={act_inactive:.3f}")
        parts.append(f"sep={act_sep:.3f}")
        # Tier-1 (activity-gated, training-matcher) per-frame metrics — same
        # semantics as valid-CSV cls_ok / pred_{azi,ele,dist}, so train vs
        # valid can be read off directly without switching columns.
        parts.append(f"cls={metrics.get('class_acc', 0):.3f}")
        parts.append(f"azi={metrics.get('azi_mae_deg', 0):.1f}°")
        parts.append(f"ele={metrics.get('ele_mae_deg', 0):.1f}°")
        parts.append(f"dist={metrics.get('dist_mae', 0):.2f}m")
        # Tier-2 oracle (upper bound ignoring activity head) for diagnostics.
        parts.append(f"ocls={metrics.get('oracle_class_acc', 0):.3f}")
        parts.append(f"oazi={metrics.get('oracle_azi_mae_deg', 0):.1f}°")
        parts.append(f"oele={metrics.get('oracle_ele_mae_deg', 0):.1f}°")
    else:
        parts.append(f"cls={metrics.get('class_acc', 0):.3f}")
        parts.append(f"azi={metrics.get('azi_mae_deg', 0):.2f}°")
        parts.append(f"ele={metrics.get('ele_mae_deg', 0):.2f}°")
        parts.append(f"dist={metrics.get('dist_mae', 0):.3f}m")
        if not is_mono:
            act_f1 = metrics.get("activity_f1", 0)
            if act_f1 > 1e-6:
                parts.append(f"act_f1={act_f1:.3f}")

    # DCASE SELD metrics.
    if "F20" in metrics:
        parts.append(
            f"| ER20={metrics['ER20']:.3f}"
            f" F20={metrics['F20']:.3f}"
            f" LE_CD={metrics['LE_CD']:.1f}°"
            f" LR_CD={metrics['LR_CD']:.3f}"
            f" SELD={metrics['SELD_score']:.3f}"
        )
    elif "seld_score" in metrics:
        parts.append(
            f"| ER={metrics['seld_er']:.3f}"
            f" F={metrics['seld_f1']:.3f}"
            f" LE={metrics['seld_le']:.1f}°"
            f" LR={metrics['seld_lr']:.3f}"
            f" SELD={metrics['seld_score']:.3f}"
        )

    return "  ".join(parts)


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


class EMAModel:
    """[D-6] Exponential moving average of model parameters.

    Maintains a shadow copy of trainable parameters and updates it after each
    optimizer step:
        shadow = decay * shadow + (1 - decay) * current
    At validation / checkpoint time, swap the model's parameters with the
    shadow copy (and restore afterwards for training to continue).

    Notes:
        - Only tracks parameters with requires_grad=True.
        - Uses Adam/SGD-style decay (constant). Typical values: 0.999, 0.9995,
          0.9999. Larger decay = more smoothing = more lag.
        - DDP-safe: works on the un-wrapped module; caller must sync across
          ranks by broadcasting shadow state if needed (typically not done
          because all ranks see identical gradients).
        - Zero additional memory: ~= 1 extra copy of the model on CPU or GPU.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9995) -> None:
        self.decay = float(decay)
        self.shadow: Dict[str, Tensor] = {}
        for name, p in _unwrap_model(model).named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update shadow params from current model weights.

        Call after every optimizer.step().
        """
        unwrapped = _unwrap_model(model)
        for name, p in unwrapped.named_parameters():
            if name in self.shadow:
                # shadow := decay * shadow + (1 - decay) * p
                self.shadow[name].mul_(self.decay).add_(
                    p.detach(), alpha=1.0 - self.decay
                )

    @torch.no_grad()
    def apply_to(self, model: nn.Module) -> Dict[str, Tensor]:
        """Swap model's params with the EMA shadow.

        Returns a backup dict so you can call ``restore(model, backup)`` to
        put training weights back afterwards.
        """
        unwrapped = _unwrap_model(model)
        backup: Dict[str, Tensor] = {}
        for name, p in unwrapped.named_parameters():
            if name in self.shadow:
                backup[name] = p.data.clone()
                p.data.copy_(self.shadow[name])
        return backup

    @torch.no_grad()
    def restore(self, model: nn.Module, backup: Dict[str, Tensor]) -> None:
        """Restore model's training weights from the backup dict."""
        unwrapped = _unwrap_model(model)
        for name, p in unwrapped.named_parameters():
            if name in backup:
                p.data.copy_(backup[name])

    def state_dict(self) -> Dict[str, Tensor]:
        """Return a serialisable state dict for checkpointing."""
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state: Dict[str, Tensor]) -> None:
        """Load a previously saved shadow dict."""
        for name in self.shadow.keys():
            if name in state:
                self.shadow[name].copy_(state[name])


@dataclass
class TrainSOBackboneConfig:
    """High-level training configuration for Spatial-BEATs stage 1.

    Stage 1 goal:
        Train the FOA front-end, BEATs trunk adaptation, temporal readout,
        fixed-slot heads, and optionally only later the LLM projector.

    Qwen-like mel front-end alignment:
        These settings should be copied into SOBackboneConfig and
        SpatialDatasetConfig so the acoustic front-end remains consistent:
            - sample_rate = 16000
            - num_mel_bins = 128
            - n_fft = 400
            - win_length = 400
            - hop_length = 160
            - dither = 0.0
    """

    train_manifest_path: str = ""
    val_manifest_path: Optional[str] = None
    test_manifest_path: Optional[str] = None
    train_manifest_paths: Tuple[str, ...] = ()
    val_manifest_paths: Tuple[str, ...] = ()
    test_manifest_paths: Tuple[str, ...] = ()
    # Per-manifest replication factors (parallel to train_manifest_paths).
    # When provided, each manifest's SpatialDataset is wrapped with
    # torch.utils.data.ConcatDataset so that manifest i is repeated
    # train_manifest_replication[i] times per epoch.  DistributedSampler /
    # shuffle work as before.  Default empty = no replication (preserves
    # existing behavior).  Example: (1, 3, 3) for ov1:ov2:ov3 = 1:3:3.
    train_manifest_replication: Tuple[int, ...] = ()
    # Hungarian class-cost warmup (frame-track supervision only).
    # Epochs < frame_match_class_cost_warmup_epochs: class cost disabled
    # (weight=0). Then linearly ramps up to frame_match_class_cost_max_weight
    # over frame_match_class_cost_ramp_epochs. Set warmup=0 to disable.
    frame_match_class_cost_warmup_epochs: int = 0
    frame_match_class_cost_ramp_epochs: int = 3
    frame_match_class_cost_max_weight: float = 1.0

    # Two-stage loss schedule for frame-track supervision.
    # Stage 1 (epoch < frame_spatial_loss_warmup_epochs): lambda_dir and
    #   lambda_dist are scaled to frame_spatial_loss_warmup_scale (e.g. 0.0 or
    #   0.1) to let the class head learn on clean signal first.
    # Stage 2:
    #   - if frame_spatial_loss_ramp_epochs == 0, full lambda values from the
    #     loss config are restored immediately at epoch == warmup_epochs.
    #   - otherwise the dir/dist lambdas and matching weights linearly ramp
    #     from frame_spatial_loss_warmup_scale to 1.0 over
    #     frame_spatial_loss_ramp_epochs epochs.
    # Set frame_spatial_loss_warmup_epochs=0 to disable (default).
    frame_spatial_loss_warmup_epochs: int = 0
    frame_spatial_loss_warmup_scale: float = 0.0  # 0.0 = fully off in stage 1
    frame_spatial_loss_ramp_epochs: int = 0
    pretrained_beats_ckpt: str = "pretrain_ckpt/BEATs_iter3_plus_AS2M.pt/BEATs_iter3_plus_AS2M.pt"
    class_finetuned_ckpt: str = ""
    # Optional path to a prior SOBackbone checkpoint (e.g. the ov1
    # local_spatial best.pt). Used by the ov123 frame-level presets to warm
    # start local_spatial_encoder/fusion/aux-head weights.
    init_from_spatial_ckpt: str = ""
    # Optional path to a BEATs-trunk-only fine-tune checkpoint produced by
    # ``train_beats_multilabel_trunk.py``. The checkpoint is expected to
    # contain a ``beats_only`` key whose state-dict has the ``beats.``
    # prefix already stripped (i.e. keys look like ``encoder.layers.0...``).
    # When set, it overrides the AS2M trunk AFTER ``load_beats_pretrained``
    # runs.  This is the v13_F hot-start route: multi-label trunk → spatial.
    trunk_finetuned_ckpt: str = ""

    batch_size: int = 32
    num_workers: int = 4
    num_epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 0.05

    # Mixed precision: "fp32" (default, no autocast), "bf16", or "fp16".
    # bf16 keeps parameters in fp32; only forward activations are cast.
    amp_dtype: str = "fp32"

    # Layer-wise LR decay for the BEATs trunk.
    # trunk_lr_scale: multiplier applied to all trunk layers (encoder.*,
    #   layer_norm, post_extract_proj). Default 1.0 = same LR as heads.
    # spatial_lr_scale: multiplier applied to the local_spatial_* / preprocessor
    #   / spatial_patch_adapter parameters.  Default 1.0.
    # When both are 1.0 the optimizer behaves exactly as before (single group).
    trunk_lr_scale: float = 1.0
    spatial_lr_scale: float = 1.0
    # local_spatial_lr_scale: multiplier for the from-scratch
    # ``local_spatial_*`` modules (LocalSpatialEncoder CNN/transformer,
    # resampler, projection, fusion).  These are NOT BEATs-adjacent —
    # they're trained from scratch and historically were lumped under
    # ``spatial_lr_scale=0.3`` together with BEATs preprocessor adapters,
    # which kept their absolute LR below the head LR even though the
    # heads are also from scratch.  Setting this >0 splits the group and
    # gives them an independent multiplier.  ``None`` (default) preserves
    # the legacy behaviour of inheriting ``spatial_lr_scale``.
    local_spatial_lr_scale: Optional[float] = None
    # v9: isolated LR multiplier for the class_head inside
    # frame_track_prediction_heads. When < 1.0 the class head is put in its
    # own param group with lr = base_lr * class_head_lr_scale.  Used during
    # DOA ramp (stage 2) to prevent class binding from being perturbed by
    # the newly-unlocked dir/dist gradients.  1.0 = legacy behaviour.
    class_head_lr_scale: float = 1.0
    # Optional epoch-range override that further scales the class head LR
    # specifically during the DOA ramp.  When set, between
    # frame_spatial_loss_warmup_epochs and frame_spatial_loss_warmup_epochs
    # + class_head_freeze_during_ramp_epochs the class head LR is set to
    # class_head_lr_scale_during_ramp (defaults to 0.0 = frozen).  After the
    # ramp window the LR returns to class_head_lr_scale.
    class_head_freeze_during_ramp_epochs: int = 0
    class_head_lr_scale_during_ramp: float = 0.0

    # v10: phase-1 freezes the spatial prediction sub-heads (direction_head,
    # distance_head) on FrameTrackPredictionHeads so only activity + class +
    # num_active train while the backbone adapts to the v10 class-focused
    # objective.  Purely phase-1 plumbing — default False keeps the old
    # behaviour for every other preset.
    freeze_frame_track_spatial_heads: bool = False

    train_projector_in_stage1: bool = False
    unfreeze_full_trunk: bool = True
    freeze_trunk_in_stage1: bool = False
    # Number of top transformer layers to unfreeze (0 = use legacy logic).
    # When > 0, layers [12 - N .. 11] + layer_norm + post_extract_proj are
    # unfrozen.  Takes precedence over unfreeze_full_trunk when non-zero.
    unfreeze_top_n_layers: int = 0
    train_patch_embedding_in_stage1: bool = True
    train_spatial_adapter_in_stage1: bool = True
    freeze_projector_by_default: bool = True
    # Freeze the local_spatial_encoder/proj during classwarmup so
    # local_update ≈ 0 and class head sees near-pure semantic tokens.
    # Only effective when readout_scheme='local_spatial'.
    freeze_local_spatial_in_classwarmup: bool = False
    train_splits: Tuple[str, ...] = ("train",)
    val_splits: Tuple[str, ...] = ("valid",)
    test_splits: Tuple[str, ...] = ("test",)
    output_dir: str = "checkpoints/so_backbone_stage1"
    save_every_n_epochs: int = 1
    save_last_checkpoint: bool = True
    save_best_checkpoint: bool = True
    best_metric_name: str = "loss_total"
    minimize_best_metric: bool = True
    resume_from_checkpoint: Optional[str] = None
    save_optimizer_state: bool = True
    load_optimizer_state_on_resume: bool = True
    reset_epoch_on_resume: bool = False
    reset_best_metric_on_resume: bool = False
    show_progress_bars: bool = True
    dump_val_predictions: bool = True
    num_val_prediction_examples: int = 16
    dump_frame_track_csv: bool = False
    frame_track_csv_max_samples_per_epoch: int = 32
    frame_track_csv_max_samples_per_group: int = 0
    distributed: bool = False
    local_rank: int = 0
    distributed_backend: str = "nccl"
    ddp_find_unused_parameters: bool = False

    # === v13_D [D-6]: Exponential Moving Average of model weights ==========
    # When use_ema=True, a shadow copy of trainable weights is maintained
    # with decay ``ema_decay``. Validation / best.pt save uses the EMA
    # weights; training continues with the live weights. ema_start_epoch
    # lets us skip the warmup-noise phase.
    use_ema: bool = False
    ema_decay: float = 0.9995
    ema_start_epoch: int = 3

    # === v13_D [D-1]: Cosine LR schedule ===================================
    # When use_cosine_lr=True, LR follows:
    #   - linear warmup from 0 → peak over first ``cosine_lr_warmup_epochs`` eps
    #   - cosine decay from peak → peak * cosine_lr_min_ratio over remaining
    # Default (use_cosine_lr=False) preserves existing constant-LR behaviour.
    use_cosine_lr: bool = False
    cosine_lr_warmup_epochs: int = 3
    cosine_lr_min_ratio: float = 0.05

    model: SOBackboneConfig = field(default_factory=SOBackboneConfig)
    dataset: SpatialDatasetConfig = field(default_factory=SpatialDatasetConfig)
    loss: SpatialLossConfig = field(default_factory=SpatialLossConfig)


def make_ov123_stage1_config(
    ov1_manifest_path: str = DEFAULT_OV1_MANIFEST,
    ov2_manifest_path: str = DEFAULT_OV2_MANIFEST,
    ov3_manifest_path: str = DEFAULT_OV3_MANIFEST,
) -> TrainSOBackboneConfig:
    """Build the default stage-1 training config for ov1+ov2+ov3 FOA manifests.

    Design choices:
        - use train/valid/test split filtering from each manifest
        - cap every clip to at most 20 seconds
        - use deterministic start truncation so train/val/test share the same
          sequence policy for mixed-length clips
        - keep projector frozen in stage 1 and focus training on the encoder

    Returns:
        TrainSOBackboneConfig:
            Ready-to-run config object for stage-1 Spatial-BEATs training.
    """
    cfg = TrainSOBackboneConfig(
        train_manifest_paths=(ov1_manifest_path, ov2_manifest_path, ov3_manifest_path),
        val_manifest_paths=(ov1_manifest_path, ov2_manifest_path, ov3_manifest_path),
        test_manifest_paths=(ov1_manifest_path, ov2_manifest_path, ov3_manifest_path),
        train_splits=("train",),
        val_splits=("valid",),
        test_splits=("test",),
        batch_size=32,
        num_workers=4,
        num_epochs=20,
        learning_rate=1e-4,
        weight_decay=0.05,
        train_projector_in_stage1=False,
        unfreeze_full_trunk=True,
        freeze_projector_by_default=True,
        output_dir="checkpoints/so_backbone_ov123_stage1",
    )
    cfg.dataset.max_clip_duration_seconds = 20.0
    cfg.dataset.crop_mode = "start"
    cfg.loss.lambda_activity = 20.0
    cfg.loss.lambda_azi = 0.75
    cfg.loss.lambda_ele = 0.75
    cfg.loss.lambda_dist = 0.75
    cfg.loss.lambda_cls_aux = 6.0
    return cfg


def make_ov23_stage1_config(
    ov2_manifest_path: str = DEFAULT_OV2_MANIFEST,
    ov3_manifest_path: str = DEFAULT_OV3_MANIFEST,
) -> TrainSOBackboneConfig:
    """Build the safe baseline stage-1 config for ov2+ov3 only."""
    cfg = TrainSOBackboneConfig(
        train_manifest_paths=(ov2_manifest_path, ov3_manifest_path),
        val_manifest_paths=(ov2_manifest_path, ov3_manifest_path),
        test_manifest_paths=(ov2_manifest_path, ov3_manifest_path),
        train_splits=("train",),
        val_splits=("valid",),
        test_splits=("test",),
        batch_size=4,
        num_workers=4,
        num_epochs=20,
        learning_rate=1e-4,
        weight_decay=0.05,
        train_projector_in_stage1=False,
        unfreeze_full_trunk=True,
        freeze_projector_by_default=True,
        output_dir="checkpoints/so_backbone_ov23_stage1",
    )
    cfg.dataset.max_clip_duration_seconds = 20.0
    cfg.dataset.crop_mode = "start"
    cfg.loss.lambda_activity = 2.0
    cfg.loss.lambda_azi = 0.75
    cfg.loss.lambda_ele = 0.75
    cfg.loss.lambda_dist = 0.75
    cfg.loss.lambda_cls_aux = 6.0
    return cfg


def make_ov1_stage1_config(
    ov1_manifest_path: str = DEFAULT_OV1_MANIFEST,
) -> TrainSOBackboneConfig:
    """Build the single-source ov1 warmup config used to sanity-check the architecture."""
    cfg = TrainSOBackboneConfig(
        train_manifest_paths=(ov1_manifest_path,),
        val_manifest_paths=(ov1_manifest_path,),
        test_manifest_paths=(ov1_manifest_path,),
        train_splits=("train",),
        val_splits=("valid",),
        test_splits=("test",),
        batch_size=8,
        num_workers=4,
        num_epochs=20,
        learning_rate=1e-4,
        weight_decay=0.05,
        train_projector_in_stage1=False,
        unfreeze_full_trunk=True,
        freeze_projector_by_default=True,
        output_dir="checkpoints/so_backbone_ov1_stage1",
    )
    cfg.dataset.max_clip_duration_seconds = 20.0
    cfg.dataset.crop_mode = "start"
    cfg.loss.lambda_activity = 8.0
    cfg.loss.lambda_azi = 1.0
    cfg.loss.lambda_ele = 1.0
    cfg.loss.lambda_dist = 1.0
    cfg.loss.lambda_cls_aux = 4.0
    return cfg


def make_ov23_spatial_finetune_config(
    ov2_manifest_path: str = DEFAULT_OV2_MANIFEST,
    ov3_manifest_path: str = DEFAULT_OV3_MANIFEST,
) -> TrainSOBackboneConfig:
    """Build a spatial-focused finetuning config after warmup.

    Intended use:
        1. warm up the channel-mixer / readout stack with the safer stage-1 run
        2. resume from that checkpoint using this config
        3. shift optimization pressure from class/activity toward spatial errors
    """
    cfg = TrainSOBackboneConfig(
        train_manifest_paths=(ov2_manifest_path, ov3_manifest_path),
        val_manifest_paths=(ov2_manifest_path, ov3_manifest_path),
        test_manifest_paths=(ov2_manifest_path, ov3_manifest_path),
        train_splits=("train",),
        val_splits=("valid",),
        test_splits=("test",),
        batch_size=4,
        num_workers=4,
        num_epochs=20,
        learning_rate=3e-5,
        weight_decay=0.05,
        train_projector_in_stage1=False,
        unfreeze_full_trunk=False,
        freeze_projector_by_default=True,
        output_dir="checkpoints/so_backbone_ov23_spatial_finetune",
        best_metric_name="azi_mae_deg",
        minimize_best_metric=True,
    )
    cfg.dataset.max_clip_duration_seconds = 20.0
    cfg.dataset.crop_mode = "start"
    cfg.loss.lambda_activity = 1.0
    cfg.loss.lambda_azi = 3.0
    cfg.loss.lambda_ele = 2.0
    cfg.loss.lambda_dist = 1.5
    cfg.loss.lambda_cls_aux = 1.0
    cfg.loss.lambda_temp = 0.05
    cfg.loss.azi_soft_label_sigma_deg = 7.5
    cfg.loss.ele_soft_label_sigma_deg = 7.5
    return cfg


def make_ov123_spatial_finetune_config(
    ov1_manifest_path: str = DEFAULT_OV1_MANIFEST,
    ov2_manifest_path: str = DEFAULT_OV2_MANIFEST,
    ov3_manifest_path: str = DEFAULT_OV3_MANIFEST,
) -> TrainSOBackboneConfig:
    """Build a spatial-focused finetuning config for ov1+ov2+ov3."""
    cfg = TrainSOBackboneConfig(
        train_manifest_paths=(ov1_manifest_path, ov2_manifest_path, ov3_manifest_path),
        val_manifest_paths=(ov1_manifest_path, ov2_manifest_path, ov3_manifest_path),
        test_manifest_paths=(ov1_manifest_path, ov2_manifest_path, ov3_manifest_path),
        train_splits=("train",),
        val_splits=("valid",),
        test_splits=("test",),
        batch_size=8,
        num_workers=4,
        num_epochs=20,
        learning_rate=3e-5,
        weight_decay=0.05,
        train_projector_in_stage1=False,
        unfreeze_full_trunk=False,
        freeze_projector_by_default=True,
        output_dir="checkpoints/so_backbone_ov123_spatial_finetune",
        best_metric_name="azi_mae_deg",
        minimize_best_metric=True,
    )
    cfg.dataset.max_clip_duration_seconds = 20.0
    cfg.dataset.crop_mode = "start"
    cfg.loss.lambda_activity = 1.0
    cfg.loss.lambda_azi = 3.0
    cfg.loss.lambda_ele = 2.0
    cfg.loss.lambda_dist = 1.5
    cfg.loss.lambda_cls_aux = 1.0
    cfg.loss.lambda_temp = 0.05
    cfg.loss.azi_soft_label_sigma_deg = 7.5
    cfg.loss.ele_soft_label_sigma_deg = 7.5
    return cfg


def make_ov1_spatial_finetune_config(
    ov1_manifest_path: str = DEFAULT_OV1_MANIFEST,
) -> TrainSOBackboneConfig:
    """Build a single-source spatial finetuning config for architecture validation."""
    cfg = TrainSOBackboneConfig(
        train_manifest_paths=(ov1_manifest_path,),
        val_manifest_paths=(ov1_manifest_path,),
        test_manifest_paths=(ov1_manifest_path,),
        train_splits=("train",),
        val_splits=("valid",),
        test_splits=("test",),
        batch_size=8,
        num_workers=4,
        num_epochs=20,
        learning_rate=3e-5,
        weight_decay=0.05,
        train_projector_in_stage1=False,
        unfreeze_full_trunk=False,
        freeze_projector_by_default=True,
        output_dir="checkpoints/so_backbone_ov1_spatial_finetune",
        best_metric_name="azi_mae_deg",
        minimize_best_metric=True,
    )
    cfg.dataset.max_clip_duration_seconds = 20.0
    cfg.dataset.crop_mode = "start"
    cfg.loss.lambda_activity = 1.0
    cfg.loss.lambda_azi = 3.0
    cfg.loss.lambda_ele = 2.0
    cfg.loss.lambda_dist = 1.5
    cfg.loss.lambda_cls_aux = 0.5
    cfg.loss.lambda_temp = 0.05
    cfg.loss.azi_soft_label_sigma_deg = 7.5
    cfg.loss.ele_soft_label_sigma_deg = 7.5
    return cfg


def make_ov1_ast_config(
    ov1_manifest_path: str = DEFAULT_OV1_MANIFEST,
) -> TrainSOBackboneConfig:
    """Build the simplified single-source Spatial-AST-style ov1 config.

    This preset discards the multi-slot matching recipe entirely and instead
    uses:
        - one class task token
        - one spatial task token
        - direct class / direction / distance supervision
    """
    cfg = TrainSOBackboneConfig(
        train_manifest_paths=(ov1_manifest_path,),
        val_manifest_paths=(ov1_manifest_path,),
        test_manifest_paths=(ov1_manifest_path,),
        train_splits=("train",),
        val_splits=("valid",),
        test_splits=("test",),
        batch_size=8,
        num_workers=4,
        num_epochs=20,
        learning_rate=5e-5,
        weight_decay=0.05,
        train_projector_in_stage1=False,
        unfreeze_full_trunk=False,
        freeze_trunk_in_stage1=True,
        train_patch_embedding_in_stage1=False,
        freeze_projector_by_default=True,
        output_dir="checkpoints/so_backbone_ov1_ast",
        best_metric_name="azi_mae_deg",
        minimize_best_metric=True,
    )
    cfg.model.readout_scheme = "mono_ast"
    cfg.model.mono_task_readout_layers = 1
    cfg.model.patch_adapter_residual_alpha_init = 1.0
    cfg.model.patch_adapter_out_proj_scale_init = 1.0
    cfg.dataset.max_clip_duration_seconds = 20.0
    cfg.dataset.crop_mode = "start"
    cfg.loss.supervision_mode = "mono_ast"
    cfg.loss.lambda_cls_aux = 0.25
    cfg.loss.lambda_direction = 10.0
    cfg.loss.lambda_dist = 2.0
    cfg.loss.lambda_activity = 0.0
    cfg.loss.lambda_azi = 0.0
    cfg.loss.lambda_ele = 0.0
    cfg.loss.lambda_temp = 0.0
    return cfg


def make_ov1_ast_classwarmup_config(
    ov1_manifest_path: str = DEFAULT_OV1_MANIFEST,
) -> TrainSOBackboneConfig:
    """Build the class-first warmup config for the mono_ast ov1 path.

    This keeps the patch-delta architecture but uses class-dominant loss so the
    new 65-way source classifier learns a usable decision boundary before the
    spatial-first stage pushes hard on direction and distance.
    """
    cfg = make_ov1_ast_config(ov1_manifest_path=ov1_manifest_path)
    cfg.num_epochs = 8
    cfg.learning_rate = 5e-5
    cfg.output_dir = "checkpoints/so_backbone_ov1_ast_classwarmup"
    cfg.best_metric_name = "class_acc"
    cfg.minimize_best_metric = False
    cfg.loss.lambda_cls_aux = 6.0
    cfg.loss.lambda_direction = 1.0
    cfg.loss.lambda_dist = 0.5
    return cfg


def make_ov1_ast_spatial_config(
    ov1_manifest_path: str = DEFAULT_OV1_MANIFEST,
) -> TrainSOBackboneConfig:
    """Build the spatial-focused follow-up config for the mono_ast ov1 path."""
    cfg = make_ov1_ast_config(ov1_manifest_path=ov1_manifest_path)
    cfg.learning_rate = 3e-5
    cfg.num_epochs = 20
    cfg.output_dir = "checkpoints/so_backbone_ov1_ast_spatial"
    cfg.loss.lambda_cls_aux = 0.1
    cfg.loss.lambda_direction = 12.0
    cfg.loss.lambda_dist = 2.0
    return cfg


def make_ov1_ast_balanced_config(
    ov1_manifest_path: str = DEFAULT_OV1_MANIFEST,
) -> TrainSOBackboneConfig:
    """Build the balanced follow-up config for the mono_ast ov1 path.

    Intended use:
        resume from the spatial-first best checkpoint and recover class
        accuracy without letting classification dominate direction learning.
    """
    cfg = make_ov1_ast_config(ov1_manifest_path=ov1_manifest_path)
    cfg.learning_rate = 3e-5
    cfg.num_epochs = 10
    cfg.output_dir = "checkpoints/so_backbone_ov1_ast_balanced"
    cfg.best_metric_name = "loss_total"
    cfg.minimize_best_metric = True
    cfg.loss.lambda_cls_aux = 2.0
    cfg.loss.lambda_direction = 8.0
    cfg.loss.lambda_dist = 2.0
    return cfg


def make_ov1_pretrunk_ast_class_config(
    ov1_manifest_path: str = DEFAULT_OV1_MANIFEST,
) -> TrainSOBackboneConfig:
    """Build the class-only warmup for the BAT/Spatial-AST-style branch.

    This branch puts distance/DoA/class task tokens inside the BEATs trunk
    before self-attention and uses CE heads, closer to the local Spatial-AST
    implementation than the previous post-trunk mono_ast readout.
    """
    cfg = TrainSOBackboneConfig(
        train_manifest_paths=(ov1_manifest_path,),
        val_manifest_paths=(ov1_manifest_path,),
        test_manifest_paths=(ov1_manifest_path,),
        train_splits=("train",),
        val_splits=("valid",),
        test_splits=("test",),
        batch_size=8,
        num_workers=4,
        num_epochs=8,
        learning_rate=5e-5,
        weight_decay=0.05,
        train_projector_in_stage1=False,
        unfreeze_full_trunk=False,
        freeze_trunk_in_stage1=False,
        train_patch_embedding_in_stage1=False,
        freeze_projector_by_default=True,
        output_dir="checkpoints/so_backbone_ov1_pretrunk_ast_class",
        best_metric_name="class_acc",
        minimize_best_metric=False,
    )
    cfg.model.readout_scheme = "pretrunk_ast"
    cfg.model.patch_adapter_residual_alpha_init = 1.0
    cfg.model.patch_adapter_out_proj_scale_init = 1.0
    cfg.dataset.max_clip_duration_seconds = 20.0
    cfg.dataset.crop_mode = "start"
    cfg.loss.supervision_mode = "pretrunk_ast"
    cfg.loss.num_distance_bins = cfg.model.num_distance_bins
    cfg.loss.distance_bin_size_m = cfg.model.distance_bin_size_m
    cfg.loss.lambda_cls_aux = 6.0
    cfg.loss.lambda_dist = 0.0
    cfg.loss.lambda_azi = 0.0
    cfg.loss.lambda_ele = 0.0
    cfg.loss.lambda_activity = 0.0
    cfg.loss.lambda_temp = 0.0
    cfg.loss.lambda_direction = 0.0
    return cfg


def make_ov1_pretrunk_ast_phase0_config(
    ov1_manifest_path: str = DEFAULT_OV1_MANIFEST,
) -> TrainSOBackboneConfig:
    """Build the strict W-only class probe for preprocessing alignment.

    This is intentionally narrower than the class warmup: trunk, pretrained
    patch embedding, and spatial delta adapter are frozen; the spatial delta is
    initialized to zero. Only pre-trunk task tokens and the new 65-way CE head
    are trained.
    """
    cfg = make_ov1_pretrunk_ast_class_config(ov1_manifest_path=ov1_manifest_path)
    cfg.num_epochs = 2
    cfg.learning_rate = 1e-4
    cfg.output_dir = "checkpoints/so_backbone_ov1_pretrunk_ast_phase0"
    cfg.freeze_trunk_in_stage1 = True
    cfg.unfreeze_full_trunk = False
    cfg.train_patch_embedding_in_stage1 = False
    cfg.train_spatial_adapter_in_stage1 = False
    cfg.model.patch_adapter_residual_alpha_init = 0.0
    cfg.model.patch_adapter_out_proj_scale_init = 0.0
    return cfg


def make_ov1_pretrunk_ast_spatial_config(
    ov1_manifest_path: str = DEFAULT_OV1_MANIFEST,
) -> TrainSOBackboneConfig:
    """Build the spatial stage for the BAT/Spatial-AST-style branch."""
    cfg = make_ov1_pretrunk_ast_class_config(ov1_manifest_path=ov1_manifest_path)
    cfg.num_epochs = 12
    cfg.learning_rate = 3e-5
    cfg.output_dir = "checkpoints/so_backbone_ov1_pretrunk_ast_spatial"
    cfg.best_metric_name = "azi_mae_deg"
    cfg.minimize_best_metric = True
    cfg.loss.lambda_cls_aux = 2.0
    cfg.loss.lambda_dist = 1.0
    cfg.loss.lambda_azi = 2.0
    cfg.loss.lambda_ele = 2.0
    return cfg




def make_so_encoder_config(
    train_manifest_path: str = "",
    valid_manifest_path: str = "",
    *,
    output_dir: str = "checkpoints/so_encoder",
    pretrained_beats_ckpt: str = "",
    class_finetuned_ckpt: str = "",
    source_vocab_path: str = "",
    source_num_classes: int = 63,
) -> TrainSOBackboneConfig:
    """Canonical SO-Encoder training config (flat, no inheritance chain).

    This is the single supported recipe for training the FOA-pretrained
    SO-Encoder used by SO-7B / SO-30B downstream. It materialises the
    settings that produced the released SO-Encoder checkpoint:

      * 10 Hz native token rate (post-projector 2.5 Hz)
      * Frame-level track supervision (matched + ontology-smoothed CE)
      * Class-head freeze during DOA ramp, then small-LR finetune
      * Cosine LR with 3-epoch warmup, EMA weights
      * Top-K rank activity loss

    Caller provides the data manifests, the upstream BEATs trunk ckpt,
    a class-finetuned trunk hot-start, and the 63-class vocabulary CSV;
    everything else is hard-coded here.
    """
    cfg = TrainSOBackboneConfig()

    # --- Data manifests ----------------------------------------------------
    if train_manifest_path:
        cfg.train_manifest_paths = (train_manifest_path,)
        cfg.train_manifest_replication = (1,)
    if valid_manifest_path:
        cfg.val_manifest_paths = ("", "", "", "", "", "", "", valid_manifest_path)
        cfg.test_manifest_paths = ("", "", "", "", "", "", "", valid_manifest_path)

    # --- Optimiser / schedule ---------------------------------------------
    cfg.batch_size = 8
    cfg.num_epochs = 25
    cfg.learning_rate = 1.5e-5
    cfg.use_cosine_lr = True
    cfg.trunk_lr_scale = 0.1
    cfg.spatial_lr_scale = 0.3
    cfg.local_spatial_lr_scale = 1.0
    cfg.class_head_lr_scale = 0.3
    cfg.class_head_freeze_during_ramp_epochs = 4
    cfg.use_ema = True

    # --- Trunk / spatial freezing ----------------------------------------
    cfg.unfreeze_full_trunk = False
    cfg.unfreeze_top_n_layers = 4
    cfg.train_patch_embedding_in_stage1 = False
    cfg.train_spatial_adapter_in_stage1 = False

    # --- Loss schedule -----------------------------------------------------
    cfg.frame_match_class_cost_warmup_epochs = 1
    cfg.frame_match_class_cost_ramp_epochs = 2
    cfg.frame_spatial_loss_warmup_epochs = 8
    cfg.frame_spatial_loss_ramp_epochs = 2

    # --- Best-checkpoint policy ------------------------------------------
    cfg.best_metric_name = "F20"
    cfg.minimize_best_metric = False

    # --- Diagnostics -------------------------------------------------------
    cfg.dump_frame_track_csv = True
    cfg.frame_track_csv_max_samples_per_epoch = 48
    cfg.frame_track_csv_max_samples_per_group = 16

    # --- Distributed -------------------------------------------------------
    cfg.ddp_find_unused_parameters = True

    # --- Output / external ckpts ------------------------------------------
    cfg.output_dir = output_dir
    if pretrained_beats_ckpt:
        cfg.pretrained_beats_ckpt = pretrained_beats_ckpt
    if class_finetuned_ckpt:
        cfg.class_finetuned_ckpt = class_finetuned_ckpt

    # --- Dataset config ---------------------------------------------------
    cfg.dataset.target_token_rate = 10.0
    cfg.dataset.max_clip_duration_seconds = 20.0
    cfg.dataset.crop_mode = "start"
    if source_vocab_path:
        cfg.dataset.source_vocab.vocab_path = source_vocab_path
    cfg.dataset.source_vocab.num_classes = source_num_classes

    # --- Loss weights -----------------------------------------------------
    cfg.loss.lambda_activity = 0.0
    cfg.loss.lambda_azi = 0.0
    cfg.loss.lambda_ele = 0.0
    cfg.loss.lambda_cls_aux = 2.0
    cfg.loss.lambda_direction = 6.0
    cfg.loss.supervision_mode = "local_spatial_track"
    cfg.loss.lambda_frame_direction = 4.0
    cfg.loss.lambda_clip_aux = 0.0
    cfg.loss.frame_activity_pos_weight = 3.0
    cfg.loss.use_segment_matching = True
    cfg.loss.label_smoothing = 0.1
    cfg.loss.lambda_sem_anchor = 1.5
    cfg.loss.frame_activity_loss_type = "topk_rank"

    # Per-class loss weights (63 classes, indexed by class_id from
    # final_vocabulary.csv). Catch-all classes (bird, machine,
    # human_vocalization, rain, breathing, singing) are down-weighted;
    # rare confusable classes (frog, crackle, tape, knock, drawer_cabinet,
    # aircraft, vehicle, train) get a moderate boost.
    cfg.loss.frame_class_loss_weights = [
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.4, 1.0, 0.6, 1.0,
        0.5, 1.0, 1.0, 0.5, 1.0, 1.0, 2.0, 1.0, 1.0, 1.5,
        1.0, 1.0, 1.5, 1.0, 1.0, 1.5, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 0.6, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 0.7, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0,
        2.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 0.7, 2.0, 1.0,
        1.0, 1.0, 3.0,
    ]
    # Ontology groups: classes that share an AudioSet parent get soft
    # cross-entropy mass between siblings (frame_class_ontology_smoothing).
    cfg.loss.frame_class_ontology_smoothing = 0.1
    cfg.loss.frame_class_ontology_groups = [
        [55, 19, 22, 44],                                # transportation
        [16, 6, 31, 32, 13, 14],                          # human voice
        [8, 62, 35, 18, 33, 27],                          # animal vocal
        [9, 10, 48, 57, 34, 30, 50, 26, 38, 39,
         36, 37, 58, 61],                                 # indoor mechanical
        [52, 21, 60, 53, 56, 46, 54, 42, 43, 49],         # percussive
        [25, 45, 29, 51, 5, 40, 24, 12, 59],              # weather/water
        [0, 1, 2, 4, 7, 15, 28, 47, 17, 41],              # musical instruments
        [20, 23, 11],                                     # alarms / signals
    ]

    # --- Backbone (SOBackboneConfig) -------------------------------------
    cfg.model.target_token_rate = 10.0
    cfg.model.bypass_spatial_delta = True
    cfg.model.enable_clip_aux_head = False
    cfg.model.enable_frame_track = True
    cfg.model.head_dropout = 0.3
    cfg.model.local_spatial_fusion_mode = "cross_attn_gated"
    cfg.model.patch_adapter_out_proj_scale_init = 0.0
    cfg.model.patch_adapter_residual_alpha_init = 0.0
    cfg.model.readout_scheme = "local_spatial_track"
    cfg.model.spec_augment_freq_masks = 2
    cfg.model.spec_augment_freq_width = 27
    cfg.model.spec_augment_time_masks = 2
    cfg.model.spec_augment_time_width = 100
    cfg.model.use_class_head_demixer = True
    cfg.model.use_class_head_mlp_residual = True
    cfg.model.use_kaldi_w_channel = True
    cfg.model.use_semantic_anchor = True
    cfg.model.use_spatial_head_demixer = True

    return cfg


def _resolve_manifest_paths(single_path: Optional[str], multi_paths: Sequence[str]) -> Tuple[str, ...]:
    paths = [path for path in multi_paths if path]
    if not paths and single_path:
        paths = [single_path]
    return tuple(paths)


def initialize_distributed_mode(train_cfg: TrainSOBackboneConfig) -> torch.device:
    """Initialize DDP from torchrun environment variables when requested."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(train_cfg.local_rank)))
    train_cfg.distributed = bool(train_cfg.distributed or world_size > 1)

    if train_cfg.distributed:
        backend = train_cfg.distributed_backend
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            if backend == "auto":
                backend = "nccl"
        else:
            device = torch.device("cpu")
            if backend == "auto":
                backend = "gloo"
        if not _is_dist_initialized():
            dist.init_process_group(backend=backend)
        train_cfg.local_rank = local_rank
        train_cfg.show_progress_bars = train_cfg.show_progress_bars and _is_main_process()
        train_cfg.dataset.show_progress = train_cfg.show_progress_bars
        _log(f"[DDP] Initialized rank={_get_rank()} world_size={_get_world_size()} local_rank={local_rank}")
        return device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_cfg.dataset.show_progress = train_cfg.show_progress_bars
    return device


def cleanup_distributed() -> None:
    """Tear down the process group after training finishes."""
    if _is_dist_initialized():
        dist.destroy_process_group()


def build_model_config(train_cfg: TrainSOBackboneConfig) -> SOBackboneConfig:
    """Create the model config used by Spatial-BEATs.

    Responsibilities:
        - propagate Qwen-like low-level mel parameters
        - propagate target token rate and source vocabulary settings
        - keep source_num_classes aligned with final_vocabulary.csv

    Returns:
        SOBackboneConfig:
            Model configuration object consumed by SOBackbone.
    """
    model_cfg = copy.deepcopy(train_cfg.model)
    dataset_cfg = train_cfg.dataset

    model_cfg.sample_rate = dataset_cfg.mel_config.sample_rate
    model_cfg.num_mel_bins = dataset_cfg.mel_config.num_mel_bins
    model_cfg.n_fft = dataset_cfg.mel_config.n_fft
    model_cfg.win_length = dataset_cfg.mel_config.win_length
    model_cfg.hop_length = dataset_cfg.mel_config.hop_length
    model_cfg.dither = dataset_cfg.mel_config.dither
    model_cfg.waveform_scale = dataset_cfg.mel_config.waveform_scale
    model_cfg.fbank_mean = dataset_cfg.mel_config.fbank_mean
    model_cfg.fbank_std = dataset_cfg.mel_config.fbank_std
    model_cfg.normalize_logmel = dataset_cfg.mel_config.normalize_logmel
    model_cfg.target_token_rate = dataset_cfg.target_token_rate
    model_cfg.max_sources = dataset_cfg.max_sources
    model_cfg.source_vocab_path = dataset_cfg.source_vocab.vocab_path
    model_cfg.source_label_id_field = dataset_cfg.source_vocab.label_id_field
    model_cfg.source_label_name_field = dataset_cfg.source_vocab.label_name_field
    model_cfg.source_num_classes = dataset_cfg.source_vocab.num_classes
    return model_cfg


def build_dataset_config(train_cfg: TrainSOBackboneConfig) -> SpatialDatasetConfig:
    """Create the dataset config used by SpatialDataset.

    Responsibilities:
        - keep mel front-end parameters aligned with the model
        - keep target token rate aligned with the model
        - keep source vocabulary path aligned with the model
    """
    dataset_cfg = copy.deepcopy(train_cfg.dataset)
    model_cfg = train_cfg.model

    dataset_cfg.mel_config.sample_rate = model_cfg.sample_rate
    dataset_cfg.mel_config.num_mel_bins = model_cfg.num_mel_bins
    dataset_cfg.mel_config.n_fft = model_cfg.n_fft
    dataset_cfg.mel_config.win_length = model_cfg.win_length
    dataset_cfg.mel_config.hop_length = model_cfg.hop_length
    dataset_cfg.mel_config.dither = model_cfg.dither
    dataset_cfg.mel_config.waveform_scale = model_cfg.waveform_scale
    dataset_cfg.mel_config.fbank_mean = model_cfg.fbank_mean
    dataset_cfg.mel_config.fbank_std = model_cfg.fbank_std
    dataset_cfg.mel_config.normalize_logmel = model_cfg.normalize_logmel
    dataset_cfg.target_token_rate = model_cfg.target_token_rate
    dataset_cfg.max_sources = model_cfg.max_sources
    dataset_cfg.source_vocab.vocab_path = model_cfg.source_vocab_path
    dataset_cfg.source_vocab.label_id_field = model_cfg.source_label_id_field
    dataset_cfg.source_vocab.label_name_field = model_cfg.source_label_name_field
    dataset_cfg.source_vocab.num_classes = model_cfg.source_num_classes
    return dataset_cfg


def build_dataloaders(
    train_cfg: TrainSOBackboneConfig,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Build training and validation dataloaders.

    Returns:
        Tuple[DataLoader, Optional[DataLoader]]:
            train_loader and optional val_loader.
    """
    dataset_cfg = build_dataset_config(train_cfg)
    train_paths = _resolve_manifest_paths(train_cfg.train_manifest_path, train_cfg.train_manifest_paths)
    if not train_paths:
        raise ValueError("At least one training manifest path must be provided.")
    _log(f"[Train] Build train datasets from {len(train_paths)} manifest(s)")

    train_dataset_cfg = copy.deepcopy(dataset_cfg)
    train_dataset_cfg.allowed_splits = train_cfg.train_splits
    train_datasets = [
        SpatialDataset(manifest_path=path, config=train_dataset_cfg)
        for path in train_paths
    ]
    # Optional per-manifest replication: parallel tuple of multipliers.
    # Only applied when length matches the number of train manifests.
    replication = train_cfg.train_manifest_replication
    if replication and len(replication) == len(train_datasets):
        replicated: List[SpatialDataset] = []
        for ds, rep in zip(train_datasets, replication):
            if rep <= 0:
                continue
            replicated.extend([ds] * int(rep))
            _log(f"[Train] Manifest {ds.manifest_path} replicated x{int(rep)}")
        if replicated:
            train_datasets = replicated
    elif replication:
        _log(
            f"[Train] WARNING: train_manifest_replication length "
            f"{len(replication)} != num train manifests {len(train_datasets)}; "
            f"ignoring replication"
        )
    train_dataset = train_datasets[0] if len(train_datasets) == 1 else ConcatDataset(train_datasets)
    train_sampler = (
        DistributedSampler(train_dataset, shuffle=True)
        if train_cfg.distributed
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=train_cfg.num_workers,
        collate_fn=lambda samples: collate_spatial_batch(samples, train_dataset_cfg),
        pin_memory=True,
        persistent_workers=train_cfg.num_workers > 0,
        prefetch_factor=4 if train_cfg.num_workers > 0 else None,
    )

    val_loader = None
    val_paths = _resolve_manifest_paths(train_cfg.val_manifest_path, train_cfg.val_manifest_paths)
    if val_paths:
        _log(f"[Train] Build val datasets from {len(val_paths)} manifest(s)")
        val_dataset_cfg = copy.deepcopy(dataset_cfg)
        val_dataset_cfg.allowed_splits = train_cfg.val_splits
        val_datasets = [
            SpatialDataset(manifest_path=path, config=val_dataset_cfg)
            for path in val_paths
        ]
        val_dataset = val_datasets[0] if len(val_datasets) == 1 else ConcatDataset(val_datasets)
        val_sampler = (
            DistributedSampler(val_dataset, shuffle=False)
            if train_cfg.distributed
            else None
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=train_cfg.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=train_cfg.num_workers,
            collate_fn=lambda samples: collate_spatial_batch(samples, val_dataset_cfg),
            pin_memory=True,
            persistent_workers=train_cfg.num_workers > 0,
            prefetch_factor=4 if train_cfg.num_workers > 0 else None,
        )

    return train_loader, val_loader


def _legacy_safe_torch_load(checkpoint_path: str):
    """Load a checkpoint that may have been pickled by a legacy module layout.

    Older checkpoints (e.g. those produced before the rename to ``so_*``) may
    embed pickled config objects whose qualified names reference modules that
    no longer exist (``spatial_beats``, ``spatial_modules``, ``train_spatial_beats``,
    ...). Standard ``torch.load(weights_only=True)`` rejects these; the unsafe
    path needs the legacy modules importable.

    This loader installs a custom ``Unpickler`` that returns a generic stub
    class whenever a legacy class is referenced — the stub holds attributes
    set on it but is otherwise inert. State-dict tensors are unaffected.
    """
    import pickle

    LEGACY_PREFIXES = (
        "spatial_beats",
        "spatial_modules",
        "spatial_loss",
        "spatial_dataset",
        "train_spatial_beats",
    )

    class _LegacyStub:
        def __init__(self, *args, **kwargs):
            self.__dict__.update(kwargs)
        def __setstate__(self, state):
            if isinstance(state, dict):
                self.__dict__.update(state)
        def __reduce__(self):
            return (_LegacyStub, ())

    class _CompatUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if any(module == p or module.startswith(p + ".") for p in LEGACY_PREFIXES):
                return _LegacyStub
            try:
                return super().find_class(module, name)
            except (ModuleNotFoundError, AttributeError):
                return _LegacyStub

    class _PickleModule:
        Unpickler = _CompatUnpickler
        @staticmethod
        def load(f, **kw):
            return _CompatUnpickler(f, **kw).load()

    return torch.load(
        checkpoint_path, map_location="cpu", weights_only=False,
        pickle_module=_PickleModule,
    )


def _load_spatial_init_checkpoint(model: SOBackbone, checkpoint_path: str) -> None:
    """Warm-start compatible weights from a prior SOBackbone checkpoint.

    Used by the ov123 frame-level presets to initialize the trunk + local
    spatial fusion stack + aux clip head from the ov1 local_spatial best.pt.
    Only keys whose names and shapes match the current model are loaded;
    scheme-specific frame-level heads (FrameSlotHead, SourceQueryDecoder,
    FrameTrackPredictionHeads, ACCDOAHeads) are intentionally skipped.
    """
    _log(f"[Train] Warm-starting from spatial checkpoint {checkpoint_path}")
    checkpoint = _legacy_safe_torch_load(checkpoint_path)
    state_dict = checkpoint.get(
        "model_state_dict",
        checkpoint.get("model", checkpoint.get("state_dict", checkpoint)),
    )
    state_dict = _seed_frame_track_heads_from_clip_head(
        model=model,
        state_dict=state_dict,
        log_prefix="[Train] Spatial warm-start",
    )
    current_state = model.state_dict()
    loadable = {
        key: value
        for key, value in state_dict.items()
        if key in current_state and current_state[key].shape == value.shape
    }
    missing, unexpected = model.load_state_dict(loadable, strict=False)
    _log(
        f"[Train] Spatial warm-start loaded={len(loadable)} "
        f"skipped_unexpected={len(unexpected)} remaining_missing={len(missing)}"
    )


def _seed_frame_track_heads_from_clip_head(
    model: nn.Module,
    state_dict: Dict[str, Tensor],
    log_prefix: str,
) -> Dict[str, Tensor]:
    """Copy compatible clip-head weights into the per-frame track heads.

    Only the output projections that share semantics are transferred:
      - class_head
      - direction_head
      - distance_head

    Pooling-related layers, query decoder, activity head, and input_norm stay
    untouched. If the checkpoint already contains frame_track keys, they win.
    """
    current_state = _unwrap_model(model).state_dict()
    if "frame_track_prediction_heads.class_head.weight" not in current_state:
        return state_dict
    if "local_spatial_prediction_heads.class_head.weight" not in state_dict:
        return state_dict

    remapped = dict(state_dict)
    mapping = {
        "local_spatial_prediction_heads.class_head.weight": "frame_track_prediction_heads.class_head.weight",
        "local_spatial_prediction_heads.class_head.bias": "frame_track_prediction_heads.class_head.bias",
        "local_spatial_prediction_heads.direction_head.0.weight": "frame_track_prediction_heads.direction_head.0.weight",
        "local_spatial_prediction_heads.direction_head.0.bias": "frame_track_prediction_heads.direction_head.0.bias",
        "local_spatial_prediction_heads.direction_head.2.weight": "frame_track_prediction_heads.direction_head.3.weight",
        "local_spatial_prediction_heads.direction_head.2.bias": "frame_track_prediction_heads.direction_head.3.bias",
        "local_spatial_prediction_heads.distance_head.0.weight": "frame_track_prediction_heads.distance_head.0.weight",
        "local_spatial_prediction_heads.distance_head.0.bias": "frame_track_prediction_heads.distance_head.0.bias",
        "local_spatial_prediction_heads.distance_head.2.weight": "frame_track_prediction_heads.distance_head.3.weight",
        "local_spatial_prediction_heads.distance_head.2.bias": "frame_track_prediction_heads.distance_head.3.bias",
    }

    copied: List[str] = []
    for source_key, target_key in mapping.items():
        if target_key in remapped:
            continue
        if source_key not in state_dict or target_key not in current_state:
            continue
        if current_state[target_key].shape != state_dict[source_key].shape:
            continue
        remapped[target_key] = state_dict[source_key]
        copied.append(f"{source_key}->{target_key}")

    if copied:
        _log(
            f"{log_prefix} seeded {len(copied)} frame-track tensor(s) "
            "from local_spatial_prediction_heads"
        )
    return remapped


def build_model(train_cfg: TrainSOBackboneConfig) -> SOBackbone:
    """Instantiate the Spatial-BEATs model and load pretrained BEATs weights.

    Responsibilities:
        - create SOBackbone from SOBackboneConfig
        - call load_beats_pretrained()
        - freeze or unfreeze modules according to stage-1 settings
    """
    model_cfg = build_model_config(train_cfg)
    _log("[Train] Build Spatial-BEATs model")
    model = SOBackbone(model_cfg)
    model.load_beats_pretrained(train_cfg.pretrained_beats_ckpt)
    if train_cfg.trunk_finetuned_ckpt:
        model.load_trunk_finetuned_checkpoint(train_cfg.trunk_finetuned_ckpt)
    if train_cfg.class_finetuned_ckpt:
        model.load_event_classifier_checkpoint(train_cfg.class_finetuned_ckpt)
    if train_cfg.init_from_spatial_ckpt:
        _load_spatial_init_checkpoint(model, train_cfg.init_from_spatial_ckpt)
    configure_stage1_trainable_parameters(model, train_cfg)
    num_trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    _log(f"[Train] Trainable parameters: {num_trainable}")
    return model


def configure_stage1_trainable_parameters(
    model: SOBackbone,
    train_cfg: TrainSOBackboneConfig,
) -> None:
    """Set requires_grad flags for encoder-only stage 1.

    Default intent:
        Train:
            - preprocessor
            - patch_embedding
            - temporal_resampler
            - temporal_readout
            - fixed-slot heads
        Optionally train:
            - trunk (full or partial)
        Default do not train:
            - projector
    """
    for param in model.parameters():
        param.requires_grad = False

    always_train_prefixes = (
        "preprocessor",
        "spatial_patch_adapter",
        "patch_embedding",
        "frequency_pool",
        "temporal_resampler",
        "temporal_readout",
        "slot_readout",
        "mono_task_readout",
        "mono_prediction_heads",
        "pretrunk_task_tokens",
        "pretrunk_prediction_heads",
        "local_spatial_encoder",
        "local_spatial_resampler",
        "local_spatial_proj",
        "local_spatial_fusion_norm",
        "local_spatial_fuser",
        "local_spatial_prediction_heads",
        "frame_wise_heads",
        "prediction_heads",
        "source_query_decoder",
        "frame_track_prediction_heads",
        "frame_slot_head",
        "accdoa_heads",
    )
    for name, param in model.named_parameters():
        if name.startswith(always_train_prefixes):
            param.requires_grad = True

    if not train_cfg.train_patch_embedding_in_stage1:
        for name, param in model.named_parameters():
            if name.startswith("patch_embedding"):
                param.requires_grad = False

    if not train_cfg.train_spatial_adapter_in_stage1:
        for name, param in model.named_parameters():
            if name.startswith("spatial_patch_adapter"):
                param.requires_grad = False

    if train_cfg.freeze_trunk_in_stage1:
        pass
    elif train_cfg.unfreeze_top_n_layers > 0:
        # Unfreeze the top N transformer layers + layer_norm + post_extract_proj
        num_layers = 12  # BEATs has 12 transformer layers
        n = train_cfg.unfreeze_top_n_layers
        start_layer = max(num_layers - n, 0)
        unfrozen_prefixes = tuple(
            f"encoder.layers.{i}" for i in range(start_layer, num_layers)
        )
        for name, param in model.named_parameters():
            if (
                name.startswith("post_extract_proj")
                or name.startswith("layer_norm")
                or name.startswith("encoder.layer_norm")
                or name.startswith(unfrozen_prefixes)
            ):
                param.requires_grad = True
    elif train_cfg.unfreeze_full_trunk:
        for name, param in model.named_parameters():
            if name.startswith(("layer_norm", "post_extract_proj", "encoder")):
                param.requires_grad = True
    else:
        for name, param in model.named_parameters():
            if (
                name.startswith("post_extract_proj")
                or name.startswith("layer_norm")
                or name.startswith("encoder.layers.10")
                or name.startswith("encoder.layers.11")
                or name.startswith("encoder.layer_norm")
            ):
                param.requires_grad = True

    if train_cfg.train_projector_in_stage1 and not train_cfg.freeze_projector_by_default:
        for name, param in model.named_parameters():
            if name.startswith("projector"):
                param.requires_grad = True

    if train_cfg.freeze_local_spatial_in_classwarmup:
        # Freeze local_spatial CNN/proj so local_update ≈ 0 and the class head
        # reads near-pure BEATs semantic tokens.  Prediction heads and fusion
        # norm remain trainable; local_spatial_resampler is stateless.
        for name, param in model.named_parameters():
            if name.startswith(("local_spatial_encoder", "local_spatial_proj")):
                param.requires_grad = False

    if train_cfg.model.readout_scheme == "pretrunk_ast":
        # Pre-trunk AST supervision reads distance/DoA/class directly from
        # task tokens that pass through the BEATs trunk. The later temporal
        # readout is still computed for interface compatibility, but it is not
        # connected to the pretrunk_ast loss and must not be trainable under DDP.
        for name, param in model.named_parameters():
            if name.startswith("temporal_readout"):
                param.requires_grad = False

    if train_cfg.model.readout_scheme in LOCAL_SPATIAL_FRAME_SCHEMES and train_cfg.loss.lambda_clip_aux == 0.0:
        # local_spatial_track / slot / accdoa / framewise 模式下，
        # 如果 lambda_clip_aux=0.0 且模型仍然包含 local_spatial_prediction_heads（旧的 enable_clip_aux_head=True 的 preset），
        # 则冻结该模块以避免 DDP find_unused_parameters 报错。
        # 推荐做法：在 preset 里设置 cfg.model.enable_clip_aux_head = False，让模型根本不构建这个 head。
        for name, param in model.named_parameters():
            if name.startswith("local_spatial_prediction_heads"):
                param.requires_grad = False

    if train_cfg.freeze_frame_track_spatial_heads:
        # v10 phase-1: freeze direction_head + distance_head on the frame-track
        # prediction heads so spatial targets don't perturb the class / activity
        # learning.  Requires lambda_frame_direction = lambda_frame_distance = 0
        # so the loss gradients into these heads are already zero; this line
        # enforces it at the parameter level (also excludes them from the
        # optimizer state, avoiding DDP unused-param warnings).
        for name, param in model.named_parameters():
            if name.startswith((
                "frame_track_prediction_heads.direction_head",
                "frame_track_prediction_heads.distance_head",
            )):
                param.requires_grad = False

def build_optimizer(
    model: SOBackbone,
    train_cfg: TrainSOBackboneConfig,
) -> Optimizer:
    """Create AdamW optimizer with optional per-group LR scaling.

    Parameter groups:
        trunk:   encoder.*, layer_norm, post_extract_proj, encoder.pos_conv
                 → lr * trunk_lr_scale
        spatial: preprocessor.*, spatial_patch_adapter.*, local_spatial_*,
                 patch_embedding.*
                 → lr * spatial_lr_scale
        heads:   everything else (prediction heads, readout, etc.)
                 → lr (full)

    When both scales are 1.0 this collapses to a single param group and
    behaves identically to the original implementation.
    """
    base_lr = train_cfg.learning_rate
    wd = train_cfg.weight_decay
    trunk_scale = train_cfg.trunk_lr_scale
    spatial_scale = train_cfg.spatial_lr_scale
    # local_spatial_lr_scale: split the from-scratch local_spatial group
    # away from BEATs-adjacent spatial group.  None preserves legacy
    # behaviour (both groups share spatial_lr_scale).
    local_spatial_scale = (
        train_cfg.local_spatial_lr_scale
        if train_cfg.local_spatial_lr_scale is not None
        else spatial_scale
    )
    cls_head_scale = train_cfg.class_head_lr_scale

    _TRUNK_PREFIXES = (
        "encoder.",
        "layer_norm.",
        "post_extract_proj.",
    )
    # BEATs-adjacent (mel preprocessor + patch-embedding-side adapters):
    # historically slow because they sit on the pretrained input path.
    _SPATIAL_PREFIXES = (
        "preprocessor.",
        "spatial_patch_adapter.",
        "patch_embedding.",
        "trunk_spatial_adapters.",
    )
    # From-scratch local-spatial branch: should train at ~head LR,
    # not at BEATs-adjacent slow LR.
    _LOCAL_SPATIAL_PREFIXES = (
        "local_spatial_encoder.",
        "local_spatial_resampler.",
        "local_spatial_proj.",
        "local_spatial_pre_pool_proj.",
        "local_spatial_fusion_norm.",
        "local_spatial_fuser.",
    )
    # v9: per-name prefixes for the class head inside
    # frame_track_prediction_heads.  Matches the Linear class_head and the
    # optional v9 class_head_mlp/class_head_demixer modules.
    _CLASS_HEAD_PREFIXES = (
        "frame_track_prediction_heads.class_head.",
        "frame_track_prediction_heads.class_head_mlp.",
        "frame_track_prediction_heads.class_head_demixer.",
    )

    if (
        trunk_scale == 1.0
        and spatial_scale == 1.0
        and local_spatial_scale == 1.0
        and cls_head_scale == 1.0
    ):
        # Fast path: single group, identical to original code
        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            raise ValueError("No trainable parameters.")
        return AdamW(params, lr=base_lr, weight_decay=wd)

    trunk_params, spatial_params, local_spatial_params, head_params, cls_head_params = (
        [], [], [], [], []
    )
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith(_CLASS_HEAD_PREFIXES) and cls_head_scale != 1.0:
            cls_head_params.append(param)
        elif name.startswith(_TRUNK_PREFIXES):
            trunk_params.append(param)
        elif name.startswith(_LOCAL_SPATIAL_PREFIXES):
            local_spatial_params.append(param)
        elif name.startswith(_SPATIAL_PREFIXES):
            spatial_params.append(param)
        else:
            head_params.append(param)

    param_groups = []
    if trunk_params:
        param_groups.append({"params": trunk_params,   "lr": base_lr * trunk_scale,   "weight_decay": wd, "group_name": "trunk"})
    if spatial_params:
        param_groups.append({"params": spatial_params, "lr": base_lr * spatial_scale, "weight_decay": wd, "group_name": "spatial"})
    if local_spatial_params:
        param_groups.append({"params": local_spatial_params, "lr": base_lr * local_spatial_scale, "weight_decay": wd, "group_name": "local_spatial"})
    if head_params:
        param_groups.append({"params": head_params,    "lr": base_lr,                 "weight_decay": wd, "group_name": "head"})
    if cls_head_params:
        param_groups.append({"params": cls_head_params, "lr": base_lr * cls_head_scale, "weight_decay": wd, "group_name": "cls_head"})

    if not param_groups:
        raise ValueError("No trainable parameters.")

    _log(
        f"[Optimizer] trunk_lr={base_lr * trunk_scale:.2e}  "
        f"spatial_lr={base_lr * spatial_scale:.2e}  "
        f"head_lr={base_lr:.2e}  "
        f"cls_head_lr={base_lr * cls_head_scale:.2e}  "
        f"(trunk={len(trunk_params)} spatial={len(spatial_params)} head={len(head_params)} cls={len(cls_head_params)} params)"
    )
    return AdamW(param_groups, lr=base_lr, weight_decay=wd)


def _is_better_metric(
    candidate: float,
    best_so_far: Optional[float],
    minimize: bool,
) -> bool:
    """Decide whether the new metric improves over the current best."""
    if best_so_far is None:
        return True
    return candidate < best_so_far if minimize else candidate > best_so_far


def _build_checkpoint_state(
    model: nn.Module,
    optimizer: Optimizer,
    train_cfg: TrainSOBackboneConfig,
    epoch: int,
    best_metric_value: Optional[float],
    train_metrics: Dict[str, float],
    val_metrics: Optional[Dict[str, float]],
) -> Dict[str, object]:
    """Build the serialized checkpoint payload."""
    model_to_save = _unwrap_model(model)
    return {
        "epoch": int(epoch),
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if train_cfg.save_optimizer_state else None,
        "best_metric_name": train_cfg.best_metric_name,
        "best_metric_value": best_metric_value,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "train_cfg": asdict(train_cfg),
    }


def save_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optimizer,
    train_cfg: TrainSOBackboneConfig,
    epoch: int,
    best_metric_value: Optional[float],
    train_metrics: Dict[str, float],
    val_metrics: Optional[Dict[str, float]],
) -> None:
    """Save a full training checkpoint to disk."""
    if not _is_main_process():
        return
    checkpoint = _build_checkpoint_state(
        model=model,
        optimizer=optimizer,
        train_cfg=train_cfg,
        epoch=epoch,
        best_metric_value=best_metric_value,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
    )
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _log(f"[Checkpoint] Save {path}")
    torch.save(checkpoint, path)


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    load_optimizer_state: bool = True,
) -> Tuple[int, Optional[float], Optional[str]]:
    """Load a training checkpoint and restore model and optimizer state.

    Returns:
        Tuple[int, Optional[float], Optional[str]]:
            Next epoch to run, restored best metric value, and the checkpoint's
            best metric name.
    """
    _log(f"[Checkpoint] Load {checkpoint_path}")
    checkpoint = _legacy_safe_torch_load(checkpoint_path)
    checkpoint_state = _seed_frame_track_heads_from_clip_head(
        model=model,
        state_dict=checkpoint["model_state_dict"],
        log_prefix="[Checkpoint]",
    )
    # Filter out keys with shape mismatches (e.g. V1→V2 adapter upgrade)
    current_state = _unwrap_model(model).state_dict()
    shape_mismatched = [
        k for k, v in checkpoint_state.items()
        if k in current_state and current_state[k].shape != v.shape
    ]
    if shape_mismatched:
        _log(
            f"[Checkpoint] Skipping {len(shape_mismatched)} key(s) with "
            f"shape mismatch: {shape_mismatched}"
        )
        for k in shape_mismatched:
            del checkpoint_state[k]
    missing, unexpected = _unwrap_model(model).load_state_dict(
        checkpoint_state, strict=False,
    )
    if missing:
        _log(
            f"[Checkpoint] WARNING: {len(missing)} missing key(s) — "
            f"newly initialized: {missing}"
        )
    if unexpected:
        _log(
            f"[Checkpoint] WARNING: {len(unexpected)} unexpected key(s) — "
            f"ignored: {unexpected}"
        )

    if (
        optimizer is not None
        and load_optimizer_state
        and checkpoint.get("optimizer_state_dict") is not None
    ):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    next_epoch = int(checkpoint.get("epoch", -1)) + 1
    best_metric_value = checkpoint.get("best_metric_value")
    best_metric_name = checkpoint.get("best_metric_name")
    return next_epoch, best_metric_value, best_metric_name


def _select_reference_metrics(
    train_metrics: Dict[str, float],
    val_metrics: Optional[Dict[str, float]],
) -> Dict[str, float]:
    """Pick the metric dictionary used for best-model selection."""
    return val_metrics if val_metrics is not None else train_metrics


def _reduce_metric_sums(
    running: Dict[str, float],
    num_batches: int,
    device: torch.device,
) -> Tuple[Dict[str, float], int]:
    """All-reduce metric sums and batch count across DDP workers."""
    if not _is_dist_initialized():
        return running, num_batches

    keys = list(running.keys())
    values = [running[key] for key in keys] + [float(num_batches)]
    tensor = torch.tensor(values, device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    reduced_running = {key: float(tensor[idx].item()) for idx, key in enumerate(keys)}
    reduced_batches = int(tensor[-1].item())
    return reduced_running, reduced_batches


def _save_epoch_checkpoints(
    model: SOBackbone,
    optimizer: Optimizer,
    train_cfg: TrainSOBackboneConfig,
    epoch: int,
    best_metric_value: Optional[float],
    train_metrics: Dict[str, float],
    val_metrics: Optional[Dict[str, float]],
    is_best: bool,
) -> None:
    """Write periodic, last, and best checkpoints for the current epoch."""
    output_dir = Path(train_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if train_cfg.save_every_n_epochs > 0 and (epoch + 1) % train_cfg.save_every_n_epochs == 0:
        save_checkpoint(
            checkpoint_path=str(output_dir / f"epoch_{epoch:04d}.pt"),
            model=model,
            optimizer=optimizer,
            train_cfg=train_cfg,
            epoch=epoch,
            best_metric_value=best_metric_value,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
        )

    if train_cfg.save_last_checkpoint:
        save_checkpoint(
            checkpoint_path=str(output_dir / "last.pt"),
            model=model,
            optimizer=optimizer,
            train_cfg=train_cfg,
            epoch=epoch,
            best_metric_value=best_metric_value,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
        )

    if train_cfg.save_best_checkpoint and is_best:
        save_checkpoint(
            checkpoint_path=str(output_dir / "best.pt"),
            model=model,
            optimizer=optimizer,
            train_cfg=train_cfg,
            epoch=epoch,
            best_metric_value=best_metric_value,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
        )


def build_train_config_from_args(args: argparse.Namespace) -> TrainSOBackboneConfig:
    """Construct the training config from a normal CLI interface."""
    if args.preset == "ov123":
        cfg = make_ov123_stage1_config(
            ov1_manifest_path=args.ov1_manifest,
            ov2_manifest_path=args.ov2_manifest,
            ov3_manifest_path=args.ov3_manifest,
        )
    elif args.preset == "ov23":
        cfg = make_ov23_stage1_config(
            ov2_manifest_path=args.ov2_manifest,
            ov3_manifest_path=args.ov3_manifest,
        )
    elif args.preset == "ov23_spatial":
        cfg = make_ov23_spatial_finetune_config(
            ov2_manifest_path=args.ov2_manifest,
            ov3_manifest_path=args.ov3_manifest,
        )
    elif args.preset == "ov123_spatial":
        cfg = make_ov123_spatial_finetune_config(
            ov1_manifest_path=args.ov1_manifest,
            ov2_manifest_path=args.ov2_manifest,
            ov3_manifest_path=args.ov3_manifest,
        )
    elif args.preset == "ov1":
        cfg = make_ov1_stage1_config(
            ov1_manifest_path=args.ov1_manifest,
        )
    elif args.preset == "ov1_spatial":
        cfg = make_ov1_spatial_finetune_config(
            ov1_manifest_path=args.ov1_manifest,
        )
    elif args.preset == "ov1_ast":
        cfg = make_ov1_ast_config(
            ov1_manifest_path=args.ov1_manifest,
        )
    elif args.preset == "ov1_ast_classwarmup":
        cfg = make_ov1_ast_classwarmup_config(
            ov1_manifest_path=args.ov1_manifest,
        )
    elif args.preset == "ov1_ast_spatial":
        cfg = make_ov1_ast_spatial_config(
            ov1_manifest_path=args.ov1_manifest,
        )
    elif args.preset == "ov1_ast_balanced":
        cfg = make_ov1_ast_balanced_config(
            ov1_manifest_path=args.ov1_manifest,
        )
    elif args.preset == "ov1_pretrunk_ast_class":
        cfg = make_ov1_pretrunk_ast_class_config(
            ov1_manifest_path=args.ov1_manifest,
        )
    elif args.preset == "ov1_pretrunk_ast_phase0":
        cfg = make_ov1_pretrunk_ast_phase0_config(
            ov1_manifest_path=args.ov1_manifest,
        )
    elif args.preset == "ov1_pretrunk_ast_spatial":
        cfg = make_ov1_pretrunk_ast_spatial_config(
            ov1_manifest_path=args.ov1_manifest,
        )
    elif args.preset == "so_encoder":
        cfg = make_so_encoder_config(
            train_manifest_path=(
                args.train_manifest
                or args.unified_train_manifest
                or args.ov1_manifest
            ),
            valid_manifest_path=(args.valid_manifest or args.unified_valid_manifest),
        )
    else:
        raise ValueError(f"Unsupported preset: {args.preset}")

    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.amp is not None:
        cfg.amp_dtype = args.amp
    if args.num_epochs is not None:
        cfg.num_epochs = args.num_epochs
    if args.learning_rate is not None:
        cfg.learning_rate = args.learning_rate
    if args.weight_decay is not None:
        cfg.weight_decay = args.weight_decay
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if getattr(args, "source_vocab_path", None):
        cfg.dataset.source_vocab.vocab_path = args.source_vocab_path
        cfg.model.source_vocab_path = args.source_vocab_path
    if getattr(args, "source_num_classes", None) is not None:
        cfg.dataset.source_vocab.num_classes = args.source_num_classes
        cfg.model.source_num_classes = args.source_num_classes
    if getattr(args, "pretrained_beats_ckpt", None):
        cfg.pretrained_beats_ckpt = args.pretrained_beats_ckpt
    if args.class_finetuned_ckpt is not None:
        cfg.class_finetuned_ckpt = args.class_finetuned_ckpt
    if getattr(args, "trunk_finetuned_ckpt", None) is not None:
        cfg.trunk_finetuned_ckpt = args.trunk_finetuned_ckpt
    if args.init_from_spatial_ckpt is not None:
        cfg.init_from_spatial_ckpt = args.init_from_spatial_ckpt
    if args.resume is not None:
        cfg.resume_from_checkpoint = args.resume
    if args.no_resume_optimizer:
        cfg.load_optimizer_state_on_resume = False
    if args.reset_epoch_on_resume:
        cfg.reset_epoch_on_resume = True
    if args.reset_best_on_resume:
        cfg.reset_best_metric_on_resume = True
    if args.crop_mode is not None:
        cfg.dataset.crop_mode = args.crop_mode
    if args.max_clip_duration_seconds is not None:
        cfg.dataset.max_clip_duration_seconds = args.max_clip_duration_seconds
    if args.save_every_n_epochs is not None:
        cfg.save_every_n_epochs = args.save_every_n_epochs
    if args.train_projector_in_stage1:
        cfg.train_projector_in_stage1 = True
        cfg.freeze_projector_by_default = False
    if args.freeze_trunk:
        cfg.unfreeze_full_trunk = False
    if args.no_progress:
        cfg.show_progress_bars = False
    if args.distributed:
        cfg.distributed = True
    if args.local_rank is not None:
        cfg.local_rank = args.local_rank
    if args.distributed_backend is not None:
        cfg.distributed_backend = args.distributed_backend
    if args.ddp_find_unused_parameters:
        cfg.ddp_find_unused_parameters = True

    return cfg


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for direct training launches."""
    parser = argparse.ArgumentParser(description="Train SO-Encoder.")
    parser.add_argument(
        "--preset",
        choices=(
            "ov123",
            "ov23",
            "ov123_spatial",
            "ov23_spatial",
            "ov1",
            "ov1_spatial",
            "ov1_ast",
            "ov1_ast_classwarmup",
            "ov1_ast_spatial",
            "ov1_ast_balanced",
            "ov1_pretrunk_ast_class",
            "ov1_pretrunk_ast_phase0",
            "ov1_pretrunk_ast_spatial",
            "so_encoder",
        ),
        default="so_encoder",
    )
    parser.add_argument(
        "--train-manifest",
        default="",
        help="Path to the training manifest (.jsonl). "
             "Records follow SO-Dataset's metadata schema.",
    )
    parser.add_argument(
        "--valid-manifest",
        default="",
        help="Path to the validation manifest (.jsonl).",
    )
    # Back-compat aliases (kept so older shell scripts and the ov*/ast/pretrunk
    # presets that take separate ov1/ov2/ov3 split paths still work).
    parser.add_argument("--ov1-manifest", default="")
    parser.add_argument("--ov2-manifest", default="")
    parser.add_argument("--ov3-manifest", default="")
    parser.add_argument("--ov1-real-manifest", default="")
    parser.add_argument("--ov2-real-manifest", default="")
    parser.add_argument("--ov3-real-manifest", default="")
    parser.add_argument("--unified-train-manifest", default="")
    parser.add_argument("--unified-valid-manifest", default="")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--amp",
        choices=("fp32", "bf16", "fp16"),
        default=None,
        help="Mixed precision mode for forward/loss. Default fp32 (no autocast).",
    )
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--source-vocab-path", type=str, default=os.environ.get("SO_VOCAB", ""),
        help="Path to source-class vocabulary CSV (columns: label_id, "
             "final_label). Falls back to env SO_VOCAB. SO-Dataset = 63 classes.",
    )
    parser.add_argument(
        "--source-num-classes", type=int, default=None,
        help="Number of source classes; must match the vocab CSV row count.",
    )
    parser.add_argument(
        "--pretrained-beats-ckpt", type=str,
        default=os.environ.get(
            "SO_BEATS_TRUNK_CKPT",
            "pretrain_ckpt/BEATs_iter3_plus_AS2M.pt/BEATs_iter3_plus_AS2M.pt",
        ),
        help="Upstream BEATs trunk checkpoint (BEATs_iter3_plus_AS2M.pt). "
             "Falls back to env SO_BEATS_TRUNK_CKPT.",
    )
    parser.add_argument("--class-finetuned-ckpt", type=str, default=None)
    parser.add_argument(
        "--trunk-finetuned-ckpt",
        type=str,
        default=None,
        help="Optional path to a BEATs trunk-only fine-tune checkpoint "
        "(produced by train_beats_multilabel_trunk.py) to hot-start the "
        "trunk. Loaded after load_beats_pretrained; keys under 'beats_only' "
        "are copied over the AS2M baseline where shapes match.",
    )
    parser.add_argument(
        "--init-from-spatial-ckpt",
        type=str,
        default=None,
        help="Optional prior SOBackbone checkpoint to warm-start the trunk "
        "and local_spatial fusion weights (used by ov123 frame-level presets).",
    )
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no-resume-optimizer", action="store_true")
    parser.add_argument("--reset-epoch-on-resume", action="store_true")
    parser.add_argument("--reset-best-on-resume", action="store_true")
    parser.add_argument("--crop-mode", choices=("none", "start", "center", "random"), default=None)
    parser.add_argument("--max-clip-duration-seconds", type=float, default=None)
    parser.add_argument("--save-every-n-epochs", type=int, default=None)
    parser.add_argument("--train-projector-in-stage1", action="store_true")
    parser.add_argument("--freeze-trunk", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--local-rank", "--local_rank", dest="local_rank", type=int, default=None)
    parser.add_argument("--distributed-backend", type=str, default=None)
    parser.add_argument("--ddp-find-unused-parameters", action="store_true")
    return parser.parse_args()


def _move_batch_to_device(batch: SpatialBatch, device: torch.device) -> SpatialBatch:
    return SpatialBatch(
        waveform=batch.waveform.to(device),
        waveform_padding_mask=batch.waveform_padding_mask.to(device)
        if batch.waveform_padding_mask is not None
        else None,
        clip_duration_seconds=batch.clip_duration_seconds.to(device),
        target_num_steps=batch.target_num_steps.to(device),
        source_class_indices=batch.source_class_indices.to(device),
        source_azimuth_deg=batch.source_azimuth_deg.to(device),
        source_elevation_deg=batch.source_elevation_deg.to(device),
        source_distance=batch.source_distance.to(device),
        source_distance_valid=batch.source_distance_valid.to(device),
        source_ele_sign_only=batch.source_ele_sign_only.to(device)
        if hasattr(batch, "source_ele_sign_only") and batch.source_ele_sign_only is not None
        else None,
        source_start_time_seconds=batch.source_start_time_seconds.to(device),
        source_end_time_seconds=batch.source_end_time_seconds.to(device),
        source_valid_mask=batch.source_valid_mask.to(device),
        sample_ids=batch.sample_ids,
        source_class_labels=batch.source_class_labels,
    )


def _init_running_metrics() -> Dict[str, float]:
    """Create a metric accumulator shared across training modes."""
    return {
        "loss_total": 0.0,
        "loss_activity": 0.0,
        "loss_azi": 0.0,
        "loss_ele": 0.0,
        "loss_dist": 0.0,
        "loss_cls_aux": 0.0,
        "loss_temp": 0.0,
        "loss_direction": 0.0,
        "activity_acc": 0.0,
        "activity_precision": 0.0,
        "activity_recall": 0.0,
        "class_acc": 0.0,
        "azi_mae_deg": 0.0,
        "ele_mae_deg": 0.0,
        "dist_mae": 0.0,
        "matched_count": 0.0,
        # Tier-2 oracle metrics (frame_track route only)
        "oracle_class_acc": 0.0,
        "oracle_azi_mae_deg": 0.0,
        "oracle_ele_mae_deg": 0.0,
        "oracle_dist_mae": 0.0,
    }


def _infer_frame_track_csv_group(sample_id: str) -> str:
    """Infer ov-family from the manifest-derived sample id."""
    sid = sample_id.lower()
    if "hm3d" in sid:
        return "ov1"
    if "ov2_" in sid:
        return "ov2"
    if "ov3_" in sid:
        return "ov3"
    return "other"


def _append_frame_track_csv_samples(
    csv_samples: List[Dict[str, object]],
    rows_for_batch: Sequence[Dict[str, object]],
    total_quota: int,
    per_group_quota: int,
    group_counts: Dict[str, int],
) -> None:
    """Append CSV dump samples with optional ov1/ov2/ov3 balancing.

    Quota semantics:
      total_quota <= 0      → unlimited (dump every sample passed in).
      per_group_quota <= 0  → no per-group cap (only the total cap applies).
    This lets presets opt into "dump the entire validation set" by setting
    ``frame_track_csv_max_samples_per_epoch = 0``.
    """
    unlimited_total = total_quota <= 0
    unlimited_per_group = per_group_quota <= 0
    for row in rows_for_batch:
        if (not unlimited_total) and len(csv_samples) >= total_quota:
            break
        if not unlimited_per_group:
            group = _infer_frame_track_csv_group(str(row["sample_id"]))
            if group not in ("ov1", "ov2", "ov3"):
                continue
            if group_counts.get(group, 0) >= per_group_quota:
                continue
            group_counts[group] = group_counts.get(group, 0) + 1
        csv_samples.append(row)


def _amp_context(amp_dtype: str):
    """Return an autocast context for the requested dtype, or nullcontext for fp32."""
    if amp_dtype == "bf16":
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    if amp_dtype == "fp16":
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def run_train_step(
    model: SOBackbone,
    batch: SpatialBatch,
    loss_cfg: SpatialLossConfig,
) -> Tuple[SOBackboneOutput, object, SpatialLossOutput]:
    """Run one forward-and-loss pass for stage-1 encoder-only training.

    Expected flow:
        1. model.forward(batch.waveform, ...)
        2. fixed-slot matching
        3. multi-task spatial loss computation

    Returns:
        Tuple[SOBackboneOutput, SpatialLossOutput]:
            Model outputs and structured loss outputs for the current batch.
    """
    mono_window_mask = None
    if loss_cfg.supervision_mode == "mono_ast":
        mono_window_mask = build_primary_source_window_mask(
            batch=batch,
            t_s_max=int(batch.target_num_steps.max().item()),
        ).to(batch.waveform.device)
    model_output = model(
        waveform=batch.waveform,
        padding_mask=batch.waveform_padding_mask,
        clip_duration_seconds=batch.clip_duration_seconds,
        mono_window_mask=mono_window_mask,
    )
    if loss_cfg.supervision_mode == "mono_ast":
        if model_output.mono_prediction_output is None:
            raise RuntimeError("mono_ast supervision requires mono_prediction_output.")
        matching_result = None
        loss_output = compute_mono_ast_losses(
            prediction_output=model_output.mono_prediction_output,
            batch=batch,
            config=loss_cfg,
        )
        # Optional parallel frame-level track supervision.
        if (
            loss_cfg.enable_frame_track_loss
            and model_output.frame_track_prediction_output is not None
        ):
            frame_track_loss = compute_frame_track_losses(
                prediction_output=model_output.frame_track_prediction_output,
                batch=batch,
                temporal_padding_mask=model_output.temporal_padding_mask,
                config=loss_cfg,
            )
            # Add frame track loss to the total.
            loss_output = SpatialLossOutput(
                loss_total=loss_output.loss_total + frame_track_loss.loss_total,
                loss_activity=loss_output.loss_activity + frame_track_loss.loss_activity,
                loss_azi=loss_output.loss_azi,
                loss_ele=loss_output.loss_ele,
                loss_dist=loss_output.loss_dist + frame_track_loss.loss_dist,
                loss_cls_aux=loss_output.loss_cls_aux + frame_track_loss.loss_cls_aux,
                loss_temp=loss_output.loss_temp,
                loss_direction=loss_output.loss_direction + frame_track_loss.loss_direction,
            )
    elif loss_cfg.supervision_mode == "pretrunk_ast":
        if model_output.pretrunk_prediction_output is None:
            raise RuntimeError("pretrunk_ast supervision requires pretrunk_prediction_output.")
        matching_result = None
        loss_output = compute_pretrunk_ast_losses(
            prediction_output=model_output.pretrunk_prediction_output,
            batch=batch,
            config=loss_cfg,
        )
    elif loss_cfg.supervision_mode == "local_spatial_slot":
        if model_output.frame_slot_prediction_output is None:
            raise RuntimeError("local_spatial_slot supervision requires frame_slot_prediction_output.")
        matching_result = None
        loss_output = compute_frame_slot_losses(
            prediction_output=model_output.frame_slot_prediction_output,
            batch=batch,
            temporal_padding_mask=model_output.temporal_padding_mask,
            config=loss_cfg,
            clip_aux_prediction=model_output.clip_aux_prediction_output,
        )
    elif loss_cfg.supervision_mode == "local_spatial_track":
        if model_output.frame_track_prediction_output is None:
            raise RuntimeError("local_spatial_track supervision requires frame_track_prediction_output.")
        matching_result = None
        loss_output = compute_frame_track_losses(
            prediction_output=model_output.frame_track_prediction_output,
            batch=batch,
            temporal_padding_mask=model_output.temporal_padding_mask,
            config=loss_cfg,
        )
    elif loss_cfg.supervision_mode == "local_spatial_accdoa":
        if model_output.frame_accdoa_prediction_output is None:
            raise RuntimeError("local_spatial_accdoa supervision requires frame_accdoa_prediction_output.")
        matching_result = None
        loss_output = compute_frame_accdoa_losses(
            prediction_output=model_output.frame_accdoa_prediction_output,
            batch=batch,
            temporal_padding_mask=model_output.temporal_padding_mask,
            config=loss_cfg,
            clip_aux_prediction=model_output.clip_aux_prediction_output,
        )
    elif loss_cfg.supervision_mode == "local_spatial_framewise":
        if model_output.frame_wise_prediction_output is None:
            raise RuntimeError("local_spatial_framewise supervision requires frame_wise_prediction_output.")
        matching_result = None
        loss_output = compute_framewise_losses(
            prediction_output=model_output.frame_wise_prediction_output,
            batch=batch,
            config=loss_cfg,
            temporal_padding_mask=model_output.temporal_padding_mask,
            clip_aux_prediction=model_output.clip_aux_prediction_output,
        )
    else:
        matching_result = match_fixed_slots(
            prediction_output=model_output.prediction_output,
            batch=batch,
            temporal_padding_mask=model_output.temporal_padding_mask,
            config=loss_cfg,
        )
        loss_output = compute_spatial_losses(
            prediction_output=model_output.prediction_output,
            matching_result=matching_result,
            batch=batch,
            temporal_padding_mask=model_output.temporal_padding_mask,
            config=loss_cfg,
        )
    return model_output, matching_result, loss_output


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    train_cfg: TrainSOBackboneConfig,
    ema_model: Optional["EMAModel"] = None,
) -> Dict[str, float]:
    """Run one training epoch and return aggregated metrics.

    When ``ema_model`` is supplied, its shadow is updated after each optimizer
    step ([D-6]).
    """
    model.train()
    device = next(model.parameters()).device
    running = _init_running_metrics()
    num_batches = 0
    progress = tqdm(
        train_loader,
        total=len(train_loader),
        desc="Train",
        leave=False,
        disable=not (train_cfg.show_progress_bars and _is_main_process()),
    )

    for batch in progress:
        batch = _move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with _amp_context(train_cfg.amp_dtype):
            model_output, matching_result, loss_output = run_train_step(model, batch, train_cfg.loss)
        if train_cfg.loss.supervision_mode == "mono_ast":
            metric_output = compute_mono_ast_validation_metrics(
                prediction_output=model_output.mono_prediction_output,
                batch=batch,
            )
        elif train_cfg.loss.supervision_mode == "pretrunk_ast":
            metric_output = compute_pretrunk_ast_validation_metrics(
                prediction_output=model_output.pretrunk_prediction_output,
                batch=batch,
                config=train_cfg.loss,
            )
        elif train_cfg.loss.supervision_mode == "local_spatial_slot":
            metric_output = compute_frame_slot_validation_metrics(
                prediction_output=model_output.frame_slot_prediction_output,
                batch=batch,
                temporal_padding_mask=model_output.temporal_padding_mask,
                config=train_cfg.loss,
            )
        elif train_cfg.loss.supervision_mode == "local_spatial_track":
            metric_output = compute_frame_track_validation_metrics(
                prediction_output=model_output.frame_track_prediction_output,
                batch=batch,
                temporal_padding_mask=model_output.temporal_padding_mask,
                config=train_cfg.loss,
            )
        elif train_cfg.loss.supervision_mode == "local_spatial_accdoa":
            metric_output = compute_frame_accdoa_validation_metrics(
                prediction_output=model_output.frame_accdoa_prediction_output,
                batch=batch,
                temporal_padding_mask=model_output.temporal_padding_mask,
                config=train_cfg.loss,
            )
        elif train_cfg.loss.supervision_mode == "local_spatial_framewise":
            metric_output = compute_framewise_validation_metrics(
                prediction_output=model_output.frame_wise_prediction_output,
                batch=batch,
                temporal_padding_mask=model_output.temporal_padding_mask,
            )
        else:
            metric_output = compute_spatial_validation_metrics(
                prediction_output=model_output.prediction_output,
                matching_result=matching_result,
                batch=batch,
                temporal_padding_mask=model_output.temporal_padding_mask,
            )
        loss_output.loss_total.backward()
        optimizer.step()
        # v13_D [D-6]: update EMA shadow after each optimizer step
        if ema_model is not None:
            ema_model.update(model)

        running["loss_total"] += float(loss_output.loss_total.item())
        running["loss_activity"] += float(loss_output.loss_activity.item())
        running["loss_azi"] += float(loss_output.loss_azi.item())
        running["loss_ele"] += float(loss_output.loss_ele.item())
        running["loss_dist"] += float(loss_output.loss_dist.item())
        running["loss_cls_aux"] += float(loss_output.loss_cls_aux.item())
        running["loss_temp"] += float(loss_output.loss_temp.item())
        running["loss_direction"] += float(loss_output.loss_direction.item())
        running["activity_acc"] += float(metric_output.activity_acc.item())
        running["activity_precision"] += float(metric_output.activity_precision.item())
        running["activity_recall"] += float(metric_output.activity_recall.item())
        running["class_acc"] += float(metric_output.class_acc.item())
        running["azi_mae_deg"] += float(metric_output.azi_mae_deg.item())
        running["ele_mae_deg"] += float(metric_output.ele_mae_deg.item())
        running["dist_mae"] += float(metric_output.dist_mae.item())
        running["matched_count"] += float(metric_output.matched_count.item())
        if hasattr(metric_output, "oracle_class_acc"):
            running["oracle_class_acc"] += float(metric_output.oracle_class_acc.item())
            running["oracle_azi_mae_deg"] += float(metric_output.oracle_azi_mae_deg.item())
            running["oracle_ele_mae_deg"] += float(metric_output.oracle_ele_mae_deg.item())
            running["oracle_dist_mae"] += float(metric_output.oracle_dist_mae.item())
        num_batches += 1
        if train_cfg.loss.supervision_mode == "local_spatial_track":
            # Two rows of per-frame metrics:
            #   - tier-1 (gated): cls/azi/ele/dist → same semantics as valid CSV
            #     (activity>=0.5 ∧ training matcher). These are what will appear
            #     in the epoch summary and can be compared 1:1 to validation
            #     CSV cls_ok / pred_azi_* / pred_dist columns.
            #   - oracle + sep: kept for diagnostics (upper bound on class head
            #     quality and activity separation proxy).
            postfix: Dict[str, str] = {
                "loss": f"{loss_output.loss_total.item():.4f}",
                "cls":  f"{metric_output.class_acc.item():.3f}",
                "azi":  f"{metric_output.azi_mae_deg.item():.1f}°",
                "ele":  f"{metric_output.ele_mae_deg.item():.1f}°",
                "dist": f"{metric_output.dist_mae.item():.2f}m",
                "ocls": f"{metric_output.oracle_class_acc.item():.3f}",
                "sep":  f"{metric_output.activity_acc.item():.3f}",
            }
        else:
            postfix = {
                "loss": f"{loss_output.loss_total.item():.4f}",
                "sep": f"{metric_output.activity_acc.item():.3f}",   # separation = active_mean - inactive_mean
                "cls": f"{metric_output.oracle_class_acc.item():.3f}" if hasattr(metric_output, "oracle_class_acc") else f"{metric_output.class_acc.item():.3f}",
                "azi": f"{metric_output.oracle_azi_mae_deg.item():.1f}°" if hasattr(metric_output, "oracle_azi_mae_deg") else f"{metric_output.azi_mae_deg.item():.1f}°",
            }
        if float(loss_output.loss_temp.item()) > 1e-6 and train_cfg.loss.supervision_mode != "local_spatial_track":
            postfix["anc"] = f"{loss_output.loss_temp.item():.4f}"
        progress.set_postfix(postfix)

    running, num_batches = _reduce_metric_sums(running, num_batches, device)
    if num_batches == 0:
        return running
    return {key: value / num_batches for key, value in running.items()}


def evaluate_one_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    train_cfg: TrainSOBackboneConfig,
) -> Tuple[Dict[str, float], List[Dict[str, object]], List[Dict[str, object]]]:
    """Run one validation epoch and return aggregated metrics.

    Returns (metrics, qualitative_examples, frame_track_csv_samples).
    The third element is non-empty only when local_spatial_track supervision
    is active and `dump_frame_track_csv` is enabled.
    """
    model.eval()
    device = next(model.parameters()).device
    running = _init_running_metrics()
    num_batches = 0
    examples: List[Dict[str, object]] = []
    csv_samples: List[Dict[str, object]] = []
    # CSV dump is enabled whenever dump_frame_track_csv is True in track mode.
    # Quota semantics (matches _append_frame_track_csv_samples):
    #   frame_track_csv_max_samples_per_epoch <= 0  → unlimited (dump ALL
    #     validation samples); we encode that internally with csv_quota = -1
    #     so callers downstream can still treat it as "no cap".
    #   > 0                                         → cap at that many samples.
    csv_dump_enabled = (
        train_cfg.dump_frame_track_csv
        and train_cfg.loss.supervision_mode == "local_spatial_track"
    )
    if csv_dump_enabled:
        _raw_total = int(train_cfg.frame_track_csv_max_samples_per_epoch)
        csv_quota = _raw_total if _raw_total > 0 else -1  # -1 = unlimited
        _raw_group = int(train_cfg.frame_track_csv_max_samples_per_group)
        csv_group_quota = _raw_group if _raw_group > 0 else -1
    else:
        csv_quota = 0
        csv_group_quota = 0
    csv_group_counts: Dict[str, int] = {}

    # DCASE SELD accumulators:
    #   - mono_ast / pretrunk_ast: legacy single-source scalar accumulator
    #   - local_spatial_track: official DCASE evaluator adapter
    is_mono_mode = train_cfg.loss.supervision_mode in ("mono_ast", "pretrunk_ast")
    is_frame_track_mode = train_cfg.loss.supervision_mode == "local_spatial_track"
    seld_acc = None
    if is_mono_mode:
        seld_acc = SELDMetricsAccumulator()
    elif is_frame_track_mode:
        seld_acc = OfficialDCASEMetricsAccumulator()

    with torch.no_grad():
        progress = tqdm(
            val_loader,
            total=len(val_loader),
            desc="Valid",
            leave=False,
            disable=not (train_cfg.show_progress_bars and _is_main_process()),
        )
        for batch in progress:
            batch = _move_batch_to_device(batch, device)
            with _amp_context(train_cfg.amp_dtype):
                model_output, matching_result, loss_output = run_train_step(model, batch, train_cfg.loss)
            if train_cfg.loss.supervision_mode == "mono_ast":
                metric_output = compute_mono_ast_validation_metrics(
                    prediction_output=model_output.mono_prediction_output,
                    batch=batch,
                )
                if seld_acc is not None and _is_main_process():
                    accumulate_mono_ast_seld(
                        prediction_output=model_output.mono_prediction_output,
                        batch=batch,
                        accumulator=seld_acc,
                    )
            elif train_cfg.loss.supervision_mode == "pretrunk_ast":
                metric_output = compute_pretrunk_ast_validation_metrics(
                    prediction_output=model_output.pretrunk_prediction_output,
                    batch=batch,
                    config=train_cfg.loss,
                )
            elif train_cfg.loss.supervision_mode == "local_spatial_slot":
                metric_output = compute_frame_slot_validation_metrics(
                    prediction_output=model_output.frame_slot_prediction_output,
                    batch=batch,
                    temporal_padding_mask=model_output.temporal_padding_mask,
                    config=train_cfg.loss,
                )
            elif train_cfg.loss.supervision_mode == "local_spatial_track":
                metric_output = compute_frame_track_validation_metrics(
                    prediction_output=model_output.frame_track_prediction_output,
                    batch=batch,
                    temporal_padding_mask=model_output.temporal_padding_mask,
                    config=train_cfg.loss,
                )
                if seld_acc is not None:
                    accumulate_frame_track_seld(
                        prediction_output=model_output.frame_track_prediction_output,
                        batch=batch,
                        temporal_padding_mask=model_output.temporal_padding_mask,
                        accumulator=seld_acc,
                        activity_threshold=0.5,
                        # v13_E: OR top-K̂ gate into the SELD evaluator when
                        # the num_active head is enabled in loss supervision.
                        use_num_active_gate=bool(
                            getattr(train_cfg.loss, "lambda_frame_num_active", 0.0) > 0.0
                            and getattr(train_cfg.model, "use_num_active_head", False)
                        ),
                    )
            elif train_cfg.loss.supervision_mode == "local_spatial_accdoa":
                metric_output = compute_frame_accdoa_validation_metrics(
                    prediction_output=model_output.frame_accdoa_prediction_output,
                    batch=batch,
                    temporal_padding_mask=model_output.temporal_padding_mask,
                    config=train_cfg.loss,
                )
            elif train_cfg.loss.supervision_mode == "local_spatial_framewise":
                metric_output = compute_framewise_validation_metrics(
                    prediction_output=model_output.frame_wise_prediction_output,
                    batch=batch,
                    temporal_padding_mask=model_output.temporal_padding_mask,
                )
            else:
                metric_output = compute_spatial_validation_metrics(
                    prediction_output=model_output.prediction_output,
                    matching_result=matching_result,
                    batch=batch,
                    temporal_padding_mask=model_output.temporal_padding_mask,
                )
            running["loss_total"] += float(loss_output.loss_total.item())
            running["loss_activity"] += float(loss_output.loss_activity.item())
            running["loss_azi"] += float(loss_output.loss_azi.item())
            running["loss_ele"] += float(loss_output.loss_ele.item())
            running["loss_dist"] += float(loss_output.loss_dist.item())
            running["loss_cls_aux"] += float(loss_output.loss_cls_aux.item())
            running["loss_temp"] += float(loss_output.loss_temp.item())
            running["loss_direction"] += float(loss_output.loss_direction.item())
            running["activity_acc"] += float(metric_output.activity_acc.item())
            running["activity_precision"] += float(metric_output.activity_precision.item())
            running["activity_recall"] += float(metric_output.activity_recall.item())
            running["class_acc"] += float(metric_output.class_acc.item())
            running["azi_mae_deg"] += float(metric_output.azi_mae_deg.item())
            running["ele_mae_deg"] += float(metric_output.ele_mae_deg.item())
            running["dist_mae"] += float(metric_output.dist_mae.item())
            running["matched_count"] += float(metric_output.matched_count.item())
            if hasattr(metric_output, "oracle_class_acc"):
                running["oracle_class_acc"] += float(metric_output.oracle_class_acc.item())
                running["oracle_azi_mae_deg"] += float(metric_output.oracle_azi_mae_deg.item())
                running["oracle_ele_mae_deg"] += float(metric_output.oracle_ele_mae_deg.item())
                running["oracle_dist_mae"] += float(metric_output.oracle_dist_mae.item())
            num_batches += 1
            if _is_main_process() and len(examples) < train_cfg.num_val_prediction_examples:
                remaining = train_cfg.num_val_prediction_examples - len(examples)
                if train_cfg.loss.supervision_mode == "mono_ast":
                    examples.extend(
                        build_mono_ast_validation_examples(
                            prediction_output=model_output.mono_prediction_output,
                            batch=batch,
                            max_examples=remaining,
                        )
                    )
                    # Also dump frame-track examples when parallel frame head is on
                    if (
                        train_cfg.loss.enable_frame_track_loss
                        and model_output.frame_track_prediction_output is not None
                    ):
                        examples.extend(
                            build_frame_track_validation_examples(
                                prediction_output=model_output.frame_track_prediction_output,
                                batch=batch,
                                temporal_padding_mask=model_output.temporal_padding_mask,
                                config=train_cfg.loss,
                                max_examples=remaining,
                            )
                        )
                elif train_cfg.loss.supervision_mode == "pretrunk_ast":
                    examples.extend(
                        build_pretrunk_ast_validation_examples(
                            prediction_output=model_output.pretrunk_prediction_output,
                            batch=batch,
                            config=train_cfg.loss,
                            max_examples=remaining,
                        )
                    )
                elif train_cfg.loss.supervision_mode == "local_spatial_slot":
                    examples.extend(
                        build_frame_slot_validation_examples(
                            prediction_output=model_output.frame_slot_prediction_output,
                            batch=batch,
                            temporal_padding_mask=model_output.temporal_padding_mask,
                            max_examples=remaining,
                        )
                    )
                elif train_cfg.loss.supervision_mode == "local_spatial_track":
                    examples.extend(
                        build_frame_track_validation_examples(
                            prediction_output=model_output.frame_track_prediction_output,
                            batch=batch,
                            temporal_padding_mask=model_output.temporal_padding_mask,
                            config=train_cfg.loss,
                            max_examples=remaining,
                        )
                    )
                elif train_cfg.loss.supervision_mode == "local_spatial_accdoa":
                    examples.extend(
                        build_frame_accdoa_validation_examples(
                            prediction_output=model_output.frame_accdoa_prediction_output,
                            batch=batch,
                            temporal_padding_mask=model_output.temporal_padding_mask,
                            config=train_cfg.loss,
                            max_examples=remaining,
                        )
                    )
                elif train_cfg.loss.supervision_mode == "local_spatial_framewise":
                    examples.extend(
                        build_framewise_validation_examples(
                            prediction_output=model_output.frame_wise_prediction_output,
                            batch=batch,
                            temporal_padding_mask=model_output.temporal_padding_mask,
                            max_examples=remaining,
                        )
                    )
                else:
                    examples.extend(
                        build_validation_examples(
                            prediction_output=model_output.prediction_output,
                            matching_result=matching_result,
                            batch=batch,
                            max_examples=remaining,
                        )
                    )
            if (
                _is_main_process()
                and csv_dump_enabled
                and (csv_quota < 0 or len(csv_samples) < csv_quota)
                and model_output.frame_track_prediction_output is not None
            ):
                rows_for_batch = collect_frame_track_csv_rows(
                    prediction_output=model_output.frame_track_prediction_output,
                    batch=batch,
                    temporal_padding_mask=model_output.temporal_padding_mask,
                )
                _append_frame_track_csv_samples(
                    csv_samples=csv_samples,
                    rows_for_batch=rows_for_batch,
                    total_quota=csv_quota,
                    per_group_quota=csv_group_quota,
                    group_counts=csv_group_counts,
                )
            if train_cfg.loss.supervision_mode == "local_spatial_track":
                progress.set_postfix(
                    loss=f"{loss_output.loss_total.item():.4f}",
                    act_on=f"{metric_output.activity_precision.item():.3f}",
                    act_off=f"{metric_output.activity_recall.item():.3f}",
                    sep=f"{metric_output.activity_acc.item():.3f}",
                    ocls=f"{metric_output.oracle_class_acc.item():.3f}",
                    oazi=f"{metric_output.oracle_azi_mae_deg.item():.1f}°",
                    oele=f"{metric_output.oracle_ele_mae_deg.item():.1f}°",
                )
            else:
                progress.set_postfix(
                    loss=f"{loss_output.loss_total.item():.4f}",
                    sep=f"{metric_output.activity_acc.item():.3f}",
                    cls=f"{metric_output.oracle_class_acc.item():.3f}" if hasattr(metric_output, "oracle_class_acc") else f"{metric_output.class_acc.item():.3f}",
                    azi=f"{metric_output.oracle_azi_mae_deg.item():.1f}°" if hasattr(metric_output, "oracle_azi_mae_deg") else f"{metric_output.azi_mae_deg.item():.1f}°",
                )

    running, num_batches = _reduce_metric_sums(running, num_batches, device)
    if num_batches == 0:
        return running, examples, csv_samples
    metrics = {key: value / num_batches for key, value in running.items()}
    if seld_acc is not None:
        # Sum SELD counters across all DDP ranks so every rank agrees on the
        # global validation metric used for best-checkpoint selection.
        seld_acc.all_reduce(device=device)
        metrics.update(seld_acc.compute())
    return metrics, examples, csv_samples


def dump_validation_examples(
    output_dir: str,
    epoch: int,
    examples: Sequence[Dict[str, object]],
) -> None:
    """Write a small set of validation predictions for qualitative inspection."""
    if not _is_main_process() or not examples:
        return
    dump_dir = Path(output_dir) / "val_predictions"
    dump_dir.mkdir(parents=True, exist_ok=True)
    path = dump_dir / f"epoch_{epoch:04d}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=True) + "\n")
    _log(f"[Validation] Dump {path}")


def dump_frame_track_csvs(
    output_dir: str,
    epoch: int,
    samples_data: Sequence[Dict[str, object]],
    train_cfg: TrainSOBackboneConfig,
) -> None:
    """Write per-(sample, gt|pred) DCASE-style frame-level CSV pairs.

    Layout: <output_dir>/val_predictions/epoch_XXXX_csv/<sample_id>__{gt,pred}.csv
    Pred file contains all K tracks × all valid frames with `activity_prob`
    so any post-hoc threshold can be applied without re-running validation.
    """
    if not _is_main_process() or not samples_data:
        return
    import csv as _csv

    index_to_label: List[str] = []
    try:
        vocab = load_source_vocabulary(
            train_cfg.dataset.source_vocab, show_progress=False
        )
        index_to_label = list(vocab.get("index_to_label", []))
    except Exception as exc:
        _log(f"[Validation] CSV dump: vocab load failed ({exc}); class_name will be empty.")

    epoch_dir = Path(output_dir) / "val_predictions" / f"epoch_{epoch:04d}_csv"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    # Columns: legacy DCASE-style schema.  `num_active_pred` is a v10 optional
    # field carried only by predicted rows (not GT rows); we include it in the
    # fieldnames list so DictWriter won't reject rows that carry it, and pair
    # it with extrasaction='ignore' as a forward-compat guard against future
    # additional per-row keys.
    columns = [
        "frame_idx",
        "frame_time_s",
        "src_or_track_idx",
        "class_idx",
        "class_name",
        "azimuth_deg",
        "elevation_deg",
        "distance_m",
        "activity_prob",
        "num_active_pred",
    ]
    for entry in samples_data:
        sid = str(entry["sample_id"]).replace("/", "__").replace("\\", "__")
        for kind in ("gt", "pred"):
            rows = list(entry[f"{kind}_rows"])
            for row in rows:
                if not row.get("class_name"):
                    cidx = int(row["class_idx"])
                    if 0 <= cidx < len(index_to_label):
                        row["class_name"] = index_to_label[cidx]
            path = epoch_dir / f"{sid}__{kind}.csv"
            with path.open("w", encoding="utf-8", newline="") as fh:
                writer = _csv.DictWriter(
                    fh, fieldnames=columns, extrasaction="ignore"
                )
                writer.writeheader()
                writer.writerows(rows)
    _log(
        f"[Validation] Dump frame-track CSVs to {epoch_dir} "
        f"({len(samples_data)} samples)"
    )


def main(train_cfg: Optional[TrainSOBackboneConfig] = None) -> None:
    """Entry point for stage-1 Spatial-BEATs training."""
    train_cfg = train_cfg or TrainSOBackboneConfig()
    train_paths = _resolve_manifest_paths(train_cfg.train_manifest_path, train_cfg.train_manifest_paths)
    if not train_paths:
        raise ValueError("At least one train manifest path must be provided.")

    device = initialize_distributed_mode(train_cfg)

    try:
        train_loader, val_loader = build_dataloaders(train_cfg)
        model = build_model(train_cfg)
        _log(f"[Train] Use device: {device}")
        model.to(device)
        if train_cfg.distributed:
            ddp_device_ids = [device.index] if device.type == "cuda" else None
            # frozen 参数（requires_grad=False）需要从 DDP 的 reduction 里排除，
            # 否则 DDP 会等待它们的 gradient 同步导致 "reduction 未完成" 错误。
            # 通过 _ddp_params_and_buffers_to_ignore 告诉 DDP 跳过这些参数。
            frozen_param_names = {
                name for name, p in model.named_parameters() if not p.requires_grad
            }
            model._ddp_params_and_buffers_to_ignore = frozen_param_names
            model = DDP(
                model,
                device_ids=ddp_device_ids,
                output_device=device.index if device.type == "cuda" else None,
                find_unused_parameters=train_cfg.ddp_find_unused_parameters,
            )
        optimizer = build_optimizer(_unwrap_model(model), train_cfg)
        start_epoch = 0
        best_metric_value: Optional[float] = None

        if train_cfg.resume_from_checkpoint:
            start_epoch, best_metric_value, loaded_best_metric_name = load_checkpoint(
                checkpoint_path=train_cfg.resume_from_checkpoint,
                model=model,
                optimizer=optimizer,
                load_optimizer_state=train_cfg.load_optimizer_state_on_resume,
            )
            if train_cfg.reset_epoch_on_resume:
                start_epoch = 0
            if train_cfg.reset_best_metric_on_resume:
                best_metric_value = None
            elif (
                loaded_best_metric_name is not None
                and loaded_best_metric_name != train_cfg.best_metric_name
            ):
                _log(
                    "[Checkpoint] Reset best metric because checkpoint used "
                    f"{loaded_best_metric_name} but current run uses {train_cfg.best_metric_name}"
                )
                best_metric_value = None
            _log(
                f"Resumed from {train_cfg.resume_from_checkpoint} "
                f"at epoch {start_epoch} with best {train_cfg.best_metric_name}={best_metric_value}"
            )

        # v13_D [D-6]: EMA shadow weights (created lazily so the model is
        # already loaded from resume). Validation / best-checkpoint save use
        # the shadow weights; training continues with the live weights.
        ema_model: Optional["EMAModel"] = None
        if getattr(train_cfg, "use_ema", False):
            ema_model = EMAModel(model, decay=float(train_cfg.ema_decay))
            _log(
                f"[EMA] Enabled with decay={train_cfg.ema_decay} "
                f"(start at epoch {train_cfg.ema_start_epoch})"
            )

        for epoch in range(start_epoch, train_cfg.num_epochs):
            if isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)
            if val_loader is not None and isinstance(val_loader.sampler, DistributedSampler):
                val_loader.sampler.set_epoch(epoch)

            # Hungarian class-cost warmup (frame-track supervision only).
            # Linear ramp from 0.0 → frame_match_class_cost_max_weight over
            # [warmup_epochs, warmup_epochs + ramp_epochs). No-op when
            # warmup_epochs == 0.
            _warmup = train_cfg.frame_match_class_cost_warmup_epochs
            if _warmup > 0:
                _ramp = max(1, train_cfg.frame_match_class_cost_ramp_epochs)
                _max_w = train_cfg.frame_match_class_cost_max_weight
                if epoch < _warmup:
                    _class_w = 0.0
                elif epoch < _warmup + _ramp:
                    _class_w = _max_w * (epoch - _warmup + 1) / _ramp
                else:
                    _class_w = _max_w
                train_cfg.loss.frame_match_class_cost_weight = _class_w
                _log(
                    f"[Epoch {epoch}] frame_match_class_cost_weight="
                    f"{_class_w:.3f}"
                )

            # v13_B [B-4] Soft macro-F1 weight warmup.
            # When frame_soft_f1_warmup_epochs > 0, use
            # frame_soft_f1_weight_warmup for ep < warmup, then
            # frame_soft_f1_weight afterwards.
            _f1_warmup = getattr(train_cfg.loss, "frame_soft_f1_warmup_epochs", 0)
            if _f1_warmup > 0:
                _f1_final = float(getattr(train_cfg.loss, "frame_soft_f1_weight", 0.0))
                _f1_warm = float(
                    getattr(train_cfg.loss, "frame_soft_f1_weight_warmup", 0.0)
                )
                _f1_w = _f1_warm if epoch < _f1_warmup else _f1_final
                train_cfg.loss.frame_soft_f1_weight = _f1_w
                _log(f"[Epoch {epoch}] frame_soft_f1_weight={_f1_w:.3f}")

            # Two-stage / gradual spatial loss schedule.
            # Stage 1: dir/dist lambda scaled down (or to 0) so class head
            #   learns on clean signal; cost weights also zeroed so DOA noise
            #   does not drive assignment.
            # Stage 2:
            #   - ramp_epochs == 0: restore full lambda values immediately
            #   - ramp_epochs > 0: linearly ramp lambdas and cost weights
            #     from warmup_scale to 1.0
            _sp_warmup = train_cfg.frame_spatial_loss_warmup_epochs
            if _sp_warmup > 0:
                _sp_scale = train_cfg.frame_spatial_loss_warmup_scale
                _sp_ramp = max(0, train_cfg.frame_spatial_loss_ramp_epochs)
                # Store original full-value lambdas once (before any override).
                if not hasattr(train_cfg, "_full_lambda_dir"):
                    train_cfg._full_lambda_dir  = train_cfg.loss.lambda_frame_direction  # type: ignore[attr-defined]
                    train_cfg._full_lambda_dist = train_cfg.loss.lambda_frame_distance   # type: ignore[attr-defined]
                if epoch < _sp_warmup:
                    _cur_scale = _sp_scale
                    train_cfg.loss.lambda_frame_direction   = train_cfg._full_lambda_dir  * _cur_scale  # type: ignore[attr-defined]
                    train_cfg.loss.lambda_frame_distance    = train_cfg._full_lambda_dist * _cur_scale  # type: ignore[attr-defined]
                    train_cfg.loss.frame_match_dir_cost_weight  = _cur_scale
                    train_cfg.loss.frame_match_dist_cost_weight = _cur_scale
                    _log(
                        f"[Epoch {epoch}] spatial stage 1 (class-warmup): "
                        f"lambda_dir={train_cfg.loss.lambda_frame_direction:.3f}  "
                        f"dir_cost_w={_cur_scale:.3f}"
                    )
                elif _sp_ramp > 0 and epoch < _sp_warmup + _sp_ramp:
                    _progress = (epoch - _sp_warmup + 1) / _sp_ramp
                    _cur_scale = _sp_scale + (1.0 - _sp_scale) * _progress
                    train_cfg.loss.lambda_frame_direction   = train_cfg._full_lambda_dir  * _cur_scale  # type: ignore[attr-defined]
                    train_cfg.loss.lambda_frame_distance    = train_cfg._full_lambda_dist * _cur_scale  # type: ignore[attr-defined]
                    train_cfg.loss.frame_match_dir_cost_weight  = _cur_scale
                    train_cfg.loss.frame_match_dist_cost_weight = _cur_scale
                    _log(
                        f"[Epoch {epoch}] spatial stage 2 (DOA ramp): "
                        f"scale={_cur_scale:.3f}  "
                        f"lambda_dir={train_cfg.loss.lambda_frame_direction:.3f}  "
                        f"lambda_dist={train_cfg.loss.lambda_frame_distance:.3f}"
                    )
                else:
                    train_cfg.loss.lambda_frame_direction   = train_cfg._full_lambda_dir   # type: ignore[attr-defined]
                    train_cfg.loss.lambda_frame_distance    = train_cfg._full_lambda_dist  # type: ignore[attr-defined]
                    train_cfg.loss.frame_match_dir_cost_weight  = 1.0
                    train_cfg.loss.frame_match_dist_cost_weight = 1.0
                    if epoch == _sp_warmup or (_sp_ramp > 0 and epoch == _sp_warmup + _sp_ramp):
                        _log(
                            f"[Epoch {epoch}] spatial stage 3 (DOA full): "
                            f"lambda_dir={train_cfg.loss.lambda_frame_direction:.3f}  "
                            f"lambda_dist={train_cfg.loss.lambda_frame_distance:.3f}"
                        )

            # v9: dynamically override class_head LR during the DOA ramp
            # window.  When class_head_freeze_during_ramp_epochs > 0 the
            # cls_head param group's LR is driven to
            # class_head_lr_scale_during_ramp (default 0.0) for the first N
            # epochs of stage 2 (right after the class-only warmup), then
            # returns to class_head_lr_scale.  Requires
            # class_head_lr_scale != 1.0 so the cls_head group exists.
            _cls_ramp_len = int(train_cfg.class_head_freeze_during_ramp_epochs)
            if _cls_ramp_len > 0 and _sp_warmup > 0 and train_cfg.class_head_lr_scale != 1.0:
                in_ramp = _sp_warmup <= epoch < _sp_warmup + _cls_ramp_len
                if in_ramp:
                    _cls_scale = train_cfg.class_head_lr_scale_during_ramp
                else:
                    _cls_scale = train_cfg.class_head_lr_scale
                for _g in optimizer.param_groups:
                    if _g.get("group_name") == "cls_head":
                        _g["lr"] = train_cfg.learning_rate * _cls_scale
                _log(
                    f"[Epoch {epoch}] cls_head_lr scale={_cls_scale:.3f}  "
                    f"lr={train_cfg.learning_rate * _cls_scale:.2e}  "
                    f"(ramp_window={_sp_warmup}..{_sp_warmup + _cls_ramp_len - 1})"
                )

            _log(f"[Epoch {epoch}] start")
            # v13_D [D-1]: cosine LR schedule (optional)
            if getattr(train_cfg, "use_cosine_lr", False):
                import math as _math
                _warmup_eps = max(0, int(getattr(train_cfg, "cosine_lr_warmup_epochs", 0)))
                _min_ratio = float(getattr(train_cfg, "cosine_lr_min_ratio", 0.0))
                _total_eps = max(1, int(train_cfg.num_epochs))
                _peak_lr = float(train_cfg.learning_rate)
                if epoch < _warmup_eps:
                    # linear warmup 0 → peak
                    _lr_scale = (epoch + 1) / max(1, _warmup_eps)
                else:
                    # cosine from peak → peak*min_ratio
                    progress = (epoch - _warmup_eps) / max(1, _total_eps - _warmup_eps)
                    progress = min(max(progress, 0.0), 1.0)
                    _lr_scale = _min_ratio + 0.5 * (1.0 - _min_ratio) * (
                        1.0 + _math.cos(_math.pi * progress)
                    )
                _new_lr = _peak_lr * _lr_scale
                # Each param_group may have its own lr scale (trunk/spatial/head).
                # We multiply _lr_scale onto each group's *current* scale-multiplier
                # baseline. To make this simple and robust, we store the original
                # lr as "base_lr" on each group on first touch, then set lr = base_lr * scale.
                for pg in optimizer.param_groups:
                    if "base_lr" not in pg:
                        pg["base_lr"] = float(pg["lr"])
                    pg["lr"] = float(pg["base_lr"]) * _lr_scale
                _log(
                    f"[Epoch {epoch}] cosine-LR scale={_lr_scale:.3f}  "
                    f"peak_lr={_peak_lr:.2e}  epoch_lr≈{_new_lr:.2e}"
                )
            # v13_D [D-6]: only pass ema_model once epoch reaches ema_start_epoch
            # so the shadow is not polluted by cls-warmup noise.
            _active_ema = ema_model if (
                ema_model is not None
                and epoch >= int(getattr(train_cfg, "ema_start_epoch", 0))
            ) else None
            train_metrics = train_one_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                train_cfg=train_cfg,
                ema_model=_active_ema,
            )
            _log(f"[Epoch {epoch}] train: {_format_metrics(train_metrics, train_cfg.loss.supervision_mode)}")
            val_metrics = None
            val_examples: List[Dict[str, object]] = []
            val_csv_samples: List[Dict[str, object]] = []
            if val_loader is not None:
                # v13_D [D-6]: swap in EMA shadow for validation (only if EMA
                # has been actively updated this run).
                _ema_backup = None
                if _active_ema is not None:
                    _ema_backup = _active_ema.apply_to(model)
                    _log("[EMA] Validating with shadow weights")
                val_metrics, val_examples, val_csv_samples = evaluate_one_epoch(
                    model=model,
                    val_loader=val_loader,
                    train_cfg=train_cfg,
                )
                if _ema_backup is not None:
                    _active_ema.restore(model, _ema_backup)
                _log(f"[Epoch {epoch}] val:   {_format_metrics(val_metrics, train_cfg.loss.supervision_mode)}")
                if train_cfg.dump_val_predictions:
                    dump_validation_examples(
                        output_dir=train_cfg.output_dir,
                        epoch=epoch,
                        examples=val_examples,
                    )
                if train_cfg.dump_frame_track_csv and val_csv_samples:
                    dump_frame_track_csvs(
                        output_dir=train_cfg.output_dir,
                        epoch=epoch,
                        samples_data=val_csv_samples,
                        train_cfg=train_cfg,
                    )

            reference_metrics = _select_reference_metrics(train_metrics, val_metrics)
            if train_cfg.best_metric_name not in reference_metrics:
                raise KeyError(
                    f"best_metric_name={train_cfg.best_metric_name} was not found in metrics: "
                    f"{sorted(reference_metrics.keys())}"
                )
            current_metric_value = float(reference_metrics[train_cfg.best_metric_name])
            is_best = _is_better_metric(
                candidate=current_metric_value,
                best_so_far=best_metric_value,
                minimize=train_cfg.minimize_best_metric,
            )
            if is_best:
                best_metric_value = current_metric_value

            # v13_D [D-6]: checkpoint saving also uses EMA weights when active,
            # so best.pt / last.pt reflect the validation-time weights.
            _ema_backup2 = None
            if _active_ema is not None:
                _ema_backup2 = _active_ema.apply_to(model)
            _save_epoch_checkpoints(
                model=model,
                optimizer=optimizer,
                train_cfg=train_cfg,
                epoch=epoch,
                best_metric_value=best_metric_value,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                is_best=is_best,
            )
            if _ema_backup2 is not None:
                _active_ema.restore(model, _ema_backup2)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main(build_train_config_from_args(parse_args()))
