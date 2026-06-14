"""Loss and matching skeleton for Spatial-BEATs encoder-only training.

This file defines the target contracts and loss interfaces that supervise the
fixed-slot readout heads. The internal matching and loss computation logic is
left unimplemented on purpose.
"""

from dataclasses import dataclass
import math
import itertools
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .so_modules import (
    FrameACCDOAPredictionOutput,
    FrameSlotPredictionOutput,
    FrameTrackPredictionOutput,
    FrameWisePredictionOutput,
    MonoTaskPredictionOutput,
    PreTrunkASTPredictionOutput,
    SpatialPredictionOutput,
)

if TYPE_CHECKING:
    from spatial_dataset import SpatialBatch


@dataclass
class SpatialLossConfig:
    """Configuration for slot matching and multi-task losses."""

    num_azi_bins: int = 360
    num_ele_bins: int = 180
    num_distance_bins: int = 21
    distance_bin_size_m: float = 0.5
    target_token_rate: float = 2.5
    max_sources: int = 4

    lambda_activity: float = 1.0
    lambda_azi: float = 1.0
    lambda_ele: float = 1.0
    lambda_dist: float = 1.0
    lambda_cls_aux: float = 1.0
    lambda_temp: float = 0.0
    lambda_direction: float = 0.0

    distance_loss_type: str = "smooth_l1"
    activity_loss_type: str = "bce"
    matching_strategy: str = "per_step_fixed_slot"
    supervision_mode: str = "fixed_slot"
    azi_loss_type: str = "soft_circular_ce"
    ele_loss_type: str = "soft_gaussian_ce"
    azi_soft_label_sigma_deg: float = 10.0
    ele_soft_label_sigma_deg: float = 10.0

    # Frame-level multi-source supervision (routes A/B/C). Only used when
    # supervision_mode is one of {"local_spatial_slot", "local_spatial_track",
    # "local_spatial_accdoa"}; OR when enable_frame_track_loss=True alongside
    # mono_ast supervision (parallel clip + frame training).
    enable_frame_track_loss: bool = False
    lambda_frame_activity: float = 1.0
    lambda_frame_class: float = 1.0
    lambda_frame_direction: float = 1.0
    lambda_frame_distance: float = 1.0
    lambda_clip_aux: float = 0.1
    frame_num_slots: int = 4
    frame_accdoa_activity_threshold: float = 0.5
    # ACCDOA direction-only supervision (bugfix).  When >0, adds
    # (1 - cos(pred, target)) on the ACTIVE (b,t,c) cells only, separate from
    # the raw MSE.  This provides a direction-pure gradient that is not
    # diluted by the 98% of inactive (b,t,c) cells in the per-class grid.
    lambda_frame_accdoa_direction: float = 0.0
    # ACCDOA inactive-cell down-weighting (bugfix).  The inactive target is
    # zero and occupies ~98% of the per-class grid; without down-weighting
    # the MSE is dominated by "predict 0 everywhere".  Set <1.0 to shrink
    # the contribution of inactive cells and let active-cell gradient
    # actually drive DOA learning.  1.0 = legacy behavior.
    frame_accdoa_inactive_weight: float = 1.0
    # pos_weight for activity BCE: counter-acts the heavy inactive-frame imbalance.
    # In K=4 track setup with OV1 (1 active source / clip), ~3/4 of tracks are
    # always negative → pos_weight ≈ 3 is a reasonable starting point.
    # Set to 0.0 (default) to disable weighting.
    frame_activity_pos_weight: float = 0.0

    # Focal BCE for frame-track activity head.  When True, replaces the plain
    # BCE-with-logits with a focal variant:
    #     FL = alpha * (1 - p_t)^gamma * CE
    # Targets the "duplicate" failure mode: confidently-wrong negatives
    # (prob≈0.9 on a duplicate slot) are heavily up-weighted, while
    # confidently-correct (prob≈0.05, 0.95) get down-weighted.  pos_weight is
    # still respected when > 0.0.
    frame_activity_use_focal: bool = False
    frame_activity_focal_gamma: float = 2.0
    frame_activity_focal_alpha: float = 0.25

    # Hungarian-cost class-NLL weight for per-frame matching.
    # Set < 1.0 (e.g. 0.0) to disable class contribution to the assignment
    # decision — useful for DETR-style warmup where class predictions are
    # noisy early in training.  Does NOT affect the class CE loss itself.
    frame_match_class_cost_weight: float = 1.0

    # Hungarian-cost dir/dist weights — decoupled from lambda_frame_direction/
    # lambda_frame_distance so that cost and loss can be controlled independently.
    # Set to 0.0 during class-warmup stage so DOA noise does not pollute matching.
    # Default 1.0 preserves existing behavior.
    frame_match_dir_cost_weight: float = 1.0
    frame_match_dist_cost_weight: float = 1.0

    # Segment-level matching: instead of per-frame independent Hungarian, find
    # contiguous segments where the set of active GT sources is constant, then
    # run one Hungarian assignment per segment with continuity preference.
    # Eliminates track identity flip between adjacent frames.
    # Set to True for legacy and later presets.
    use_segment_matching: bool = False

    # ADPIT-style duplicate supervision for single-source frames (A0-A0-A0 trick).
    # When True and a frame has exactly 1 active GT source, that source is
    # assigned to ALL K tracks simultaneously (not just the Hungarian winner).
    # This ensures K-1 "dead" tracks receive positive gradients on single-source
    # frames, directly countering the track-dead failure mode observed in ov1-heavy
    # training.  Does not affect frames with 2+ active sources.
    # WARNING: This is only suitable for ACCDOA/slot-output formats, NOT for
    # track-query decoders that threshold each track independently — it will cause
    # all tracks to fire together on every frame.
    use_adpit_duplicate: bool = False

    # Soft activity regularization for non-winner tracks (legacy).
    # When > 0.0, non-winner tracks (tracks NOT selected by Hungarian matching)
    # receive a soft activity target of `nonwinner_activity_soft_target` on
    # GT-active frames instead of the hard 0.0 target.
    # This gives a weak positive signal to keep K-1 tracks alive without
    # teaching them the class/direction/distance of the source (those targets
    # are only set on the Hungarian winner track).
    # Typical value: 0.05 ~ 0.15.  0.0 = disabled (default, preserves legacy behavior).
    nonwinner_activity_soft_target: float = 0.0

    # Dynamic pos_weight for activity BCE: replaces the fixed frame_activity_pos_weight
    # with a per-batch sqrt(neg/pos) balance, clamped to [1, dynamic_pos_weight_cap].
    # When True, frame_activity_pos_weight is used as a base multiplier.
    # Matches the DCASE baseline's adpit_dynamic_weight strategy.
    use_dynamic_pos_weight: bool = False
    dynamic_pos_weight_cap: float = 20.0

    # Class-weighted cross-entropy for the frame-track class head.
    # When non-empty, must be a list of per-class weights (length = num_classes).
    # Rare classes (aircraft, insect, vehicle) get higher weight to compensate
    # for imbalanced occurrence in ov2/ov3 data.
    # Empty list (default) = uniform weights, identical to current behavior.
    frame_class_loss_weights: List[float] = None  # type: ignore[assignment]

    # legacy hierarchical label smoothing for the frame-track class head.
    # When frame_class_ontology_smoothing > 0, the CE target becomes a soft
    # label distribution:
    #   target[c_gt] = 1 - eps
    #   target[c_sib] = eps / |siblings|  (for each sibling in same ontology
    #                                       group as c_gt; excludes c_gt itself)
    #   target[c_other] = 0
    # where `siblings` are other class indices in the same AudioSet-style
    # parent group.  Classes not listed in any group fall back to hard CE
    # (eps splits over 0 siblings = 0 contribution), so behaviour is backward
    # compatible when the list is empty.
    #
    # This addresses the legacy finding that 40%+ of class errors are
    # "sibling collapse" (aircraft->speech, frog->bird, vehicle->machine,
    # etc.).  Class-weighted CE cannot fix sibling collapse on its own
    # because both classes remain mutually-exclusive under hard CE.
    frame_class_ontology_smoothing: float = 0.0
    # Parallel list of sibling groups.  Each entry is a list of class indices
    # belonging to the same AudioSet ontology parent (e.g. "transportation"
    # = {aircraft, vehicle, train, car}).  A class may appear in only one
    # group.  Empty list = no hierarchical smoothing (even if
    # frame_class_ontology_smoothing > 0).
    frame_class_ontology_groups: List[List[int]] = None  # type: ignore[assignment]

    # weight on the new per-frame num-active-source CE loss.  Only fires
    # when the model exposes ``pred_num_active_logits`` (use_num_active_head=True).
    # 0.0 = loss disabled (default, backward compatible).
    lambda_frame_num_active: float = 0.0
    # optional focal variant for the num_active CE.  Motivation
    # (ov3 under-report diagnosis):
    #   - n_gt=3 frames are ~6% of training ov3 data; vanilla CE learns
    #     "always predict K̂=2" because the majority class of [0..4] is 2.
    #   - Focal CE: (1 - p_gt)^gamma * CE up-weights frames where the head
    #     is already wrong (e.g. predicts 2 when truth is 3).
    #   - Optional class weights (indexed 0..num_active_max) further
    #     up-weight rare high-K targets: [0.5, 1.0, 1.0, 1.5, 2.0].
    # Defaults off to preserve legacy phase-1 behaviour.
    frame_num_active_use_focal: bool = False
    frame_num_active_focal_gamma: float = 2.0
    frame_num_active_class_weights: List[float] = None  # type: ignore[assignment]

    # Sign-only elevation hemisphere loss.
    # When a GT elevation is only known as positive/negative (±inf sentinel in
    # the new unified dataset), the direction cosine loss is disabled for those
    # frames and a 2-class upper/lower hemisphere BCE is applied instead.
    # lambda_frame_hemisphere weights this BCE relative to lambda_frame_direction;
    # 0.0 = disabled (falls back to skipping those frames entirely).
    # Default 1.0 provides equal weighting to the direction cosine loss.
    lambda_frame_hemisphere: float = 1.0

    def __post_init__(self) -> None:
        if self.frame_class_loss_weights is None:
            self.frame_class_loss_weights = []
        if self.frame_class_ontology_groups is None:
            self.frame_class_ontology_groups = []
        if self.frame_num_active_class_weights is None:
            self.frame_num_active_class_weights = []

    # Regularization. Defaults to 0.0 (off) to preserve existing behavior.
    label_smoothing: float = 0.0
    # Semantic anchor: weight for the auxiliary class loss on pre-fusion
    # BEATs tokens. 0.0 = off (default, preserves existing behavior).
    lambda_sem_anchor: float = 0.0
    # Frame-wise activity BCE loss (local_spatial_framewise only).
    # Applied to all non-padded frames: active=1, inactive=0.
    lambda_framewise_activity: float = 1.0

    # === v13_B [B-2]: Asymmetric Loss (ASL) for activity head ===============
    # When frame_activity_loss_type == "asymmetric", the frame-track activity
    # head uses Asymmetric Loss (ICCV 2021) instead of BCE:
    #   positive:  -((1-p)**gamma_pos) * log(p)
    #   negative:  p_shifted = max(p - margin, 0)
    #              -(p_shifted**gamma_neg) * log(1 - p_shifted)
    # Suppresses easy negatives while keeping positive grad strong → raises
    # activity_recall. Default "bce" preserves existing behaviour.
    #
    # v13_D [D-2]: Top-K rank activity loss.  When set to "topk_rank", uses
    # pairwise margin hinge between active/inactive slots per frame. Directly
    # aligned with DCASE's "take top-K̂ per frame" decision rule, bypassing the
    # per-element logprob training objective that BCE/ASL both optimize.
    frame_activity_loss_type: str = "bce"         # "bce" | "asymmetric" | "topk_rank"
    asl_gamma_neg: float = 4.0
    asl_gamma_pos: float = 0.0
    asl_probability_margin: float = 0.05

    # === v13_D [D-2]: Top-K rank activity loss hyperparameters ==============
    topk_rank_margin: float = 2.0                 # hinge margin in logit space
    topk_rank_bce_weight: float = 0.1             # anchor BCE co-efficient

    # === v13_B [B-4]: Soft macro-F1 auxiliary loss ==========================
    # When > 0, adds a soft-F1 surrogate loss computed from class-conditional
    # activity probabilities: p_c = sigmoid(act) * softmax(class)[c].
    # Directly aligned with DCASE F20 metric (macro averaging).
    # Set both weights to the same value for constant schedule; otherwise
    # train loop ramps from warmup → final at soft_f1_warmup_epochs.
    frame_soft_f1_weight: float = 0.0
    frame_soft_f1_weight_warmup: float = 0.0      # weight during ep < warmup_epochs
    frame_soft_f1_warmup_epochs: int = 0          # 0 = no warmup, use final directly

    # === v13_C [C-4]: Laplace NLL for log-distance head =====================
    # When "laplace_nll", the frame-track distance loss is computed from
    # pred_distance + pred_distance_log_var via Laplace negative log-likelihood.
    # Requires the model's FrameTrackPredictionHeads to have
    # use_log_distance_head=True (outputs pred_distance_log_var).
    # Default "l1" preserves existing behaviour.
    frame_distance_loss_type: str = "l1"          # "l1" | "laplace_nll"


@dataclass
class FixedSlotMatchingResult:
    """Assignment object for fixed-slot supervision.

    Attributes:
        matched_slot_indices:
            [B, N_gt_max, T_s_max] integer slot index selected for each GT source
            at each valid time step. Invalid entries may be filled with -1.
        matched_valid_mask:
            [B, N_gt_max, T_s_max] boolean mask where True marks a valid
            assignment between a GT source and one of the K fixed slots.
        window_mask:
            [B, N_gt_max, T_s_max] boolean mask of weak valid supervision windows.
    """

    matched_slot_indices: Tensor
    matched_valid_mask: Tensor
    window_mask: Tensor


@dataclass
class SpatialLossOutput:
    """Structured multi-task loss output."""

    loss_total: Tensor
    loss_activity: Tensor
    loss_azi: Tensor
    loss_ele: Tensor
    loss_dist: Tensor
    loss_cls_aux: Tensor
    loss_temp: Tensor
    loss_direction: Tensor


@dataclass
class SpatialMetricOutput:
    """Structured validation metrics for matched spatial predictions."""

    activity_acc: Tensor
    activity_precision: Tensor
    activity_recall: Tensor
    class_acc: Tensor
    azi_mae_deg: Tensor
    ele_mae_deg: Tensor
    dist_mae: Tensor
    matched_count: Tensor


def build_time_window_mask(
    source_start_time_seconds: Tensor,
    source_end_time_seconds: Tensor,
    source_valid_mask: Tensor,
    clip_duration_seconds: Tensor,
    target_num_steps: Tensor,
    t_s_max: int,
) -> Tensor:
    """Project weak source time windows to the padded temporal token axis.

    Args:
        source_start_time_seconds:
            [B, N_gt_max] source start times in seconds.
        source_end_time_seconds:
            [B, N_gt_max] source end times in seconds.
        source_valid_mask:
            [B, N_gt_max] boolean mask where True marks real sources.
        clip_duration_seconds:
            [B] valid clip durations.
        target_num_steps:
            [B] per-sample valid temporal token counts:
                T_s_i = round(duration_i * target_token_rate)
        t_s_max:
            Max temporal token length in the current batch.

    Returns:
        Tensor:
            [B, N_gt_max, T_s_max] boolean weak supervision window mask.
    """
    batch_size, num_gt = source_start_time_seconds.shape
    device = source_start_time_seconds.device
    window_mask = torch.zeros(batch_size, num_gt, t_s_max, dtype=torch.bool, device=device)

    for batch_index in range(batch_size):
        valid_steps = int(target_num_steps[batch_index].item())
        valid_steps = max(valid_steps, 1)
        clip_duration = float(max(clip_duration_seconds[batch_index].item(), 1e-6))
        for gt_index in range(num_gt):
            if not bool(source_valid_mask[batch_index, gt_index]):
                continue
            start = float(source_start_time_seconds[batch_index, gt_index].item())
            end = float(source_end_time_seconds[batch_index, gt_index].item())
            start = min(max(start, 0.0), clip_duration)
            end = min(max(end, start), clip_duration)
            start_idx = int(torch.floor(torch.tensor(start / clip_duration * valid_steps)).item())
            end_idx = int(torch.ceil(torch.tensor(end / clip_duration * valid_steps)).item())
            start_idx = max(0, min(start_idx, valid_steps - 1))
            end_idx = max(start_idx + 1, min(end_idx, valid_steps))
            window_mask[batch_index, gt_index, start_idx:end_idx] = True

    return window_mask


def build_primary_source_window_mask(
    batch: "SpatialBatch",
    t_s_max: int,
) -> Tensor:
    """Build a single-source weak time mask for mono_ast supervision.

    This path is intended for ov1-style data where each clip contains exactly
    one source after cropping. The first valid source window is projected to the
    temporal token axis and returned as [B, T_s_max].
    """
    valid_source_counts = batch.source_valid_mask.sum(dim=1)
    if not torch.all(valid_source_counts == 1):
        raise ValueError(
            "mono_ast supervision expects exactly one valid source per sample; "
            f"got counts={valid_source_counts.tolist()}"
        )
    window_mask = build_time_window_mask(
        source_start_time_seconds=batch.source_start_time_seconds[:, :1],
        source_end_time_seconds=batch.source_end_time_seconds[:, :1],
        source_valid_mask=batch.source_valid_mask[:, :1],
        clip_duration_seconds=batch.clip_duration_seconds,
        target_num_steps=batch.target_num_steps,
        t_s_max=t_s_max,
    )
    return window_mask[:, 0]


def discretize_direction_targets(
    source_azimuth_deg: Tensor,
    source_elevation_deg: Tensor,
    num_azi_bins: int,
    num_ele_bins: int,
) -> Dict[str, Tensor]:
    """Convert continuous direction labels to classification bins.

    Args:
        source_azimuth_deg:
            [B, N_gt_max] azimuth values in degrees.
        source_elevation_deg:
            [B, N_gt_max] elevation values in degrees.

    Returns:
        Dict[str, Tensor]:
            Suggested keys:
                - azi_bin_indices: [B, N_gt_max]
                - ele_bin_indices: [B, N_gt_max]
    """
    azi = torch.remainder(source_azimuth_deg, 360.0)
    azi_bin_indices = torch.floor(azi).long().clamp(min=0, max=num_azi_bins - 1)

    ele = torch.clamp(source_elevation_deg + 90.0, min=0.0, max=179.999)
    ele_bin_indices = torch.floor(ele).long().clamp(min=0, max=num_ele_bins - 1)

    return {
        "azi_bin_indices": azi_bin_indices,
        "ele_bin_indices": ele_bin_indices,
    }


def discretize_distance_targets(
    source_distance_m: Tensor,
    num_distance_bins: int,
    distance_bin_size_m: float,
) -> Tensor:
    """Convert continuous distance labels to fixed-width distance bins."""
    dist_bin = torch.round(source_distance_m / max(float(distance_bin_size_m), 1e-6))
    return dist_bin.long().clamp(min=0, max=num_distance_bins - 1)


def _circular_distance_deg(
    a_deg: Tensor,
    b_deg: Tensor,
) -> Tensor:
    """Compute wrapped circular distance in degrees on [0, 360)."""
    return torch.abs(torch.remainder(a_deg - b_deg + 180.0, 360.0) - 180.0)


def _to_dcase_azimuth(azi_deg: Tensor) -> Tensor:
    """Wrap any azimuth to DCASE convention: [-180, 180)."""
    return torch.remainder(azi_deg + 180.0, 360.0) - 180.0


def _expected_azimuth_deg_from_logits(logits: Tensor) -> Tensor:
    """Compute the circular mean azimuth from classification logits.

    Args:
        logits:
            [..., num_azi_bins]

    Returns:
        Tensor:
            [...] circular-mean azimuth in [0, 360).
    """
    probs = torch.softmax(logits, dim=-1)
    centers_deg = torch.arange(logits.size(-1), device=logits.device, dtype=logits.dtype)
    centers_rad = torch.deg2rad(centers_deg)
    sin_mean = torch.sum(probs * torch.sin(centers_rad), dim=-1)
    cos_mean = torch.sum(probs * torch.cos(centers_rad), dim=-1)
    mean_rad = torch.atan2(sin_mean, cos_mean)
    return torch.remainder(torch.rad2deg(mean_rad), 360.0)


def _expected_elevation_deg_from_logits(logits: Tensor) -> Tensor:
    """Compute the expected elevation in degrees from classification logits."""
    probs = torch.softmax(logits, dim=-1)
    centers_deg = torch.arange(logits.size(-1), device=logits.device, dtype=logits.dtype) - 90.0
    return torch.sum(probs * centers_deg, dim=-1)


def _direction_vector_from_azi_ele_deg(
    azimuth_deg: Tensor,
    elevation_deg: Tensor,
) -> Tensor:
    """Convert spherical azimuth/elevation labels to unit Cartesian vectors."""
    azi_rad = torch.deg2rad(azimuth_deg)
    ele_rad = torch.deg2rad(elevation_deg)
    cos_ele = torch.cos(ele_rad)
    return torch.stack(
        [
            cos_ele * torch.cos(azi_rad),
            cos_ele * torch.sin(azi_rad),
            torch.sin(ele_rad),
        ],
        dim=-1,
    )


def _azi_ele_deg_from_direction_vector(direction: Tensor) -> tuple[Tensor, Tensor]:
    """Convert Cartesian direction vectors to azimuth/elevation in degrees.

    Returns azimuth in [-180, 180) (DCASE convention) and elevation in [-90, 90].
    """
    direction = F.normalize(direction, dim=-1)
    x = direction[..., 0]
    y = direction[..., 1]
    z = torch.clamp(direction[..., 2], min=-1.0, max=1.0)
    # atan2 returns [-180, 180] — keep in DCASE convention so val_predictions
    # GT and pred are in the same coordinate system.
    azimuth_deg = torch.rad2deg(torch.atan2(y, x))
    elevation_deg = torch.rad2deg(torch.asin(z))
    return azimuth_deg, elevation_deg


def _build_circular_soft_targets(
    target_deg: Tensor,
    num_bins: int,
    sigma_deg: float,
) -> Tensor:
    """Build circular Gaussian soft labels over azimuth bins."""
    centers_deg = torch.arange(num_bins, device=target_deg.device, dtype=target_deg.dtype)
    delta = _circular_distance_deg(centers_deg.unsqueeze(0), target_deg.unsqueeze(-1))
    sigma = max(float(sigma_deg), 1e-3)
    soft_targets = torch.exp(-0.5 * (delta / sigma) ** 2)
    soft_targets = soft_targets / torch.clamp(soft_targets.sum(dim=-1, keepdim=True), min=1e-8)
    return soft_targets


def _build_gaussian_soft_targets(
    target_deg: Tensor,
    num_bins: int,
    sigma_deg: float,
    offset_deg: float,
) -> Tensor:
    """Build Gaussian soft labels over a linear elevation bin axis."""
    centers_deg = torch.arange(num_bins, device=target_deg.device, dtype=target_deg.dtype) + offset_deg
    delta = centers_deg.unsqueeze(0) - target_deg.unsqueeze(-1)
    sigma = max(float(sigma_deg), 1e-3)
    soft_targets = torch.exp(-0.5 * (delta / sigma) ** 2)
    soft_targets = soft_targets / torch.clamp(soft_targets.sum(dim=-1, keepdim=True), min=1e-8)
    return soft_targets


def _soft_cross_entropy(
    logits: Tensor,
    soft_targets: Tensor,
) -> Tensor:
    """Soft-label cross-entropy loss."""
    log_probs = F.log_softmax(logits, dim=-1)
    return -(soft_targets * log_probs).sum(dim=-1).mean()


def _match_active_sources_to_slots(cost_matrix: Tensor) -> Tensor:
    """Brute-force assignment for up to K=4 slots.

    Args:
        cost_matrix:
            [num_active_gt, K] pairwise costs.

    Returns:
        Tensor:
            [num_active_gt] slot index assigned to each active GT row.
    """
    num_active_gt, num_slots = cost_matrix.shape
    slot_indices = list(range(num_slots))
    best_perm = None
    best_cost = None
    for perm in itertools.permutations(slot_indices, num_active_gt):
        total_cost = 0.0
        for gt_row, slot_idx in enumerate(perm):
            total_cost += float(cost_matrix[gt_row, slot_idx].item())
        if best_cost is None or total_cost < best_cost:
            best_cost = total_cost
            best_perm = perm

    return torch.tensor(best_perm, device=cost_matrix.device, dtype=torch.long)


def _collect_matched_rows(
    matching_result: FixedSlotMatchingResult,
    batch: "SpatialBatch",
    device: torch.device,
) -> Dict[str, object]:
    """Collect matched slot rows and aligned GT targets.

    Returns:
        Dict[str, object]:
            Keys:
                - row_tensor: [N_match, 3] or None
                - gt_index_tensor: [N_match] or None
                - azi_target_deg: [N_match] or None
                - ele_target_deg: [N_match] or None
                - dist_target: [N_match] or None
                - cls_target: [N_match] or None
    """
    matched_rows = []
    matched_gt_indices = []
    matched_azi_targets = []
    matched_ele_targets = []
    matched_dist_targets = []
    matched_cls_targets = []

    for batch_index in range(batch.source_class_indices.size(0)):
        num_gt = batch.source_class_indices.size(1)
        for gt_index in range(num_gt):
            valid_times = torch.nonzero(
                matching_result.matched_valid_mask[batch_index, gt_index],
                as_tuple=False,
            ).flatten()
            for time_index in valid_times:
                slot_index = int(matching_result.matched_slot_indices[batch_index, gt_index, time_index].item())
                if slot_index < 0:
                    continue
                matched_rows.append((batch_index, int(time_index.item()), slot_index))
                matched_gt_indices.append(gt_index)
                # Per-frame target: index the loader tensor by (b, gt, t).
                _t_i = int(time_index.item())
                matched_azi_targets.append(float(batch.source_azimuth_deg[batch_index, gt_index, _t_i].item()))
                matched_ele_targets.append(float(batch.source_elevation_deg[batch_index, gt_index, _t_i].item()))
                matched_dist_targets.append(float(batch.source_distance[batch_index, gt_index, _t_i].item()))
                matched_cls_targets.append(int(batch.source_class_indices[batch_index, gt_index].item()))

    if not matched_rows:
        return {
            "row_tensor": None,
            "gt_index_tensor": None,
            "azi_target_deg": None,
            "ele_target_deg": None,
            "dist_target": None,
            "cls_target": None,
        }

    return {
        "row_tensor": torch.tensor(matched_rows, device=device, dtype=torch.long),
        "gt_index_tensor": torch.tensor(matched_gt_indices, device=device, dtype=torch.long),
        "azi_target_deg": torch.tensor(matched_azi_targets, device=device, dtype=torch.float32),
        "ele_target_deg": torch.tensor(matched_ele_targets, device=device, dtype=torch.float32),
        "dist_target": torch.tensor(matched_dist_targets, device=device, dtype=torch.float32),
        "cls_target": torch.tensor(matched_cls_targets, device=device, dtype=torch.long),
    }


def match_fixed_slots(
    prediction_output: SpatialPredictionOutput,
    batch: "SpatialBatch",
    temporal_padding_mask: Optional[Tensor],
    config: SpatialLossConfig,
) -> FixedSlotMatchingResult:
    """Match ground-truth sources to K fixed slots at each valid time step.

    Args:
        prediction_output:
            Slot-level prediction tensors with padded time dimension T_s_max.
        batch:
            Collated SpatialBatch with source labels and weak time windows.
        temporal_padding_mask:
            Optional [B, T_s_max] mask where True marks padded time steps.
        config:
            Loss and matching configuration.

    Returns:
        FixedSlotMatchingResult:
            Slot assignment object consumed by the loss computation.
    """
    pred_activity = prediction_output.pred_activity.detach()
    pred_azi_logits = prediction_output.pred_azi_logits.detach()
    pred_ele_logits = prediction_output.pred_ele_logits.detach()
    pred_dist = prediction_output.pred_dist.detach()
    pred_class_logits = prediction_output.pred_class_logits.detach()

    batch_size, t_s_max, num_slots = pred_activity.shape
    _, num_gt = batch.source_class_indices.shape

    window_mask = build_time_window_mask(
        source_start_time_seconds=batch.source_start_time_seconds.to(pred_activity.device),
        source_end_time_seconds=batch.source_end_time_seconds.to(pred_activity.device),
        source_valid_mask=batch.source_valid_mask.to(pred_activity.device),
        clip_duration_seconds=batch.clip_duration_seconds.to(pred_activity.device),
        target_num_steps=batch.target_num_steps.to(pred_activity.device),
        t_s_max=t_s_max,
    )
    matched_slot_indices = torch.full(
        (batch_size, num_gt, t_s_max),
        fill_value=-1,
        dtype=torch.long,
        device=pred_activity.device,
    )
    matched_valid_mask = torch.zeros(
        batch_size,
        num_gt,
        t_s_max,
        dtype=torch.bool,
        device=pred_activity.device,
    )

    for batch_index in range(batch_size):
        valid_steps = int(batch.target_num_steps[batch_index].item())
        for time_index in range(valid_steps):
            if temporal_padding_mask is not None and bool(temporal_padding_mask[batch_index, time_index]):
                continue
            active_gt = torch.nonzero(window_mask[batch_index, :, time_index], as_tuple=False).flatten()
            if active_gt.numel() == 0:
                continue
            if active_gt.numel() > num_slots:
                active_gt = active_gt[:num_slots]

            cost_matrix = torch.zeros(active_gt.numel(), num_slots, device=pred_activity.device)
            for row_index, gt_index in enumerate(active_gt):
                gt_class = int(batch.source_class_indices[batch_index, gt_index].item())
                # Per-frame DOA/distance targets: index with (b, gt, t).
                gt_azi_deg = float(batch.source_azimuth_deg[batch_index, gt_index, time_index].item())
                gt_ele_deg = float(batch.source_elevation_deg[batch_index, gt_index, time_index].item())
                gt_dist = float(batch.source_distance[batch_index, gt_index, time_index].item())

                class_nll = -F.log_softmax(pred_class_logits[batch_index, time_index], dim=-1)[:, gt_class]
                pred_azi_deg = _expected_azimuth_deg_from_logits(pred_azi_logits[batch_index, time_index])
                pred_ele_deg = _expected_elevation_deg_from_logits(pred_ele_logits[batch_index, time_index])
                azi_cost = _circular_distance_deg(
                    pred_azi_deg,
                    torch.full_like(pred_azi_deg, gt_azi_deg),
                ) / 180.0
                ele_cost = torch.abs(pred_ele_deg - gt_ele_deg) / 90.0
                dist_l1 = torch.abs(pred_dist[batch_index, time_index, :, 0] - gt_dist)
                act_cost = 1.0 - torch.sigmoid(pred_activity[batch_index, time_index])

                cost_matrix[row_index] = (
                    act_cost
                    + class_nll
                    + azi_cost
                    + ele_cost
                    + dist_l1
                )

            assignment = _match_active_sources_to_slots(cost_matrix)
            for row_index, gt_index in enumerate(active_gt):
                matched_slot_indices[batch_index, gt_index, time_index] = assignment[row_index]
                matched_valid_mask[batch_index, gt_index, time_index] = True

    return FixedSlotMatchingResult(
        matched_slot_indices=matched_slot_indices,
        matched_valid_mask=matched_valid_mask,
        window_mask=window_mask,
    )


def compute_spatial_losses(
    prediction_output: SpatialPredictionOutput,
    matching_result: FixedSlotMatchingResult,
    batch: "SpatialBatch",
    temporal_padding_mask: Optional[Tensor],
    config: SpatialLossConfig,
) -> SpatialLossOutput:
    """Compute multi-task losses for encoder-only Spatial-BEATs training.

    Expected loss terms:
        - activity / objectness loss
        - azimuth classification loss
        - elevation classification loss
        - distance regression loss
        - auxiliary source class loss
        - optional temporal consistency regularization

    Args:
        prediction_output:
            Slot-level prediction tensors with padded time dimension T_s_max.
        matching_result:
            Output of per-step fixed-slot matching.
        batch:
            Collated source-level targets.
        temporal_padding_mask:
            Optional [B, T_s_max] mask where True marks padded time steps.
        config:
            Multi-task loss configuration.

    Returns:
        SpatialLossOutput:
            Structured total loss and each component loss.
    """
    pred_activity = prediction_output.pred_activity
    pred_azi_logits = prediction_output.pred_azi_logits
    pred_ele_logits = prediction_output.pred_ele_logits
    pred_dist = prediction_output.pred_dist
    pred_class_logits = prediction_output.pred_class_logits

    batch_size, t_s_max, num_slots = pred_activity.shape
    device = pred_activity.device

    valid_time_mask = torch.ones(batch_size, t_s_max, dtype=torch.bool, device=device)
    if temporal_padding_mask is not None:
        valid_time_mask = ~temporal_padding_mask.to(device)

    activity_target = torch.zeros_like(pred_activity)
    activity_valid_mask = valid_time_mask.unsqueeze(-1).expand_as(pred_activity)
    matched = _collect_matched_rows(matching_result=matching_result, batch=batch, device=device)
    row_tensor = matched["row_tensor"]
    azi_target_deg = matched["azi_target_deg"]
    ele_target_deg = matched["ele_target_deg"]
    dist_target = matched["dist_target"]
    cls_target = matched["cls_target"]

    if row_tensor is not None:
        activity_target[row_tensor[:, 0], row_tensor[:, 1], row_tensor[:, 2]] = 1.0

    if config.activity_loss_type == "bce":
        loss_activity_all = F.binary_cross_entropy_with_logits(
            pred_activity,
            activity_target,
            reduction="none",
        )
    else:
        raise ValueError(f"Unsupported activity_loss_type: {config.activity_loss_type}")
    loss_activity = loss_activity_all[activity_valid_mask].mean() if activity_valid_mask.any() else pred_activity.sum() * 0.0

    if row_tensor is not None:
        pred_azi_sel = pred_azi_logits[row_tensor[:, 0], row_tensor[:, 1], row_tensor[:, 2]]
        pred_ele_sel = pred_ele_logits[row_tensor[:, 0], row_tensor[:, 1], row_tensor[:, 2]]
        pred_dist_sel = pred_dist[row_tensor[:, 0], row_tensor[:, 1], row_tensor[:, 2], 0]
        pred_cls_sel = pred_class_logits[row_tensor[:, 0], row_tensor[:, 1], row_tensor[:, 2]]
        assert azi_target_deg is not None and ele_target_deg is not None
        assert dist_target is not None and cls_target is not None
        azi_target_deg = azi_target_deg.to(dtype=pred_azi_sel.dtype)
        ele_target_deg = ele_target_deg.to(dtype=pred_ele_sel.dtype)
        dist_target = dist_target.to(dtype=pred_dist.dtype)

        if config.azi_loss_type == "soft_circular_ce":
            azi_soft_target = _build_circular_soft_targets(
                target_deg=torch.remainder(azi_target_deg, 360.0),
                num_bins=pred_azi_sel.size(-1),
                sigma_deg=config.azi_soft_label_sigma_deg,
            )
            loss_azi = _soft_cross_entropy(pred_azi_sel, azi_soft_target)
        elif config.azi_loss_type == "hard_ce":
            azi_target = torch.remainder(torch.floor(azi_target_deg), pred_azi_sel.size(-1)).long()
            loss_azi = F.cross_entropy(pred_azi_sel, azi_target)
        else:
            raise ValueError(f"Unsupported azi_loss_type: {config.azi_loss_type}")

        if config.ele_loss_type == "soft_gaussian_ce":
            ele_soft_target = _build_gaussian_soft_targets(
                target_deg=torch.clamp(ele_target_deg, min=-90.0, max=89.999),
                num_bins=pred_ele_sel.size(-1),
                sigma_deg=config.ele_soft_label_sigma_deg,
                offset_deg=-90.0,
            )
            loss_ele = _soft_cross_entropy(pred_ele_sel, ele_soft_target)
        elif config.ele_loss_type == "hard_ce":
            ele_target = torch.floor(torch.clamp(ele_target_deg + 90.0, min=0.0, max=179.999)).long()
            loss_ele = F.cross_entropy(pred_ele_sel, ele_target)
        else:
            raise ValueError(f"Unsupported ele_loss_type: {config.ele_loss_type}")

        loss_cls_aux = F.cross_entropy(pred_cls_sel, cls_target)
        loss_dist = F.smooth_l1_loss(pred_dist_sel, dist_target)
    else:
        zero = pred_activity.sum() * 0.0
        loss_azi = zero
        loss_ele = zero
        loss_cls_aux = zero
        loss_dist = zero
    loss_direction = pred_activity.sum() * 0.0

    temporal_smoothness_terms = []
    for batch_index in range(batch_size):
        num_gt = batch.source_class_indices.size(1)
        for gt_index in range(num_gt):
            valid_times = torch.nonzero(
                matching_result.matched_valid_mask[batch_index, gt_index],
                as_tuple=False,
            ).flatten()
            if valid_times.numel() < 2:
                continue
            prev_time = int(valid_times[0].item())
            prev_slot = int(matching_result.matched_slot_indices[batch_index, gt_index, prev_time].item())
            for next_time_tensor in valid_times[1:]:
                next_time = int(next_time_tensor.item())
                next_slot = int(matching_result.matched_slot_indices[batch_index, gt_index, next_time].item())
                prev_dist = pred_dist[batch_index, prev_time, prev_slot, 0]
                next_dist = pred_dist[batch_index, next_time, next_slot, 0]
                prev_act = torch.sigmoid(pred_activity[batch_index, prev_time, prev_slot])
                next_act = torch.sigmoid(pred_activity[batch_index, next_time, next_slot])
                temporal_smoothness_terms.append(torch.abs(next_dist - prev_dist))
                temporal_smoothness_terms.append(torch.abs(next_act - prev_act))
                prev_time, prev_slot = next_time, next_slot

    if temporal_smoothness_terms:
        loss_temp = torch.stack(temporal_smoothness_terms).mean()
    else:
        loss_temp = pred_activity.sum() * 0.0

    loss_total = (
        config.lambda_activity * loss_activity
        + config.lambda_azi * loss_azi
        + config.lambda_ele * loss_ele
        + config.lambda_dist * loss_dist
        + config.lambda_cls_aux * loss_cls_aux
        + config.lambda_temp * loss_temp
        + config.lambda_direction * loss_direction
    )

    return SpatialLossOutput(
        loss_total=loss_total,
        loss_activity=loss_activity,
        loss_azi=loss_azi,
        loss_ele=loss_ele,
        loss_dist=loss_dist,
        loss_cls_aux=loss_cls_aux,
        loss_temp=loss_temp,
        loss_direction=loss_direction,
    )


def compute_mono_ast_losses(
    prediction_output: MonoTaskPredictionOutput,
    batch: "SpatialBatch",
    config: SpatialLossConfig,
) -> SpatialLossOutput:
    """Compute single-source Spatial-AST-style losses without slot matching."""
    valid_source_counts = batch.source_valid_mask.sum(dim=1)
    if not torch.all(valid_source_counts == 1):
        raise ValueError(
            "mono_ast supervision expects exactly one valid source per sample; "
            f"got counts={valid_source_counts.tolist()}"
        )

    cls_target = batch.source_class_indices[:, 0].to(prediction_output.pred_class_logits.device)
    azi_target_deg = batch.source_azimuth_deg[:, 0, 0].to(prediction_output.pred_direction.device)
    ele_target_deg = batch.source_elevation_deg[:, 0, 0].to(prediction_output.pred_direction.device)
    dist_target = batch.source_distance[:, 0, 0].to(prediction_output.pred_distance.device)
    gt_direction = _direction_vector_from_azi_ele_deg(azi_target_deg, ele_target_deg).to(
        prediction_output.pred_direction.dtype
    )

    loss_cls_aux = F.cross_entropy(
        prediction_output.pred_class_logits, cls_target,
        label_smoothing=config.label_smoothing,
    )
    pred_direction = F.normalize(prediction_output.pred_direction, dim=-1)
    loss_direction = (1.0 - (pred_direction * gt_direction).sum(dim=-1)).mean()
    loss_dist = F.smooth_l1_loss(prediction_output.pred_distance[:, 0], dist_target)

    zero = loss_cls_aux.new_zeros(())

    # Semantic anchor: auxiliary CE on pre-fusion BEATs tokens (training only).
    # Keeps the trunk semantically grounded while spatial loss pushes fused tokens.
    loss_sem_anchor = zero
    if (
        config.lambda_sem_anchor > 0.0
        and prediction_output.sem_class_logits is not None
    ):
        loss_sem_anchor = F.cross_entropy(
            prediction_output.sem_class_logits, cls_target,
            label_smoothing=config.label_smoothing,
        )

    loss_total = (
        config.lambda_cls_aux * loss_cls_aux
        + config.lambda_direction * loss_direction
        + config.lambda_dist * loss_dist
        + config.lambda_sem_anchor * loss_sem_anchor
    )
    return SpatialLossOutput(
        loss_total=loss_total,
        loss_activity=zero,
        loss_azi=zero,
        loss_ele=zero,
        loss_dist=loss_dist,
        loss_cls_aux=loss_cls_aux,
        loss_temp=loss_sem_anchor,   # reuse loss_temp slot for anchor loss logging
        loss_direction=loss_direction,
    )


def compute_mono_ast_validation_metrics(
    prediction_output: MonoTaskPredictionOutput,
    batch: "SpatialBatch",
) -> SpatialMetricOutput:
    """Compute single-source validation metrics for the mono_ast path."""
    valid_source_counts = batch.source_valid_mask.sum(dim=1)
    if not torch.all(valid_source_counts == 1):
        raise ValueError(
            "mono_ast supervision expects exactly one valid source per sample; "
            f"got counts={valid_source_counts.tolist()}"
        )

    cls_target = batch.source_class_indices[:, 0].to(prediction_output.pred_class_logits.device)
    azi_target_deg = batch.source_azimuth_deg[:, 0, 0].to(prediction_output.pred_direction.device)
    ele_target_deg = batch.source_elevation_deg[:, 0, 0].to(prediction_output.pred_direction.device)
    dist_target = batch.source_distance[:, 0, 0].to(prediction_output.pred_distance.device)

    pred_class_idx = prediction_output.pred_class_logits.argmax(dim=-1)
    pred_azi_deg, pred_ele_deg = _azi_ele_deg_from_direction_vector(prediction_output.pred_direction)
    pred_dist = prediction_output.pred_distance[:, 0]

    zero = pred_dist.new_zeros(())
    return SpatialMetricOutput(
        activity_acc=zero,
        activity_precision=zero,
        activity_recall=zero,
        class_acc=(pred_class_idx == cls_target).float().mean(),
        azi_mae_deg=_circular_distance_deg(pred_azi_deg, azi_target_deg).mean(),
        ele_mae_deg=torch.abs(pred_ele_deg - ele_target_deg).mean(),
        dist_mae=torch.abs(pred_dist - dist_target).mean(),
        matched_count=torch.tensor(
            float(prediction_output.pred_class_logits.size(0)),
            device=pred_dist.device,
            dtype=pred_dist.dtype,
        ),
    )


def build_mono_ast_validation_examples(
    prediction_output: MonoTaskPredictionOutput,
    batch: "SpatialBatch",
    max_examples: int = 16,
) -> List[Dict[str, object]]:
    """Build qualitative validation examples for single-source mono_ast runs."""
    pred_azi_deg, pred_ele_deg = _azi_ele_deg_from_direction_vector(prediction_output.pred_direction)
    pred_class_idx = prediction_output.pred_class_logits.argmax(dim=-1)
    pred_class_prob = prediction_output.pred_class_logits.softmax(dim=-1).amax(dim=-1)

    examples: List[Dict[str, object]] = []
    limit = min(int(max_examples), len(batch.sample_ids))
    for idx in range(limit):
        examples.append(
            {
                "sample_id": batch.sample_ids[idx],
                "gt_class_index": int(batch.source_class_indices[idx, 0].item()),
                "gt_class_name": batch.source_class_labels[idx][0] if batch.source_class_labels else None,
                "pred_class_index": int(pred_class_idx[idx].item()),
                "pred_class_confidence": float(pred_class_prob[idx].item()),
                "gt_azimuth_deg": float(batch.source_azimuth_deg[idx, 0, 0].item()),
                "pred_azimuth_deg": float(pred_azi_deg[idx].item()),
                "gt_elevation_deg": float(batch.source_elevation_deg[idx, 0, 0].item()),
                "pred_elevation_deg": float(pred_ele_deg[idx].item()),
                "gt_distance_m": float(batch.source_distance[idx, 0, 0].item()),
                "pred_distance_m": float(prediction_output.pred_distance[idx, 0].item()),
            }
        )
    return examples


def compute_pretrunk_ast_losses(
    prediction_output: PreTrunkASTPredictionOutput,
    batch: "SpatialBatch",
    config: SpatialLossConfig,
) -> SpatialLossOutput:
    """Compute Spatial-AST-style CE losses from task tokens inside the trunk.

    This path is intentionally single-source. It is used to validate whether
    pre-trunk distance/DoA/class task tokens can learn ov1 before reintroducing
    multi-source matching.
    """
    valid_source_counts = batch.source_valid_mask.sum(dim=1)
    if not torch.all(valid_source_counts == 1):
        raise ValueError(
            "pretrunk_ast supervision expects exactly one valid source per sample; "
            f"got counts={valid_source_counts.tolist()}"
        )

    device = prediction_output.pred_class_logits.device
    cls_target = batch.source_class_indices[:, 0].to(device)
    azi_target_deg = batch.source_azimuth_deg[:, 0, 0].to(device)
    ele_target_deg = batch.source_elevation_deg[:, 0, 0].to(device)
    dist_target = batch.source_distance[:, 0, 0].to(device)

    direction_bins = discretize_direction_targets(
        source_azimuth_deg=azi_target_deg,
        source_elevation_deg=ele_target_deg,
        num_azi_bins=config.num_azi_bins,
        num_ele_bins=config.num_ele_bins,
    )
    distance_bins = discretize_distance_targets(
        source_distance_m=dist_target,
        num_distance_bins=config.num_distance_bins,
        distance_bin_size_m=config.distance_bin_size_m,
    )

    loss_cls_aux = F.cross_entropy(
        prediction_output.pred_class_logits, cls_target,
        label_smoothing=config.label_smoothing,
    )
    loss_dist = F.cross_entropy(prediction_output.pred_distance_logits, distance_bins)
    loss_azi = F.cross_entropy(prediction_output.pred_azi_logits, direction_bins["azi_bin_indices"])
    loss_ele = F.cross_entropy(prediction_output.pred_ele_logits, direction_bins["ele_bin_indices"])
    loss_direction = loss_azi + loss_ele

    zero = loss_cls_aux.new_zeros(())
    loss_total = (
        config.lambda_cls_aux * loss_cls_aux
        + config.lambda_dist * loss_dist
        + config.lambda_azi * loss_azi
        + config.lambda_ele * loss_ele
        + config.lambda_direction * loss_direction
    )
    return SpatialLossOutput(
        loss_total=loss_total,
        loss_activity=zero,
        loss_azi=loss_azi,
        loss_ele=loss_ele,
        loss_dist=loss_dist,
        loss_cls_aux=loss_cls_aux,
        loss_temp=zero,
        loss_direction=loss_direction,
    )


def compute_pretrunk_ast_validation_metrics(
    prediction_output: PreTrunkASTPredictionOutput,
    batch: "SpatialBatch",
    config: SpatialLossConfig,
) -> SpatialMetricOutput:
    """Compute source-level metrics for the pre-trunk Spatial-AST branch."""
    valid_source_counts = batch.source_valid_mask.sum(dim=1)
    if not torch.all(valid_source_counts == 1):
        raise ValueError(
            "pretrunk_ast supervision expects exactly one valid source per sample; "
            f"got counts={valid_source_counts.tolist()}"
        )

    device = prediction_output.pred_class_logits.device
    cls_target = batch.source_class_indices[:, 0].to(device)
    azi_target_deg = batch.source_azimuth_deg[:, 0, 0].to(device)
    ele_target_deg = batch.source_elevation_deg[:, 0, 0].to(device)
    dist_target = batch.source_distance[:, 0, 0].to(device)

    pred_class_idx = prediction_output.pred_class_logits.argmax(dim=-1)
    pred_azi_deg = prediction_output.pred_azi_logits.argmax(dim=-1).to(dtype=azi_target_deg.dtype)
    pred_ele_deg = prediction_output.pred_ele_logits.argmax(dim=-1).to(dtype=ele_target_deg.dtype) - 90.0
    pred_dist = (
        prediction_output.pred_distance_logits.argmax(dim=-1).to(dtype=dist_target.dtype)
        * float(config.distance_bin_size_m)
    )

    zero = dist_target.new_zeros(())
    return SpatialMetricOutput(
        activity_acc=zero,
        activity_precision=zero,
        activity_recall=zero,
        class_acc=(pred_class_idx == cls_target).float().mean(),
        azi_mae_deg=_circular_distance_deg(pred_azi_deg, azi_target_deg).mean(),
        ele_mae_deg=torch.abs(pred_ele_deg - ele_target_deg).mean(),
        dist_mae=torch.abs(pred_dist - dist_target).mean(),
        matched_count=torch.tensor(
            float(prediction_output.pred_class_logits.size(0)),
            device=device,
            dtype=dist_target.dtype,
        ),
    )


def build_pretrunk_ast_validation_examples(
    prediction_output: PreTrunkASTPredictionOutput,
    batch: "SpatialBatch",
    config: SpatialLossConfig,
    max_examples: int = 16,
) -> List[Dict[str, object]]:
    """Build qualitative validation examples for the pre-trunk AST branch."""
    pred_class_idx = prediction_output.pred_class_logits.argmax(dim=-1)
    pred_class_prob = prediction_output.pred_class_logits.softmax(dim=-1).amax(dim=-1)
    # bin index = azimuth in [0,360); convert to DCASE [-180,180) for logging
    pred_azi_deg = _to_dcase_azimuth(
        prediction_output.pred_azi_logits.argmax(dim=-1).to(dtype=torch.float32)
    )
    pred_ele_deg = prediction_output.pred_ele_logits.argmax(dim=-1) - 90
    pred_dist = (
        prediction_output.pred_distance_logits.argmax(dim=-1).to(dtype=torch.float32)
        * float(config.distance_bin_size_m)
    )

    examples: List[Dict[str, object]] = []
    limit = min(int(max_examples), len(batch.sample_ids))
    for idx in range(limit):
        examples.append(
            {
                "sample_id": batch.sample_ids[idx],
                "gt_class_index": int(batch.source_class_indices[idx, 0].item()),
                "gt_class_name": batch.source_class_labels[idx][0] if batch.source_class_labels else None,
                "pred_class_index": int(pred_class_idx[idx].item()),
                "pred_class_confidence": float(pred_class_prob[idx].item()),
                "gt_azimuth_deg": float(batch.source_azimuth_deg[idx, 0, 0].item()),
                "pred_azimuth_deg": float(pred_azi_deg[idx].item()),
                "gt_elevation_deg": float(batch.source_elevation_deg[idx, 0, 0].item()),
                "pred_elevation_deg": float(pred_ele_deg[idx].item()),
                "gt_distance_m": float(batch.source_distance[idx, 0, 0].item()),
                "pred_distance_m": float(pred_dist[idx].item()),
            }
        )
    return examples


def compute_spatial_validation_metrics(
    prediction_output: SpatialPredictionOutput,
    matching_result: FixedSlotMatchingResult,
    batch: "SpatialBatch",
    temporal_padding_mask: Optional[Tensor],
) -> SpatialMetricOutput:
    """Compute interpretable validation metrics from slot predictions.

    These metrics are meant for monitoring whether the model is learning
    spatial structure, independent of the weighted training loss.
    """
    pred_activity = prediction_output.pred_activity
    pred_azi_logits = prediction_output.pred_azi_logits
    pred_ele_logits = prediction_output.pred_ele_logits
    pred_dist = prediction_output.pred_dist
    pred_class_logits = prediction_output.pred_class_logits

    batch_size, t_s_max, _ = pred_activity.shape
    device = pred_activity.device

    valid_time_mask = torch.ones(batch_size, t_s_max, dtype=torch.bool, device=device)
    if temporal_padding_mask is not None:
        valid_time_mask = ~temporal_padding_mask.to(device)

    activity_target = torch.zeros_like(pred_activity)
    matched = _collect_matched_rows(matching_result=matching_result, batch=batch, device=device)
    row_tensor = matched["row_tensor"]
    if row_tensor is not None:
        activity_target[row_tensor[:, 0], row_tensor[:, 1], row_tensor[:, 2]] = 1.0

    activity_valid_mask = valid_time_mask.unsqueeze(-1).expand_as(pred_activity)
    activity_pred = (torch.sigmoid(pred_activity) >= 0.5).float()
    activity_true = activity_target
    valid_pred = activity_pred[activity_valid_mask]
    valid_true = activity_true[activity_valid_mask]

    if valid_true.numel() > 0:
        activity_acc = (valid_pred == valid_true).float().mean()
        true_positive = ((valid_pred == 1.0) & (valid_true == 1.0)).float().sum()
        pred_positive = (valid_pred == 1.0).float().sum()
        gt_positive = (valid_true == 1.0).float().sum()
        activity_precision = true_positive / torch.clamp(pred_positive, min=1.0)
        activity_recall = true_positive / torch.clamp(gt_positive, min=1.0)
    else:
        zero = pred_activity.sum() * 0.0
        activity_acc = zero
        activity_precision = zero
        activity_recall = zero

    if row_tensor is not None:
        assert matched["azi_target_deg"] is not None
        assert matched["ele_target_deg"] is not None
        assert matched["dist_target"] is not None
        assert matched["cls_target"] is not None
        pred_azi_sel = pred_azi_logits[row_tensor[:, 0], row_tensor[:, 1], row_tensor[:, 2]]
        pred_ele_sel = pred_ele_logits[row_tensor[:, 0], row_tensor[:, 1], row_tensor[:, 2]]
        pred_dist_sel = pred_dist[row_tensor[:, 0], row_tensor[:, 1], row_tensor[:, 2], 0]
        pred_cls_sel = pred_class_logits[row_tensor[:, 0], row_tensor[:, 1], row_tensor[:, 2]]

        azi_target_deg = matched["azi_target_deg"].to(device=device, dtype=pred_azi_sel.dtype)
        ele_target_deg = matched["ele_target_deg"].to(device=device, dtype=pred_ele_sel.dtype)
        dist_target = matched["dist_target"].to(device=device, dtype=pred_dist_sel.dtype)
        cls_target = matched["cls_target"].to(device=device)

        pred_azi_deg = _expected_azimuth_deg_from_logits(pred_azi_sel)
        pred_ele_deg = _expected_elevation_deg_from_logits(pred_ele_sel)
        pred_cls_idx = pred_cls_sel.argmax(dim=-1)

        azi_mae_deg = _circular_distance_deg(pred_azi_deg, azi_target_deg).mean()
        ele_mae_deg = torch.abs(pred_ele_deg - ele_target_deg).mean()
        dist_mae = torch.abs(pred_dist_sel - dist_target).mean()
        class_acc = (pred_cls_idx == cls_target).float().mean()
        matched_count = torch.tensor(float(row_tensor.size(0)), device=device, dtype=pred_activity.dtype)
    else:
        zero = pred_activity.sum() * 0.0
        azi_mae_deg = zero
        ele_mae_deg = zero
        dist_mae = zero
        class_acc = zero
        matched_count = zero

    return SpatialMetricOutput(
        activity_acc=activity_acc,
        activity_precision=activity_precision,
        activity_recall=activity_recall,
        class_acc=class_acc,
        azi_mae_deg=azi_mae_deg,
        ele_mae_deg=ele_mae_deg,
        dist_mae=dist_mae,
        matched_count=matched_count,
    )


def build_validation_examples(
    prediction_output: SpatialPredictionOutput,
    matching_result: FixedSlotMatchingResult,
    batch: "SpatialBatch",
    max_examples: int = 16,
) -> List[Dict[str, object]]:
    """Build a small list of validation examples for qualitative inspection."""
    device = prediction_output.pred_activity.device
    matched = _collect_matched_rows(matching_result=matching_result, batch=batch, device=device)
    row_tensor = matched["row_tensor"]
    gt_index_tensor = matched["gt_index_tensor"]
    if row_tensor is None or gt_index_tensor is None:
        return []

    pred_azi_logits = prediction_output.pred_azi_logits[row_tensor[:, 0], row_tensor[:, 1], row_tensor[:, 2]]
    pred_ele_logits = prediction_output.pred_ele_logits[row_tensor[:, 0], row_tensor[:, 1], row_tensor[:, 2]]
    pred_dist = prediction_output.pred_dist[row_tensor[:, 0], row_tensor[:, 1], row_tensor[:, 2], 0]
    pred_class_logits = prediction_output.pred_class_logits[row_tensor[:, 0], row_tensor[:, 1], row_tensor[:, 2]]
    pred_activity = prediction_output.pred_activity[row_tensor[:, 0], row_tensor[:, 1], row_tensor[:, 2]]

    pred_azi_deg = _to_dcase_azimuth(_expected_azimuth_deg_from_logits(pred_azi_logits))
    pred_ele_deg = _expected_elevation_deg_from_logits(pred_ele_logits)
    pred_class_idx = pred_class_logits.argmax(dim=-1)
    pred_class_prob = pred_class_logits.softmax(dim=-1).amax(dim=-1)
    pred_activity_prob = torch.sigmoid(pred_activity)

    examples: List[Dict[str, object]] = []
    limit = min(int(max_examples), int(row_tensor.size(0)))
    for idx in range(limit):
        batch_index = int(row_tensor[idx, 0].item())
        time_index = int(row_tensor[idx, 1].item())
        slot_index = int(row_tensor[idx, 2].item())
        gt_index = int(gt_index_tensor[idx].item())
        examples.append(
            {
                "sample_id": batch.sample_ids[batch_index],
                "time_index": time_index,
                "slot_index": slot_index,
                "gt_index": gt_index,
                "gt_class_index": int(batch.source_class_indices[batch_index, gt_index].item()),
                "gt_class_label": batch.source_class_labels[batch_index][gt_index],
                "pred_class_index": int(pred_class_idx[idx].item()),
                "pred_class_confidence": float(pred_class_prob[idx].item()),
                "gt_azimuth_deg": float(batch.source_azimuth_deg[batch_index, gt_index, time_index].item()),
                "pred_azimuth_deg": float(pred_azi_deg[idx].item()),
                "gt_elevation_deg": float(batch.source_elevation_deg[batch_index, gt_index, time_index].item()),
                "pred_elevation_deg": float(pred_ele_deg[idx].item()),
                "gt_distance": float(batch.source_distance[batch_index, gt_index, time_index].item()),
                "pred_distance": float(pred_dist[idx].item()),
                "pred_activity_prob": float(pred_activity_prob[idx].item()),
            }
        )
    return examples


# ---------------------------------------------------------------------------
# Frame-level multi-source supervision (routes A/B/C).
#
# These losses operate on the fused ``local_spatial`` embedding sequence and
# are only used when ``supervision_mode`` is one of:
#   * ``local_spatial_slot``   — per-frame K-slot head with per-step Hungarian
#                                matching between slots and active sources.
#   * ``local_spatial_track``  — K track queries with one clip-level match per
#                                track; frame-level supervision on matched
#                                tracks using the source's weak time window.
#   * ``local_spatial_accdoa`` — per-class ACCDOA vector field without
#                                matching (single source per class per frame).
#
# Additional utilities:
#   * ``compute_clip_aux_losses``: reuses the existing mono_ast-style clip
#     head to provide a weak single-source auxiliary gradient. Only applied
#     when every sample in the batch has exactly one active source.
# ---------------------------------------------------------------------------


@dataclass
class FrameMetricOutput:
    """Structured validation metrics for frame-level multi-source heads.

    Tier-1 (honest, detection-gated):
        activity_acc / precision / recall: threshold pred_activity >= 0.5
        class_acc / azi_mae_deg / ele_mae_deg / dist_mae:
            evaluated ONLY on frames where model predicts active AND GT is active
            (i.e., the model had to first correctly detect the frame before it is
            counted in the spatial metric)

    Tier-2 (oracle, debugging):
        oracle_class_acc / oracle_azi_mae_deg / oracle_ele_mae_deg / oracle_dist_mae:
            evaluated on GT-active frames regardless of model's activity prediction.
            Shows spatial head quality independent of the activity head.
            Useful during early training when activity has not yet converged.
    """

    activity_acc: Tensor
    activity_precision: Tensor
    activity_recall: Tensor
    class_acc: Tensor
    azi_mae_deg: Tensor
    ele_mae_deg: Tensor
    dist_mae: Tensor
    matched_count: Tensor
    # Tier-2 oracle metrics (suffixed _oracle)
    oracle_class_acc: Tensor
    oracle_azi_mae_deg: Tensor
    oracle_ele_mae_deg: Tensor
    oracle_dist_mae: Tensor


def _frame_source_target_tensors(
    batch: "SpatialBatch",
    t_s_max: int,
    device: torch.device,
) -> Dict[str, Tensor]:
    """Build the frame-level source activity/class/direction/distance tensors.

    Returns:
        Dict[str, Tensor] with shapes:
            window_mask: [B, N_gt, T_s]
            source_valid: [B, N_gt]
            source_class: [B, N_gt]
            source_direction: [B, N_gt, T_s, 3] (unit vectors, per frame)
            source_distance: [B, N_gt, T_s]
            source_distance_valid: [B, N_gt, T_s] (per-frame validity)
            source_azimuth_deg: [B, N_gt, T_s]
            source_elevation_deg: [B, N_gt, T_s]

    For static sources, per-frame tensors carry the same scalar along the T_s
    axis (set by the loader).  For dynamic sources (qa_moving, DCASE), the
    T_s axis holds the interpolated trajectory.  Downstream code should
    index per-step (e.g. ``source_direction[b, gt, t]`` not ``[b, gt]``).
    """
    source_valid = batch.source_valid_mask.to(device)
    source_class = batch.source_class_indices.to(device)
    # Per-frame targets: shape [B, N_gt, T_s_loader].  Loader may have used a
    # larger t_s_max than the current batch's t_s_max (e.g. when the longest
    # sample determines loader's t_s_max, but the frame-track head outputs
    # fewer steps).  Truncate/pad along the last axis to align with t_s_max.
    def _align_t(x: Tensor) -> Tensor:
        # Loader-side target tensors: [B, N_gt, T_s_loader].
        if x.ndim < 3:
            # Legacy path — shouldn't happen now but keep a safe fallback.
            return x.unsqueeze(-1).expand(*x.shape, t_s_max)
        if x.size(-1) == t_s_max:
            return x
        if x.size(-1) > t_s_max:
            return x[..., :t_s_max]
        # Pad along T with repeats of the last value (static tail).
        pad = t_s_max - x.size(-1)
        last = x[..., -1:].expand(*x.shape[:-1], pad)
        return torch.cat([x, last], dim=-1)

    azi_deg = _align_t(batch.source_azimuth_deg.to(device))      # [B, N_gt, T_s]
    ele_deg = _align_t(batch.source_elevation_deg.to(device))    # [B, N_gt, T_s]
    source_direction = _direction_vector_from_azi_ele_deg(azi_deg, ele_deg)  # [B, N_gt, T_s, 3]
    source_distance = _align_t(batch.source_distance.to(device))  # [B, N_gt, T_s]
    # source_distance_valid: [B, N_gt, T_s] — False for sources/frames with null distance
    # (e.g. STARSS real data, DCASE).  Falls back to all-True for legacy batches.
    if hasattr(batch, "source_distance_valid") and batch.source_distance_valid is not None:
        src_dv = batch.source_distance_valid.to(device)
        source_distance_valid = _align_t(src_dv) if src_dv.ndim >= 3 else src_dv.unsqueeze(-1).expand(*src_dv.shape, t_s_max)
    else:
        source_distance_valid = torch.ones(
            source_valid.size(0), source_valid.size(1), t_s_max,
            dtype=torch.bool, device=device,
        )

    window_mask = build_time_window_mask(
        source_start_time_seconds=batch.source_start_time_seconds.to(device),
        source_end_time_seconds=batch.source_end_time_seconds.to(device),
        source_valid_mask=source_valid,
        clip_duration_seconds=batch.clip_duration_seconds.to(device),
        target_num_steps=batch.target_num_steps.to(device),
        t_s_max=t_s_max,
    )
    # source_ele_sign_only: [B, N_gt, T_s] bool — True when only hemisphere is
    # known for that (source, frame) pair.  Falls back to all-False for batches
    # from the old dataset format that lack this field.
    if hasattr(batch, "source_ele_sign_only") and batch.source_ele_sign_only is not None:
        src_eso = batch.source_ele_sign_only.to(device)
        source_ele_sign_only = _align_t(src_eso) if src_eso.ndim >= 3 else src_eso.unsqueeze(-1).expand(*src_eso.shape, t_s_max)
    else:
        source_ele_sign_only = torch.zeros(
            source_valid.size(0), source_valid.size(1), t_s_max,
            dtype=torch.bool, device=device,
        )
    return {
        "window_mask": window_mask,
        "source_valid": source_valid,
        "source_class": source_class,
        "source_direction": source_direction,
        "source_distance": source_distance,
        "source_distance_valid": source_distance_valid,
        "source_azimuth_deg": azi_deg,
        "source_elevation_deg": ele_deg,
        "source_ele_sign_only": source_ele_sign_only,
    }


def _valid_time_mask(
    temporal_padding_mask: Optional[Tensor],
    batch_size: int,
    t_s_max: int,
    device: torch.device,
) -> Tensor:
    if temporal_padding_mask is None:
        return torch.ones(batch_size, t_s_max, dtype=torch.bool, device=device)
    return ~temporal_padding_mask.to(device=device, dtype=torch.bool)


def _masked_mean(values: Tensor, mask: Tensor, eps: float = 1e-8) -> Tensor:
    mask = mask.to(values.dtype)
    denom = mask.sum().clamp(min=eps)
    return (values * mask).sum() / denom


def compute_clip_aux_losses(
    prediction_output: Optional[MonoTaskPredictionOutput],
    batch: "SpatialBatch",
    config: SpatialLossConfig,
) -> Tensor:
    """Auxiliary clip-level single-source loss on top of frame supervision.

    Returns a scalar. The auxiliary head is defined for single-source clips, so
    mixed-source batches compute the loss on the single-source subset only. If
    a batch has no eligible samples, return a graph-connected zero scalar so
    DDP still sees every aux-head parameter participate in backward.
    """
    if prediction_output is None:
        device = batch.source_class_indices.device
        return torch.zeros((), device=device)

    def _zero_connected() -> Tensor:
        zero = prediction_output.pred_class_logits.sum() * 0.0
        zero = zero + prediction_output.pred_direction.sum() * 0.0
        zero = zero + prediction_output.pred_distance.sum() * 0.0
        if prediction_output.sem_class_logits is not None:
            zero = zero + prediction_output.sem_class_logits.sum() * 0.0
        return zero

    device = prediction_output.pred_class_logits.device
    valid_counts = batch.source_valid_mask.to(device).sum(dim=1)
    single_source_index = torch.nonzero(valid_counts == 1, as_tuple=False).flatten()
    if single_source_index.numel() == 0:
        return _zero_connected()

    cls_target = batch.source_class_indices.to(device)[single_source_index, 0]
    # Clip-level single-source path: take first frame of first source.
    azi_deg = batch.source_azimuth_deg.to(device)[single_source_index, 0, 0]
    ele_deg = batch.source_elevation_deg.to(device)[single_source_index, 0, 0]
    dist_t = batch.source_distance.to(device)[single_source_index, 0, 0]
    gt_direction = _direction_vector_from_azi_ele_deg(azi_deg, ele_deg).to(
        prediction_output.pred_direction.dtype
    )

    pred_class_logits = prediction_output.pred_class_logits[single_source_index]
    pred_direction = F.normalize(prediction_output.pred_direction[single_source_index], dim=-1)
    pred_distance = prediction_output.pred_distance[single_source_index]

    loss_cls = F.cross_entropy(pred_class_logits, cls_target)
    loss_direction = (1.0 - (pred_direction * gt_direction).sum(dim=-1)).mean()
    loss_dist = F.smooth_l1_loss(pred_distance[:, 0], dist_t)
    loss = loss_cls + loss_direction + loss_dist
    # Also include semantic_anchor_head loss if present, so its parameters
    # always receive gradients in DDP (prevents find_unused_parameters errors).
    if prediction_output.sem_class_logits is not None:
        sem_class_logits = prediction_output.sem_class_logits[single_source_index]
        loss = loss + F.cross_entropy(sem_class_logits, cls_target) * 0.0
        if config.lambda_sem_anchor > 0.0:
            loss = loss + config.lambda_sem_anchor * F.cross_entropy(
                sem_class_logits, cls_target,
                label_smoothing=config.label_smoothing,
            )
    return loss


# ---------------------------------------------------------------------------
# Route A — per-frame K-slot head with per-step Hungarian matching.
# ---------------------------------------------------------------------------


def _match_frame_slots_per_step(
    prediction_output: FrameSlotPredictionOutput,
    batch: "SpatialBatch",
    temporal_padding_mask: Optional[Tensor],
    target_direction: Tensor,
    target_distance: Tensor,
    target_class: Tensor,
    window_mask: Tensor,
) -> Tensor:
    """Return matched slot index per (b, gt, t): [B, N_gt, T_s] with -1 when unset.

    Cost combines activity, class NLL, direction cos-distance and distance L1.
    """
    pred_activity = prediction_output.pred_activity.detach()
    pred_class = prediction_output.pred_class_logits.detach()
    pred_direction = prediction_output.pred_direction.detach()
    pred_distance = prediction_output.pred_distance.detach()

    batch_size, t_s_max, num_slots = pred_activity.shape
    num_gt = target_class.size(1)
    device = pred_activity.device
    matched = torch.full(
        (batch_size, num_gt, t_s_max), -1, dtype=torch.long, device=device
    )

    valid_steps_batch = batch.target_num_steps.to(device).tolist()
    pad_mask = temporal_padding_mask.to(device=device, dtype=torch.bool) if temporal_padding_mask is not None else None

    for b in range(batch_size):
        valid_steps = min(int(valid_steps_batch[b]), t_s_max)
        for t in range(valid_steps):
            if pad_mask is not None and bool(pad_mask[b, t]):
                continue
            active_gt = torch.nonzero(window_mask[b, :, t], as_tuple=False).flatten()
            if active_gt.numel() == 0:
                continue
            if active_gt.numel() > num_slots:
                active_gt = active_gt[:num_slots]

            act_cost = 1.0 - torch.sigmoid(pred_activity[b, t])  # [K]
            cost = torch.zeros(active_gt.numel(), num_slots, device=device)
            for row, gt_index in enumerate(active_gt):
                gt_idx = int(gt_index.item())
                gt_class = int(target_class[b, gt_idx].item())
                cls_nll = -F.log_softmax(pred_class[b, t], dim=-1)[:, gt_class]  # [K]
                dir_cos = (
                    pred_direction[b, t] * target_direction[b, gt_idx].unsqueeze(0)
                ).sum(dim=-1)
                dir_cost = 1.0 - dir_cos  # [K]
                dist_cost = torch.abs(
                    pred_distance[b, t] - target_distance[b, gt_idx]
                )  # [K]
                cost[row] = act_cost + cls_nll + dir_cost + dist_cost

            assignment = _match_active_sources_to_slots(cost)
            for row, gt_index in enumerate(active_gt):
                matched[b, int(gt_index.item()), t] = assignment[row]
    return matched


def compute_frame_slot_losses(
    prediction_output: FrameSlotPredictionOutput,
    batch: "SpatialBatch",
    temporal_padding_mask: Optional[Tensor],
    config: SpatialLossConfig,
    clip_aux_prediction: Optional[MonoTaskPredictionOutput] = None,
) -> SpatialLossOutput:
    """Compute Route A (per-frame K-slot) losses."""
    device = prediction_output.pred_activity.device
    batch_size, t_s_max, num_slots = prediction_output.pred_activity.shape
    targets = _frame_source_target_tensors(batch, t_s_max, device)
    window_mask = targets["window_mask"]
    valid_time = _valid_time_mask(temporal_padding_mask, batch_size, t_s_max, device)

    matched_slot = _match_frame_slots_per_step(
        prediction_output=prediction_output,
        batch=batch,
        temporal_padding_mask=temporal_padding_mask,
        target_direction=targets["source_direction"],
        target_distance=targets["source_distance"],
        target_class=targets["source_class"],
        window_mask=window_mask,
    )

    # Activity target: for each (b, t, k) = 1 if some GT is matched to k at (b, t).
    activity_target = torch.zeros_like(prediction_output.pred_activity)
    valid_assign = matched_slot >= 0
    if valid_assign.any():
        idx_b, idx_gt, idx_t = torch.nonzero(valid_assign, as_tuple=True)
        idx_k = matched_slot[idx_b, idx_gt, idx_t]
        activity_target[idx_b, idx_t, idx_k] = 1.0

    activity_mask = valid_time.unsqueeze(-1).expand_as(prediction_output.pred_activity).to(device)
    activity_bce = F.binary_cross_entropy_with_logits(
        prediction_output.pred_activity,
        activity_target,
        reduction="none",
    )
    loss_activity = _masked_mean(activity_bce, activity_mask)

    # Matched-only losses.
    zero = loss_activity.new_zeros(())
    if valid_assign.any():
        idx_b_m, idx_gt_m, idx_t_m = torch.nonzero(valid_assign, as_tuple=True)
        idx_k_m = matched_slot[idx_b_m, idx_gt_m, idx_t_m]

        pred_class_sel = prediction_output.pred_class_logits[idx_b_m, idx_t_m, idx_k_m]
        cls_target = targets["source_class"][idx_b_m, idx_gt_m]
        loss_class = F.cross_entropy(pred_class_sel, cls_target)

        pred_dir_sel = prediction_output.pred_direction[idx_b_m, idx_t_m, idx_k_m]
        tgt_dir_sel = targets["source_direction"][idx_b_m, idx_gt_m, idx_t_m].to(pred_dir_sel.dtype)
        pred_dir_sel = F.normalize(pred_dir_sel, dim=-1)
        loss_direction = (1.0 - (pred_dir_sel * tgt_dir_sel).sum(dim=-1)).mean()

        pred_dist_sel = prediction_output.pred_distance[idx_b_m, idx_t_m, idx_k_m]
        tgt_dist_sel = targets["source_distance"][idx_b_m, idx_gt_m, idx_t_m].to(pred_dist_sel.dtype)
        loss_distance = F.smooth_l1_loss(pred_dist_sel, tgt_dist_sel)
    else:
        # Ensure graph connectivity to the heads even when no GT is active.
        loss_class = prediction_output.pred_class_logits.sum() * 0.0
        loss_direction = prediction_output.pred_direction.sum() * 0.0
        loss_distance = prediction_output.pred_distance.sum() * 0.0

    loss_clip = compute_clip_aux_losses(clip_aux_prediction, batch, config)
    loss_total = (
        config.lambda_frame_activity * loss_activity
        + config.lambda_frame_class * loss_class
        + config.lambda_frame_direction * loss_direction
        + config.lambda_frame_distance * loss_distance
        + config.lambda_clip_aux * loss_clip
    )
    return SpatialLossOutput(
        loss_total=loss_total,
        loss_activity=loss_activity,
        loss_azi=zero,
        loss_ele=zero,
        loss_dist=loss_distance,
        loss_cls_aux=loss_class,
        loss_temp=loss_clip,
        loss_direction=loss_direction,
    )


def compute_frame_slot_validation_metrics(
    prediction_output: FrameSlotPredictionOutput,
    batch: "SpatialBatch",
    temporal_padding_mask: Optional[Tensor],
    config: SpatialLossConfig,
) -> FrameMetricOutput:
    """Activity/class/angle/distance metrics for route A."""
    device = prediction_output.pred_activity.device
    batch_size, t_s_max, _ = prediction_output.pred_activity.shape
    targets = _frame_source_target_tensors(batch, t_s_max, device)
    window_mask = targets["window_mask"]
    valid_time = _valid_time_mask(temporal_padding_mask, batch_size, t_s_max, device)

    matched_slot = _match_frame_slots_per_step(
        prediction_output=prediction_output,
        batch=batch,
        temporal_padding_mask=temporal_padding_mask,
        target_direction=targets["source_direction"],
        target_distance=targets["source_distance"],
        target_class=targets["source_class"],
        window_mask=window_mask,
    )
    activity_target = torch.zeros_like(prediction_output.pred_activity)
    valid_assign = matched_slot >= 0
    if valid_assign.any():
        idx_b, idx_gt, idx_t = torch.nonzero(valid_assign, as_tuple=True)
        idx_k = matched_slot[idx_b, idx_gt, idx_t]
        activity_target[idx_b, idx_t, idx_k] = 1.0

    activity_mask = valid_time.unsqueeze(-1).expand_as(prediction_output.pred_activity)
    activity_pred = (torch.sigmoid(prediction_output.pred_activity) >= 0.5).float()
    valid_pred = activity_pred[activity_mask]
    valid_true = activity_target[activity_mask]
    if valid_true.numel() > 0:
        activity_acc = (valid_pred == valid_true).float().mean()
        tp = ((valid_pred == 1.0) & (valid_true == 1.0)).float().sum()
        pp = (valid_pred == 1.0).float().sum()
        gp = (valid_true == 1.0).float().sum()
        activity_precision = tp / torch.clamp(pp, min=1.0)
        activity_recall = tp / torch.clamp(gp, min=1.0)
    else:
        z = prediction_output.pred_activity.sum() * 0.0
        activity_acc = z
        activity_precision = z
        activity_recall = z

    if valid_assign.any():
        idx_b_m, idx_gt_m, idx_t_m = torch.nonzero(valid_assign, as_tuple=True)
        idx_k_m = matched_slot[idx_b_m, idx_gt_m, idx_t_m]
        pred_class_idx = prediction_output.pred_class_logits[idx_b_m, idx_t_m, idx_k_m].argmax(dim=-1)
        cls_target = targets["source_class"][idx_b_m, idx_gt_m]
        class_acc = (pred_class_idx == cls_target).float().mean()

        pred_dir = F.normalize(
            prediction_output.pred_direction[idx_b_m, idx_t_m, idx_k_m], dim=-1
        )
        pred_azi_deg, pred_ele_deg = _azi_ele_deg_from_direction_vector(pred_dir)
        azi_tgt = targets["source_azimuth_deg"][idx_b_m, idx_gt_m, idx_t_m].to(pred_azi_deg.dtype)
        ele_tgt = targets["source_elevation_deg"][idx_b_m, idx_gt_m, idx_t_m].to(pred_ele_deg.dtype)
        azi_mae = _circular_distance_deg(pred_azi_deg, azi_tgt).mean()
        ele_mae = torch.abs(pred_ele_deg - ele_tgt).mean()

        pred_dist = prediction_output.pred_distance[idx_b_m, idx_t_m, idx_k_m]
        dist_tgt = targets["source_distance"][idx_b_m, idx_gt_m, idx_t_m].to(pred_dist.dtype)
        dist_mae = torch.abs(pred_dist - dist_tgt).mean()
        matched_count = torch.tensor(
            float(idx_b_m.numel()), device=device, dtype=prediction_output.pred_activity.dtype
        )
    else:
        z = prediction_output.pred_activity.sum() * 0.0
        class_acc = z
        azi_mae = z
        ele_mae = z
        dist_mae = z
        matched_count = z

    return FrameMetricOutput(
        activity_acc=activity_acc,
        activity_precision=activity_precision,
        activity_recall=activity_recall,
        class_acc=class_acc,
        azi_mae_deg=azi_mae,
        ele_mae_deg=ele_mae,
        dist_mae=dist_mae,
        matched_count=matched_count,
        # Slot route has no separate oracle pipeline; the matched-slot metrics
        # above are evaluated against GT-active assignments, so they double as
        # the oracle tier.  This keeps the FrameMetricOutput contract uniform.
        oracle_class_acc=class_acc,
        oracle_azi_mae_deg=azi_mae,
        oracle_ele_mae_deg=ele_mae,
        oracle_dist_mae=dist_mae,
    )


def build_frame_slot_validation_examples(
    prediction_output: FrameSlotPredictionOutput,
    batch: "SpatialBatch",
    temporal_padding_mask: Optional[Tensor],
    max_examples: int = 16,
) -> List[Dict[str, object]]:
    """Qualitative per-(clip, time, gt) examples for route A."""
    device = prediction_output.pred_activity.device
    batch_size, t_s_max, _ = prediction_output.pred_activity.shape
    targets = _frame_source_target_tensors(batch, t_s_max, device)
    window_mask = targets["window_mask"]
    matched_slot = _match_frame_slots_per_step(
        prediction_output=prediction_output,
        batch=batch,
        temporal_padding_mask=temporal_padding_mask,
        target_direction=targets["source_direction"],
        target_distance=targets["source_distance"],
        target_class=targets["source_class"],
        window_mask=window_mask,
    )
    examples: List[Dict[str, object]] = []
    valid_assign = matched_slot >= 0
    if not valid_assign.any():
        return examples
    idx_b, idx_gt, idx_t = torch.nonzero(valid_assign, as_tuple=True)
    idx_k = matched_slot[idx_b, idx_gt, idx_t]
    limit = min(int(max_examples), int(idx_b.numel()))
    for i in range(limit):
        b = int(idx_b[i].item())
        gt = int(idx_gt[i].item())
        t = int(idx_t[i].item())
        k = int(idx_k[i].item())
        pred_dir = F.normalize(prediction_output.pred_direction[b, t, k], dim=-1)
        pred_azi_deg, pred_ele_deg = _azi_ele_deg_from_direction_vector(pred_dir)
        pred_cls = int(prediction_output.pred_class_logits[b, t, k].argmax().item())
        examples.append(
            {
                "sample_id": batch.sample_ids[b],
                "time_index": t,
                "slot_index": k,
                "gt_index": gt,
                "gt_class_index": int(batch.source_class_indices[b, gt].item()),
                "pred_class_index": pred_cls,
                "gt_azimuth_deg": float(batch.source_azimuth_deg[b, gt, t].item()),
                "pred_azimuth_deg": float(pred_azi_deg.item()),
                "gt_elevation_deg": float(batch.source_elevation_deg[b, gt, t].item()),
                "pred_elevation_deg": float(pred_ele_deg.item()),
                "gt_distance": float(batch.source_distance[b, gt, t].item()),
                "pred_distance": float(prediction_output.pred_distance[b, t, k].item()),
                "pred_activity_prob": float(
                    torch.sigmoid(prediction_output.pred_activity[b, t, k]).item()
                ),
            }
        )
    return examples


# ---------------------------------------------------------------------------
# Route B — K track queries with PER-FRAME Hungarian matching.
#
# DCASE-style: at every frame t, at most 3 sources can be simultaneously active,
# so K=4 query slots always cover the maximum overlap.  We Hungarian-match the
# set of GT-active sources at frame t against the K track queries at the same
# frame t, independently for every (b, t).  No clip-level aggregation, no GT
# truncation, no clip-level head.
# ---------------------------------------------------------------------------


def _match_frame_tracks_per_frame(
    prediction_output: FrameTrackPredictionOutput,
    target_class: Tensor,
    target_direction: Tensor,
    target_distance: Tensor,
    source_valid: Tensor,
    window_mask: Tensor,
    valid_time: Tensor,
    include_activity_cost: bool = True,
    class_cost_weight: float = 1.0,
    dir_cost_weight: float = 1.0,
    dist_cost_weight: float = 1.0,
) -> Tensor:
    """Match K tracks to GT active sources at EVERY frame independently.

    Returns:
        Tensor [B, N_gt, T_s] with the matched track index (0..K-1) or -1.
        A value >= 0 at (b, n, t) means: GT source n is active at frame t of
        clip b, and it is assigned to track ``matched[b, n, t]``.
    """
    # Fully vectorized matcher — no Python per-(b, t) loop, no `.item()` syncs.
    # Equivalence with the reference CPU-loop implementation is covered by
    # test_vectorized_matching.py (400+ random trials on both CPU and CUDA).
    pred_activity = prediction_output.pred_activity.detach()  # [B, K, T_s]
    pred_class = prediction_output.pred_class_logits.detach()  # [B, K, T_s, C]
    pred_direction = prediction_output.pred_direction.detach()  # [B, K, T_s, 3]
    pred_distance = prediction_output.pred_distance.detach()  # [B, K, T_s]

    B, K, T = pred_activity.shape
    N = target_class.size(1)
    device = pred_activity.device

    # ---- Full cost tensor [B, N, K, T] via broadcasting ----
    cls_log = F.log_softmax(pred_class, dim=-1)  # [B, K, T, C]
    tc = target_class.clamp(min=0)  # [B, N]
    tc_exp = tc.view(B, N, 1, 1, 1).expand(B, N, K, T, 1)
    cls_log_exp = cls_log.unsqueeze(1).expand(B, N, K, T, cls_log.size(-1))
    cls_nll = -cls_log_exp.gather(-1, tc_exp).squeeze(-1)  # [B, N, K, T]

    pd = pred_direction.unsqueeze(1)  # [B, 1, K, T, 3]
    # target_direction: [B, N, T, 3] (per-frame).  Broadcast to [B, N, 1, T, 3].
    td = target_direction.unsqueeze(2)
    dir_cost = 1.0 - (pd * td).sum(dim=-1)  # [B, N, K, T]

    pdist = pred_distance.unsqueeze(1)  # [B, 1, K, T]
    # target_distance: [B, N, T] (per-frame).  Broadcast to [B, N, 1, T].
    tdist = target_distance.unsqueeze(2)
    dist_cost = torch.abs(pdist - tdist)  # [B, N, K, T]

    cost = class_cost_weight * cls_nll + dir_cost_weight * dir_cost + dist_cost_weight * dist_cost  # [B, N, K, T]
    if include_activity_cost:
        act_cost = (1.0 - torch.sigmoid(pred_activity)).unsqueeze(1)  # [B, 1, K, T]
        cost = cost + act_cost

    # ---- GT active mask, grouped by n_active ----
    gt_active = (
        window_mask & source_valid.unsqueeze(-1) & valid_time.unsqueeze(1)
    )  # [B, N, T]

    matched = torch.full((B, N, T), -1, dtype=torch.long, device=device)

    active_count = gt_active.sum(dim=1)  # [B, T]
    # DCASE guarantees ≤ K simultaneous sources; truncate excess to match the
    # reference behavior for the edge case.
    active_count = torch.clamp(active_count, max=K)
    active_count = torch.where(valid_time, active_count, torch.zeros_like(active_count))

    n_max = int(min(K, N))
    arange_K = torch.arange(K, device=device)
    for n_active in range(1, n_max + 1):
        mask_bt = active_count == n_active  # [B, T]
        if not mask_bt.any():
            continue

        bt = mask_bt.nonzero(as_tuple=False)  # [M, 2]
        b_idx = bt[:, 0]
        t_idx = bt[:, 1]
        M = b_idx.size(0)

        # Per-(b, t) ascending GT indices (stable argsort of bool->int picks
        # True positions in their original order to match the reference).
        ga = gt_active[b_idx, :, t_idx]  # [M, N]
        sort_idx = ga.to(torch.int8).argsort(dim=1, descending=True, stable=True)
        gt_indices = sort_idx[:, :n_active]  # [M, n_active]

        # cost_sub[m, i, k] = cost[b_idx[m], gt_indices[m, i], k, t_idx[m]]
        cost_sub = cost[
            b_idx.view(M, 1, 1).expand(M, n_active, K),
            gt_indices.view(M, n_active, 1).expand(M, n_active, K),
            arange_K.view(1, 1, K).expand(M, n_active, K),
            t_idx.view(M, 1, 1).expand(M, n_active, K),
        ]  # [M, n_active, K]

        # All K-taken-n_active permutations (≤ 24 for K=4).  Same enumeration
        # order as itertools.permutations → tie-breaking matches the reference.
        perms = torch.tensor(
            list(itertools.permutations(range(K), n_active)),
            dtype=torch.long,
            device=device,
        )  # [P, n_active]
        P = perms.size(0)

        cost_sub_exp = cost_sub.unsqueeze(1).expand(M, P, n_active, K)
        perms_idx = perms.view(1, P, n_active, 1).expand(M, P, n_active, 1)
        gathered = cost_sub_exp.gather(3, perms_idx).squeeze(-1)  # [M, P, n_active]
        perm_cost = gathered.sum(dim=2)  # [M, P]

        best_perm_idx = perm_cost.argmin(dim=1)  # [M]
        best_perms = perms[best_perm_idx]  # [M, n_active]

        for i in range(n_active):
            matched[b_idx, gt_indices[:, i], t_idx] = best_perms[:, i]

    return matched


def _hungarian_assign_n_by_k(cost_np: "np.ndarray", n: int, K: int) -> List[int]:
    """Brute-force optimal assignment of n GT sources to K tracks (n ≤ K ≤ 4).

    Returns a list of length n: assign[i] = track index for GT i.
    Uses full permutation enumeration (≤ 24 perms for K=4) so it is exact and
    avoids a scipy dependency.
    """
    best_cost = float("inf")
    best_assign: List[int] = list(range(n))
    for perm in itertools.permutations(range(K), n):
        c = sum(cost_np[i, perm[i]] for i in range(n))
        if c < best_cost:
            best_cost = c
            best_assign = list(perm)
    return best_assign


def _match_frame_tracks_per_segment(
    prediction_output: FrameTrackPredictionOutput,
    target_class: Tensor,
    target_direction: Tensor,
    target_distance: Tensor,
    source_valid: Tensor,
    window_mask: Tensor,
    valid_time: Tensor,
    include_activity_cost: bool = True,
    class_cost_weight: float = 1.0,
    dir_cost_weight: float = 1.0,
    dist_cost_weight: float = 1.0,
) -> Tensor:
    """Segment-level matching with track continuity across frames.

    Algorithm:
      For each clip b:
        1. Detect contiguous segments where the active GT source set is constant.
        2. For each segment, run Hungarian matching on the segment-averaged cost.
        3. Apply a continuity bonus (-2.0) to re-use the same track for a GT
           source that was active in the previous segment.  This means a source
           that persists across a segment boundary stays on the same track.
        4. Non-overlapping sources (active at different times) can share a track.

    Returns matched [B, N_gt, T_s] with the same semantics as
    _match_frame_tracks_per_frame:  matched[b, n, t] = track index (0..K-1)
    or -1 if source n is not active at frame t of clip b.

    Compared to per-frame matching this eliminates the track-identity flipping
    that caused activity BCE targets to look random and suppressed K-1 tracks.
    """
    pred_activity  = prediction_output.pred_activity.detach()   # [B, K, T]
    pred_class     = prediction_output.pred_class_logits.detach()  # [B, K, T, C]
    pred_direction = prediction_output.pred_direction.detach()   # [B, K, T, 3]
    pred_distance  = prediction_output.pred_distance.detach()    # [B, K, T]

    B, K, T = pred_activity.shape
    N = target_class.size(1)
    device = pred_activity.device

    # ---- Pre-build full cost tensor [B, N, K, T] ----
    cls_log = F.log_softmax(pred_class, dim=-1)       # [B, K, T, C]
    tc = target_class.clamp(min=0)                     # [B, N]
    tc_exp = tc.view(B, N, 1, 1, 1).expand(B, N, K, T, 1)
    cls_log_exp = cls_log.unsqueeze(1).expand(B, N, K, T, cls_log.size(-1))
    cls_nll = -cls_log_exp.gather(-1, tc_exp).squeeze(-1)  # [B, N, K, T]

    pd = pred_direction.unsqueeze(1)               # [B, 1, K, T, 3]
    # target_direction: [B, N, T, 3] (per-frame).  Broadcast to [B, N, 1, T, 3].
    td = target_direction.unsqueeze(2)
    dir_cost = 1.0 - (pd * td).sum(dim=-1)         # [B, N, K, T]

    pdist = pred_distance.unsqueeze(1)             # [B, 1, K, T]
    # target_distance: [B, N, T] (per-frame).  Broadcast to [B, N, 1, T].
    tdist = target_distance.unsqueeze(2)
    dist_cost = torch.abs(pdist - tdist)           # [B, N, K, T]

    cost_full = (
        class_cost_weight  * cls_nll
        + dir_cost_weight  * dir_cost
        + dist_cost_weight * dist_cost
    )  # [B, N, K, T]
    if include_activity_cost:
        act_cost = (1.0 - torch.sigmoid(pred_activity)).unsqueeze(1)  # [B, 1, K, T]
        cost_full = cost_full + act_cost

    # ---- GT active mask ----
    gt_active = (
        window_mask & source_valid.unsqueeze(-1) & valid_time.unsqueeze(1)
    )  # [B, N, T]

    matched = torch.full((B, N, T), -1, dtype=torch.long, device=device)

    cost_cpu = cost_full.cpu().numpy()       # move once to CPU for numpy ops
    gt_active_cpu = gt_active.cpu().numpy()  # [B, N, T]
    valid_time_cpu = valid_time.cpu().numpy()  # [B, T]

    for b in range(B):
        T_valid = int(valid_time_cpu[b].sum())
        if T_valid == 0:
            continue

        # active pattern: [N, T_valid] — which GT sources are active each frame
        pattern = gt_active_cpu[b, :, :T_valid]  # [N, T_valid]

        # ---- Segment detection: boundary whenever active set changes ----
        # seg_starts contains the t-indices where a new segment begins (0-indexed)
        seg_starts: List[int] = [0]
        for t in range(1, T_valid):
            if not (pattern[:, t] == pattern[:, t - 1]).all():
                seg_starts.append(t)
        seg_starts.append(T_valid)  # sentinel end

        # Track which track each GT source was last assigned to (for continuity).
        prev_assign: Dict[int, int] = {}   # gt_n -> track_k

        for si in range(len(seg_starts) - 1):
            t0 = seg_starts[si]
            t1 = seg_starts[si + 1]
            seg_len = t1 - t0

            # Which GT sources are active in this segment?
            active_ns_all = [n for n in range(N) if pattern[n, t0]]

            # Keep segment matching consistent with the per-frame matcher:
            # if more than K sources are active in an edge-case segment, only
            # the first K are supervised. Prefer already-tracked sources so we
            # do not break continuity gratuitously when the overflow appears.
            if len(active_ns_all) > K:
                carried = [
                    n for n in active_ns_all
                    if n in prev_assign and 0 <= prev_assign[n] < K
                ]
                fresh = [n for n in active_ns_all if n not in carried]
                active_ns = (carried + fresh)[:K]
            else:
                active_ns = active_ns_all

            if not active_ns:
                # Silent segment — clear continuity for sources that went inactive
                new_prev: Dict[int, int] = {}
                for n in range(N):
                    if n in prev_assign and pattern[n, t0]:
                        new_prev[n] = prev_assign[n]
                prev_assign = new_prev
                continue

            n_act = len(active_ns)

            # Segment-averaged cost [n_act, K]
            seg_cost = cost_cpu[b][active_ns][:, :, t0:t1].mean(axis=-1)  # [n_act, K]

            # Continuity bonus: strongly prefer reusing the same track
            for i, n in enumerate(active_ns):
                if n in prev_assign and 0 <= prev_assign[n] < K:
                    seg_cost[i, prev_assign[n]] -= 2.0

            assign = _hungarian_assign_n_by_k(seg_cost, n_act, K)
            # assign[i] = track for active_ns[i]

            for i, n in enumerate(active_ns):
                k = assign[i]
                matched[b, n, t0:t1] = k
                prev_assign[n] = k

            # Drop continuity for sources that are no longer active
            prev_assign = {n: k for n, k in prev_assign.items() if pattern[n, t0]}

    return matched


def _match_frame_tracks(
    prediction_output: FrameTrackPredictionOutput,
    target_class: Tensor,
    target_direction: Tensor,
    target_distance: Tensor,
    source_valid: Tensor,
    window_mask: Tensor,
    valid_time: Tensor,
    config: SpatialLossConfig,
    include_activity_cost: bool = True,
) -> Tensor:
    """Dispatch track matching using the configured matcher.

    This keeps training loss, oracle metrics, and qualitative examples on the
    same matching semantics. When segment matching is enabled, diagnostics
    should not silently fall back to per-frame Hungarian.
    """
    match_kwargs = dict(
        prediction_output=prediction_output,
        target_class=target_class,
        target_direction=target_direction,
        target_distance=target_distance,
        source_valid=source_valid,
        window_mask=window_mask,
        valid_time=valid_time,
        include_activity_cost=include_activity_cost,
        class_cost_weight=config.frame_match_class_cost_weight,
        dir_cost_weight=config.frame_match_dir_cost_weight,
        dist_cost_weight=config.frame_match_dist_cost_weight,
    )
    if config.use_segment_matching:
        return _match_frame_tracks_per_segment(**match_kwargs)
    return _match_frame_tracks_per_frame(**match_kwargs)


def _compute_track_oracle_metrics(
    prediction_output: FrameTrackPredictionOutput,
    matched_track: Tensor,
    targets: Dict[str, Tensor],
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Oracle frame-track metrics on GT-active pairs after per-frame matching.

    Matching is assumed to already be computed. Oracle here means:
      - only GT-active (b, gt, t) cells are counted
      - no activity threshold is applied to the matched track
    """
    device = prediction_output.pred_activity.device
    zero = prediction_output.pred_activity.sum() * 0.0
    valid_assign = matched_track >= 0
    if not valid_assign.any():
        return zero, zero, zero, zero, zero

    idx_b, idx_gt, idx_t = torch.nonzero(valid_assign, as_tuple=True)
    idx_k = matched_track[idx_b, idx_gt, idx_t]

    pred_class_idx = prediction_output.pred_class_logits[idx_b, idx_k, idx_t].argmax(dim=-1)
    cls_target = targets["source_class"][idx_b, idx_gt]
    oracle_class_acc = (pred_class_idx == cls_target).float().mean()

    pred_dir = F.normalize(prediction_output.pred_direction[idx_b, idx_k, idx_t], dim=-1)
    pred_azi_deg, pred_ele_deg = _azi_ele_deg_from_direction_vector(pred_dir)
    azi_tgt = targets["source_azimuth_deg"][idx_b, idx_gt, idx_t].to(pred_azi_deg.dtype)
    ele_tgt = targets["source_elevation_deg"][idx_b, idx_gt, idx_t].to(pred_ele_deg.dtype)
    oracle_azi_mae = _circular_distance_deg(pred_azi_deg, azi_tgt).mean()
    oracle_ele_mae = torch.abs(pred_ele_deg - ele_tgt).mean()

    pred_dist = prediction_output.pred_distance[idx_b, idx_k, idx_t]
    dist_tgt = targets["source_distance"][idx_b, idx_gt, idx_t].to(pred_dist.dtype)
    oracle_dist_mae = torch.abs(pred_dist - dist_tgt).mean()
    matched_count = torch.tensor(
        float(idx_b.numel()),
        device=device,
        dtype=prediction_output.pred_activity.dtype,
    )
    return oracle_class_acc, oracle_azi_mae, oracle_ele_mae, oracle_dist_mae, matched_count


# =============================================================================
# legacy helper loss functions (B-2 ASL, B-4 soft-F1, C-4 Laplace NLL)
# =============================================================================


def _asymmetric_loss_with_logits(
    logits: Tensor,
    targets: Tensor,
    gamma_neg: float = 4.0,
    gamma_pos: float = 0.0,
    margin: float = 0.05,
) -> Tensor:
    """Asymmetric Loss (Ben-Baruch et al., ICCV 2021), per-element version.

    Returns the per-element ASL (same shape as ``logits``), suitable for
    masked aggregation downstream. Numerically stable: uses log1mexp-style
    tricks for positives via F.logsigmoid.

    Positive branch (y=1):    -(1 - p)^gamma_pos * log(p)
    Negative branch (y=0):    -(max(p - m, 0))^gamma_neg * log(1 - p_shifted)

    When ``gamma_pos=gamma_neg=0`` and ``margin=0`` the loss reduces to plain
    BCE (with reduction="none").
    """
    eps = 1e-8
    p = torch.sigmoid(logits)
    p_shifted = (p - margin).clamp(min=0.0) if margin > 0 else p
    # log(p) and log(1 - p_shifted) with numerical stability.
    log_p = F.logsigmoid(logits)                 # log(sigmoid(x))
    log_1mp = F.logsigmoid(-logits)              # log(1 - sigmoid(x))
    # For the shifted negative branch:
    #   log(1 - p_shifted) = log(1 - max(p - m, 0))
    # We compute this explicitly because the shifted prob is off the
    # sigmoid surface.
    one_minus_p_shifted = (1.0 - p_shifted).clamp(min=eps)
    log_1mps = torch.log(one_minus_p_shifted)

    loss_pos = -((1.0 - p).clamp(min=0.0) ** gamma_pos) * log_p
    loss_neg = -(p_shifted ** gamma_neg) * log_1mps
    return targets * loss_pos + (1.0 - targets) * loss_neg


def _soft_macro_f1_loss(
    activity_logits: Tensor,
    class_logits: Tensor,
    supervise_mask: Tensor,
    class_target: Tensor,
) -> Tensor:
    """Soft macro-F1 loss aligned with DCASE F20 metric.

    For each class c, compute:
        p_c = sigmoid(act_logit) * softmax(class_logits)[c]   [B, K, T, C]
        y_c = 1 iff (supervise_mask is True) and (class_target == c)
        tp_c = sum(p_c * y_c)
        fp_c = sum(p_c * (1 - y_c))
        fn_c = sum((1 - p_c) * y_c)
        f1_c = 2 tp_c / (2 tp_c + fp_c + fn_c + eps)
    Loss = 1 - mean(f1_c over classes).

    Shapes:
        activity_logits: [B, K, T]
        class_logits:    [B, K, T, C]
        supervise_mask:  [B, K, T]  (True on matched positive slots)
        class_target:    [B, K, T]  (valid on supervise_mask; 0 elsewhere)
    """
    eps = 1e-8
    B, K, T = activity_logits.shape
    C = class_logits.size(-1)
    act_prob = torch.sigmoid(activity_logits).unsqueeze(-1)         # [B, K, T, 1]
    cls_prob = F.softmax(class_logits, dim=-1)                      # [B, K, T, C]
    p_c = act_prob * cls_prob                                       # [B, K, T, C]

    # Build per-class target via one-hot × supervise_mask.
    y_c = F.one_hot(class_target.clamp(min=0), num_classes=C).to(p_c.dtype)  # [B, K, T, C]
    y_c = y_c * supervise_mask.unsqueeze(-1).to(p_c.dtype)

    # Sum over B, K, T per class.
    tp = (p_c * y_c).sum(dim=(0, 1, 2))                             # [C]
    fp = (p_c * (1.0 - y_c)).sum(dim=(0, 1, 2))                     # [C]
    fn = ((1.0 - p_c) * y_c).sum(dim=(0, 1, 2))                     # [C]

    # Only average over classes that have at least one GT positive to avoid
    # 0/0 on empty classes (otherwise all-zero classes contribute f1=0 and
    # dominate the mean).
    has_gt = (tp + fn > 0).to(p_c.dtype)                            # [C]
    f1 = 2.0 * tp / (2.0 * tp + fp + fn + eps)                      # [C]
    denom = has_gt.sum().clamp(min=1.0)
    mean_f1 = (f1 * has_gt).sum() / denom
    return 1.0 - mean_f1


def _topk_rank_activity_loss(
    activity_logit: Tensor,
    target_active: Tensor,
    valid_time: Tensor,
    margin: float = 2.0,
    bce_weight: float = 0.1,
) -> Tensor:
    """[D-2] Top-K rank activity loss.

    Instead of optimising each (b, k, t) position as an independent binary
    classification, this loss enforces pairwise ranking between active and
    inactive slots within each frame:

        For every (active slot i, inactive slot j) in the same frame,
        enforce  logit[i] > logit[j] + margin  via hinge loss
            loss_rank = mean over valid pairs of max(0, margin + logit[j] - logit[i])

    This aligns with the DCASE evaluation pipeline (take top-K̂ per frame),
    and sidesteps the per-sample positive/negative imbalance that plagues BCE
    and ASL on K=4 track outputs where most slots are negative.

    A small BCE anchor term (``bce_weight``) is added to prevent logit
    magnitude from drifting to ±inf (ranking loss alone is invariant to a
    global shift).

    Shapes:
        activity_logit:  [B, K, T]   (raw logits)
        target_active:   [B, K, T]   (0/1 binary)
        valid_time:      [B, T]      (bool / 0-1, True = valid frame)

    Returns:
        scalar loss (rank + bce_weight * anchor_bce).
    """
    # Pairwise hinge: want logit[active] > logit[inactive] + margin
    # act_mask_i: [B, K_i, 1, T]  — 1 if slot i is active
    # ina_mask_j: [B, 1, K_j, T]  — 1 if slot j is inactive
    act_mask_i = target_active.unsqueeze(2)          # [B, K, 1, T]
    ina_mask_j = (1.0 - target_active).unsqueeze(1)  # [B, 1, K, T]
    pair_mask = act_mask_i * ina_mask_j              # [B, K_i, K_j, T]
    # Broadcast logits
    logit_i = activity_logit.unsqueeze(2)            # [B, K, 1, T]
    logit_j = activity_logit.unsqueeze(1)            # [B, 1, K, T]
    # Hinge: max(0, margin + logit_j - logit_i)   — 0 iff logit_i > logit_j + margin
    hinge = F.relu(margin + logit_j - logit_i)       # [B, K_i, K_j, T]
    hinge = hinge * pair_mask                        # zero out non-pair positions
    # Per-frame normalisation: sum over (K_i, K_j) then divide by pair count
    pair_count = pair_mask.sum(dim=(1, 2))           # [B, T]
    hinge_per_frame = hinge.sum(dim=(1, 2))          # [B, T]
    # Only average over frames with at least one active-inactive pair AND valid time
    valid_frame_mask = (valid_time.to(pair_count.dtype) > 0) & (pair_count > 0)
    valid_frame_mask_f = valid_frame_mask.to(hinge_per_frame.dtype)
    denom = valid_frame_mask_f.sum().clamp(min=1.0)
    # Average hinge per pair: frames with more pairs should not dominate.
    # hinge_per_frame / pair_count is the mean hinge per pair in that frame.
    per_frame_mean = hinge_per_frame / pair_count.clamp(min=1.0)
    loss_rank = (per_frame_mean * valid_frame_mask_f).sum() / denom

    # Anchor BCE term to bound logit magnitude
    if bce_weight > 0:
        bce = F.binary_cross_entropy_with_logits(
            activity_logit, target_active, reduction='none'
        )  # [B, K, T]
        bce_mask = valid_time.unsqueeze(1).to(bce.dtype)  # [B, 1, T]
        bce = bce * bce_mask
        loss_anchor = bce.sum() / bce_mask.sum().clamp(min=1.0) / activity_logit.size(1)
        loss = loss_rank + bce_weight * loss_anchor
    else:
        loss = loss_rank
    return loss


def _laplace_nll_distance_loss(
    pred_distance: Tensor,
    pred_log_var: Tensor,
    target_distance: Tensor,
) -> Tensor:
    """Laplace negative log-likelihood for the [C-4] log-distance head.

    Model:  d ~ Laplace(mu=pred_distance, b=exp(0.5 * log_var))
    NLL:    |target - pred| / b + 0.5 * log_var   (+ log(2), constant)

    The ``log_var`` term penalises the model for claiming certainty.
    """
    eps = 1e-6
    b = torch.exp(0.5 * pred_log_var).clamp(min=eps)
    abs_err = (target_distance - pred_distance).abs()
    nll = abs_err / b + 0.5 * pred_log_var
    return nll.mean()


def compute_frame_track_losses(
    prediction_output: FrameTrackPredictionOutput,
    batch: "SpatialBatch",
    temporal_padding_mask: Optional[Tensor],
    config: SpatialLossConfig,
) -> SpatialLossOutput:
    """Route B loss with per-frame or segment-level Hungarian matching.

    When config.use_segment_matching=True, uses segment-level matching which
    keeps track identity consistent within each contiguous active-set segment
    (eliminates track-identity flipping that suppresses K-1 tracks).
    When False (default), uses the original per-frame Hungarian.

    Class CE uses per-class weights from config.frame_class_loss_weights if
    provided (helps rare classes like aircraft/insect/vehicle).
    """
    device = prediction_output.pred_activity.device
    batch_size, num_tracks, t_s_max = prediction_output.pred_activity.shape
    targets = _frame_source_target_tensors(batch, t_s_max, device)
    window_mask = targets["window_mask"]        # [B, N_gt, T_s]
    source_valid = targets["source_valid"]      # [B, N_gt]
    valid_time = _valid_time_mask(temporal_padding_mask, batch_size, t_s_max, device)

    matched_track = _match_frame_tracks(
        prediction_output=prediction_output,
        target_class=targets["source_class"],
        target_direction=targets["source_direction"],
        target_distance=targets["source_distance"],
        source_valid=source_valid,
        window_mask=window_mask,
        valid_time=valid_time,
        config=config,
        include_activity_cost=True,
    )
    # [B, N_gt, T_s]

    # Build per-(b, k, t) activity target + class/dir/dist target maps.
    activity_target = torch.zeros_like(prediction_output.pred_activity)
    class_target = torch.full(
        (batch_size, num_tracks, t_s_max), -1, dtype=torch.long, device=device
    )
    direction_target = torch.zeros_like(prediction_output.pred_direction)
    distance_target = torch.zeros_like(prediction_output.pred_distance)
    supervise_mask = torch.zeros_like(prediction_output.pred_activity, dtype=torch.bool)
    # dist_supervise_mask: like supervise_mask but also requires distance_valid=True.
    # This prevents distance loss from sources with null distance (e.g. STARSS real data).
    dist_supervise_mask = torch.zeros_like(prediction_output.pred_activity, dtype=torch.bool)
    # ele_sign_only_target: [B, K, T_s] — True for (track, frame) pairs where the
    # matched GT source has only hemisphere (sign) known for elevation.
    # Those frames are excluded from direction cosine loss and instead supervised
    # with a 2-class upper/lower hemisphere BCE.
    ele_sign_only_target = torch.zeros_like(prediction_output.pred_activity, dtype=torch.bool)

    valid_assign = matched_track >= 0  # [B, N_gt, T_s]
    if valid_assign.any():
        idx_b, idx_gt, idx_t = torch.nonzero(valid_assign, as_tuple=True)
        idx_k = matched_track[idx_b, idx_gt, idx_t]
        activity_target[idx_b, idx_k, idx_t] = 1.0
        supervise_mask[idx_b, idx_k, idx_t] = True
        class_target[idx_b, idx_k, idx_t] = targets["source_class"][idx_b, idx_gt]
        # Per-frame direction/distance targets: index with (b, gt, t).
        direction_target[idx_b, idx_k, idx_t] = targets["source_direction"][idx_b, idx_gt, idx_t].to(
            direction_target.dtype
        )
        distance_target[idx_b, idx_k, idx_t] = targets["source_distance"][idx_b, idx_gt, idx_t].to(
            distance_target.dtype
        )
        # Only supervise distance for (source, frame) pairs whose distance is
        # valid (not null).  STARSS/DCASE flip the whole source to invalid; a
        # dynamic source with missing distance at some frames flips per-frame.
        gt_dist_valid = targets["source_distance_valid"][idx_b, idx_gt, idx_t]  # [M] bool
        dist_supervise_mask[idx_b, idx_k, idx_t] = gt_dist_valid
        # Sign-only elevation: propagate per-frame flag to matched (track, frame).
        gt_eso = targets["source_ele_sign_only"][idx_b, idx_gt, idx_t]  # [M] bool
        ele_sign_only_target[idx_b, idx_k, idx_t] = gt_eso

    # A0-A0-A0 trick (ADPIT-style duplicate supervision).
    # For frames where exactly 1 GT source is active, broadcast that source's
    # supervision to ALL K tracks.  This gives positive gradients to all tracks
    # on single-source frames, preventing K-1 tracks from dying.
    if config.use_adpit_duplicate:
        gt_active = (
            window_mask & source_valid.unsqueeze(-1) & valid_time.unsqueeze(1)
        )  # [B, N_gt, T_s]
        active_count = gt_active.sum(dim=1)  # [B, T_s]
        # Find (b, t) cells where exactly 1 GT source is active
        single_bt = (active_count == 1) & valid_time  # [B, T_s]
        if single_bt.any():
            sb, st = torch.nonzero(single_bt, as_tuple=True)
            # For each such (b, t), find which GT source is active
            # gt_active[b, :, t] has exactly one True → argmax gives its index
            sg = gt_active[sb, :, st].to(torch.long).argmax(dim=1)  # [M]
            # Broadcast to ALL K tracks (not just the Hungarian winner)
            for k in range(num_tracks):
                activity_target[sb, k, st] = 1.0
                supervise_mask[sb, k, st] = True
                class_target[sb, k, st] = targets["source_class"][sb, sg]
                # Per-frame direction/distance: index with (b, gt, t).
                direction_target[sb, k, st] = targets["source_direction"][sb, sg, st].to(
                    direction_target.dtype
                )
                distance_target[sb, k, st] = targets["source_distance"][sb, sg, st].to(
                    distance_target.dtype
                )
                # Per-frame distance validity.
                dist_supervise_mask[sb, k, st] = targets["source_distance_valid"][sb, sg, st]
                # Sign-only elevation flag.
                ele_sign_only_target[sb, k, st] = targets["source_ele_sign_only"][sb, sg, st]

    # Soft activity regularization for non-winner tracks (legacy).
    # For GT-active (b, t) cells, tracks NOT selected by Hungarian matching
    # receive a soft activity target instead of the hard 0.0.  This gives a
    # weak positive signal to keep K-1 tracks alive on single-source frames
    # without teaching them the source's class/direction/distance.
    if config.nonwinner_activity_soft_target > 0.0:
        soft_val = config.nonwinner_activity_soft_target
        # gt_active_bt[b, t] = True if any GT source is active at (b, t)
        gt_active_bt = (
            window_mask & source_valid.unsqueeze(-1) & valid_time.unsqueeze(1)
        ).any(dim=1)  # [B, T_s]
        # nonwinner_bt: (b, t) cells where activity_target is still 0 but GT is active
        # i.e., these are the dead-track positions on GT-active frames
        gt_active_bt_exp = gt_active_bt.unsqueeze(1).expand_as(activity_target)
        nonwinner_mask = (activity_target == 0.0) & gt_active_bt_exp  # [B, K, T_s]
        activity_target[nonwinner_mask] = soft_val

    activity_mask = valid_time.unsqueeze(1).expand_as(prediction_output.pred_activity)
    # pos_weight: dynamic (sqrt(neg/pos), capped) or fixed scalar.
    _act_pw_base = config.frame_activity_pos_weight
    if config.use_dynamic_pos_weight:
        with torch.no_grad():
            _masked_act = activity_target * activity_mask
            _pos = _masked_act.sum().clamp_min(1.0)
            _neg = ((1.0 - activity_target) * activity_mask).sum().clamp_min(1.0)
            _dyn = (_neg / _pos).sqrt().clamp(1.0, config.dynamic_pos_weight_cap)
            _effective_pw = (_act_pw_base if _act_pw_base > 0.0 else 1.0) * _dyn
        _act_pw_tensor = activity_target.new_tensor(_effective_pw)
    else:
        _act_pw_tensor = (
            activity_target.new_tensor(_act_pw_base) if _act_pw_base > 0.0 else None
        )
    activity_bce = F.binary_cross_entropy_with_logits(
        prediction_output.pred_activity,
        activity_target,
        pos_weight=_act_pw_tensor,
        reduction="none",
    )
    if config.frame_activity_use_focal:
        # Focal BCE: alpha_t * (1 - p_t)^gamma * CE
        # Preserves the pos_weight contribution already baked into activity_bce.
        with torch.no_grad():
            p = torch.sigmoid(prediction_output.pred_activity)
            p_t = p * activity_target + (1.0 - p) * (1.0 - activity_target)
            alpha = config.frame_activity_focal_alpha
            alpha_t = alpha * activity_target + (1.0 - alpha) * (1.0 - activity_target)
            focal_weight = alpha_t * (1.0 - p_t).pow(config.frame_activity_focal_gamma)
        activity_bce = focal_weight * activity_bce
    # === v13_B [B-2] Asymmetric Loss (ASL) replacement ======================
    # When enabled, overwrite activity_bce with ASL (per-element, reduction="none"
    # so the mask+mean pipeline downstream still works).
    _act_loss_type = getattr(config, "frame_activity_loss_type", "bce")
    if _act_loss_type == "asymmetric":
        activity_bce = _asymmetric_loss_with_logits(
            prediction_output.pred_activity,
            activity_target,
            gamma_neg=float(config.asl_gamma_neg),
            gamma_pos=float(config.asl_gamma_pos),
            margin=float(config.asl_probability_margin),
        )
        loss_activity = _masked_mean(activity_bce, activity_mask)
    elif _act_loss_type == "topk_rank":
        # v13_D [D-2]: pairwise rank loss over active/inactive slots per frame.
        # valid_time is [B, T_s]. activity_target is [B, K, T_s].
        loss_activity = _topk_rank_activity_loss(
            activity_logit=prediction_output.pred_activity,
            target_active=activity_target,
            valid_time=valid_time,
            margin=float(getattr(config, "topk_rank_margin", 2.0)),
            bce_weight=float(getattr(config, "topk_rank_bce_weight", 0.1)),
        )
    else:
        loss_activity = _masked_mean(activity_bce, activity_mask)

    if supervise_mask.any():
        class_logits_flat = prediction_output.pred_class_logits[supervise_mask]
        class_target_flat = class_target[supervise_mask]
        # Optional per-class weights (helps rare classes like aircraft/insect).
        _cls_weights: Optional[Tensor] = None
        if config.frame_class_loss_weights:
            _cls_weights = torch.tensor(
                config.frame_class_loss_weights,
                dtype=class_logits_flat.dtype,
                device=device,
            )
        # hierarchical (ontology-aware) label smoothing.  When enabled,
        # build a per-sample soft target that places `eps / |siblings|` mass
        # on each sibling class inside the same ontology group so that
        # sibling-collapse errors (aircraft->speech, frog->bird, ...) carry
        # less loss than cross-group confusions.
        eps_onto = float(config.frame_class_ontology_smoothing)
        onto_groups = config.frame_class_ontology_groups
        if eps_onto > 0.0 and onto_groups:
            num_classes = class_logits_flat.size(-1)
            # Build a [C, C] soft-target "mixing" table on first use and
            # cache it on the config object to avoid per-batch rebuild.
            if (
                not hasattr(config, "_onto_mixing_table")
                or config._onto_mixing_table is None  # type: ignore[attr-defined]
                or config._onto_mixing_table.shape[0] != num_classes  # type: ignore[attr-defined]
                or config._onto_mixing_table.dtype != class_logits_flat.dtype  # type: ignore[attr-defined]
                or config._onto_mixing_table.device != device  # type: ignore[attr-defined]
            ):
                table = torch.zeros(
                    (num_classes, num_classes),
                    dtype=class_logits_flat.dtype,
                    device=device,
                )
                # Default: hard one-hot.
                table.fill_diagonal_(1.0)
                # Overlay soft targets for classes inside an ontology group.
                for group in onto_groups:
                    members = [int(c) for c in group if 0 <= int(c) < num_classes]
                    if len(members) < 2:
                        continue
                    for c in members:
                        siblings = [s for s in members if s != c]
                        table[c].zero_()
                        table[c, c] = 1.0 - eps_onto
                        sib_mass = eps_onto / len(siblings)
                        for s in siblings:
                            table[c, s] = sib_mass
                config._onto_mixing_table = table  # type: ignore[attr-defined]
            soft_target = config._onto_mixing_table[class_target_flat]  # type: ignore[attr-defined]
            log_probs = F.log_softmax(class_logits_flat, dim=-1)
            if _cls_weights is not None:
                # Apply per-class weight in the "positive" fashion used by
                # F.cross_entropy(weight=...): weight the loss by the weight
                # of the (hard) GT class.
                sample_w = _cls_weights[class_target_flat]
                per_sample_loss = -(soft_target * log_probs).sum(dim=-1)
                loss_class = (per_sample_loss * sample_w).sum() / sample_w.sum().clamp_min(1e-8)
            else:
                loss_class = -(soft_target * log_probs).sum(dim=-1).mean()
        else:
            loss_class = F.cross_entropy(
                class_logits_flat, class_target_flat, weight=_cls_weights
            )

        # Direction cosine loss: only on frames where full DOA (not just hemisphere)
        # is known.  sign-only frames are handled by the hemisphere BCE below.
        dir_full_mask = supervise_mask & ~ele_sign_only_target  # [B, K, T_s]
        hemi_mask = supervise_mask & ele_sign_only_target        # [B, K, T_s]
        if dir_full_mask.any():
            pred_dir_full = F.normalize(
                prediction_output.pred_direction[dir_full_mask], dim=-1
            )
            tgt_dir_full = direction_target[dir_full_mask].to(pred_dir_full.dtype)
            loss_direction = (1.0 - (pred_dir_full * tgt_dir_full).sum(dim=-1)).mean()
        else:
            loss_direction = prediction_output.pred_direction.sum() * 0.0

        # Hemisphere BCE: for sign-only elevation frames, use only the elevation
        # component to decide upper (target=1) or lower hemisphere (target=0).
        # We predict the elevation from pred_direction's y-component (unit vector
        # sin(elevation)), which is pred_direction[..., 2] in the standard
        # (x=cos(ele)cos(azi), y=cos(ele)sin(azi), z=sin(ele)) convention.
        # target_ele in direction_target[..., 2] carries +sin(90°)=+1 for upper
        # hemisphere (ele was stored as +90°) and -1 for lower (ele was -90°).
        loss_hemisphere = prediction_output.pred_direction.sum() * 0.0
        if hemi_mask.any() and config.lambda_frame_hemisphere > 0.0:
            pred_dir_hemi = prediction_output.pred_direction[hemi_mask]  # [M, 3]
            tgt_dir_hemi = direction_target[hemi_mask].to(pred_dir_hemi.dtype)  # [M, 3]
            # elevation z-component of the unit direction vector
            pred_ele_z = pred_dir_hemi[:, 2]               # sin(pred_ele)  [M]
            tgt_ele_z = tgt_dir_hemi[:, 2]                 # ±1             [M]
            hemi_gt = (tgt_ele_z > 0.0).float()            # 1=upper, 0=lower
            # Use logit = pred_ele_z (it ranges ~[-1,+1], monotone w.r.t. elevation)
            loss_hemisphere = F.binary_cross_entropy_with_logits(
                pred_ele_z, hemi_gt, reduction="mean"
            )

        pred_dist_sel = prediction_output.pred_distance[supervise_mask]
        tgt_dist_sel = distance_target[supervise_mask].to(pred_dist_sel.dtype)
        # === v13_C [C-4] Laplace NLL distance loss ==========================
        if (
            getattr(config, "frame_distance_loss_type", "l1") == "laplace_nll"
            and prediction_output.pred_distance_log_var is not None
        ):
            log_var_sel = prediction_output.pred_distance_log_var[supervise_mask].to(
                pred_dist_sel.dtype
            )
            loss_distance = _laplace_nll_distance_loss(
                pred_dist_sel, log_var_sel, tgt_dist_sel
            )
        else:
            loss_distance = F.smooth_l1_loss(pred_dist_sel, tgt_dist_sel)

        # Re-compute distance loss using dist_supervise_mask (excludes null distances).
        # This replaces the above if any sources have unknown distance.
        if dist_supervise_mask.any():
            if not dist_supervise_mask.equal(supervise_mask):
                # Some sources have null distance — use the stricter mask.
                pred_dist_known = prediction_output.pred_distance[dist_supervise_mask]
                tgt_dist_known = distance_target[dist_supervise_mask].to(pred_dist_known.dtype)
                if (
                    getattr(config, "frame_distance_loss_type", "l1") == "laplace_nll"
                    and prediction_output.pred_distance_log_var is not None
                ):
                    log_var_known = prediction_output.pred_distance_log_var[
                        dist_supervise_mask
                    ].to(pred_dist_known.dtype)
                    loss_distance = _laplace_nll_distance_loss(
                        pred_dist_known, log_var_known, tgt_dist_known
                    )
                else:
                    loss_distance = F.smooth_l1_loss(pred_dist_known, tgt_dist_known)
        else:
            # All matched sources have null distance → zero loss, no gradient
            loss_distance = prediction_output.pred_distance.sum() * 0.0
    else:
        loss_class = prediction_output.pred_class_logits.sum() * 0.0
        loss_direction = prediction_output.pred_direction.sum() * 0.0
        loss_distance = prediction_output.pred_distance.sum() * 0.0
        loss_hemisphere = prediction_output.pred_direction.sum() * 0.0

    zero = loss_activity.new_zeros(())
    # per-frame num-active-source CE.  Fires only when the model head is
    # present AND lambda > 0.  The target is the GT active source count per
    # frame, clamped to [0, K].  Uses loss_temp as the auxiliary slot so no
    # new SpatialLossOutput field is needed.
    #
    # optional focal CE + per-class (K_target) weighting to combat the
    # n_gt=3 under-reporting observed in ov3 (vanilla CE collapses K̂ to the
    # majority class 2).  Falls back to plain CE when frame_num_active_use_focal=False.
    loss_num_active = zero
    if (
        prediction_output.pred_num_active_logits is not None
        and config.lambda_frame_num_active > 0.0
    ):
        num_active_logits = prediction_output.pred_num_active_logits  # [B, T_s, K+1]
        K_max = num_active_logits.size(-1) - 1
        num_active_gt = (
            (window_mask & source_valid.unsqueeze(-1) & valid_time.unsqueeze(1))
            .sum(dim=1)
            .clamp(max=K_max)
        )  # [B, T_s]
        num_active_logits_flat = num_active_logits.reshape(-1, num_active_logits.size(-1))
        num_active_target_flat = num_active_gt.reshape(-1)
        time_mask_flat = valid_time.reshape(-1)
        if time_mask_flat.any():
            logits_valid = num_active_logits_flat[time_mask_flat]
            target_valid = num_active_target_flat[time_mask_flat]
            num_active_weights: Optional[Tensor] = None
            if config.frame_num_active_class_weights:
                w = list(config.frame_num_active_class_weights)
                if len(w) != K_max + 1:
                    # Graceful fallback: pad/truncate to K+1 to avoid a noisy
                    # failure mid-training if the preset is momentarily mis-sized.
                    w = (w + [1.0] * (K_max + 1))[: K_max + 1]
                num_active_weights = logits_valid.new_tensor(w)
            if config.frame_num_active_use_focal:
                log_probs = F.log_softmax(logits_valid, dim=-1)
                probs = log_probs.exp()
                p_t = probs.gather(1, target_valid.unsqueeze(1)).squeeze(1).clamp_min(1e-8)
                ce = F.nll_loss(log_probs, target_valid, weight=num_active_weights, reduction="none")
                focal_weight = (1.0 - p_t).pow(config.frame_num_active_focal_gamma)
                # NLL already weights by the class weight; multiplying by focal is additive.
                loss_num_active = (focal_weight * ce).mean()
            else:
                loss_num_active = F.cross_entropy(
                    logits_valid, target_valid, weight=num_active_weights
                )

    # === v13_B [B-4] Soft macro-F1 auxiliary loss ===========================
    # Computed as: for each class c, build class-conditional activity prob
    # p_c = sigmoid(act) * softmax(class)[c] and supervise with soft-F1.
    # The target is 1 only on (b,k,t,c=gt_class) where the slot is truly active.
    loss_soft_f1 = zero
    _soft_f1_w = float(getattr(config, "frame_soft_f1_weight", 0.0))
    if _soft_f1_w > 0.0 and supervise_mask.any():
        loss_soft_f1 = _soft_macro_f1_loss(
            activity_logits=prediction_output.pred_activity,
            class_logits=prediction_output.pred_class_logits,
            supervise_mask=supervise_mask,
            class_target=class_target,
        )

    loss_total = (
        config.lambda_frame_activity * loss_activity
        + config.lambda_frame_class * loss_class
        + config.lambda_frame_direction * loss_direction
        + config.lambda_frame_hemisphere * loss_hemisphere
        + config.lambda_frame_distance * loss_distance
        + config.lambda_frame_num_active * loss_num_active
        + _soft_f1_w * loss_soft_f1
    )
    return SpatialLossOutput(
        loss_total=loss_total,
        loss_activity=loss_activity,
        loss_azi=zero,
        loss_ele=zero,
        loss_dist=loss_distance,
        loss_cls_aux=loss_class,
        loss_temp=loss_num_active,  # reuse loss_temp for num_active CE reporting
        loss_direction=loss_direction,
    )


def compute_frame_track_validation_metrics(
    prediction_output: FrameTrackPredictionOutput,
    batch: "SpatialBatch",
    temporal_padding_mask: Optional[Tensor],
    config: SpatialLossConfig,
) -> FrameMetricOutput:
    """Lightweight per-batch scalars for the progress bar.

    Everything that matters for model selection — official DCASE
    ``ER20/F20/LE_CD/LR_CD/SELD_score`` — is computed globally after all
    batches are seen, so this function returns only:
      - activity_precision: mean sigmoid(pred_activity) on GT-active (b, g, t)
      - activity_recall:    mean sigmoid(pred_activity) on GT-inactive (b, k, t)
      - activity_acc:       activity_precision - activity_recall (separation)
      - oracle_class_acc / oracle_azi_mae_deg / oracle_ele_mae_deg /
        oracle_dist_mae: per-frame Hungarian on GT-active pairs without
        activity thresholding, meant only for training-time diagnostics

    Detection-gated class/DoA metrics remain deferred to the epoch-level DCASE
    evaluator. GT-active per frame = window_mask[b, g, t] & source_valid[b, g]
    & valid_time[b, t].
    """
    device = prediction_output.pred_activity.device
    B, K, T_s = prediction_output.pred_activity.shape
    targets = _frame_source_target_tensors(batch, T_s, device)
    window_mask = targets["window_mask"]           # [B, N_gt, T_s]
    source_valid = targets["source_valid"]         # [B, N_gt]
    valid_time = _valid_time_mask(temporal_padding_mask, B, T_s, device)
    z = prediction_output.pred_activity.sum() * 0.0

    # Build [B, T_s] "any GT active at this frame" mask, then broadcast to K.
    gt_any_active = (
        window_mask & source_valid.unsqueeze(-1) & valid_time.unsqueeze(1)
    ).any(dim=1)  # [B, T_s]
    # active_slots_per_frame: [B, T_s] = number of GT sources active at (b, t).
    num_active_gt = (
        window_mask & source_valid.unsqueeze(-1) & valid_time.unsqueeze(1)
    ).sum(dim=1)  # [B, T_s]

    pred_act_prob = torch.sigmoid(prediction_output.pred_activity)  # [B, K, T_s]
    activity_mask_bkt = valid_time.unsqueeze(1).expand_as(pred_act_prob)

    # "GT-active (b, k, t)" is under-specified without matching (which track k
    # represents which GT g?).  We use a looser proxy suitable for a progress
    # bar: for each (b, t) that has ≥1 active GT, we say exactly
    # min(num_active_gt, K) tracks "should fire".  For the ACT_SEP metric we
    # therefore simply split the valid (b, k, t) mask into two buckets:
    #   - "supposed-active" bucket: (b, t) has GT active  AND  k < num_active_gt
    #     (top-`num_active_gt` tracks ranked by pred_act_prob at that frame)
    #   - "supposed-inactive" bucket: everything else in activity_mask_bkt
    # This is a heuristic for the bar only; SELD accumulator does the real job.
    top_k_thresh = pred_act_prob.topk(K, dim=1).values  # [B, K, T_s] sorted desc
    # rank of each (b, k, t): position in sorted order.  We replicate the
    # "top-num_active_gt" rule via a per-(b, t) threshold = num_active_gt-th prob.
    # Simpler: take the per-(b, t) num_active_gt-th (1-indexed) largest prob.
    sort_desc = pred_act_prob.sort(dim=1, descending=True).values  # [B, K, T_s]
    # Index = clamp(num_active_gt - 1, 0, K-1)
    idx = torch.clamp(num_active_gt - 1, min=0, max=K - 1).unsqueeze(1)  # [B,1,T_s]
    thresh_per_bt = sort_desc.gather(1, idx).squeeze(1)                  # [B, T_s]
    # A (b, k, t) is "supposed-active" if gt_any_active[b, t] is True AND the
    # track k's prob is >= thresh_per_bt[b, t].
    supposed_active_bkt = (
        gt_any_active.unsqueeze(1)
        & (pred_act_prob >= thresh_per_bt.unsqueeze(1))
        & activity_mask_bkt
    )

    active_vals = pred_act_prob[supposed_active_bkt]
    inactive_vals = pred_act_prob[(~supposed_active_bkt) & activity_mask_bkt]

    if active_vals.numel() > 0:
        activity_precision = active_vals.mean()
    else:
        activity_precision = z
    if inactive_vals.numel() > 0:
        activity_recall = inactive_vals.mean()
    else:
        activity_recall = z
    activity_acc = activity_precision - activity_recall

    oracle_matched_track = _match_frame_tracks(
        prediction_output=prediction_output,
        target_class=targets["source_class"],
        target_direction=targets["source_direction"],
        target_distance=targets["source_distance"],
        source_valid=source_valid,
        window_mask=window_mask,
        valid_time=valid_time,
        config=config,
        include_activity_cost=False,
    )
    (
        oracle_class_acc,
        oracle_azi_mae,
        oracle_ele_mae,
        oracle_dist_mae,
        oracle_matched_count,
    ) = _compute_track_oracle_metrics(
        prediction_output=prediction_output,
        matched_track=oracle_matched_track,
        targets=targets,
    )

    # -------------------------------------------------------------------
    # Tier-1 (honest) per-frame metrics: activity-gated on the *training*
    # matcher (include_activity_cost=True; segment-aware when enabled).
    # Same semantics as the valid CSV cls_ok / pred_{azi,ele,dist}:
    #   (1) run the training matcher to get matched_track[b, gt, t]
    #   (2) keep only GT-active (b, gt, t) pairs
    #   (3) further require sigmoid(pred_activity[b, k, t]) >= 0.5
    # Train and valid now use the same closed-form number.
    # -------------------------------------------------------------------
    train_matched_track = _match_frame_tracks(
        prediction_output=prediction_output,
        target_class=targets["source_class"],
        target_direction=targets["source_direction"],
        target_distance=targets["source_distance"],
        source_valid=source_valid,
        window_mask=window_mask,
        valid_time=valid_time,
        config=config,
        include_activity_cost=True,
    )
    act_thresh = float(config.frame_accdoa_activity_threshold)
    gated_valid = train_matched_track >= 0
    gated_class_acc = z
    gated_azi_mae = z
    gated_ele_mae = z
    gated_dist_mae = z
    gated_matched_count = torch.zeros((), device=device, dtype=pred_act_prob.dtype)
    if gated_valid.any():
        idx_b, idx_gt, idx_t = torch.nonzero(gated_valid, as_tuple=True)
        idx_k = train_matched_track[idx_b, idx_gt, idx_t]
        # if the model exposes pred_num_active_logits, gate by top-K̂
        # (argmax over per-frame num-active prediction).  We OR this with the
        # hard 0.5 activity threshold so the metric degrades gracefully when
        # the num_active head has not yet warmed up:
        #   - at ep0 (weight=0, bias=[+4,0,0,0,0]), K̂ ≡ 0 for every frame so
        #     the top-K̂ path adds no active cells; the 0.5 path gives the
        #     usual legacy-equivalent gating ⇒ cls/azi/ele/dist are non-zero from
        #     the very first batch.
        #   - once num_active head learns (bias shifts, weight ≠ 0), K̂ starts
        #     picking the intended number of tracks and complements 0.5 by
        #     catching under-reported multi-source frames.
        # When the head is absent entirely, fall back to pure 0.5.
        hard_active_bkt = pred_act_prob >= act_thresh
        if prediction_output.pred_num_active_logits is not None:
            num_active_pred = prediction_output.pred_num_active_logits.argmax(dim=-1)
            # num_active_pred: [B, T_s] ∈ [0, K]
            K_max = int(prediction_output.pred_activity.size(1))
            # For each (b, t), decide which of the K tracks are "active":
            # sort pred_act_prob descending along K, keep top-K̂ positions.
            sort_idx = pred_act_prob.argsort(dim=1, descending=True)  # [B, K, T_s]
            rank_in_sorted = torch.empty_like(sort_idx)
            rank_in_sorted.scatter_(
                1,
                sort_idx,
                torch.arange(K_max, device=device).view(1, K_max, 1).expand_as(sort_idx),
            )
            # track-k at (b, t) is active iff its rank < K̂_{b, t}
            k_hat_per_bt = num_active_pred.clamp(max=K_max)  # [B, T_s]
            topk_active_bkt = rank_in_sorted < k_hat_per_bt.unsqueeze(1)
            active_bkt = topk_active_bkt | hard_active_bkt
            act_ok = active_bkt[idx_b, idx_k, idx_t]
        else:
            act_ok = hard_active_bkt[idx_b, idx_k, idx_t]
        if act_ok.any():
            sel_b = idx_b[act_ok]
            sel_gt = idx_gt[act_ok]
            sel_t = idx_t[act_ok]
            sel_k = idx_k[act_ok]
            # class
            pc = prediction_output.pred_class_logits[sel_b, sel_k, sel_t].argmax(dim=-1)
            ct = targets["source_class"][sel_b, sel_gt]
            gated_class_acc = (pc == ct).float().mean()
            # azi / ele
            pdir = F.normalize(
                prediction_output.pred_direction[sel_b, sel_k, sel_t], dim=-1
            )
            pazi, pele = _azi_ele_deg_from_direction_vector(pdir)
            azi_tgt = targets["source_azimuth_deg"][sel_b, sel_gt, sel_t].to(pazi.dtype)
            ele_tgt = targets["source_elevation_deg"][sel_b, sel_gt, sel_t].to(pele.dtype)
            gated_azi_mae = _circular_distance_deg(pazi, azi_tgt).mean()
            gated_ele_mae = torch.abs(pele - ele_tgt).mean()
            # dist
            pdist = prediction_output.pred_distance[sel_b, sel_k, sel_t]
            dist_tgt = targets["source_distance"][sel_b, sel_gt, sel_t].to(pdist.dtype)
            gated_dist_mae = torch.abs(pdist - dist_tgt).mean()
            gated_matched_count = torch.tensor(
                float(act_ok.sum().item()),
                device=device,
                dtype=pred_act_prob.dtype,
            )

    return FrameMetricOutput(
        activity_acc=activity_acc,
        activity_precision=activity_precision,
        activity_recall=activity_recall,
        class_acc=gated_class_acc,
        azi_mae_deg=gated_azi_mae,
        ele_mae_deg=gated_ele_mae,
        dist_mae=gated_dist_mae,
        matched_count=gated_matched_count,
        oracle_class_acc=oracle_class_acc,
        oracle_azi_mae_deg=oracle_azi_mae,
        oracle_ele_mae_deg=oracle_ele_mae,
        oracle_dist_mae=oracle_dist_mae,
    )


def build_frame_track_validation_examples(
    prediction_output: FrameTrackPredictionOutput,
    batch: "SpatialBatch",
    temporal_padding_mask: Optional[Tensor],
    config: SpatialLossConfig,
    max_examples: int = 16,
) -> List[Dict[str, object]]:
    """Qualitative examples for route B, one per matched (clip, track) pair."""
    device = prediction_output.pred_activity.device
    batch_size, num_tracks, t_s_max = prediction_output.pred_activity.shape
    targets = _frame_source_target_tensors(batch, t_s_max, device)
    window_mask = targets["window_mask"]
    valid_time = _valid_time_mask(temporal_padding_mask, batch_size, t_s_max, device)
    matched_track = _match_frame_tracks(
        prediction_output=prediction_output,
        target_class=targets["source_class"],
        target_direction=targets["source_direction"],
        target_distance=targets["source_distance"],
        source_valid=targets["source_valid"],
        window_mask=window_mask,
        valid_time=valid_time,
        config=config,
        include_activity_cost=True,
    )  # [B, N_gt, T_s]
    examples: List[Dict[str, object]] = []
    valid_assign = matched_track >= 0
    if not valid_assign.any():
        return examples
    idx_b, idx_gt, idx_t = torch.nonzero(valid_assign, as_tuple=True)
    idx_k = matched_track[idx_b, idx_gt, idx_t]
    limit = min(int(max_examples), int(idx_b.numel()))
    for i in range(limit):
        b = int(idx_b[i].item())
        gt = int(idx_gt[i].item())
        k = int(idx_k[i].item())
        t = int(idx_t[i].item())
        pred_dir = F.normalize(prediction_output.pred_direction[b, k, t], dim=-1)
        pred_azi_deg, pred_ele_deg = _azi_ele_deg_from_direction_vector(pred_dir)
        pred_cls = int(prediction_output.pred_class_logits[b, k, t].argmax().item())
        examples.append(
            {
                "sample_id": batch.sample_ids[b],
                "track_index": k,
                "time_index": t,
                "gt_index": gt,
                "gt_class_index": int(batch.source_class_indices[b, gt].item()),
                "pred_class_index": pred_cls,
                "gt_azimuth_deg": float(batch.source_azimuth_deg[b, gt, t].item()),
                "pred_azimuth_deg": float(pred_azi_deg.item()),
                "gt_elevation_deg": float(batch.source_elevation_deg[b, gt, t].item()),
                "pred_elevation_deg": float(pred_ele_deg.item()),
                "gt_distance": float(batch.source_distance[b, gt, t].item()),
                "pred_distance": float(prediction_output.pred_distance[b, k, t].item()),
                "pred_activity_prob": float(
                    torch.sigmoid(prediction_output.pred_activity[b, k, t]).item()
                ),
            }
        )
    return examples


def collect_frame_track_csv_rows(
    prediction_output: FrameTrackPredictionOutput,
    batch: "SpatialBatch",
    temporal_padding_mask: Optional[Tensor],
) -> List[Dict[str, object]]:
    """Per-(sample, frame) DCASE-style rows for both GT and prediction.

    For each batch sample produces a dict with `gt_rows` and `pred_rows`.
    GT rows: one per active (source, frame) inside the source's window.
    Pred rows: ALL K tracks × all valid frames, each with `activity_prob`
    so downstream tooling can apply any threshold (or none).
    """
    device = prediction_output.pred_activity.device
    batch_size, num_tracks, t_s_max = prediction_output.pred_activity.shape
    targets = _frame_source_target_tensors(batch, t_s_max, device)
    window_mask = targets["window_mask"].cpu()  # [B, N_gt, T_s]
    valid_time = _valid_time_mask(
        temporal_padding_mask, batch_size, t_s_max, device
    ).cpu()

    pred_activity_prob = torch.sigmoid(prediction_output.pred_activity).cpu()
    pred_class_idx = prediction_output.pred_class_logits.argmax(dim=-1).cpu()
    pred_dir = F.normalize(prediction_output.pred_direction, dim=-1)
    pred_azi_deg, pred_ele_deg = _azi_ele_deg_from_direction_vector(pred_dir)
    pred_azi_deg = pred_azi_deg.cpu()
    pred_ele_deg = pred_ele_deg.cpu()
    pred_dist = prediction_output.pred_distance.cpu()
    # per-frame num-active prediction, when the head is present.
    # Used to annotate the CSV so downstream tooling can replay top-K̂ gating.
    pred_num_active = None
    if prediction_output.pred_num_active_logits is not None:
        pred_num_active = (
            prediction_output.pred_num_active_logits.argmax(dim=-1).cpu()
        )  # [B, T_s]

    source_valid = batch.source_valid_mask.cpu()
    src_azi = batch.source_azimuth_deg.cpu()
    src_ele = batch.source_elevation_deg.cpu()
    src_dist = batch.source_distance.cpu()
    src_cls = batch.source_class_indices.cpu()
    target_num_steps = batch.target_num_steps.cpu()
    clip_dur = batch.clip_duration_seconds.cpu()

    out: List[Dict[str, object]] = []
    for b in range(batch_size):
        valid_steps = max(int(target_num_steps[b].item()), 1)
        cd = float(clip_dur[b].item())
        step = cd / valid_steps if valid_steps > 0 else 0.0

        label_names = (
            batch.source_class_labels[b] if batch.source_class_labels else []
        )

        gt_rows: List[Dict[str, object]] = []
        for s in range(int(source_valid.size(1))):
            if not bool(source_valid[b, s]):
                continue
            cls_name = label_names[s] if s < len(label_names) else ""
            for t in range(valid_steps):
                if not bool(window_mask[b, s, t] & valid_time[b, t]):
                    continue
                gt_rows.append(
                    {
                        "frame_idx": t,
                        "frame_time_s": round(t * step, 4),
                        "src_or_track_idx": s,
                        "class_idx": int(src_cls[b, s].item()),
                        "class_name": cls_name,
                        # Per-frame GT: index with (b, s, t).  For static sources
                        # this degenerates to a constant across t.  For dynamic
                        # sources it reflects the trajectory.
                        "azimuth_deg": round(float(src_azi[b, s, t].item()), 3),
                        "elevation_deg": round(float(src_ele[b, s, t].item()), 3),
                        "distance_m": round(float(src_dist[b, s, t].item()), 3),
                        "activity_prob": 1.0,
                    }
                )

        pred_rows: List[Dict[str, object]] = []
        for k in range(num_tracks):
            for t in range(valid_steps):
                if not bool(valid_time[b, t]):
                    continue
                row = {
                    "frame_idx": t,
                    "frame_time_s": round(t * step, 4),
                    "src_or_track_idx": k,
                    "class_idx": int(pred_class_idx[b, k, t].item()),
                    "class_name": "",  # filled by dumper using vocab
                    "azimuth_deg": round(float(pred_azi_deg[b, k, t].item()), 3),
                    "elevation_deg": round(float(pred_ele_deg[b, k, t].item()), 3),
                    "distance_m": round(float(pred_dist[b, k, t].item()), 3),
                    "activity_prob": round(
                        float(pred_activity_prob[b, k, t].item()), 4
                    ),
                }
                if pred_num_active is not None:
                    row["num_active_pred"] = int(pred_num_active[b, t].item())
                pred_rows.append(row)

        out.append(
            {
                "sample_id": batch.sample_ids[b],
                "frame_step_seconds": step,
                "valid_steps": valid_steps,
                "num_tracks": num_tracks,
                "gt_rows": gt_rows,
                "pred_rows": pred_rows,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Route C — per-class ACCDOA vector field (no matching).
# ---------------------------------------------------------------------------


def _build_accdoa_targets(
    batch: "SpatialBatch",
    t_s_max: int,
    num_classes: int,
    device: torch.device,
) -> Dict[str, Tensor]:
    """Construct per-(b, t, c) ACCDOA target and per-(b, t, c) activity/distance.

    Assumes ov2/ov3 data contains no same-class overlap within a clip. When a
    class is active at a frame, its ACCDOA target equals the source unit
    direction vector and the distance target is the source distance. Otherwise
    the ACCDOA target is zero and distance is masked out.
    """
    targets = _frame_source_target_tensors(batch, t_s_max, device)
    window_mask = targets["window_mask"]  # [B, N_gt, T_s]
    source_class = targets["source_class"]  # [B, N_gt]
    source_direction = targets["source_direction"]  # [B, N_gt, 3]
    source_distance = targets["source_distance"]  # [B, N_gt]
    source_valid = targets["source_valid"]  # [B, N_gt]

    batch_size = source_class.size(0)
    accdoa_target = torch.zeros(batch_size, t_s_max, num_classes, 3, device=device)
    activity_target = torch.zeros(batch_size, t_s_max, num_classes, device=device)
    distance_target = torch.zeros(batch_size, t_s_max, num_classes, device=device)
    distance_mask = torch.zeros(
        batch_size, t_s_max, num_classes, dtype=torch.bool, device=device
    )

    num_gt = source_class.size(1)
    for b in range(batch_size):
        for gt in range(num_gt):
            if not bool(source_valid[b, gt]):
                continue
            cls = int(source_class[b, gt].item())
            if cls < 0 or cls >= num_classes:
                continue
            active_t = torch.nonzero(window_mask[b, gt], as_tuple=False).flatten()
            if active_t.numel() == 0:
                continue
            dir_vec = source_direction[b, gt].to(accdoa_target.dtype)
            dist_val = source_distance[b, gt].to(distance_target.dtype)
            accdoa_target[b, active_t, cls] = dir_vec
            activity_target[b, active_t, cls] = 1.0
            distance_target[b, active_t, cls] = dist_val
            distance_mask[b, active_t, cls] = True

    return {
        "accdoa_target": accdoa_target,
        "activity_target": activity_target,
        "distance_target": distance_target,
        "distance_mask": distance_mask,
    }


def compute_frame_accdoa_losses(
    prediction_output: FrameACCDOAPredictionOutput,
    batch: "SpatialBatch",
    temporal_padding_mask: Optional[Tensor],
    config: SpatialLossConfig,
    clip_aux_prediction: Optional[MonoTaskPredictionOutput] = None,
) -> SpatialLossOutput:
    """Compute Route C (ACCDOA) losses."""
    device = prediction_output.pred_accdoa.device
    batch_size, t_s_max, num_classes, _ = prediction_output.pred_accdoa.shape
    targets = _build_accdoa_targets(batch, t_s_max, num_classes, device)
    accdoa_target = targets["accdoa_target"]
    distance_target = targets["distance_target"]
    distance_mask = targets["distance_mask"]

    valid_time = _valid_time_mask(temporal_padding_mask, batch_size, t_s_max, device)
    time_mask = valid_time.unsqueeze(-1).unsqueeze(-1).expand_as(prediction_output.pred_accdoa)

    # ACCDOA MSE on all valid (b, t, c) locations — target magnitude encodes
    # activity, direction encodes DoA.
    accdoa_err = (prediction_output.pred_accdoa - accdoa_target) ** 2
    # Bugfix: the raw MSE is dominated by the ~98% of inactive (b,t,c) cells
    # whose target is zero.  Down-weight inactive cells so active-cell
    # direction/magnitude gradient actually drives learning.
    inactive_w = float(getattr(config, "frame_accdoa_inactive_weight", 1.0))
    active_mask = (targets["activity_target"] > 0).to(accdoa_err.dtype)  # [B, T, C]
    if inactive_w != 1.0:
        # Weight per (b,t,c): 1.0 on active cells, inactive_w on inactive.
        cell_weight = active_mask + inactive_w * (1.0 - active_mask)  # [B, T, C]
        cell_weight = cell_weight * valid_time.unsqueeze(-1).to(cell_weight.dtype)
        weighted = accdoa_err.sum(dim=-1) * cell_weight
        denom = cell_weight.sum().clamp_min(1.0)
        loss_accdoa = weighted.sum() / denom
    else:
        loss_accdoa = _masked_mean(accdoa_err.sum(dim=-1), time_mask[..., 0])

    # Distance loss only where the class is active.
    dist_err = F.smooth_l1_loss(
        prediction_output.pred_distance,
        distance_target,
        reduction="none",
    )
    dist_weight = (distance_mask & valid_time.unsqueeze(-1)).to(dist_err.dtype)
    loss_distance = _masked_mean(dist_err, dist_weight)

    # Split the ACCDOA loss into activity vs direction components (reporting
    # only; both contribute through loss_accdoa already).
    pred_mag = prediction_output.pred_accdoa.norm(dim=-1)  # [B, T, C]
    activity_err = (pred_mag - targets["activity_target"]) ** 2
    loss_activity = _masked_mean(activity_err, valid_time.unsqueeze(-1).expand_as(activity_err))

    # Direction-only cos-distance on active classes.
    direction_weight = (targets["activity_target"] > 0).to(prediction_output.pred_accdoa.dtype)
    direction_weight = direction_weight * valid_time.unsqueeze(-1).to(direction_weight.dtype)
    pred_unit = F.normalize(
        prediction_output.pred_accdoa + 1e-8, dim=-1
    )
    tgt_unit = F.normalize(accdoa_target + 1e-8, dim=-1)
    cos_sim = (pred_unit * tgt_unit).sum(dim=-1)  # [B, T, C]
    loss_direction = _masked_mean(1.0 - cos_sim, direction_weight)

    loss_clip = compute_clip_aux_losses(clip_aux_prediction, batch, config)
    zero = loss_accdoa.new_zeros(())
    # The ACCDOA MSE is the primary driver; use lambda_frame_activity as its
    # weight since it couples activity + direction.  Bugfix: also add a
    # direction-pure term that goes into backward when
    # lambda_frame_accdoa_direction > 0, since loss_direction was previously
    # report-only.
    lambda_dir = float(getattr(config, "lambda_frame_accdoa_direction", 0.0))
    loss_total = (
        config.lambda_frame_activity * loss_accdoa
        + lambda_dir * loss_direction
        + config.lambda_frame_distance * loss_distance
        + config.lambda_clip_aux * loss_clip
    )
    return SpatialLossOutput(
        loss_total=loss_total,
        loss_activity=loss_activity,
        loss_azi=zero,
        loss_ele=zero,
        loss_dist=loss_distance,
        loss_cls_aux=zero,
        loss_temp=loss_clip,
        loss_direction=loss_direction,
    )


def compute_frame_accdoa_validation_metrics(
    prediction_output: FrameACCDOAPredictionOutput,
    batch: "SpatialBatch",
    temporal_padding_mask: Optional[Tensor],
    config: SpatialLossConfig,
) -> FrameMetricOutput:
    """Metrics for route C. Activity is derived from ACCDOA vector magnitude."""
    device = prediction_output.pred_accdoa.device
    batch_size, t_s_max, num_classes, _ = prediction_output.pred_accdoa.shape
    targets = _build_accdoa_targets(batch, t_s_max, num_classes, device)
    activity_target = targets["activity_target"]
    accdoa_target = targets["accdoa_target"]
    distance_target = targets["distance_target"]
    distance_mask = targets["distance_mask"]

    valid_time = _valid_time_mask(temporal_padding_mask, batch_size, t_s_max, device)
    time_mask = valid_time.unsqueeze(-1).expand_as(activity_target)
    threshold = float(config.frame_accdoa_activity_threshold)
    pred_mag = prediction_output.pred_accdoa.norm(dim=-1)
    pred_active = (pred_mag >= threshold).float()

    valid_pred = pred_active[time_mask]
    valid_true = activity_target[time_mask]
    if valid_true.numel() > 0:
        activity_acc = (valid_pred == valid_true).float().mean()
        tp = ((valid_pred == 1.0) & (valid_true == 1.0)).float().sum()
        pp = (valid_pred == 1.0).float().sum()
        gp = (valid_true == 1.0).float().sum()
        activity_precision = tp / torch.clamp(pp, min=1.0)
        activity_recall = tp / torch.clamp(gp, min=1.0)
    else:
        z = prediction_output.pred_accdoa.sum() * 0.0
        activity_acc = z
        activity_precision = z
        activity_recall = z

    active_mask = (activity_target > 0) & valid_time.unsqueeze(-1)
    if active_mask.any():
        pred_unit = F.normalize(prediction_output.pred_accdoa[active_mask] + 1e-8, dim=-1)
        tgt_unit = F.normalize(accdoa_target[active_mask] + 1e-8, dim=-1)
        pred_azi_deg, pred_ele_deg = _azi_ele_deg_from_direction_vector(pred_unit)
        tgt_azi_deg, tgt_ele_deg = _azi_ele_deg_from_direction_vector(tgt_unit)
        azi_mae = _circular_distance_deg(pred_azi_deg, tgt_azi_deg).mean()
        ele_mae = torch.abs(pred_ele_deg - tgt_ele_deg).mean()

        # Class acc: was the right class activated? Compare argmax of per-frame
        # activity over classes with GT.
        matched_count = torch.tensor(
            float(active_mask.sum().item()),
            device=device,
            dtype=prediction_output.pred_accdoa.dtype,
        )
    else:
        z = prediction_output.pred_accdoa.sum() * 0.0
        azi_mae = z
        ele_mae = z
        matched_count = z

    dist_valid = distance_mask & valid_time.unsqueeze(-1)
    if dist_valid.any():
        dist_mae = torch.abs(
            prediction_output.pred_distance[dist_valid]
            - distance_target[dist_valid]
        ).mean()
    else:
        dist_mae = prediction_output.pred_distance.sum() * 0.0

    # Class accuracy = fraction of active-frame GT classes where the predicted
    # ACCDOA magnitude of that class is >= magnitude of every other class.
    if active_mask.any():
        # Reduce to (b, t) where any class is active, then argmax.
        frame_active = active_mask.any(dim=-1) & valid_time
        if frame_active.any():
            pred_cls_idx = pred_mag[frame_active].argmax(dim=-1)
            # GT: the (b, t) frame has exactly one active class by design; use
            # argmax on activity_target.
            gt_cls_idx = activity_target[frame_active].argmax(dim=-1)
            class_acc = (pred_cls_idx == gt_cls_idx).float().mean()
        else:
            class_acc = prediction_output.pred_accdoa.sum() * 0.0
    else:
        class_acc = prediction_output.pred_accdoa.sum() * 0.0

    return FrameMetricOutput(
        activity_acc=activity_acc,
        activity_precision=activity_precision,
        activity_recall=activity_recall,
        class_acc=class_acc,
        azi_mae_deg=azi_mae,
        ele_mae_deg=ele_mae,
        dist_mae=dist_mae,
        matched_count=matched_count,
        # ACCDOA's class_acc / azi_mae / ele_mae / dist_mae are already
        # evaluated on GT-active frames regardless of predicted activity,
        # so they ARE the oracle metrics.  Activity and DOA are coupled
        # through ||v_c||, there is no separate detection-gated tier here.
        oracle_class_acc=class_acc,
        oracle_azi_mae_deg=azi_mae,
        oracle_ele_mae_deg=ele_mae,
        oracle_dist_mae=dist_mae,
    )


def build_frame_accdoa_validation_examples(
    prediction_output: FrameACCDOAPredictionOutput,
    batch: "SpatialBatch",
    temporal_padding_mask: Optional[Tensor],
    config: SpatialLossConfig,
    max_examples: int = 16,
) -> List[Dict[str, object]]:
    """Qualitative examples for route C at sampled (clip, time, class) cells."""
    device = prediction_output.pred_accdoa.device
    batch_size, t_s_max, num_classes, _ = prediction_output.pred_accdoa.shape
    targets = _build_accdoa_targets(batch, t_s_max, num_classes, device)
    activity_target = targets["activity_target"]
    valid_time = _valid_time_mask(temporal_padding_mask, batch_size, t_s_max, device)
    active_mask = (activity_target > 0) & valid_time.unsqueeze(-1)
    if not active_mask.any():
        return []
    idx_b, idx_t, idx_c = torch.nonzero(active_mask, as_tuple=True)
    limit = min(int(max_examples), int(idx_b.numel()))
    examples: List[Dict[str, object]] = []
    for i in range(limit):
        b = int(idx_b[i].item())
        t = int(idx_t[i].item())
        c = int(idx_c[i].item())
        pred_vec = prediction_output.pred_accdoa[b, t, c]
        pred_mag = float(pred_vec.norm().item())
        pred_unit = F.normalize(pred_vec + 1e-8, dim=-1)
        pred_azi_deg, pred_ele_deg = _azi_ele_deg_from_direction_vector(pred_unit)
        # Lookup GT source sharing this class at this (b, t).
        gt_direction = targets["accdoa_target"][b, t, c]
        gt_azi_deg, gt_ele_deg = _azi_ele_deg_from_direction_vector(
            F.normalize(gt_direction + 1e-8, dim=-1)
        )
        examples.append(
            {
                "sample_id": batch.sample_ids[b],
                "time_index": t,
                "class_index": c,
                "pred_activity_mag": pred_mag,
                "pred_azimuth_deg": float(pred_azi_deg.item()),
                "gt_azimuth_deg": float(gt_azi_deg.item()),
                "pred_elevation_deg": float(pred_ele_deg.item()),
                "gt_elevation_deg": float(gt_ele_deg.item()),
                "pred_distance": float(prediction_output.pred_distance[b, t, c].item()),
                "gt_distance": float(targets["distance_target"][b, t, c].item()),
            }
        )
    return examples


# ---------------------------------------------------------------------------
# Legacy single-source SELD metrics (mono_ast / pretrunk_ast).
# ---------------------------------------------------------------------------
# Reference: DCASE2022-2024 SELD evaluation framework
#   ER  = (FP + FN) / N_ref          — detection error rate (0=best)
#   F   = 2TP / (2TP + FP + FN)      — location-sensitive F1 (1=best)
#         TP only if class matches AND angular_error <= doa_threshold
#   LE  = mean angular error over TP  — localization error in degrees (0=best)
#         = 180 if no TP
#   LR  = TP_loc / N_ref              — localization recall, class-agnostic
#         TP_loc: angular_error <= doa_threshold regardless of class
#   SELD = (ER + (1-F) + LE/180 + (1-LR)) / 4  — joint score (0=best)
# ---------------------------------------------------------------------------


@dataclass
class SELDMetricsAccumulator:
    """Epoch-level accumulator for DCASE SELD metrics.

    Accumulate across all batches, then call ``compute()``.
    Not DDP-aware — gather to rank-0 before feeding batches.
    """
    tp_cls: int = 0
    fp: int = 0
    fn: int = 0
    tp_loc: int = 0
    angular_err_sum: float = 0.0
    n_ref: int = 0
    doa_threshold_deg: float = 20.0

    def update(
        self,
        pred_class_idx: int,
        pred_azi: float,
        pred_ele: float,
        gt_class_idx: int,
        gt_azi: float,
        gt_ele: float,
    ) -> None:
        """Update for one (prediction, GT) pair from a single-source sample."""
        self.n_ref += 1
        ang_err = _seld_angular_distance_deg(pred_azi, pred_ele, gt_azi, gt_ele)
        localized = ang_err <= self.doa_threshold_deg
        if localized:
            self.tp_loc += 1
        if pred_class_idx == gt_class_idx and localized:
            self.tp_cls += 1
            self.angular_err_sum += ang_err
        else:
            self.fp += 1
            self.fn += 1

    def update_frame_multi(
        self,
        preds: List[Tuple[int, float, float]],
        gts: List[Tuple[int, float, float]],
    ) -> None:
        """Accumulate one frame of multi-source pred vs GT.

        Args:
            preds: list of (pred_class_idx, pred_azi_deg, pred_ele_deg) for every
                track currently considered ACTIVE at this frame.
            gts:   list of (gt_class_idx, gt_azi_deg, gt_ele_deg) for every
                GT source active at this frame.

        Matching rule (DCASE):
            1. n_ref += len(gts)
            2. Greedy min-angular-error assignment between preds and gts with
               a 20° angular gate (same as doa_threshold_deg).
            3. A (pred, gt) pair with angular_err <= threshold counts:
                 - tp_loc (class-agnostic localization recall)
                 - tp_cls if classes also match (location-sensitive TP)
                 - else tp_loc only; both FP and FN incremented
            4. Leftover preds -> pure FP; leftover gts -> pure FN.
        """
        n_gt = len(gts)
        n_pred = len(preds)
        self.n_ref += n_gt

        if n_pred == 0 and n_gt == 0:
            return
        if n_pred == 0:
            self.fn += n_gt
            return
        if n_gt == 0:
            self.fp += n_pred
            return

        # Compute angular error matrix [n_pred, n_gt].
        ang_mat = [[0.0] * n_gt for _ in range(n_pred)]
        for i, (_, pa, pe) in enumerate(preds):
            for j, (_, ga, ge) in enumerate(gts):
                ang_mat[i][j] = _seld_angular_distance_deg(pa, pe, ga, ge)

        # Greedy one-to-one matching, smallest angular error first.
        used_pred: set = set()
        used_gt: set = set()
        candidates = [
            (ang_mat[i][j], i, j) for i in range(n_pred) for j in range(n_gt)
        ]
        candidates.sort()
        for err, i, j in candidates:
            if i in used_pred or j in used_gt:
                continue
            if err > self.doa_threshold_deg:
                break  # remaining candidates are all worse
            used_pred.add(i)
            used_gt.add(j)
            self.tp_loc += 1
            if preds[i][0] == gts[j][0]:
                self.tp_cls += 1
                self.angular_err_sum += err
            else:
                # Wrong class → not a location-sensitive TP, counts as both FP and FN
                self.fp += 1
                self.fn += 1

        self.fp += n_pred - len(used_pred)
        self.fn += n_gt - len(used_gt)

    def compute(self) -> Dict[str, float]:
        """Return dict of DCASE SELD scores."""
        tp = self.tp_cls
        fp = self.fp
        fn = self.fn
        n = max(self.n_ref, 1)
        er = (fp + fn) / n
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2.0 * prec * rec / max(prec + rec, 1e-9)
        le = (self.angular_err_sum / tp) if tp > 0 else 180.0
        lr = self.tp_loc / n
        seld = (er + (1.0 - f1) + le / 180.0 + (1.0 - lr)) / 4.0
        return {
            "seld_er": round(er, 4),
            "seld_f1": round(f1, 4),
            "seld_le": round(le, 2),
            "seld_lr": round(lr, 4),
            "seld_score": round(seld, 4),
        }

    def all_reduce(self, device: Optional[torch.device] = None) -> None:
        """Sum this accumulator's counters across all DDP ranks in-place.

        Call once at the end of validation, before ``compute()``.  After this
        every rank holds the globally aggregated tp/fp/fn/etc., so ``compute()``
        returns the same seld_score on all ranks and best-checkpoint logic is
        consistent.  Safe to call outside DDP (no-op if not initialized).
        """
        try:
            import torch.distributed as dist
        except ImportError:
            return
        if not dist.is_available() or not dist.is_initialized():
            return
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # [tp_cls, fp, fn, tp_loc, ang_sum, n_ref]
        buf = torch.tensor(
            [
                float(self.tp_cls),
                float(self.fp),
                float(self.fn),
                float(self.tp_loc),
                float(self.angular_err_sum),
                float(self.n_ref),
            ],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        b = buf.tolist()
        self.tp_cls = int(b[0])
        self.fp = int(b[1])
        self.fn = int(b[2])
        self.tp_loc = int(b[3])
        self.angular_err_sum = float(b[4])
        self.n_ref = int(b[5])


def _official_distance_between_spherical_coordinates_rad(
    az1: np.ndarray,
    ele1: np.ndarray,
    az2: np.ndarray,
    ele2: np.ndarray,
) -> np.ndarray:
    """Official DCASE great-circle distance in degrees."""
    dist = (
        np.sin(ele1) * np.sin(ele2)
        + np.cos(ele1) * np.cos(ele2) * np.cos(np.abs(az1 - az2))
    )
    dist = np.clip(dist, -1.0, 1.0)
    return np.arccos(dist) * 180.0 / np.pi


def _official_least_distance_between_gt_pred(
    gt_list: np.ndarray,
    pred_list: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Official DCASE Hungarian matcher for one class in one segment."""
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:
        raise ImportError(
            "Official DCASE evaluator requires scipy.optimize.linear_sum_assignment."
        ) from exc

    gt_len, pred_len = gt_list.shape[0], pred_list.shape[0]
    ind_pairs = np.array([[x, y] for y in range(pred_len) for x in range(gt_len)])
    cost_mat = np.zeros((gt_len, pred_len), dtype=np.float64)
    if gt_len and pred_len:
        az1 = gt_list[ind_pairs[:, 0], 0]
        ele1 = gt_list[ind_pairs[:, 0], 1]
        az2 = pred_list[ind_pairs[:, 1], 0]
        ele2 = pred_list[ind_pairs[:, 1], 1]
        cost_mat[ind_pairs[:, 0], ind_pairs[:, 1]] = (
            _official_distance_between_spherical_coordinates_rad(az1, ele1, az2, ele2)
        )
    row_ind, col_ind = linear_sum_assignment(cost_mat)
    return cost_mat[row_ind, col_ind], row_ind, col_ind


class OfficialDCASESELDMetrics:
    """Official DCASE 2023 SELD evaluator with DDP all-reduce support.

    Adapted from the official baseline:
    https://github.com/sharathadavanne/seld-dcase2023/blob/master/SELD_evaluation_metrics.py
    """

    def __init__(
        self,
        doa_threshold: float = 20.0,
        nb_classes: int = 11,
        average: str = "macro",
    ) -> None:
        self._nb_classes = int(nb_classes)
        self._TP = np.zeros(self._nb_classes, dtype=np.float64)
        self._FP = np.zeros(self._nb_classes, dtype=np.float64)
        self._FP_spatial = np.zeros(self._nb_classes, dtype=np.float64)
        self._FN = np.zeros(self._nb_classes, dtype=np.float64)
        self._Nref = np.zeros(self._nb_classes, dtype=np.float64)
        self._spatial_T = float(doa_threshold)
        self._S = 0.0
        self._D = 0.0
        self._I = 0.0
        self._total_DE = np.zeros(self._nb_classes, dtype=np.float64)
        self._DE_TP = np.zeros(self._nb_classes, dtype=np.float64)
        self._DE_FP = np.zeros(self._nb_classes, dtype=np.float64)
        self._DE_FN = np.zeros(self._nb_classes, dtype=np.float64)
        self._average = average
        self._eps = np.finfo(np.float64).eps

    def early_stopping_metric(
        self,
        er: np.ndarray,
        f: np.ndarray,
        le: np.ndarray,
        lr: np.ndarray,
    ) -> np.ndarray:
        return np.mean([er, 1.0 - f, le / 180.0, 1.0 - lr], axis=0)

    def compute_seld_scores(
        self,
    ) -> Tuple[float, float, float, float, float, np.ndarray]:
        er = float((self._S + self._D + self._I) / (self._Nref.sum() + self._eps))
        classwise_results = np.array([])
        if self._average == "micro":
            f = float(
                self._TP.sum()
                / (
                    self._eps
                    + self._TP.sum()
                    + self._FP_spatial.sum()
                    + 0.5 * (self._FP.sum() + self._FN.sum())
                )
            )
            le = (
                float(self._total_DE.sum() / (self._DE_TP.sum() + self._eps))
                if self._DE_TP.sum() > 0
                else 180.0
            )
            lr = float(self._DE_TP.sum() / (self._eps + self._DE_TP.sum() + self._DE_FN.sum()))
            seld = float(self.early_stopping_metric(np.array(er), np.array(f), np.array(le), np.array(lr)))
        else:
            f_arr = self._TP / (
                self._eps
                + self._TP
                + self._FP_spatial
                + 0.5 * (self._FP + self._FN)
            )
            le_arr = self._total_DE / (self._DE_TP + self._eps)
            le_arr[self._DE_TP == 0] = 180.0
            lr_arr = self._DE_TP / (self._eps + self._DE_TP + self._DE_FN)
            seld_arr = self.early_stopping_metric(
                np.repeat(er, self._nb_classes),
                f_arr,
                le_arr,
                lr_arr,
            )
            classwise_results = np.array(
                [np.repeat(er, self._nb_classes), f_arr, le_arr, lr_arr, seld_arr]
            )
            f = float(f_arr.mean())
            le = float(le_arr.mean())
            lr = float(lr_arr.mean())
            seld = float(seld_arr.mean())
        return er, f, le, lr, seld, classwise_results

    def update_seld_scores(
        self,
        pred: Dict[int, Dict[int, List[List[object]]]],
        gt: Dict[int, Dict[int, List[List[object]]]],
    ) -> None:
        for block_cnt in range(len(gt.keys())):
            loc_fn, loc_fp = 0.0, 0.0
            for class_cnt in range(self._nb_classes):
                nb_gt_doas = (
                    max(len(val) for val in gt[block_cnt][class_cnt][0][1])
                    if class_cnt in gt[block_cnt]
                    else None
                )
                nb_pred_doas = (
                    max(len(val) for val in pred[block_cnt][class_cnt][0][1])
                    if class_cnt in pred[block_cnt]
                    else None
                )
                if nb_gt_doas is not None:
                    self._Nref[class_cnt] += nb_gt_doas
                if class_cnt in gt[block_cnt] and class_cnt in pred[block_cnt]:
                    matched_track_dist: Dict[int, List[float]] = {}
                    matched_track_cnt: Dict[int, List[int]] = {}
                    gt_ind_list = gt[block_cnt][class_cnt][0][0]
                    pred_ind_list = pred[block_cnt][class_cnt][0][0]
                    for gt_ind, gt_val in enumerate(gt_ind_list):
                        if gt_val not in pred_ind_list:
                            continue
                        gt_arr = np.array(gt[block_cnt][class_cnt][0][1][gt_ind], dtype=np.float64)
                        gt_ids = np.arange(len(gt_arr[:, -1]))
                        gt_doas = gt_arr[:, 1:] * np.pi / 180.0

                        pred_ind = pred_ind_list.index(gt_val)
                        pred_arr = np.array(
                            pred[block_cnt][class_cnt][0][1][pred_ind],
                            dtype=np.float64,
                        )
                        pred_doas = pred_arr[:, 1:] * np.pi / 180.0

                        dist_list, row_inds, col_inds = _official_least_distance_between_gt_pred(
                            gt_doas,
                            pred_doas,
                        )
                        for dist_cnt, dist_val in enumerate(dist_list):
                            matched_gt_track = int(gt_ids[row_inds[dist_cnt]])
                            if matched_gt_track not in matched_track_dist:
                                matched_track_dist[matched_gt_track] = []
                                matched_track_cnt[matched_gt_track] = []
                            matched_track_dist[matched_gt_track].append(float(dist_val))
                            matched_track_cnt[matched_gt_track].append(int(col_inds[dist_cnt]))

                    if len(matched_track_dist) == 0:
                        loc_fn += float(nb_gt_doas or 0)
                        self._FN[class_cnt] += float(nb_gt_doas or 0)
                        self._DE_FN[class_cnt] += float(nb_gt_doas or 0)
                    else:
                        for track_id in matched_track_dist:
                            total_spatial_dist = sum(matched_track_dist[track_id])
                            total_framewise_matching_doa = len(matched_track_cnt[track_id])
                            avg_spatial_dist = total_spatial_dist / total_framewise_matching_doa
                            self._total_DE[class_cnt] += avg_spatial_dist
                            self._DE_TP[class_cnt] += 1.0
                            if avg_spatial_dist <= self._spatial_T:
                                self._TP[class_cnt] += 1.0
                            else:
                                loc_fp += 1.0
                                self._FP_spatial[class_cnt] += 1.0
                        if (nb_pred_doas or 0) > (nb_gt_doas or 0):
                            extra = float((nb_pred_doas or 0) - (nb_gt_doas or 0))
                            loc_fp += extra
                            self._FP[class_cnt] += extra
                            self._DE_FP[class_cnt] += extra
                        elif (nb_pred_doas or 0) < (nb_gt_doas or 0):
                            miss = float((nb_gt_doas or 0) - (nb_pred_doas or 0))
                            loc_fn += miss
                            self._FN[class_cnt] += miss
                            self._DE_FN[class_cnt] += miss
                elif class_cnt in gt[block_cnt] and class_cnt not in pred[block_cnt]:
                    miss = float(nb_gt_doas or 0)
                    loc_fn += miss
                    self._FN[class_cnt] += miss
                    self._DE_FN[class_cnt] += miss
                elif class_cnt not in gt[block_cnt] and class_cnt in pred[block_cnt]:
                    extra = float(nb_pred_doas or 0)
                    loc_fp += extra
                    self._FP[class_cnt] += extra
                    self._DE_FP[class_cnt] += extra
            self._S += min(loc_fp, loc_fn)
            self._D += max(0.0, loc_fn - loc_fp)
            self._I += max(0.0, loc_fp - loc_fn)

    def all_reduce(self, device: Optional[torch.device] = None) -> None:
        try:
            import torch.distributed as dist
        except ImportError:
            return
        if not dist.is_available() or not dist.is_initialized():
            return
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        parts = [
            self._TP,
            self._FP,
            self._FP_spatial,
            self._FN,
            self._Nref,
            np.array([self._S, self._D, self._I], dtype=np.float64),
            self._total_DE,
            self._DE_TP,
            self._DE_FP,
            self._DE_FN,
        ]
        buffer = torch.tensor(
            np.concatenate(parts),
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(buffer, op=dist.ReduceOp.SUM)
        vals = buffer.cpu().numpy()
        n = self._nb_classes
        offset = 0
        self._TP = vals[offset : offset + n].copy()
        offset += n
        self._FP = vals[offset : offset + n].copy()
        offset += n
        self._FP_spatial = vals[offset : offset + n].copy()
        offset += n
        self._FN = vals[offset : offset + n].copy()
        offset += n
        self._Nref = vals[offset : offset + n].copy()
        offset += n
        self._S, self._D, self._I = vals[offset : offset + 3].tolist()
        offset += 3
        self._total_DE = vals[offset : offset + n].copy()
        offset += n
        self._DE_TP = vals[offset : offset + n].copy()
        offset += n
        self._DE_FP = vals[offset : offset + n].copy()
        offset += n
        self._DE_FN = vals[offset : offset + n].copy()


class OfficialDCASEMetricsAccumulator:
    """Official DCASE track evaluator adapter for local_spatial_track."""

    def __init__(
        self,
        doa_threshold_deg: float = 20.0,
        average: str = "macro",
    ) -> None:
        self._doa_threshold_deg = float(doa_threshold_deg)
        self._average = average
        self._metrics: Optional[OfficialDCASESELDMetrics] = None

    def _ensure_metrics(self, nb_classes: int) -> None:
        if self._metrics is None:
            self._metrics = OfficialDCASESELDMetrics(
                doa_threshold=self._doa_threshold_deg,
                nb_classes=nb_classes,
                average=self._average,
            )
        elif self._metrics._nb_classes != int(nb_classes):
            raise ValueError(
                f"OfficialDCASEMetricsAccumulator expected {self._metrics._nb_classes} classes "
                f"but got {nb_classes}."
            )

    def update(
        self,
        pred: Dict[int, Dict[int, List[List[object]]]],
        gt: Dict[int, Dict[int, List[List[object]]]],
        nb_classes: int,
    ) -> None:
        self._ensure_metrics(nb_classes)
        assert self._metrics is not None
        self._metrics.update_seld_scores(pred, gt)

    def compute(self) -> Dict[str, float]:
        if self._metrics is None:
            return {
                "ER20": 0.0,
                "F20": 0.0,
                "LE_CD": 180.0,
                "LR_CD": 0.0,
                "SELD_score": 1.0,
            }
        er, f20, le_cd, lr_cd, seld, _ = self._metrics.compute_seld_scores()
        return {
            "ER20": round(float(er), 4),
            "F20": round(float(f20), 4),
            "LE_CD": round(float(le_cd), 2),
            "LR_CD": round(float(lr_cd), 4),
            "SELD_score": round(float(seld), 4),
        }

    def all_reduce(self, device: Optional[torch.device] = None) -> None:
        if self._metrics is not None:
            self._metrics.all_reduce(device=device)


def _empty_official_segment_dict(
    num_blocks: int,
) -> Dict[int, Dict[int, List[List[object]]]]:
    return {block_idx: {} for block_idx in range(max(int(num_blocks), 1))}


def _buffer_event_for_official_segments(
    buffer: Dict[int, Dict[int, Dict[int, List[List[float]]]]],
    block_idx: int,
    class_idx: int,
    frame_idx_in_block: int,
    track_or_src_idx: int,
    azimuth_deg: float,
    elevation_deg: float,
) -> None:
    class_map = buffer.setdefault(block_idx, {}).setdefault(class_idx, {})
    class_map.setdefault(frame_idx_in_block, []).append(
        [int(track_or_src_idx), float(azimuth_deg), float(elevation_deg)]
    )


def _finalize_official_segment_buffer(
    buffer: Dict[int, Dict[int, Dict[int, List[List[float]]]]],
    num_blocks: int,
) -> Dict[int, Dict[int, List[List[object]]]]:
    out = _empty_official_segment_dict(num_blocks)
    for block_idx, class_map in buffer.items():
        if block_idx not in out:
            continue
        for class_idx, frame_map in class_map.items():
            frame_keys = sorted(frame_map.keys())
            out[block_idx][class_idx] = [[frame_keys, [frame_map[k] for k in frame_keys]]]
    return out


def _build_frame_track_official_segment_dicts(
    prediction_output: FrameTrackPredictionOutput,
    batch: "SpatialBatch",
    temporal_padding_mask: Optional[Tensor],
    activity_threshold: float,
    use_num_active_gate: bool = False,
) -> List[Tuple[Dict[int, Dict[int, List[List[object]]]], Dict[int, Dict[int, List[List[object]]]]]]:
    """Convert one batch to official DCASE segment dictionaries per sample.

    When ``use_num_active_gate`` is True and the model exposes
    ``pred_num_active_logits``, a per-frame top-K̂ gate is OR'd with the
    hard ``activity_threshold`` gate. This matches how the validation
    metrics (``compute_frame_track_validation_metrics``) already use the
    num-active head, so the official DCASE SELD pipeline (F20/ER20/LE_CD/
    LR_CD) will actually benefit from training lambda_frame_num_active > 0.
    Default False preserves the legacy behaviour.
    """
    device = prediction_output.pred_activity.device
    batch_size, num_tracks, t_s_max = prediction_output.pred_activity.shape
    targets = _frame_source_target_tensors(batch, t_s_max, device)
    window_mask = targets["window_mask"].cpu()
    source_valid = targets["source_valid"].cpu()
    valid_time = _valid_time_mask(temporal_padding_mask, batch_size, t_s_max, device).cpu()

    pred_act_prob = torch.sigmoid(prediction_output.pred_activity).cpu()
    pred_cls_idx = prediction_output.pred_class_logits.argmax(dim=-1).cpu()
    pred_dir = F.normalize(prediction_output.pred_direction, dim=-1)
    pred_azi_deg, pred_ele_deg = _azi_ele_deg_from_direction_vector(pred_dir)
    pred_azi_deg = pred_azi_deg.cpu()
    pred_ele_deg = pred_ele_deg.cpu()

    # v13_E: per-frame top-K̂ gate. active_bkt[b, k, t] = (prob >= threshold)
    # OR (rank_in_sorted_by_prob < K̂_bt).
    if (
        use_num_active_gate
        and prediction_output.pred_num_active_logits is not None
    ):
        # sort activity prob per frame descending; pick rank indices
        sort_idx = pred_act_prob.argsort(dim=1, descending=True)  # [B, K, T]
        rank = torch.empty_like(sort_idx)
        rank.scatter_(
            1,
            sort_idx,
            torch.arange(num_tracks).view(1, num_tracks, 1).expand_as(sort_idx),
        )
        num_active_pred = (
            prediction_output.pred_num_active_logits.argmax(dim=-1).cpu()
        )  # [B, T]
        k_hat = num_active_pred.clamp(max=num_tracks)
        topk_active = rank < k_hat.unsqueeze(1)   # [B, K, T]
        hard_active = pred_act_prob >= activity_threshold
        active_mask_bkt = (topk_active | hard_active)
    else:
        active_mask_bkt = pred_act_prob >= activity_threshold

    src_cls = batch.source_class_indices.cpu()
    src_azi = batch.source_azimuth_deg.cpu()
    src_ele = batch.source_elevation_deg.cpu()
    target_num_steps = batch.target_num_steps.cpu()
    clip_duration = batch.clip_duration_seconds.cpu()

    outputs: List[
        Tuple[
            Dict[int, Dict[int, List[List[object]]]],
            Dict[int, Dict[int, List[List[object]]]],
        ]
    ] = []
    num_gt = source_valid.size(1)
    for b in range(batch_size):
        valid_steps = max(int(target_num_steps[b].item()), 1)
        clip_seconds = float(max(clip_duration[b].item(), 1e-6))
        step_seconds = clip_seconds / float(valid_steps)
        num_blocks = max(1, int(math.floor(max(clip_seconds - 1e-6, 0.0))) + 1)
        pred_buffer: Dict[int, Dict[int, Dict[int, List[List[float]]]]] = {}
        gt_buffer: Dict[int, Dict[int, Dict[int, List[List[float]]]]] = {}

        block_frame_counts: Dict[int, int] = {}
        block_frame_idx: List[int] = [0] * valid_steps
        block_idx_for_t: List[int] = [0] * valid_steps
        for t in range(valid_steps):
            frame_time_s = t * step_seconds
            block_idx = min(int(math.floor(frame_time_s + 1e-8)), num_blocks - 1)
            local_frame_idx = block_frame_counts.get(block_idx, 0)
            block_frame_counts[block_idx] = local_frame_idx + 1
            block_idx_for_t[t] = block_idx
            block_frame_idx[t] = local_frame_idx

        for t in range(valid_steps):
            if not bool(valid_time[b, t]):
                continue
            block_idx = block_idx_for_t[t]
            local_idx = block_frame_idx[t]
            for k in range(num_tracks):
                if not bool(active_mask_bkt[b, k, t].item()):
                    continue
                _buffer_event_for_official_segments(
                    pred_buffer,
                    block_idx=block_idx,
                    class_idx=int(pred_cls_idx[b, k, t].item()),
                    frame_idx_in_block=local_idx,
                    track_or_src_idx=k,
                    azimuth_deg=float(pred_azi_deg[b, k, t].item()),
                    elevation_deg=float(pred_ele_deg[b, k, t].item()),
                )
            for g in range(num_gt):
                if not bool(source_valid[b, g]):
                    continue
                if not bool(window_mask[b, g, t]):
                    continue
                _buffer_event_for_official_segments(
                    gt_buffer,
                    block_idx=block_idx,
                    class_idx=int(src_cls[b, g].item()),
                    frame_idx_in_block=local_idx,
                    track_or_src_idx=g,
                    # Per-frame GT for SELD official metric.
                    azimuth_deg=float(src_azi[b, g, t].item()),
                    elevation_deg=float(src_ele[b, g, t].item()),
                )

        outputs.append(
            (
                _finalize_official_segment_buffer(pred_buffer, num_blocks),
                _finalize_official_segment_buffer(gt_buffer, num_blocks),
            )
        )
    return outputs


def _seld_angular_distance_deg(
    azi1: float, ele1: float, azi2: float, ele2: float
) -> float:
    """Great-circle angular distance in degrees between two directions."""
    import math
    a1, e1 = math.radians(azi1), math.radians(ele1)
    a2, e2 = math.radians(azi2), math.radians(ele2)
    dot = (
        math.cos(e1) * math.cos(a1) * math.cos(e2) * math.cos(a2)
        + math.cos(e1) * math.sin(a1) * math.cos(e2) * math.sin(a2)
        + math.sin(e1) * math.sin(e2)
    )
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


def accumulate_mono_ast_seld(
    prediction_output: MonoTaskPredictionOutput,
    batch: "SpatialBatch",
    accumulator: SELDMetricsAccumulator,
) -> None:
    """Feed a mono_ast batch into a SELDMetricsAccumulator (CPU, no grad)."""
    pred_cls = prediction_output.pred_class_logits.argmax(dim=-1).tolist()
    pred_azi, pred_ele = _azi_ele_deg_from_direction_vector(
        prediction_output.pred_direction
    )
    pred_azi = pred_azi.tolist()
    pred_ele = pred_ele.tolist()
    gt_cls = batch.source_class_indices[:, 0].tolist()
    gt_azi = batch.source_azimuth_deg[:, 0, 0].tolist()
    gt_ele = batch.source_elevation_deg[:, 0, 0].tolist()
    for i in range(len(gt_cls)):
        accumulator.update(
            pred_class_idx=int(pred_cls[i]),
            pred_azi=float(pred_azi[i]),
            pred_ele=float(pred_ele[i]),
            gt_class_idx=int(gt_cls[i]),
            gt_azi=float(gt_azi[i]),
            gt_ele=float(gt_ele[i]),
        )


def accumulate_frame_track_seld(
    prediction_output: FrameTrackPredictionOutput,
    batch: "SpatialBatch",
    temporal_padding_mask: Optional[Tensor],
    accumulator: OfficialDCASEMetricsAccumulator,
    activity_threshold: float = 0.5,
    use_num_active_gate: bool = False,
) -> None:
    """Feed a frame-track batch into the official DCASE evaluator.

    When ``use_num_active_gate=True`` and the model exposes a num_active head,
    gating OR's the per-frame top-K̂ mask with the hard activity threshold.
    This lets the official DCASE F20 actually benefit from training the
    num_active head (v13_E).
    """
    nb_classes = int(prediction_output.pred_class_logits.size(-1))
    sample_dicts = _build_frame_track_official_segment_dicts(
        prediction_output=prediction_output,
        batch=batch,
        temporal_padding_mask=temporal_padding_mask,
        activity_threshold=activity_threshold,
        use_num_active_gate=use_num_active_gate,
    )
    for pred_dict, gt_dict in sample_dicts:
        accumulator.update(pred=pred_dict, gt=gt_dict, nb_classes=nb_classes)


# ---------------------------------------------------------------------------
# Frame-wise single-source supervision (``local_spatial_framewise``)
#
# 每帧独立预测 activity + class + direction + distance：
#   - activity BCE: 所有非 padding 帧都参与，活跃帧 GT=1，非活跃帧 GT=0
#   - class / direction / distance loss: 只在活跃帧上计算
# ---------------------------------------------------------------------------

def compute_framewise_losses(
    prediction_output: FrameWisePredictionOutput,
    batch: "SpatialBatch",
    config: SpatialLossConfig,
    temporal_padding_mask: Optional[Tensor] = None,
    clip_aux_prediction: Optional[MonoTaskPredictionOutput] = None,
) -> "SpatialLossOutput":
    """Frame-level activity + cls + direction + distance loss.

    Activity BCE: all non-padded frames, active window = 1, else = 0.
    Cls / dir / dist: active frames only.
    """
    valid_counts = batch.source_valid_mask.sum(dim=1)
    if not torch.all(valid_counts == 1):
        raise ValueError(
            "framewise supervision expects exactly one valid source per sample; "
            f"got counts={valid_counts.tolist()}"
        )

    device = prediction_output.pred_class_logits.device
    B, T_s, _ = prediction_output.pred_class_logits.shape

    cls_target = batch.source_class_indices[:, 0].to(device)   # [B]
    azi_deg    = batch.source_azimuth_deg[:, 0, 0].to(device)     # [B]
    ele_deg    = batch.source_elevation_deg[:, 0, 0].to(device)   # [B]
    dist_tgt   = batch.source_distance[:, 0, 0].to(device)        # [B]
    gt_dir     = _direction_vector_from_azi_ele_deg(azi_deg, ele_deg).to(
        prediction_output.pred_direction.dtype
    )  # [B, 3]

    # Active-window mask [B, T_s]: True = source is active in this frame
    win_mask = build_primary_source_window_mask(batch, T_s).to(device)

    # Non-padded mask [B, T_s]: True = this frame is not padding
    valid_mask = torch.ones(B, T_s, dtype=torch.bool, device=device)
    if temporal_padding_mask is not None:
        valid_mask = valid_mask & (~temporal_padding_mask.to(device).bool())

    # Active mask must be within valid frames
    active_mask = win_mask & valid_mask   # [B, T_s]

    # Fallback: if some samples have no active frame, treat all valid frames as active
    has_active = active_mask.any(dim=1)
    if not has_active.all():
        active_mask[~has_active] = valid_mask[~has_active]

    zero = prediction_output.pred_class_logits.new_zeros(())

    # --- Activity BCE: all non-padded frames ---
    # GT: active_mask.float() = 1 for active, 0 for inactive
    act_gt    = active_mask.float().unsqueeze(-1)   # [B, T_s, 1]
    act_valid = valid_mask.unsqueeze(-1)             # [B, T_s, 1]
    act_logits = prediction_output.pred_activity     # [B, T_s, 1]
    if act_valid.any():
        loss_activity = F.binary_cross_entropy_with_logits(
            act_logits[act_valid],
            act_gt[act_valid],
        )
    else:
        loss_activity = zero

    # --- Cls / dir / dist: active frames only ---
    cls_losses, dir_losses, dist_losses = [], [], []
    for b in range(B):
        mask_b = active_mask[b]   # [T_s]
        if not mask_b.any():
            continue

        cls_logits_b = prediction_output.pred_class_logits[b][mask_b]   # [N, C]
        cls_tgt_b    = cls_target[b].expand(mask_b.sum())
        cls_losses.append(F.cross_entropy(
            cls_logits_b, cls_tgt_b, label_smoothing=config.label_smoothing
        ))

        pred_dir_b = F.normalize(
            prediction_output.pred_direction[b][mask_b], dim=-1
        )   # [N, 3]
        gt_dir_b = gt_dir[b].unsqueeze(0).expand_as(pred_dir_b)
        dir_losses.append((1.0 - (pred_dir_b * gt_dir_b).sum(dim=-1)).mean())

        pred_dist_b = prediction_output.pred_distance[b][mask_b, 0]   # [N]
        dist_losses.append(F.smooth_l1_loss(
            pred_dist_b, dist_tgt[b].expand_as(pred_dist_b)
        ))

    if not cls_losses:
        return SpatialLossOutput(
            loss_total=config.lambda_framewise_activity * loss_activity,
            loss_activity=loss_activity, loss_azi=zero,
            loss_ele=zero, loss_dist=zero, loss_cls_aux=zero,
            loss_temp=zero, loss_direction=zero,
        )

    loss_cls  = torch.stack(cls_losses).mean()
    loss_dir  = torch.stack(dir_losses).mean()
    loss_dist = torch.stack(dist_losses).mean()

    # Semantic anchor loss (pre-fusion BEATs tokens, active frames)
    loss_sem_anchor = zero
    if config.lambda_sem_anchor > 0.0 and prediction_output.sem_class_logits is not None:
        anc_losses = []
        for b in range(B):
            mask_b = active_mask[b]
            if not mask_b.any():
                continue
            anc_logits_b = prediction_output.sem_class_logits[b][mask_b]
            anc_tgt_b    = cls_target[b].expand(mask_b.sum())
            anc_losses.append(F.cross_entropy(
                anc_logits_b, anc_tgt_b, label_smoothing=config.label_smoothing
            ))
        if anc_losses:
            loss_sem_anchor = torch.stack(anc_losses).mean()

    loss_total = (
        config.lambda_framewise_activity * loss_activity
        + config.lambda_cls_aux          * loss_cls
        + config.lambda_direction        * loss_dir
        + config.lambda_dist             * loss_dist
        + config.lambda_sem_anchor       * loss_sem_anchor
    )

    # Clip-level aux head must participate in loss to avoid DDP errors
    if clip_aux_prediction is not None:
        clip_aux_loss = compute_clip_aux_losses(clip_aux_prediction, batch, config)
        loss_total = loss_total + config.lambda_clip_aux * clip_aux_loss

    return SpatialLossOutput(
        loss_total=loss_total,
        loss_activity=loss_activity,
        loss_azi=zero,
        loss_ele=zero,
        loss_dist=loss_dist,
        loss_cls_aux=loss_cls,
        loss_temp=loss_sem_anchor,   # reuse loss_temp 槽记录 anchor loss
        loss_direction=loss_dir,
    )


def compute_framewise_validation_metrics(
    prediction_output: FrameWisePredictionOutput,
    batch: "SpatialBatch",
    temporal_padding_mask: Optional[Tensor] = None,
) -> "SpatialMetricOutput":
    """Validation metrics for framewise path.

    逐帧评测逻辑：
    1. activity 评测：以 GT 活跃窗口为正样本，计算 precision/recall/acc
    2. cls/dir/dist 评测：对模型预测为活跃的帧做 mean-pool，
       fallback 到 GT 活跃帧（确保每个样本都有预测结果）
    """
    device = prediction_output.pred_class_logits.device
    B, T_s, _ = prediction_output.pred_class_logits.shape

    valid_counts = batch.source_valid_mask.sum(dim=1)
    if not torch.all(valid_counts == 1):
        raise ValueError(
            "framewise supervision expects exactly one valid source per sample; "
            f"got counts={valid_counts.tolist()}"
        )

    cls_target  = batch.source_class_indices[:, 0].to(device)
    azi_tgt_deg = batch.source_azimuth_deg[:, 0, 0].to(device)
    ele_tgt_deg = batch.source_elevation_deg[:, 0, 0].to(device)
    dist_tgt    = batch.source_distance[:, 0, 0].to(device)

    # Non-padded mask
    valid_mask = torch.ones(B, T_s, dtype=torch.bool, device=device)
    if temporal_padding_mask is not None:
        valid_mask = valid_mask & (~temporal_padding_mask.to(device).bool())

    # GT active mask
    gt_active = build_primary_source_window_mask(batch, T_s).to(device) & valid_mask

    # Predicted active mask: frames where sigmoid(activity) > 0.5
    pred_active = (prediction_output.pred_activity.squeeze(-1) > 0.0) & valid_mask  # [B, T_s]

    # --- Activity metrics (frame-level) ---
    tp = (pred_active & gt_active).float().sum()
    fp = (pred_active & ~gt_active).float().sum()
    fn = (~pred_active & gt_active).float().sum()
    tn = (~pred_active & ~gt_active & valid_mask).float().sum()
    total_valid = valid_mask.float().sum().clamp_min(1.0)
    activity_acc       = (tp + tn) / total_valid
    activity_precision = tp / (tp + fp).clamp_min(1e-6)
    activity_recall    = tp / (tp + fn).clamp_min(1e-6)

    # --- Cls / dir / dist: mean-pool over predicted active frames ---
    # fallback: use GT active if model predicts nothing active for a sample
    pool_mask = pred_active.clone()
    no_pred = ~pool_mask.any(dim=1)
    if no_pred.any():
        pool_mask[no_pred] = gt_active[no_pred]
        # last resort: all valid frames
        still_empty = ~pool_mask.any(dim=1)
        if still_empty.any():
            pool_mask[still_empty] = valid_mask[still_empty]

    keep_f = pool_mask.unsqueeze(-1).float()           # [B, T_s, 1]
    denom  = keep_f.sum(dim=1).clamp_min(1.0)         # [B, 1]

    cls_logits_pooled = (
        prediction_output.pred_class_logits * keep_f
    ).sum(dim=1) / denom                               # [B, C]
    dir_pooled = F.normalize(
        (prediction_output.pred_direction * keep_f).sum(dim=1) / denom, dim=-1
    )                                                  # [B, 3]
    dist_pooled = (
        prediction_output.pred_distance * keep_f
    ).sum(dim=1) / denom                               # [B, 1]

    pred_class_idx = cls_logits_pooled.argmax(dim=-1)
    pred_azi_deg, pred_ele_deg = _azi_ele_deg_from_direction_vector(dir_pooled)
    pred_dist = dist_pooled[:, 0]

    return SpatialMetricOutput(
        activity_acc=activity_acc,
        activity_precision=activity_precision,
        activity_recall=activity_recall,
        class_acc=(pred_class_idx == cls_target).float().mean(),
        azi_mae_deg=_circular_distance_deg(pred_azi_deg, azi_tgt_deg).mean(),
        ele_mae_deg=(pred_ele_deg - ele_tgt_deg).abs().mean(),
        dist_mae=(pred_dist - dist_tgt).abs().mean(),
        matched_count=torch.tensor(float(B), device=device),
    )


def build_framewise_validation_examples(
    prediction_output: FrameWisePredictionOutput,
    batch: "SpatialBatch",
    temporal_padding_mask: Optional[Tensor],
    vocabulary: Optional[List[str]] = None,
    max_examples: int = 16,
) -> List[dict]:
    """Build JSONL-serialisable qualitative examples for the framewise path.

    Mean-pools over predicted active frames (sigmoid > 0.5),
    with fallback to GT active window if model predicts nothing.
    """
    B, T_s, _ = prediction_output.pred_class_logits.shape
    device = prediction_output.pred_class_logits.device

    valid_mask = torch.ones(B, T_s, dtype=torch.bool, device=device)
    if temporal_padding_mask is not None:
        valid_mask = valid_mask & (~temporal_padding_mask.to(device).bool())

    gt_active   = build_primary_source_window_mask(batch, T_s).to(device) & valid_mask
    pred_active = (prediction_output.pred_activity.squeeze(-1) > 0.0) & valid_mask

    pool_mask = pred_active.clone()
    no_pred = ~pool_mask.any(dim=1)
    if no_pred.any():
        pool_mask[no_pred] = gt_active[no_pred]
        still_empty = ~pool_mask.any(dim=1)
        if still_empty.any():
            pool_mask[still_empty] = valid_mask[still_empty]

    keep_f = pool_mask.unsqueeze(-1).float()
    denom  = keep_f.sum(dim=1).clamp_min(1.0)

    cls_logits_pooled = (
        prediction_output.pred_class_logits * keep_f
    ).sum(dim=1) / denom
    dir_pooled = F.normalize(
        (prediction_output.pred_direction * keep_f).sum(dim=1) / denom, dim=-1
    )
    dist_pooled = (
        (prediction_output.pred_distance * keep_f).sum(dim=1) / denom
    )[:, 0]

    pred_class_idx = cls_logits_pooled.argmax(dim=-1)
    pred_azi_deg, pred_ele_deg = _azi_ele_deg_from_direction_vector(dir_pooled)
    pred_azi_deg = _to_dcase_azimuth(pred_azi_deg)

    examples: List[dict] = []
    for i in range(min(B, max_examples)):
        gt_cls_idx = int(batch.source_class_indices[i, 0].item())
        examples.append({
            "sample_id": batch.sample_ids[i] if batch.sample_ids else str(i),
            "gt_class_index":   gt_cls_idx,
            "gt_class_name":    (vocabulary[gt_cls_idx] if vocabulary and gt_cls_idx < len(vocabulary) else str(gt_cls_idx)),
            "pred_class_index": int(pred_class_idx[i].item()),
            "pred_class_confidence": float(cls_logits_pooled[i].softmax(dim=-1).max().item()),
            "gt_azimuth_deg":   float(batch.source_azimuth_deg[i, 0, 0].item()),
            "pred_azimuth_deg": float(pred_azi_deg[i].item()),
            "gt_elevation_deg": float(batch.source_elevation_deg[i, 0, 0].item()),
            "pred_elevation_deg": float(pred_ele_deg[i].item()),
            "gt_distance_m":    float(batch.source_distance[i, 0, 0].item()),
            "pred_distance_m":  float(dist_pooled[i].item()),
        })
    return examples
