"""Dataset skeleton for the simplified Spatial-BEATs training pipeline.

This file defines the data interfaces, vocabulary contracts, and batch tensor
shapes for Spatial-BEATs. The actual data loading logic is intentionally left
unimplemented so the I/O structure can be reviewed first.
"""

from dataclasses import dataclass, field
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset
from tqdm.auto import tqdm


@dataclass
class SourceVocabularyConfig:
    """Source vocabulary settings for auxiliary class supervision.

    Defaults to the SO-Dataset 63-class vocabulary produced by
    ``scripts/data/build_so_vocab.py`` (63 rows). Users can override
    via the ``SO_VOCAB`` env var or the ``--source-vocab-path`` /
    ``--source-num-classes`` CLI flags on ``train_so_pretrain.py``.
    """

    vocab_path: str = os.environ.get("SO_VOCAB", "")
    label_id_field: str = "label_id"
    label_name_field: str = "final_label"
    num_classes: int = 63


@dataclass
class QwenLikeMelConfig:
    """Low-level mel parameter contract aligned with Qwen-2.5-Omni.

    Only the front-end acoustic parameters are aligned:
        - sample_rate = 16000
        - num_mel_bins = 128
        - n_fft = 400
        - win_length = 400
        - hop_length = 160
        - dither = 0.0

    The downstream encoder architecture remains Spatial-BEATs, not Qwen.
    """

    sample_rate: int = 16000
    num_mel_bins: int = 128
    n_fft: int = 400
    win_length: int = 400
    hop_length: int = 160
    dither: float = 0.0
    waveform_scale: float = float(2**15)
    fbank_mean: float = 15.41663
    fbank_std: float = 6.55582
    normalize_logmel: bool = True


@dataclass
class SpatialDatasetConfig:
    """Configuration for dataset loading and collation."""

    source_vocab: SourceVocabularyConfig = field(default_factory=SourceVocabularyConfig)
    mel_config: QwenLikeMelConfig = field(default_factory=QwenLikeMelConfig)
    target_token_rate: float = 2.5
    max_sources: int = 4
    padding_side: str = "right"
    padding_value: float = 0.0
    max_clip_duration_seconds: Optional[float] = None
    min_crop_duration_seconds: Optional[float] = None  # random duration range lower bound
    crop_mode: str = "none"
    allowed_splits: Optional[tuple[str, ...]] = None
    show_progress: bool = True

    # === v13_B [B-5] Real-distribution augment ==============================
    # All augment flags default to "off" so existing presets are unaffected.
    # Augments are ONLY applied when the dataset's allowed_splits contains
    # "train" (i.e. training data), never on valid/test.
    use_spec_augment: bool = False
    spec_augment_time_mask_ratio: float = 0.0     # max fraction of T masked per stripe
    spec_augment_freq_mask_ratio: float = 0.0     # max fraction of F masked per stripe
    spec_augment_num_time_stripes: int = 2
    spec_augment_num_freq_stripes: int = 2
    # Waveform-level augment
    random_gain_db: float = 0.0                   # sample gain from U[-x, +x] dB
    channel_dropout_prob: float = 0.0             # P(mask 1 FOA channel, per sample)
    lowpass_sim_real_prob: float = 0.0            # P(apply lowpass, per sample)
    lowpass_cutoff_min_hz: float = 4000.0
    lowpass_cutoff_max_hz: float = 8000.0


@dataclass
class SourceEvent:
    """One source-level annotation inside a clip.

    Attributes:
        class_index:
            Integer class index in the final_vocabulary.csv space.
        class_label:
            Human-readable label name from final_vocabulary.csv.
        azimuth_deg:
            Source azimuth in degrees.  For dynamic sources this is the
            per-clip fallback (e.g. first frame's DOA); per-frame targets
            live in ``frame_azi_deg``.
        elevation_deg:
            Source elevation in degrees.  Per-frame targets live in
            ``frame_ele_deg`` for dynamic sources.
        distance:
            Continuous source distance (metres).  Per-frame targets live in
            ``frame_distance_m`` for dynamic sources.
        distance_valid:
            False when the manifest marks distance as unknown (STARSS real,
            DCASE).  Suppresses distance loss at the clip level.
        start_time_seconds:
            Weak start time for the source inside the clip.
        end_time_seconds:
            Weak end time for the source inside the clip.
        frame_times_s:
            Optional [N_frames] tensor of frame timestamps in seconds,
            relative to the start of the (un-cropped) clip.  When present,
            the loader builds per-frame DOA targets by linear interpolation;
            when None, static scalar DOA is broadcast to every time step.
        frame_azi_deg:
            Optional [N_frames] per-frame azimuth in degrees, aligned with
            ``frame_times_s``.  Continuous (not wrapped).
        frame_ele_deg:
            Optional [N_frames] per-frame elevation in degrees.
        frame_distance_m:
            Optional [N_frames] per-frame distance in metres.  Values are
            only consumed where ``frame_distance_valid`` is True.
        frame_distance_valid:
            Optional [N_frames] boolean mask.  True when that frame carries
            a reliable distance (e.g. sim_moving), False for DCASE/STARSS
            where distance is unknown.
        frame_ele_sign_only:
            Optional [N_frames] boolean mask.  True when the elevation for
            that frame is only known as a sign (upper/lower hemisphere),
            not as an exact angle.  Derived from ±inf elevation in the new
            unified dataset.  When True, the loss uses a 2-class
            upper/lower BCE instead of the full 180-bin Gaussian CE.
    """

    class_index: int
    class_label: str
    azimuth_deg: float
    elevation_deg: float
    distance: float
    distance_valid: bool
    start_time_seconds: float
    end_time_seconds: float
    frame_times_s: Optional[Tensor] = None
    frame_azi_deg: Optional[Tensor] = None
    frame_ele_deg: Optional[Tensor] = None
    frame_distance_m: Optional[Tensor] = None
    frame_distance_valid: Optional[Tensor] = None
    frame_ele_sign_only: Optional[Tensor] = None


@dataclass
class SpatialSample:
    """One training sample before collation.

    Attributes:
        sample_id:
            Stable identifier for debugging and loss analysis.
        waveform:
            [4, T] FOA waveform ordered as W / X / Y / Z.
        clip_duration_seconds:
            Scalar duration of the valid clip before any batch padding.
        sources:
            Variable-length list of source-level annotations.
    """

    sample_id: str
    waveform: Tensor
    clip_duration_seconds: float
    sources: List[SourceEvent]


@dataclass
class SpatialBatch:
    """Collated batch contract used by Spatial-BEATs.

    Tensor fields:
        waveform:
            [B, 4, T_max_wave] FOA waveform padded across the batch.
        waveform_padding_mask:
            [B, T_max_wave] boolean mask where True marks padded waveform samples.
        clip_duration_seconds:
            [B] valid clip durations in seconds.
        target_num_steps:
            [B] valid temporal token counts after resampling:
                T_s_i = round(duration_i * target_token_rate)
        source_class_indices:
            [B, N_gt_max] class indices in the final_vocabulary.csv space.
        source_azimuth_deg:
            [B, N_gt_max, T_s_max] per-frame azimuth targets in degrees.  For
            static sources the scalar is broadcast along the T_s axis; for
            dynamic sources the values come from linear interpolation of
            ``SourceEvent.frame_azi_deg`` onto the model's 10 Hz grid.
        source_elevation_deg:
            [B, N_gt_max, T_s_max] per-frame elevation targets in degrees.
        source_distance:
            [B, N_gt_max, T_s_max] per-frame distance targets in metres.
        source_distance_valid:
            [B, N_gt_max, T_s_max] boolean mask where True means the distance
            target at that time step is reliable.  False for sources whose
            distance is null (STARSS real, DCASE) or for frames outside a
            dynamic source's known-distance range.
        source_ele_sign_only:
            [B, N_gt_max, T_s_max] boolean mask where True means the
            elevation for that (source, frame) is only known as a
            sign/hemisphere (±inf in the new unified dataset), not as an
            exact angle.  Loss code uses a 2-class upper/lower CE for these
            frames instead of the full 180-bin Gaussian CE.
        source_start_time_seconds:
            [B, N_gt_max] weak start times.
        source_end_time_seconds:
            [B, N_gt_max] weak end times.
        source_valid_mask:
            [B, N_gt_max] boolean mask where True marks real sources.

    Non-tensor fields:
        sample_ids:
            List[str] of length B.
        source_class_labels:
            Optional nested label names aligned with source_class_indices.
    """

    waveform: Tensor
    waveform_padding_mask: Optional[Tensor]
    clip_duration_seconds: Tensor
    target_num_steps: Tensor
    source_class_indices: Tensor
    source_azimuth_deg: Tensor
    source_elevation_deg: Tensor
    source_distance: Tensor
    source_distance_valid: Tensor
    source_ele_sign_only: Tensor
    source_start_time_seconds: Tensor
    source_end_time_seconds: Tensor
    source_valid_mask: Tensor
    sample_ids: List[str]
    source_class_labels: Optional[List[List[str]]]


def load_source_vocabulary(
    config: SourceVocabularyConfig,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Load the source vocabulary metadata from final_vocabulary.csv.

    Expected CSV fields:
        - label_id
        - final_label

    Returns:
        Dict[str, Any]:
            A structured vocabulary object. Suggested keys:
                - index_to_label
                - label_to_index
                - label_id_to_index
                - raw_rows
    """
    rows: List[Dict[str, Any]] = []
    vocab_path = Path(config.vocab_path)
    with open(vocab_path, "r", encoding="utf-8") as handle:
        total_rows = max(sum(1 for _ in handle) - 1, 0)
    with open(vocab_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in tqdm(
            reader,
            total=total_rows,
            desc=f"Load vocabulary {vocab_path.name}",
            leave=False,
            disable=not show_progress,
        ):
            rows.append(row)

    rows = sorted(rows, key=lambda row: int(row[config.label_id_field]))
    index_to_label: List[str] = []
    label_to_index: Dict[str, int] = {}
    label_id_to_index: Dict[int, int] = {}

    for index, row in enumerate(rows):
        label_name = str(row[config.label_name_field])
        label_id = int(row[config.label_id_field])
        index_to_label.append(label_name)
        label_to_index[label_name] = index
        label_id_to_index[label_id] = index

    if config.num_classes != len(index_to_label):
        raise ValueError(
            f"Configured source_num_classes={config.num_classes}, "
            f"but vocabulary contains {len(index_to_label)} rows."
        )

    return {
        "index_to_label": index_to_label,
        "label_to_index": label_to_index,
        "label_id_to_index": label_id_to_index,
        "raw_rows": rows,
    }


def compute_target_num_steps(
    clip_duration_seconds: Tensor,
    target_token_rate: float,
) -> Tensor:
    """Convert clip durations into valid temporal token counts.

    Args:
        clip_duration_seconds:
            [B] clip durations in seconds.
        target_token_rate:
            Final spatial token rate, e.g. 2.5 Hz.

    Returns:
        Tensor:
            [B] valid number of temporal steps for each sample:
                T_s_i = round(duration_i * target_token_rate)
    """
    target_num_steps = torch.round(clip_duration_seconds * target_token_rate).long()
    return torch.clamp(target_num_steps, min=1)


def _load_manifest_entries(
    manifest_path: Path,
    show_progress: bool = True,
) -> List[Dict[str, Any]]:
    suffix = manifest_path.suffix.lower()
    if suffix == ".jsonl":
        entries = []
        with open(manifest_path, "r", encoding="utf-8") as handle:
            total_lines = sum(1 for _ in handle)
        with open(manifest_path, "r", encoding="utf-8") as handle:
            for line in tqdm(
                handle,
                total=total_lines,
                desc=f"Load manifest {manifest_path.name}",
                leave=False,
                disable=not show_progress,
            ):
                line = line.strip()
                if not line:
                    continue
                entries.append(json.loads(line))
        return entries
    if suffix == ".json":
        with open(manifest_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict) and "data" in data:
            data = data["data"]
        if not isinstance(data, list):
            raise ValueError("JSON manifest must be a list or a dict with a 'data' list.")
        return list(
            tqdm(
                data,
                total=len(data),
                desc=f"Load manifest {manifest_path.name}",
                leave=False,
                disable=not show_progress,
            )
        )
    raise ValueError(f"Unsupported manifest format: {manifest_path}")


def _load_audio_file(path: str, expected_sample_rate: int) -> Tensor:
    try:
        import soundfile as sf  # type: ignore

        waveform, sample_rate = sf.read(path, always_2d=True)
        waveform = torch.from_numpy(waveform.T).float()
    except Exception:
        try:
            from scipy.io import wavfile  # type: ignore

            sample_rate, waveform_np = wavfile.read(path)
            if waveform_np.ndim == 1:
                waveform_np = waveform_np[:, None]
            if not str(waveform_np.dtype).startswith("float"):
                max_val = max(float(torch.iinfo(torch.from_numpy(waveform_np).dtype).max), 1.0)
                waveform = torch.from_numpy(waveform_np.astype("float32")) / max_val
            else:
                waveform = torch.from_numpy(waveform_np.astype("float32"))
            waveform = waveform.transpose(0, 1)
        except Exception as exc:
            try:
                import wave

                with wave.open(path, "rb") as handle:
                    sample_rate = handle.getframerate()
                    num_channels = handle.getnchannels()
                    sample_width = handle.getsampwidth()
                    num_frames = handle.getnframes()
                    raw_bytes = handle.readframes(num_frames)

                if sample_width == 2:
                    dtype = torch.int16
                    max_val = float(torch.iinfo(dtype).max)
                elif sample_width == 4:
                    dtype = torch.int32
                    max_val = float(torch.iinfo(dtype).max)
                else:
                    raise RuntimeError(f"Unsupported PCM sample width: {sample_width} bytes")

                waveform = torch.frombuffer(raw_bytes, dtype=dtype).reshape(num_frames, num_channels)
                waveform = waveform.float().transpose(0, 1) / max_val
            except Exception as wave_exc:
                raise RuntimeError(
                    f"Failed to load audio file '{path}'. Install soundfile/scipy or provide PCM wav."
                ) from wave_exc

    if int(sample_rate) != int(expected_sample_rate):
        raise ValueError(
            f"Expected sample_rate={expected_sample_rate}, but got {sample_rate} for {path}"
        )
    if waveform.ndim != 2:
        raise ValueError(f"Expected audio with shape [C, T], got {tuple(waveform.shape)}")
    if waveform.size(0) != 4 and waveform.size(1) == 4:
        waveform = waveform.transpose(0, 1)
    if waveform.size(0) != 4:
        raise ValueError(f"Expected FOA waveform with 4 channels, got {tuple(waveform.shape)}")
    return waveform.contiguous()


# ---------------------------------------------------------------------------
# Label aliases — fine-grained labels collapsed to the 63-class FSD50K vocab.
# Some of our generated manifests (qa_moving, qa_counting, ...) carry the
# original gendered/specialised labels from FSD50K's raw ontology (e.g.
# "male_singing" / "female_singing"), but the Spatial-BEATs vocabulary used in
# training only covers the 63 collapsed classes (e.g. "singing").  Rather
# than regenerating every manifest we normalise the raw label string here
# before vocabulary lookup.
# ---------------------------------------------------------------------------
_LABEL_ALIASES: Dict[str, str] = {
    "male_singing": "singing",
    "female_singing": "singing",
}


def _apply_label_alias(label: str) -> str:
    return _LABEL_ALIASES.get(label, label)


def _resolve_class_index(
    source_entry: Dict[str, Any],
    vocabulary: Dict[str, Any],
) -> int:
    # Explicit numeric index fields (highest priority, always unambiguous).
    if "class_index" in source_entry:
        return int(source_entry["class_index"])
    if "label_index" in source_entry:
        return int(source_entry["label_index"])
    if "source_label_index" in source_entry:
        return int(source_entry["source_label_index"])
    # String label fields (second priority).  This correctly handles datasets
    # whose numeric label_id uses a different numbering scheme from the
    # vocabulary (e.g. the unified_spatial_foa_fsd63_all dataset uses 0-based
    # label_ids that don't match the 1-based ids in final_vocabulary.csv, but
    # always supplies a human-readable 'label' string).
    for key in (
        "class_label",
        "final_label",
        "label",
        "source_label",
        "mono_target_label",
        "mono_primary_label",
    ):
        if key in source_entry:
            raw = str(source_entry[key])
            canonical = _apply_label_alias(raw)
            return int(vocabulary["label_to_index"][canonical])
    # Numeric label_id / class_id fallback (legacy datasets that don't carry a
    # string label).  Look up in the vocabulary map; the map was built from the
    # CSV and handles 1-based IDs correctly.
    for id_key in ("label_id", "class_id"):
        if id_key in source_entry:
            lid = int(source_entry[id_key])
            if lid in vocabulary["label_id_to_index"]:
                return int(vocabulary["label_id_to_index"][lid])
    raise KeyError("Unable to resolve source class index from source entry.")


def _resolve_class_label(
    source_entry: Dict[str, Any],
    vocabulary: Dict[str, Any],
    class_index: int,
) -> str:
    for key in (
        "class_label",
        "final_label",
        "label",
        "source_label",
        "mono_target_label",
        "mono_primary_label",
    ):
        if key in source_entry:
            return _apply_label_alias(str(source_entry[key]))
    return str(vocabulary["index_to_label"][class_index])


def _maybe_get_float(source_entry: Dict[str, Any], keys: Sequence[str]) -> Optional[float]:
    for key in keys:
        if key in source_entry:
            value = source_entry[key]
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def _get_float(source_entry: Dict[str, Any], keys: Sequence[str], default: Optional[float] = None) -> float:
    value = _maybe_get_float(source_entry, keys)
    if value is not None:
        return value
    if default is not None:
        return float(default)
    raise KeyError(f"Missing required keys {keys} in source entry.")


def _resolve_distance_m(source_entry: Dict[str, Any], default: float = 1.0) -> float:
    value = _maybe_get_float(source_entry, ("distance", "distance_m"))
    if value is not None:
        return value

    value = _maybe_get_float(
        source_entry,
        ("distance_cm", "rir_distance_cm", "horizontal_distance_cm", "rir_horizontal_distance_cm"),
    )
    if value is not None:
        return value / 100.0

    listener_position = source_entry.get("listener_position_cm", source_entry.get("rir_listener_position_cm"))
    source_position = source_entry.get("source_position_cm", source_entry.get("rir_source_position_cm"))
    if (
        isinstance(listener_position, list | tuple)
        and isinstance(source_position, list | tuple)
        and len(listener_position) >= 3
        and len(source_position) >= 3
    ):
        try:
            return math.sqrt(
                sum(
                    (float(source_position[idx]) - float(listener_position[idx])) ** 2
                    for idx in range(3)
                )
            ) / 100.0
        except (TypeError, ValueError):
            pass

    return float(default)


def _is_distance_valid(source_entry: Dict[str, Any]) -> bool:
    """Return False when the manifest explicitly marks distance as unknown/null.

    A source is considered to have a *valid* distance when:
      - the manifest entry contains ``"distance_valid": true`` (generated by
        map_real_manifest.py for STARSS real data), OR
      - ``"distance_valid"`` is absent (legacy sim manifests — all have real
        distances), OR
      - any numeric distance field is present and non-null.

    A source is *invalid* only when ``"distance_valid": false`` is explicitly
    set, or when all numeric fields resolve to None.
    """
    explicit = source_entry.get("distance_valid")
    if explicit is not None:
        return bool(explicit)

    # Legacy: if any numeric distance field is non-null, treat as valid.
    for key in ("distance", "distance_m", "distance_cm", "rir_distance_cm",
                "horizontal_distance_cm", "rir_horizontal_distance_cm"):
        v = source_entry.get(key)
        if v is not None:
            return True

    # Has 3-D positions → can compute distance
    lp = source_entry.get("listener_position_cm", source_entry.get("rir_listener_position_cm"))
    sp = source_entry.get("source_position_cm", source_entry.get("rir_source_position_cm"))
    if isinstance(lp, (list, tuple)) and isinstance(sp, (list, tuple)):
        return True

    return False


def _merge_doa_fields(source_entry: Dict[str, Any]) -> Dict[str, Any]:
    doa = source_entry.get("doa")
    if not isinstance(doa, dict):
        return dict(source_entry)
    merged = dict(source_entry)
    if "azimuth_deg" in doa:
        merged["azimuth_deg"] = doa.get("azimuth_deg")
    if "elevation_deg" in doa:
        merged["elevation_deg"] = doa.get("elevation_deg")
    return merged


def _source_has_valid_doa(source_entry: Dict[str, Any]) -> bool:
    merged = _merge_doa_fields(source_entry)
    azimuth = _maybe_get_float(merged, ("azimuth_deg", "azimuth"))
    elevation = _maybe_get_float(merged, ("elevation_deg", "elevation"))
    if azimuth is not None and elevation is not None:
        return True
    # New unified dataset: trajectory lives in an external CSV file.
    # The CSV always contains per-frame azi/ele, so treat as valid.
    if source_entry.get("source_trajectory_csv_path"):
        return True
    # Dynamic sources (qa_moving / DCASE) carry per-frame trajectories under
    # "frames" and may have null top-level doa.  Treat them as valid when any
    # frame provides azi+ele.
    frames = source_entry.get("frames")
    if isinstance(frames, (list, tuple)) and frames:
        for fr in frames:
            if not isinstance(fr, dict):
                continue
            fdoa = fr.get("doa") or {}
            az = fr.get("azimuth_deg", fdoa.get("azimuth_deg"))
            el = fr.get("elevation_deg", fdoa.get("elevation_deg"))
            if az is not None and el is not None:
                return True
    return False


def _entry_has_valid_geometry(entry: Dict[str, Any]) -> bool:
    if "sources" in entry and isinstance(entry["sources"], list):
        if not entry["sources"]:
            return False
        return all(isinstance(source_entry, dict) and _source_has_valid_doa(source_entry) for source_entry in entry["sources"])

    source_like = {
        "azimuth_deg": entry.get("rir_doa_azimuth_deg"),
        "elevation_deg": entry.get("rir_doa_elevation_deg"),
    }
    return _source_has_valid_doa(source_like)


def _build_source_event_from_top_level_entry(
    entry: Dict[str, Any],
    vocabulary: Dict[str, Any],
) -> SourceEvent:
    source_like = {
        "final_label": entry.get("mono_target_label", entry.get("mono_primary_label")),
        "azimuth_deg": entry.get("rir_doa_azimuth_deg"),
        "elevation_deg": entry.get("rir_doa_elevation_deg"),
        "distance_cm": entry.get("rir_distance_cm"),
        "horizontal_distance_cm": entry.get("rir_horizontal_distance_cm"),
        "listener_position_cm": entry.get("rir_listener_position_cm"),
        "source_position_cm": entry.get("rir_source_position_cm"),
        "start_time_seconds": 0.0,
        "end_time_seconds": entry.get("output_duration_seconds"),
    }
    class_index = _resolve_class_index(source_like, vocabulary)
    class_label = _resolve_class_label(source_like, vocabulary, class_index)
    return SourceEvent(
        class_index=class_index,
        class_label=class_label,
        azimuth_deg=_get_float(source_like, ("azimuth_deg", "azimuth")),
        elevation_deg=_get_float(source_like, ("elevation_deg", "elevation")),
        distance=_resolve_distance_m(source_like),
        distance_valid=_is_distance_valid(source_like),
        start_time_seconds=float(source_like["start_time_seconds"]),
        end_time_seconds=float(source_like["end_time_seconds"]),
    )


def _load_csv_trajectory(
    csv_path: str,
    clip_duration_seconds: float,
    frame_rate: float = 10.0,
) -> Dict[str, Optional[Tensor]]:
    """Parse a 6-column per-frame CSV trajectory file from the unified dataset.

    Expected columns (no header):
        frame_idx, class_id, track_id, azimuth_deg, elevation_deg, distance_cm

    Special values:
        distance_cm == -1  → distance unknown, mark frame invalid
        elevation_deg == inf or -inf  → hemisphere known, exact angle unknown
                                        → set frame_ele_sign_only = True
                                        → clamp stored elevation to ±90 sentinel

    Returns the same dict shape as ``_parse_frame_trajectory``:
        frame_times_s, frame_azi_deg, frame_ele_deg, frame_distance_m,
        frame_distance_valid, frame_ele_sign_only
    All tensors are 1-D float32 / bool.
    """
    _empty: Dict[str, Optional[Tensor]] = {
        "frame_times_s": None,
        "frame_azi_deg": None,
        "frame_ele_deg": None,
        "frame_distance_m": None,
        "frame_distance_valid": None,
        "frame_ele_sign_only": None,
    }
    try:
        times: List[float] = []
        azi: List[float] = []
        ele: List[float] = []
        dist_m: List[float] = []
        dist_valid: List[bool] = []
        ele_sign_only: List[bool] = []

        with open(csv_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) < 6:
                    continue
                try:
                    frame_idx = int(parts[0])
                    az_val = float(parts[3])
                    el_str = parts[4].strip()
                    d_cm_str = parts[5].strip()
                except (ValueError, IndexError):
                    continue

                t = frame_idx / frame_rate
                times.append(t)
                azi.append(az_val)

                # Elevation: handle ±inf (sign-only frames)
                if el_str in ("inf", "+inf", "Inf", "+Inf"):
                    ele.append(90.0)        # sentinel: upper hemisphere
                    ele_sign_only.append(True)
                elif el_str in ("-inf", "-Inf"):
                    ele.append(-90.0)       # sentinel: lower hemisphere
                    ele_sign_only.append(True)
                else:
                    try:
                        el_val = float(el_str)
                    except ValueError:
                        el_val = 0.0
                    # Clamp finite values to valid range
                    el_val = max(-90.0, min(90.0, el_val))
                    ele.append(el_val)
                    ele_sign_only.append(False)

                # Distance: -1 → unknown
                try:
                    d_cm = float(d_cm_str)
                except ValueError:
                    d_cm = -1.0
                if d_cm >= 0:
                    dist_m.append(d_cm / 100.0)
                    dist_valid.append(True)
                else:
                    dist_m.append(0.0)
                    dist_valid.append(False)

        if not times:
            return _empty

        return {
            "frame_times_s": torch.tensor(times, dtype=torch.float32),
            "frame_azi_deg": torch.tensor(azi, dtype=torch.float32),
            "frame_ele_deg": torch.tensor(ele, dtype=torch.float32),
            "frame_distance_m": torch.tensor(dist_m, dtype=torch.float32),
            "frame_distance_valid": torch.tensor(dist_valid, dtype=torch.bool),
            "frame_ele_sign_only": torch.tensor(ele_sign_only, dtype=torch.bool),
        }
    except OSError:
        return _empty


def _parse_frame_trajectory(
    source_entry: Dict[str, Any],
    clip_duration_seconds: float,
) -> Dict[str, Optional[Tensor]]:
    """Extract per-frame DOA trajectory from a manifest source entry.

    Supports three manifest layouts:

    1. New unified dataset (spatial_foa_scene_v1): ``source_trajectory_csv_path``
       points to an external 6-column CSV (frame_idx,class_id,track_id,azi,ele,dist_cm).
    2. qa_moving.jsonl style: ``frames`` is a list of dicts with keys
       ``frame_idx`` and ``doa.azimuth_deg/elevation_deg`` and optional
       ``distance_cm``; clip-level ``num_frames`` / ``frame_rate`` / a top-level
       ``duration_sec`` determine the time axis.
    3. DCASE-style converter output (see tools/dcase_starss_to_jsonl.py):
       ``frames`` rows carry ``time_s`` directly.

    Returns a dict with keys ``frame_times_s``, ``frame_azi_deg``,
    ``frame_ele_deg``, ``frame_distance_m``, ``frame_distance_valid``,
    ``frame_ele_sign_only`` — each either a 1-D Tensor or ``None`` when the
    source has no trajectory.
    Returned distances are in metres; frames with missing distance are
    marked invalid.
    """
    # --- Branch 1: external CSV file (new unified dataset) ---
    csv_path = source_entry.get("source_trajectory_csv_path")
    if csv_path:
        clip_frame_rate = float(source_entry.get("frame_rate", 10.0))
        return _load_csv_trajectory(
            csv_path=str(csv_path),
            clip_duration_seconds=clip_duration_seconds,
            frame_rate=clip_frame_rate,
        )

    # --- Branch 2 & 3: inline frames[] list ---
    frames = source_entry.get("frames")
    if not isinstance(frames, (list, tuple)) or len(frames) == 0:
        return {
            "frame_times_s": None,
            "frame_azi_deg": None,
            "frame_ele_deg": None,
            "frame_distance_m": None,
            "frame_distance_valid": None,
            "frame_ele_sign_only": None,
        }

    # Determine time axis.  Prefer explicit time_s per row, then fall back to
    # frame_idx / frame_rate, finally to linspace over clip_duration.
    n = len(frames)
    explicit_time = all(isinstance(f, dict) and ("time_s" in f or "time" in f) for f in frames)
    frame_rate = source_entry.get("frame_rate")
    # Fall back: parent manifest sometimes sets frame_rate on the clip record,
    # not per source.  Caller can pre-fill source_entry["frame_rate"].
    if frame_rate is not None:
        try:
            frame_rate = float(frame_rate)
        except (TypeError, ValueError):
            frame_rate = None
    if frame_rate is None or frame_rate <= 0:
        frame_rate = None

    times: List[float] = []
    azi: List[float] = []
    ele: List[float] = []
    dist_m: List[float] = []
    dist_valid: List[bool] = []
    ele_sign_only: List[bool] = []

    for i, fr in enumerate(frames):
        if not isinstance(fr, dict):
            continue
        # time axis
        if explicit_time:
            t = float(fr.get("time_s", fr.get("time", i)))
        else:
            fi = fr.get("frame_idx", i)
            if frame_rate is not None:
                t = float(fi) / float(frame_rate)
            elif clip_duration_seconds > 0.0 and n > 1:
                t = float(i) * float(clip_duration_seconds) / float(n - 1)
            else:
                t = float(i)
        times.append(t)

        # DOA can live either nested under "doa": {...} or flat
        doa = fr.get("doa") or {}
        az = fr.get("azimuth_deg", doa.get("azimuth_deg"))
        el = fr.get("elevation_deg", doa.get("elevation_deg"))
        azi.append(float(az) if az is not None else 0.0)
        if el is not None:
            try:
                el_f = float(el)
            except (TypeError, ValueError):
                el_f = 0.0
            # inline frames[] are not expected to carry ±inf in practice,
            # but handle gracefully just in case
            if math.isinf(el_f):
                ele.append(90.0 if el_f > 0 else -90.0)
                ele_sign_only.append(True)
            else:
                ele.append(max(-90.0, min(90.0, el_f)))
                ele_sign_only.append(False)
        else:
            ele.append(0.0)
            ele_sign_only.append(False)

        # distance: accept distance_cm (preferred) or distance_m; -1 → invalid
        d_cm = fr.get("distance_cm")
        d_m = fr.get("distance_m")
        if d_cm is not None and d_cm != -1 and d_cm >= 0:
            dist_m.append(float(d_cm) / 100.0)
            dist_valid.append(True)
        elif d_m is not None and d_m >= 0:
            dist_m.append(float(d_m))
            dist_valid.append(True)
        else:
            dist_m.append(0.0)
            dist_valid.append(False)

    if not times:
        return {
            "frame_times_s": None,
            "frame_azi_deg": None,
            "frame_ele_deg": None,
            "frame_distance_m": None,
            "frame_distance_valid": None,
            "frame_ele_sign_only": None,
        }

    return {
        "frame_times_s": torch.tensor(times, dtype=torch.float32),
        "frame_azi_deg": torch.tensor(azi, dtype=torch.float32),
        "frame_ele_deg": torch.tensor(ele, dtype=torch.float32),
        "frame_distance_m": torch.tensor(dist_m, dtype=torch.float32),
        "frame_distance_valid": torch.tensor(dist_valid, dtype=torch.bool),
        "frame_ele_sign_only": torch.tensor(ele_sign_only, dtype=torch.bool),
    }


def _build_source_event_from_nested_entry(
    source_entry: Dict[str, Any],
    vocabulary: Dict[str, Any],
    clip_duration_seconds: float,
) -> SourceEvent:
    class_index = _resolve_class_index(source_entry, vocabulary)
    class_label = _resolve_class_label(source_entry, vocabulary, class_index)

    # "active_times" is a list-of-intervals in the new unified dataset
    # (e.g. [[1.0, 5.05]]).  Use the first interval as the clip-level window.
    # Legacy fields: "active_time" (singular list [start, end]) or "full_time".
    active_times = source_entry.get("active_times")
    active_time = source_entry.get("active_time")
    full_time = source_entry.get("full_time")
    if active_times is not None and isinstance(active_times, (list, tuple)) and len(active_times) > 0:
        # take the first interval
        first_interval = active_times[0]
        start_time_seconds = float(first_interval[0])
        end_time_seconds = float(first_interval[-1])
    elif active_time is not None:
        start_time_seconds, end_time_seconds = float(active_time[0]), float(active_time[1])
    elif full_time is not None:
        start_time_seconds, end_time_seconds = float(full_time[0]), float(full_time[1])
    else:
        start_time_seconds, end_time_seconds = 0.0, float(clip_duration_seconds)

    doa = source_entry.get("doa") or {}
    # Use _maybe_get_float — dynamic sources may carry doa=None at the top
    # level, with the real DOA living inside frames[].  Missing here is OK;
    # trajectory-based fallback fills the scalar below.
    azimuth_deg = _maybe_get_float(
        {**source_entry, **({"azimuth_deg": doa.get("azimuth_deg")} if isinstance(doa, dict) and "azimuth_deg" in doa else {})},
        ("azimuth_deg", "azimuth"),
    )
    elevation_deg = _maybe_get_float(
        {**source_entry, **({"elevation_deg": doa.get("elevation_deg")} if isinstance(doa, dict) and "elevation_deg" in doa else {})},
        ("elevation_deg", "elevation"),
    )

    # Extract per-frame trajectory for dynamic sources (qa_moving, DCASE, ...).
    traj = _parse_frame_trajectory(source_entry, clip_duration_seconds)

    # For dynamic sources the top-level ``doa`` may be null; fall back to the
    # first trajectory frame so the scalar fields stay usable as a default.
    if azimuth_deg is None and traj["frame_azi_deg"] is not None:
        azimuth_deg = float(traj["frame_azi_deg"][0].item())
    if elevation_deg is None and traj["frame_ele_deg"] is not None:
        elevation_deg = float(traj["frame_ele_deg"][0].item())

    # Distance: prefer explicit source-level distance, else first valid frame.
    distance_m = _resolve_distance_m(source_entry)
    distance_valid = _is_distance_valid(source_entry)
    if not distance_valid and traj["frame_distance_valid"] is not None:
        valid_mask = traj["frame_distance_valid"]
        if bool(valid_mask.any().item()):
            first_valid = int(torch.nonzero(valid_mask, as_tuple=False)[0].item())
            distance_m = float(traj["frame_distance_m"][first_valid].item())
            distance_valid = True

    return SourceEvent(
        class_index=class_index,
        class_label=class_label,
        azimuth_deg=float(azimuth_deg) if azimuth_deg is not None else 0.0,
        elevation_deg=float(elevation_deg) if elevation_deg is not None else 0.0,
        distance=distance_m,
        distance_valid=distance_valid,
        start_time_seconds=start_time_seconds,
        end_time_seconds=end_time_seconds,
        frame_times_s=traj["frame_times_s"],
        frame_azi_deg=traj["frame_azi_deg"],
        frame_ele_deg=traj["frame_ele_deg"],
        frame_distance_m=traj["frame_distance_m"],
        frame_distance_valid=traj["frame_distance_valid"],
        frame_ele_sign_only=traj.get("frame_ele_sign_only"),
    )


def _maybe_crop_sample(
    waveform: Tensor,
    clip_duration_seconds: float,
    sources: List[SourceEvent],
    sample_rate: int,
    max_clip_duration_seconds: Optional[float],
    crop_mode: str,
    min_crop_duration_seconds: Optional[float] = None,
) -> tuple[Tensor, float, List[SourceEvent]]:
    if max_clip_duration_seconds is None:
        return waveform, clip_duration_seconds, sources

    total_num_samples = waveform.size(-1)

    # Random duration crop: sample duration from [min, min(max, actual_length)]
    if min_crop_duration_seconds is not None and crop_mode == "random":
        min_samples = max(int(round(min_crop_duration_seconds * sample_rate)), 1)
        max_samples = min(int(round(max_clip_duration_seconds * sample_rate)), total_num_samples)
        min_samples = min(min_samples, max_samples)
        if min_samples >= total_num_samples:
            # Audio shorter than min_crop_duration — use as-is
            return waveform, clip_duration_seconds, sources
        crop_num_samples = int(torch.randint(min_samples, max_samples + 1, (1,)).item())
    else:
        if clip_duration_seconds <= max_clip_duration_seconds:
            return waveform, clip_duration_seconds, sources
        crop_num_samples = int(round(max_clip_duration_seconds * sample_rate))

    if crop_num_samples >= total_num_samples:
        return waveform, clip_duration_seconds, sources

    if crop_mode == "random":
        # Constrain random start so the crop window covers at least one source.
        # Use the first source as anchor (ov1 = single source per clip).
        lo = 0
        hi = total_num_samples - crop_num_samples
        if sources:
            anchor = sources[0]
            src_start_sample = int(round(anchor.start_time_seconds * sample_rate))
            src_end_sample = int(round(anchor.end_time_seconds * sample_rate))
            # Window must start before src_end and end after src_start:
            #   start_sample + crop_num_samples > src_start_sample
            #   start_sample < src_end_sample
            lo = max(lo, src_start_sample - crop_num_samples + 1)
            hi = min(hi, src_end_sample - 1)
            lo = max(lo, 0)
            hi = min(hi, total_num_samples - crop_num_samples)
            if lo > hi:
                # Source is shorter than one sample inside any window — use full clip
                return waveform, clip_duration_seconds, sources
        start_sample = int(torch.randint(lo, hi + 1, (1,)).item())
    elif crop_mode == "start":
        start_sample = 0
    elif crop_mode == "center":
        start_sample = max((total_num_samples - crop_num_samples) // 2, 0)
    elif crop_mode == "none":
        return waveform, clip_duration_seconds, sources
    else:
        raise ValueError(f"Unsupported crop_mode: {crop_mode}")

    end_sample = start_sample + crop_num_samples
    crop_start_seconds = start_sample / float(sample_rate)
    crop_end_seconds = end_sample / float(sample_rate)
    cropped_waveform = waveform[:, start_sample:end_sample]

    cropped_sources: List[SourceEvent] = []
    for source in sources:
        new_start = max(source.start_time_seconds, crop_start_seconds)
        new_end = min(source.end_time_seconds, crop_end_seconds)
        if new_end <= new_start:
            continue

        # Crop the per-frame trajectory to the new time window and re-base the
        # timestamps to the cropped clip's start.
        frame_times_s = source.frame_times_s
        frame_azi_deg = source.frame_azi_deg
        frame_ele_deg = source.frame_ele_deg
        frame_distance_m = source.frame_distance_m
        frame_distance_valid = source.frame_distance_valid
        frame_ele_sign_only = source.frame_ele_sign_only
        if frame_times_s is not None and frame_times_s.numel() > 0:
            # Keep frames whose timestamp falls inside [crop_start, crop_end].
            mask = (frame_times_s >= crop_start_seconds) & (frame_times_s <= crop_end_seconds)
            if bool(mask.any().item()):
                frame_times_s = frame_times_s[mask] - crop_start_seconds
                frame_azi_deg = frame_azi_deg[mask] if frame_azi_deg is not None else None
                frame_ele_deg = frame_ele_deg[mask] if frame_ele_deg is not None else None
                frame_distance_m = frame_distance_m[mask] if frame_distance_m is not None else None
                frame_distance_valid = (
                    frame_distance_valid[mask] if frame_distance_valid is not None else None
                )
                frame_ele_sign_only = (
                    frame_ele_sign_only[mask] if frame_ele_sign_only is not None else None
                )
            else:
                # No frames survive the crop — drop trajectory and keep scalar.
                frame_times_s = None
                frame_azi_deg = None
                frame_ele_deg = None
                frame_distance_m = None
                frame_distance_valid = None
                frame_ele_sign_only = None

        cropped_sources.append(
            SourceEvent(
                class_index=source.class_index,
                class_label=source.class_label,
                azimuth_deg=source.azimuth_deg,
                elevation_deg=source.elevation_deg,
                distance=source.distance,
                distance_valid=source.distance_valid,
                start_time_seconds=new_start - crop_start_seconds,
                end_time_seconds=new_end - crop_start_seconds,
                frame_times_s=frame_times_s,
                frame_azi_deg=frame_azi_deg,
                frame_ele_deg=frame_ele_deg,
                frame_distance_m=frame_distance_m,
                frame_distance_valid=frame_distance_valid,
                frame_ele_sign_only=frame_ele_sign_only,
            )
        )

    return cropped_waveform, crop_num_samples / float(sample_rate), cropped_sources


# =============================================================================
# v13_B [B-5] Waveform-level augmentation
# =============================================================================


def _apply_waveform_augment(
    waveform: Tensor,
    config: "SpatialDatasetConfig",
) -> Tensor:
    """Apply waveform-level augments in-place-free style.

    Augments (each independently sampled, all training-only):
      - random gain:     multiply by 10^(g/20), g ~ U[-x, +x] dB
      - channel dropout: zero one FOA channel with prob p
      - time mask:       zero a contiguous waveform chunk (SpecAugment-like)
      - lowpass:         first-order IIR lowpass with random cutoff

    Args:
        waveform: [C, N] tensor (C=4 for FOA, N=time samples)
        config:   SpatialDatasetConfig with augment flags

    Returns:
        Augmented waveform, same shape/dtype/device as input.
    """
    if waveform.ndim != 2:
        return waveform
    import random

    # --- random gain ---
    if config.random_gain_db > 0.0:
        g_db = (random.random() * 2.0 - 1.0) * float(config.random_gain_db)
        gain = 10.0 ** (g_db / 20.0)
        waveform = waveform * gain

    # --- channel dropout ---
    if config.channel_dropout_prob > 0.0 and random.random() < config.channel_dropout_prob:
        C = waveform.size(0)
        if C > 1:
            idx = random.randint(0, C - 1)
            mask = torch.ones(C, 1, dtype=waveform.dtype, device=waveform.device)
            mask[idx] = 0.0
            waveform = waveform * mask

    # --- time mask (SpecAugment-equivalent on waveform) ---
    if config.use_spec_augment and config.spec_augment_time_mask_ratio > 0.0:
        N = waveform.size(-1)
        for _ in range(max(1, int(config.spec_augment_num_time_stripes))):
            max_len = int(float(config.spec_augment_time_mask_ratio) * N)
            if max_len < 1:
                break
            mask_len = random.randint(1, max_len)
            if mask_len >= N:
                continue
            start = random.randint(0, N - mask_len)
            waveform = waveform.clone()
            waveform[:, start : start + mask_len] = 0.0

    # --- lowpass (first-order IIR) ---
    if config.lowpass_sim_real_prob > 0.0 and random.random() < config.lowpass_sim_real_prob:
        cutoff = random.uniform(
            float(config.lowpass_cutoff_min_hz),
            float(config.lowpass_cutoff_max_hz),
        )
        # First-order IIR: y[n] = a * y[n-1] + (1-a) * x[n]
        # a = exp(-2*pi*cutoff/sr). Assume sr=16000.
        import math
        sr = 16000.0
        a = math.exp(-2.0 * math.pi * cutoff / sr)
        # Apply per-channel via torch.cumulative IIR using torch.lfilter-like.
        # Fallback: simple loop is OK for short clips (<=10s, 160k samples).
        # For speed we use torchaudio.functional.lfilter when available.
        try:
            import torchaudio.functional as TAF
            b_coeffs = torch.tensor([1.0 - a, 0.0], dtype=waveform.dtype, device=waveform.device)
            a_coeffs = torch.tensor([1.0, -a], dtype=waveform.dtype, device=waveform.device)
            waveform = TAF.lfilter(waveform, a_coeffs, b_coeffs, clamp=False)
        except Exception:
            # Skip silently if torchaudio not available / lfilter fails
            pass

    return waveform


class SpatialDataset(Dataset):
    """Base dataset interface for Spatial-BEATs.

    A concrete implementation is expected to read manifest-style metadata and
    return SpatialSample objects with FOA waveform plus source-level labels.
    """

    def __init__(
        self,
        manifest_path: str,
        config: SpatialDatasetConfig,
    ) -> None:
        super().__init__()
        self.manifest_path = Path(manifest_path)
        self.config = config
        if self.config.show_progress:
            tqdm.write(f"[SpatialDataset] Initialize from {self.manifest_path}")
        self.vocabulary = load_source_vocabulary(
            config.source_vocab,
            show_progress=self.config.show_progress,
        )
        entries = _load_manifest_entries(
            self.manifest_path,
            show_progress=self.config.show_progress,
        )
        if config.allowed_splits is not None:
            allowed = set(config.allowed_splits)
            entries = [
                entry
                for entry in tqdm(
                    entries,
                    total=len(entries),
                    desc=f"Filter splits {self.manifest_path.name}",
                    leave=False,
                    disable=not self.config.show_progress,
                )
                if entry.get("split") in allowed
            ]
        valid_entries = []
        dropped_invalid_geometry = 0
        for entry in tqdm(
            entries,
            total=len(entries),
            desc=f"Validate geometry {self.manifest_path.name}",
            leave=False,
            disable=not self.config.show_progress,
        ):
            if _entry_has_valid_geometry(entry):
                valid_entries.append(entry)
            else:
                dropped_invalid_geometry += 1
        entries = valid_entries
        self.entries = entries
        if self.config.show_progress:
            tqdm.write(
                f"[SpatialDataset] {self.manifest_path.name}: "
                f"{len(self.entries)} entries after split/geometry filtering"
            )
            if dropped_invalid_geometry:
                tqdm.write(
                    f"[SpatialDataset] {self.manifest_path.name}: "
                    f"dropped {dropped_invalid_geometry} entries with missing DOA geometry"
                )

        # v13_B [B-5]: enable augment only on training splits.
        _splits = config.allowed_splits or ()
        self._is_train_split = ("train" in set(_splits))
        self._augment_enabled = bool(
            self._is_train_split
            and (
                config.use_spec_augment
                or config.random_gain_db > 0.0
                or config.channel_dropout_prob > 0.0
                or config.lowpass_sim_real_prob > 0.0
            )
        )

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.entries)

    def __getitem__(self, index: int) -> SpatialSample:
        """Load one FOA clip and its source-level annotations.

        Returns:
            SpatialSample:
                The uncollated sample object described above.
        """
        entry = self.entries[index]
        # Resolve waveform path: support top-level fields and the new
        # spatial_foa_scene_v1 layout where it lives under entry["audio"]["foa_path"].
        waveform_path = (
            entry.get("output_foa_path")
            or entry.get("waveform_path")
            or entry.get("audio_path")
            or entry.get("foa_path")
        )
        if waveform_path is None:
            audio_meta = entry.get("audio")
            if isinstance(audio_meta, dict):
                waveform_path = audio_meta.get("foa_path") or audio_meta.get("path")
        if waveform_path is None:
            raise KeyError(
                "Manifest entry must contain output_foa_path/waveform_path/audio_path/"
                "foa_path or audio.foa_path."
            )
        waveform = _load_audio_file(str(waveform_path), self.config.mel_config.sample_rate)

        clip_duration_seconds = entry.get("clip_duration_seconds")
        if clip_duration_seconds is None:
            clip_duration_seconds = entry.get("output_duration_seconds")
        if clip_duration_seconds is None:
            clip_duration_seconds = entry.get("duration")
        # New unified dataset: duration under audio.duration_seconds
        if clip_duration_seconds is None:
            audio_meta = entry.get("audio")
            if isinstance(audio_meta, dict):
                clip_duration_seconds = audio_meta.get("duration_seconds")
        if clip_duration_seconds is None:
            clip_duration_seconds = float(waveform.size(-1)) / float(self.config.mel_config.sample_rate)

        sources: List[SourceEvent] = []
        if "sources" in entry and isinstance(entry["sources"], list):
            # Clip-level frame_rate (e.g. qa_moving.jsonl has 25) is carried on
            # the top-level record, not per source.  Inject it into the source
            # entry so ``_parse_frame_trajectory`` can recover the time axis.
            clip_frame_rate = entry.get("frame_rate")
            for source_entry in entry["sources"]:
                source_entry_with_rate = source_entry
                if clip_frame_rate is not None and isinstance(source_entry, dict):
                    source_entry_with_rate = {**source_entry, "frame_rate": source_entry.get("frame_rate", clip_frame_rate)}
                sources.append(
                    _build_source_event_from_nested_entry(
                        source_entry=source_entry_with_rate,
                        vocabulary=self.vocabulary,
                        clip_duration_seconds=float(clip_duration_seconds),
                    )
                )
        else:
            sources.append(
                _build_source_event_from_top_level_entry(
                    entry=entry,
                    vocabulary=self.vocabulary,
                )
            )

        waveform, clip_duration_seconds, sources = _maybe_crop_sample(
            waveform=waveform,
            clip_duration_seconds=float(clip_duration_seconds),
            sources=sources,
            sample_rate=self.config.mel_config.sample_rate,
            max_clip_duration_seconds=self.config.max_clip_duration_seconds,
            crop_mode=self.config.crop_mode,
            min_crop_duration_seconds=self.config.min_crop_duration_seconds,
        )

        # v13_B [B-5]: waveform-level augment (training only)
        if self._augment_enabled:
            waveform = _apply_waveform_augment(waveform, self.config)

        sample_id = str(entry.get("scene_id", entry.get("pair_id", entry.get("sample_id", entry.get("id", index)))))
        return SpatialSample(
            sample_id=sample_id,
            waveform=waveform,
            clip_duration_seconds=float(clip_duration_seconds),
            sources=sources,
        )


def _linear_interp_1d(
    query_t: Tensor,
    keys_t: Tensor,
    values: Tensor,
) -> Tensor:
    """Linear interpolation of ``values`` sampled at ``keys_t`` onto ``query_t``.

    Edge handling: queries before/after the keyframe range clamp to the nearest
    endpoint (common for dynamic sources whose active_time is a superset of
    their trajectory support — e.g. qa_moving records cover the full 4 s even
    when the wav is longer after padding).

    Args:
        query_t: [Q] query timestamps in seconds, expected sorted ascending.
        keys_t:  [K] key timestamps in seconds, sorted ascending (K >= 1).
        values:  [K] values aligned with ``keys_t``.

    Returns:
        Tensor of shape [Q] containing interpolated values.
    """
    if keys_t.numel() == 1:
        return values[0].expand_as(query_t).clone()
    # torch.searchsorted returns indices such that keys_t[idx-1] <= q < keys_t[idx].
    idx_right = torch.searchsorted(keys_t, query_t, right=True).clamp(1, keys_t.numel() - 1)
    idx_left = idx_right - 1
    kl = keys_t[idx_left]
    kr = keys_t[idx_right]
    # Avoid divide-by-zero for repeated keys (degenerate).
    span = (kr - kl).clamp_min(1e-9)
    w = ((query_t - kl) / span).clamp(0.0, 1.0)
    vl = values[idx_left]
    vr = values[idx_right]
    return vl + (vr - vl) * w


def _linear_interp_valid_mask(
    query_t: Tensor,
    keys_t: Tensor,
    valid_mask: Tensor,
) -> Tensor:
    """Piecewise-constant resampling of a boolean validity mask.

    For interpolation points, a query is valid only if both of its surrounding
    keyframes are valid — this prevents inferring a "valid" distance in a
    segment where at least one endpoint was unknown.
    """
    if keys_t.numel() == 1:
        return valid_mask[0].expand_as(query_t).clone()
    idx_right = torch.searchsorted(keys_t, query_t, right=True).clamp(1, keys_t.numel() - 1)
    idx_left = idx_right - 1
    return valid_mask[idx_left] & valid_mask[idx_right]


def _build_per_frame_targets(
    source: "SourceEvent",
    t_axis: Tensor,
    t_s_max: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Build [T_s_max] per-frame (azi, ele, dist, dist_valid, ele_sign_only) rows for one source.

    - Indices [0, T_s_i) hold the real per-step targets:
        * Static source (no ``frame_times_s``): scalar broadcast.
        * Dynamic source: linear interpolation of the trajectory onto ``t_axis``.
    - Indices [T_s_i, T_s_max) hold zeros (padding beyond the sample's valid
      time range; loss masks out these steps via ``source_valid_mask`` combined
      with ``window_mask``).

    The source's ``active_time`` window is **not** applied here — the window
    mask is applied separately by the loss layer.  That keeps this function
    purely a per-frame target builder.

    Returns (azi_row, ele_row, dist_row, dist_valid_row, ele_sign_only_row).
    """
    t_s_i = t_axis.numel()
    azi_row = torch.zeros(t_s_max, dtype=torch.float32)
    ele_row = torch.zeros(t_s_max, dtype=torch.float32)
    dist_row = torch.zeros(t_s_max, dtype=torch.float32)
    dist_valid_row = torch.zeros(t_s_max, dtype=torch.bool)
    ele_sign_only_row = torch.zeros(t_s_max, dtype=torch.bool)

    if source.frame_times_s is None or source.frame_times_s.numel() == 0:
        # Static fallback: broadcast the scalar over the sample's valid axis.
        azi_row[:t_s_i] = float(source.azimuth_deg)
        ele_row[:t_s_i] = float(source.elevation_deg)
        dist_row[:t_s_i] = float(source.distance)
        dist_valid_row[:t_s_i] = bool(source.distance_valid)
        # Static sources never carry sign-only elevation
        # ele_sign_only_row stays False
        return azi_row, ele_row, dist_row, dist_valid_row, ele_sign_only_row

    # Dynamic: interpolate frames to the model time axis.
    keys_t = source.frame_times_s
    azi_vals = source.frame_azi_deg
    ele_vals = source.frame_ele_deg
    # Azimuth lives on a circle; linear interp can wrap badly across the
    # ±180° boundary.  Unwrap keyframes first so we interpolate on the
    # continuous axis, then wrap the output back to [-180, 180].
    azi_unwrapped = _unwrap_deg(azi_vals)
    azi_interp = _linear_interp_1d(t_axis, keys_t, azi_unwrapped)
    # Wrap to [-180, 180] to match GT convention everywhere else.
    azi_interp = ((azi_interp + 180.0) % 360.0) - 180.0
    ele_interp = _linear_interp_1d(t_axis, keys_t, ele_vals)
    azi_row[:t_s_i] = azi_interp
    ele_row[:t_s_i] = ele_interp

    if source.frame_distance_m is not None and source.frame_distance_valid is not None:
        dist_interp = _linear_interp_1d(t_axis, keys_t, source.frame_distance_m)
        dist_valid_interp = _linear_interp_valid_mask(t_axis, keys_t, source.frame_distance_valid)
        dist_row[:t_s_i] = dist_interp
        dist_valid_row[:t_s_i] = dist_valid_interp
    else:
        dist_row[:t_s_i] = float(source.distance)
        dist_valid_row[:t_s_i] = bool(source.distance_valid)

    # ele_sign_only: nearest-neighbour resample of per-frame boolean mask.
    # A query inherits the sign-only flag of the nearest keyframe.
    if source.frame_ele_sign_only is not None:
        sign_only_interp = _linear_interp_nearest_bool(t_axis, keys_t, source.frame_ele_sign_only)
        ele_sign_only_row[:t_s_i] = sign_only_interp

    return azi_row, ele_row, dist_row, dist_valid_row, ele_sign_only_row


def _linear_interp_nearest_bool(
    query_t: Tensor,
    keys_t: Tensor,
    bool_values: Tensor,
) -> Tensor:
    """Nearest-neighbour resampling of a boolean 1-D mask.

    Each query timestamp inherits the value of the nearest keyframe.
    """
    if keys_t.numel() == 1:
        return bool_values[0].expand_as(query_t).clone()
    # Use searchsorted to find bracket, then pick the closer neighbour.
    idx_right = torch.searchsorted(keys_t, query_t, right=True).clamp(0, keys_t.numel() - 1)
    idx_left = (idx_right - 1).clamp(0)
    # For each query pick the nearer key.
    kl = keys_t[idx_left]
    kr = keys_t[idx_right]
    # If left is closer (or equal), use left; otherwise right.
    use_left = (query_t - kl) <= (kr - query_t)
    idx = torch.where(use_left, idx_left, idx_right)
    return bool_values[idx]


def _unwrap_deg(azi_deg: Tensor) -> Tensor:
    """Unwrap a [N] azimuth trajectory so linear interp doesn't jump ±180°.

    Trajectories in qa_moving and DCASE are continuous but labels are wrapped
    to [-180, 180]; a source moving from 170° to -170° actually swept 20°
    across the back, not -340°.  Detect and remove the 360° discontinuities.
    """
    if azi_deg.numel() <= 1:
        return azi_deg
    d = torch.diff(azi_deg)
    # Jumps > 180° are unwrap artifacts; add multiples of 360° to cancel them.
    jumps = torch.zeros_like(d)
    jumps[d > 180.0] = -360.0
    jumps[d < -180.0] = 360.0
    offsets = torch.cat([torch.zeros(1), torch.cumsum(jumps, dim=0)])
    return azi_deg + offsets


def collate_spatial_batch(
    samples: Sequence[SpatialSample],
    config: SpatialDatasetConfig,
) -> SpatialBatch:
    """Collate variable-length SpatialSample objects into a padded batch.

    Responsibilities:
        - Pad raw FOA waveforms to T_max_wave
        - Build waveform_padding_mask
        - Pad source annotations to N_gt_max
        - Compute target_num_steps = round(duration_i * target_token_rate)
        - Keep sample_ids and optional class label strings for debugging

    Args:
        samples:
            Sequence of SpatialSample objects.
        config:
            Dataset configuration with vocabulary and token-rate settings.

    Returns:
        SpatialBatch:
            Batch object consumed by the model, dataset utilities, and loss code.
    """
    if len(samples) == 0:
        raise ValueError("collate_spatial_batch received an empty sample list.")

    batch_size = len(samples)
    max_wave_len = max(sample.waveform.size(-1) for sample in samples)
    waveform = torch.full(
        (batch_size, 4, max_wave_len),
        fill_value=float(config.padding_value),
        dtype=torch.float32,
    )
    waveform_padding_mask = torch.ones(batch_size, max_wave_len, dtype=torch.bool)

    clip_duration_seconds = torch.tensor(
        [sample.clip_duration_seconds for sample in samples],
        dtype=torch.float32,
    )
    target_num_steps = compute_target_num_steps(
        clip_duration_seconds=clip_duration_seconds,
        target_token_rate=config.target_token_rate,
    )

    max_num_sources = max(max(len(sample.sources), 1) for sample in samples)
    # Per-frame target tensors: shape [B, N_gt_max, T_s_max].  For static
    # sources, the scalar is broadcast along the T_s axis (identical behaviour
    # to the legacy [B, N_gt_max] tensors).  For dynamic sources (frames[] in
    # the manifest), values are resampled onto the model's token-rate grid by
    # linear interpolation.
    t_s_max = int(target_num_steps.max().item()) if batch_size > 0 else 1
    t_s_max = max(t_s_max, 1)
    target_token_rate = float(config.target_token_rate)

    source_class_indices = torch.zeros(batch_size, max_num_sources, dtype=torch.long)
    source_azimuth_deg = torch.zeros(batch_size, max_num_sources, t_s_max, dtype=torch.float32)
    source_elevation_deg = torch.zeros(batch_size, max_num_sources, t_s_max, dtype=torch.float32)
    source_distance = torch.zeros(batch_size, max_num_sources, t_s_max, dtype=torch.float32)
    # Default True; flipped to False for null-distance sources or frames.
    source_distance_valid = torch.ones(batch_size, max_num_sources, t_s_max, dtype=torch.bool)
    # Default False; True for frames where only the hemisphere (sign) is known.
    source_ele_sign_only = torch.zeros(batch_size, max_num_sources, t_s_max, dtype=torch.bool)
    source_start_time_seconds = torch.zeros(batch_size, max_num_sources, dtype=torch.float32)
    source_end_time_seconds = torch.zeros(batch_size, max_num_sources, dtype=torch.float32)
    source_valid_mask = torch.zeros(batch_size, max_num_sources, dtype=torch.bool)

    sample_ids: List[str] = []
    source_class_labels: List[List[str]] = []

    for batch_index, sample in enumerate(samples):
        length = sample.waveform.size(-1)
        waveform[batch_index, :, :length] = sample.waveform
        waveform_padding_mask[batch_index, :length] = False
        sample_ids.append(sample.sample_id)

        # Per-sample model time axis: first T_s_i time steps at target_token_rate.
        t_s_i = int(target_num_steps[batch_index].item())
        t_s_i = max(t_s_i, 1)
        if target_token_rate > 0:
            t_axis = torch.arange(t_s_i, dtype=torch.float32) / float(target_token_rate)
        else:
            t_axis = torch.zeros(t_s_i, dtype=torch.float32)

        label_names: List[str] = []
        for source_index, source in enumerate(sample.sources):
            source_class_indices[batch_index, source_index] = int(source.class_index)
            source_start_time_seconds[batch_index, source_index] = float(source.start_time_seconds)
            source_end_time_seconds[batch_index, source_index] = float(source.end_time_seconds)
            source_valid_mask[batch_index, source_index] = True
            label_names.append(source.class_label)

            # Fill per-frame DOA/distance targets.
            azi_row, ele_row, dist_row, dist_valid_row, ele_sign_only_row = _build_per_frame_targets(
                source=source,
                t_axis=t_axis,
                t_s_max=t_s_max,
            )
            source_azimuth_deg[batch_index, source_index] = azi_row
            source_elevation_deg[batch_index, source_index] = ele_row
            source_distance[batch_index, source_index] = dist_row
            source_distance_valid[batch_index, source_index] = dist_valid_row
            source_ele_sign_only[batch_index, source_index] = ele_sign_only_row
        source_class_labels.append(label_names)

    return SpatialBatch(
        waveform=waveform,
        waveform_padding_mask=waveform_padding_mask,
        clip_duration_seconds=clip_duration_seconds,
        target_num_steps=target_num_steps,
        source_class_indices=source_class_indices,
        source_azimuth_deg=source_azimuth_deg,
        source_elevation_deg=source_elevation_deg,
        source_distance=source_distance,
        source_distance_valid=source_distance_valid,
        source_ele_sign_only=source_ele_sign_only,
        source_start_time_seconds=source_start_time_seconds,
        source_end_time_seconds=source_end_time_seconds,
        source_valid_mask=source_valid_mask,
        sample_ids=sample_ids,
        source_class_labels=source_class_labels,
    )
