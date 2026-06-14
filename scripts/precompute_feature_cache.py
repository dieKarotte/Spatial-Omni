import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


DEFAULT_LEGACY_REPO = ""
DEFAULT_QA_ROOT = (
    "${DCASE_BASELINE_REPO}/"
    "prepared_datasets/starss23_foa_plus_29cls_20s/qa_pairs_supported"
)
DEFAULT_SELD233_CKPT = (
    "${DCASE_BASELINE_REPO}/"
    "3_1_dev_split0_multiaccdoa_foa_model.h5"
)
DEFAULT_SELD233_STATS_DIR = (
    "${SELD_FEATURE_STATS_DIR}"
)
DEFAULT_CACHE_DIR = (
    "${DCASE_BASELINE_REPO}/"
    "prepared_datasets/starss23_foa_plus_29cls_20s/seld_feature_cache"
)
SAMPLE_RATE = 16000
MAX_AUDIO_SECONDS = 20
MAX_AUDIO_SAMPLES = SAMPLE_RATE * MAX_AUDIO_SECONDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute SELD233 feature caches from QA audio files."
    )
    parser.add_argument("--legacy-repo-path", type=str, default=DEFAULT_LEGACY_REPO)
    parser.add_argument("--qa-root", type=str, default=DEFAULT_QA_ROOT)
    parser.add_argument("--splits", nargs="+", default=["train", "valid"])
    parser.add_argument("--max-samples-per-split", type=int, default=None)
    parser.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--cache-dtype",
        type=str,
        default="float16",
        choices=("float16", "float32", "bfloat16"),
    )
    parser.add_argument(
        "--baseline-repo-path",
        type=str,
        default="${DCASE_BASELINE_REPO}",
    )
    parser.add_argument("--seld-task-id", type=str, default="235")
    parser.add_argument("--seld-checkpoint-path", type=str, default=DEFAULT_SELD233_CKPT)
    parser.add_argument("--seld-feature-stats-dir", type=str, default=DEFAULT_SELD233_STATS_DIR)
    return parser.parse_args()


def add_legacy_repo_to_path(legacy_repo_path: str) -> None:
    if legacy_repo_path not in sys.path:
        sys.path.insert(0, legacy_repo_path)


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[name]


def collect_unique_audio_paths(
    qa_root: str,
    splits: List[str],
    max_samples_per_split: Optional[int],
) -> List[str]:
    unique_paths: Dict[str, None] = {}
    for split in splits:
        split_path = os.path.join(qa_root, f"{split}.jsonl")
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"Missing split file: {split_path}")
        with open(split_path, "r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if max_samples_per_split is not None and index >= max_samples_per_split:
                    break
                record = json.loads(line)
                audio_path = os.path.abspath(record["audio_path"])
                if not os.path.exists(audio_path):
                    raise FileNotFoundError(f"Missing audio referenced by {split_path}: {audio_path}")
                unique_paths[audio_path] = None
    return sorted(unique_paths.keys())


class UniqueAudioDataset(Dataset):
    def __init__(self, audio_paths: List[str]) -> None:
        self.audio_paths = audio_paths

    def __len__(self) -> int:
        return len(self.audio_paths)

    def __getitem__(self, index: int) -> str:
        return self.audio_paths[index]


@dataclass
class FoaBatchCollator:
    sample_rate: int = SAMPLE_RATE
    max_audio_samples: int = MAX_AUDIO_SAMPLES

    def __call__(self, audio_paths: List[str]) -> Dict[str, torch.Tensor]:
        batch_size = len(audio_paths)
        spatial_audio = np.zeros((batch_size, self.max_audio_samples, 4), dtype=np.float32)
        spatial_lengths = np.zeros((batch_size,), dtype=np.int64)

        for index, audio_path in enumerate(audio_paths):
            waveform, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
            if sample_rate != self.sample_rate:
                raise ValueError(
                    f"Expected {self.sample_rate} Hz audio, got {sample_rate} for {audio_path}"
                )
            if waveform.shape[1] != 4:
                raise ValueError(
                    f"Expected 4-channel FOA audio, got shape {tuple(waveform.shape)} for {audio_path}"
                )
            waveform = waveform[: self.max_audio_samples, :]
            valid_samples = waveform.shape[0]
            spatial_audio[index, :valid_samples, :] = waveform
            spatial_lengths[index] = valid_samples

        return {
            "audio_paths": audio_paths,
            "spatial_audio": torch.from_numpy(spatial_audio),
            "spatial_audio_lengths": torch.from_numpy(spatial_lengths),
        }


def build_cache_relpath(audio_path: str) -> str:
    audio_path = os.path.abspath(audio_path)
    stem = os.path.splitext(os.path.basename(audio_path))[0]
    digest = hashlib.sha1(audio_path.encode("utf-8")).hexdigest()[:16]
    return f"{stem}-{digest}.pt"


def main() -> None:
    args = parse_args()
    add_legacy_repo_to_path(args.legacy_repo_path)

    from spatial_omni.modules.seld_feature_bridge import SeldFeatureBridge

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device {args.device}, but CUDA is not available.")

    os.makedirs(args.cache_dir, exist_ok=True)
    manifest_path = os.path.join(args.cache_dir, "manifest.json")
    cache_dtype = dtype_from_name(args.cache_dtype)

    audio_paths = collect_unique_audio_paths(
        qa_root=args.qa_root,
        splits=list(args.splits),
        max_samples_per_split=args.max_samples_per_split,
    )
    print(
        f"Collected {len(audio_paths)} unique audio files from splits={args.splits} "
        f"under {args.qa_root}"
    )

    dataset = UniqueAudioDataset(audio_paths)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=FoaBatchCollator(),
    )

    device = torch.device(args.device)
    feature_bridge = SeldFeatureBridge(
        sample_rate=SAMPLE_RATE,
        max_audio_seconds=MAX_AUDIO_SECONDS,
        num_feature_channels=7,
        num_mel_bins=128,
        hop_length=160,
        baseline_repo_path=args.baseline_repo_path,
        task_id=str(args.seld_task_id),
        feature_stats_dir=args.seld_feature_stats_dir,
    ).to(device)
    feature_bridge.eval()

    entries: Dict[str, str] = {}
    with torch.no_grad():
        for batch in tqdm(loader, desc="precompute_seld_feature_cache"):
            spatial_audio = batch["spatial_audio"].to(device=device)
            spatial_audio_lengths = batch["spatial_audio_lengths"].to(device=device)

            feature_output = feature_bridge(
                spatial_audio=spatial_audio,
                spatial_audio_lengths=spatial_audio_lengths,
            )

            for index, audio_path in enumerate(batch["audio_paths"]):
                feature_length = int(feature_output.feature_lengths[index].item())
                features = feature_output.features[index, :, :feature_length].detach().cpu()
                relpath = build_cache_relpath(audio_path)
                out_path = os.path.join(args.cache_dir, relpath)
                torch.save(
                    {
                        "audio_path": os.path.abspath(audio_path),
                        "seld_features": features.to(dtype=cache_dtype).contiguous(),
                        "seld_feature_lengths": torch.tensor(feature_length, dtype=torch.long),
                    },
                    out_path,
                )
                entries[os.path.abspath(audio_path)] = relpath

    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "cache_dir": os.path.abspath(args.cache_dir),
                "entries": entries,
                "metadata": {
                    "qa_root": os.path.abspath(args.qa_root),
                    "splits": list(args.splits),
                    "num_unique_audio": len(entries),
                    "sample_rate": SAMPLE_RATE,
                    "max_audio_seconds": MAX_AUDIO_SECONDS,
                    "cache_dtype": args.cache_dtype,
                    "task_id": str(args.seld_task_id),
                },
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
