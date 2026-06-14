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


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-Omni-7B"
DEFAULT_SO_REPO = os.environ.get("SO_REPO", "")
DEFAULT_QA_ROOT = (
    "${DCASE_BASELINE_REPO}/"
    "prepared_datasets/qa"
)
DEFAULT_CACHE_DIR = (
    "${DCASE_BASELINE_REPO}/"
    "prepared_datasets/qa/qwen_audio_cache"
)
SAMPLE_RATE = 16000
MAX_AUDIO_SECONDS = 20
MAX_AUDIO_SAMPLES = SAMPLE_RATE * MAX_AUDIO_SECONDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute Qwen audio feature cache for QA training.")
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--so-repo", type=str, default=DEFAULT_SO_REPO)
    parser.add_argument("--qa-root", type=str, default=DEFAULT_QA_ROOT)
    parser.add_argument("--splits", nargs="+", default=["train", "valid"])
    parser.add_argument("--max-samples-per-split", type=int, default=None)
    parser.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--cache-dtype",
        type=str,
        default="float16",
        choices=("float16", "float32", "bfloat16"),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="跳过磁盘上已存在的 .pt 文件（支持中断续传，不会重做）。",
    )
    parser.add_argument(
        "--no-spatial-cache",
        action="store_true",
        help="不缓存 spatial_audio 原始波形。强烈推荐开启！"
             "原始 20s 4ch FOA 单条 2.5MB × 398K 条 ≈ 1TB，"
             "而训练时直接 sf.read 只需 5ms（与 Qwen mel 的 400ms 相比可忽略）。"
             "开启后 cache 仅含 input_features，398K 条仅 ~100GB。",
    )
    # 以下两个用于分片并行；手动指定优先级最高，
    # 其次自动从 env RANK/WORLD_SIZE（torchrun 场景）读取。
    parser.add_argument("--shard-index", type=int, default=None,
                        help="当前 rank，从 0 开始；省略则从 env RANK/LOCAL_RANK 读取。")
    parser.add_argument("--num-shards", type=int, default=None,
                        help="总 rank 数；省略则从 env WORLD_SIZE 读取。")
    return parser.parse_args()


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
        split_path = None
        for ext in (".jsonl", ".json"):
            candidate = os.path.join(qa_root, f"{split}{ext}")
            if os.path.exists(candidate):
                split_path = candidate
                break
        if split_path is None:
            raise FileNotFoundError(f"Missing split file for {split} under {qa_root}")
        with open(split_path, "r", encoding="utf-8") as handle:
            if split_path.endswith(".jsonl"):
                iterator = handle
            else:
                payload = json.load(handle)
                iterator = payload if isinstance(payload, list) else payload.get("records", payload.get("data", []))
            for index, item in enumerate(iterator):
                if max_samples_per_split is not None and index >= max_samples_per_split:
                    break
                record = json.loads(item) if isinstance(item, str) else item
                audio_path = os.path.abspath(record["audio_path"])
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
class AudioBatchCollator:
    sample_rate: int = SAMPLE_RATE
    max_audio_samples: int = MAX_AUDIO_SAMPLES

    def __call__(self, audio_paths: List[str]) -> Dict[str, List[np.ndarray]]:
        mono_audio: List[np.ndarray] = []
        foa_audio: List[np.ndarray] = []
        foa_lengths: List[int] = []
        normalized_paths: List[str] = []
        for audio_path in audio_paths:
            waveform, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
            if sample_rate != self.sample_rate:
                raise ValueError(
                    f"Expected {self.sample_rate} Hz audio, got {sample_rate} for {audio_path}"
                )
            waveform = waveform[: self.max_audio_samples]
            if waveform.shape[1] == 4:
                foa = waveform.T
                mono = waveform[:, 0]
            elif waveform.shape[1] == 1:
                raise ValueError(f"Expected FOA audio with 4 channels, got shape {tuple(waveform.shape)} for {audio_path}")
            else:
                raise ValueError(f"Expected FOA audio with 4 channels, got shape {tuple(waveform.shape)} for {audio_path}")
            if foa.shape[0] != 4:
                raise ValueError(f"Expected cached spatial audio shape [4, T], got {tuple(foa.shape)} for {audio_path}")
            foa_audio.append(np.asarray(foa, dtype=np.float32))
            foa_lengths.append(int(foa.shape[1]))
            mono_audio.append(np.asarray(mono, dtype=np.float32))
            normalized_paths.append(os.path.abspath(audio_path))
        return {
            "audio_paths": normalized_paths,
            "mono_audio": mono_audio,
            "foa_audio": foa_audio,
            "foa_lengths": foa_lengths,
        }


def build_cache_relpath(audio_path: str) -> str:
    audio_path = os.path.abspath(audio_path)
    stem = os.path.splitext(os.path.basename(audio_path))[0]
    digest = hashlib.sha1(audio_path.encode("utf-8")).hexdigest()[:16]
    return f"{stem}-{digest}.pt"


def main() -> None:
    args = parse_args()
    if args.so_repo not in sys.path:
        sys.path.insert(0, args.so_repo)

    from spatial_omni.model.processing_qwen2_5_omni import Qwen2_5OmniProcessor

    # ---------- Sharding for parallel execution ----------
    # 手动指定 > env var（torchrun 场景）> 单进程 fallback
    shard_index = args.shard_index
    num_shards = args.num_shards
    if shard_index is None:
        env_rank = os.environ.get("RANK") or os.environ.get("LOCAL_RANK")
        shard_index = int(env_rank) if env_rank is not None else 0
    if num_shards is None:
        env_ws = os.environ.get("WORLD_SIZE")
        num_shards = int(env_ws) if env_ws is not None else 1
    if num_shards <= 0:
        num_shards = 1
    if not (0 <= shard_index < num_shards):
        raise ValueError(f"shard_index={shard_index} 不在 [0, {num_shards}) 范围内")
    is_rank0 = shard_index == 0

    os.makedirs(args.cache_dir, exist_ok=True)
    manifest_path = os.path.join(args.cache_dir, "manifest.json")
    cache_dtype = dtype_from_name(args.cache_dtype)

    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_id)
    feature_extractor = processor.feature_extractor

    all_audio_paths = collect_unique_audio_paths(
        qa_root=args.qa_root,
        splits=list(args.splits),
        max_samples_per_split=args.max_samples_per_split,
    )
    # 按 rank 均匀切分，保证不重复
    audio_paths = all_audio_paths[shard_index::num_shards]
    if is_rank0:
        print(
            f"[rank {shard_index}/{num_shards}] Total unique audios = {len(all_audio_paths):,}; "
            f"this shard = {len(audio_paths):,}"
        )
    else:
        print(
            f"[rank {shard_index}/{num_shards}] shard size = {len(audio_paths):,}"
        )

    dataset = UniqueAudioDataset(audio_paths)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=AudioBatchCollator(),
    )

    entries: Dict[str, str] = {}
    skipped = 0
    for batch in tqdm(
        loader,
        desc=f"precompute[rank{shard_index}]",
        disable=not is_rank0,
    ):
        # 快速 skip：若所有 path 都已在 cache 存在，整个 batch 跳过提取
        if args.skip_existing:
            all_exist = True
            planned_relpaths = []
            for audio_path in batch["audio_paths"]:
                relpath = build_cache_relpath(audio_path)
                out_path = os.path.join(args.cache_dir, relpath)
                planned_relpaths.append(relpath)
                if not os.path.exists(out_path):
                    all_exist = False
            if all_exist:
                for audio_path, relpath in zip(batch["audio_paths"], planned_relpaths):
                    entries[audio_path] = relpath
                skipped += len(batch["audio_paths"])
                continue

        audio_inputs = feature_extractor(
            batch["mono_audio"],
            sampling_rate=SAMPLE_RATE,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_features = audio_inputs["input_features"]
        feature_attention_mask = audio_inputs["attention_mask"]
        input_lengths = (feature_attention_mask.sum(-1) - 1) // 2 + 1
        audio_token_lengths = (input_lengths - 2) // 2 + 1

        for index, audio_path in enumerate(batch["audio_paths"]):
            relpath = build_cache_relpath(audio_path)
            out_path = os.path.join(args.cache_dir, relpath)
            if args.skip_existing and os.path.exists(out_path):
                entries[audio_path] = relpath
                skipped += 1
                continue
            feature_length = int(feature_attention_mask[index].sum().item())
            spatial_audio = torch.from_numpy(batch["foa_audio"][index])
            spatial_audio_length = int(batch["foa_lengths"][index])
            payload = {
                "audio_path": audio_path,
                "input_features": input_features[index, :, :feature_length].to(dtype=cache_dtype).contiguous(),
                "feature_length": torch.tensor(feature_length, dtype=torch.long),
                "audio_token_length": torch.tensor(int(audio_token_lengths[index].item()), dtype=torch.long),
            }
            if not args.no_spatial_cache:
                payload["spatial_audio"] = spatial_audio.to(dtype=cache_dtype).contiguous()
                payload["spatial_audio_length"] = torch.tensor(spatial_audio_length, dtype=torch.long)
            torch.save(payload, out_path)
            entries[audio_path] = relpath

    # 每个 rank 写自己的 shard manifest；rank 0 在所有人完成后再合并
    shard_manifest_path = os.path.join(
        args.cache_dir, f"manifest.shard{shard_index:03d}of{num_shards:03d}.json"
    )
    with open(shard_manifest_path, "w", encoding="utf-8") as handle:
        json.dump({"entries": entries, "skipped": skipped}, handle, indent=2, sort_keys=True)
    print(
        f"[rank {shard_index}] Wrote shard manifest with {len(entries):,} entries "
        f"(skipped {skipped:,}) -> {shard_manifest_path}"
    )

    if num_shards == 1:
        # 单进程直接写出完整 manifest
        _merge_manifests(args, manifest_path, entries, shard_paths=[shard_manifest_path])
        return

    # 分布式模式：只在 rank0 上做最终 merge，其他 rank 退出。
    # 由于不依赖 NCCL barrier（本脚本不引入 torch.distributed.init_process_group），
    # 我们用轮询等待所有 shard manifest 都出现，然后 merge。
    if not is_rank0:
        return
    import time as _time
    expected = [
        os.path.join(args.cache_dir, f"manifest.shard{i:03d}of{num_shards:03d}.json")
        for i in range(num_shards)
    ]
    waited_s = 0
    print(f"[rank 0] Waiting for {num_shards} shard manifests ...")
    while True:
        missing = [p for p in expected if not os.path.exists(p)]
        if not missing:
            break
        _time.sleep(5)
        waited_s += 5
        if waited_s % 60 == 0:
            print(f"[rank 0] Still waiting, missing {len(missing)} shards after {waited_s}s")
    # 合并
    merged: Dict[str, str] = {}
    for p in expected:
        with open(p, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for audio_path, relpath in payload["entries"].items():
            merged[audio_path] = relpath
    _merge_manifests(args, manifest_path, merged, shard_paths=expected)


def _merge_manifests(
    args: argparse.Namespace,
    manifest_path: str,
    entries: Dict[str, str],
    shard_paths: List[str],
) -> None:
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "cache_dir": os.path.abspath(args.cache_dir),
                "entries": entries,
                "metadata": {
                    "model_id": args.model_id,
                    "qa_root": os.path.abspath(args.qa_root),
                    "splits": list(args.splits),
                    "num_unique_audio": len(entries),
                    "sample_rate": SAMPLE_RATE,
                    "max_audio_seconds": MAX_AUDIO_SECONDS,
                    "cache_dtype": args.cache_dtype,
                },
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    print(f"Wrote merged manifest to {manifest_path} with {len(entries):,} entries")
    # 留着 shard manifest 以便调试，如需清理请手动删除


if __name__ == "__main__":
    main()
