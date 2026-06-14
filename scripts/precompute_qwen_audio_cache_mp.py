#!/usr/bin/env python3
"""纯 CPU 多进程版 Qwen Whisper mel 特征预计算。

相比 `precompute_qwen_audio_cache.py` 的 torchrun 版：
  - 不占用 GPU（Whisper STFT 在 CPU 上跑本来就够快，CPU 并行是正确方向）
  - 进程数不受 GPU 卡数限制（192 核机器可以开 64+ 进程）
  - `multiprocessing.Pool` + `imap_unordered` 负载自动均衡
  - 预期 8h（单进程版）→ 10~15 分钟

典型用法（398K 条音频，32 进程，约 15 分钟）：
    python scripts/precompute_qwen_audio_cache_mp.py \\
        --qa-root /path/to/SO-Dataset/qa \\
        --splits train valid test \\
        --cache-dir /path/to/qwen_audio_cache \\
        --num-procs 32 \\
        --no-spatial-cache \\
        --skip-existing

核心设计：
  - Pool initializer 里每个 worker 加载自己的 Qwen feature_extractor（一次，常驻）
  - 强制每个 worker `OMP_NUM_THREADS=1`，避免 numpy/MKL 多线程互相抢核
  - 用 imap_unordered + chunksize=16 降低 IPC 开销
  - `--skip-existing`：主进程先扫已完成的 .pt，只把真正 TODO 的任务下发给 worker
  - 中途中断后再跑，加 `--skip-existing` 即可续传
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from tqdm.auto import tqdm


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-Omni-7B"
DEFAULT_SO_REPO = os.environ.get("SO_REPO", "")
SAMPLE_RATE = 16000
MAX_AUDIO_SECONDS = 20
MAX_AUDIO_SAMPLES = SAMPLE_RATE * MAX_AUDIO_SECONDS


def dtype_from_name(name: str) -> torch.dtype:
    return {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[name]


def build_cache_relpath(audio_path: str) -> str:
    """与 precompute_qwen_audio_cache.py 的命名完全一致，生成的 cache 两版互通。"""
    audio_path = os.path.abspath(audio_path)
    stem = os.path.splitext(os.path.basename(audio_path))[0]
    digest = hashlib.sha1(audio_path.encode("utf-8")).hexdigest()[:16]
    return f"{stem}-{digest}.pt"


def collect_unique_audio_paths(
    qa_root: str,
    splits: List[str],
    max_samples_per_split: Optional[int],
) -> List[str]:
    unique: Dict[str, None] = {}
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
                for idx, line in enumerate(handle):
                    if max_samples_per_split is not None and idx >= max_samples_per_split:
                        break
                    record = json.loads(line)
                    unique[os.path.abspath(record["audio_path"])] = None
            else:
                payload = json.load(handle)
                iterator = payload if isinstance(payload, list) else payload.get(
                    "records", payload.get("data", [])
                )
                for idx, record in enumerate(iterator):
                    if max_samples_per_split is not None and idx >= max_samples_per_split:
                        break
                    unique[os.path.abspath(record["audio_path"])] = None
    return sorted(unique.keys())


# ------------------------------------------------------------------
# Worker-level globals（每个 worker 进程独立一份）
# ------------------------------------------------------------------
_FE = None  # Qwen WhisperFeatureExtractor
_CACHE_DIR: Optional[str] = None
_CACHE_DTYPE: Optional[torch.dtype] = None
_NO_SPATIAL_CACHE: bool = False


def _init_worker(
    model_id: str,
    so_repo: str,
    cache_dir_abs: str,
    cache_dtype_name: str,
    no_spatial_cache: bool,
) -> None:
    global _FE, _CACHE_DIR, _CACHE_DTYPE, _NO_SPATIAL_CACHE
    # 每个 worker 强制单线程，避免 numpy/MKL/torch 内部多线程和其他 worker 抢核
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    if so_repo not in sys.path:
        sys.path.insert(0, so_repo)
    from spatial_omni.model.processing_qwen2_5_omni import Qwen2_5OmniProcessor
    processor = Qwen2_5OmniProcessor.from_pretrained(model_id)
    _FE = processor.feature_extractor
    _CACHE_DIR = cache_dir_abs
    _CACHE_DTYPE = dtype_from_name(cache_dtype_name)
    _NO_SPATIAL_CACHE = no_spatial_cache


def _process_one(audio_path: str) -> Tuple[str, str, Optional[str]]:
    """处理单条音频。

    Returns:
        (audio_path, relpath, error_msg)
        - error_msg=None 表示成功（或磁盘上已存在、跳过）。
    """
    global _FE, _CACHE_DIR, _CACHE_DTYPE, _NO_SPATIAL_CACHE
    try:
        relpath = build_cache_relpath(audio_path)
        out_path = os.path.join(_CACHE_DIR, relpath)

        # 在 worker 内再做一次存在检查：主进程扫完到 worker 开工之间可能已跑过
        if os.path.exists(out_path):
            return audio_path, relpath, None

        # 读 FOA 波形
        wav, sr = sf.read(audio_path, dtype="float32", always_2d=True)
        if sr != SAMPLE_RATE:
            return audio_path, "", f"bad_sample_rate={sr}"
        # sf 读出 [T, C]，我们要 [C, T]
        foa = wav.T
        if foa.ndim != 2 or foa.shape[0] != 4:
            return audio_path, "", f"bad_shape={tuple(foa.shape)}"
        if foa.shape[1] > MAX_AUDIO_SAMPLES:
            foa = foa[:, :MAX_AUDIO_SAMPLES]
        foa_length = int(foa.shape[1])
        mono = foa.mean(axis=0).astype(np.float32, copy=False)

        # Qwen Whisper mel 特征
        audio_inputs = _FE(
            [mono],
            sampling_rate=SAMPLE_RATE,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_features = audio_inputs["input_features"]             # [1, D, T_pad]
        feature_attention_mask = audio_inputs["attention_mask"]     # [1, T_pad]
        input_lengths = (feature_attention_mask.sum(-1) - 1) // 2 + 1
        audio_token_lengths = (input_lengths - 2) // 2 + 1
        feature_length = int(feature_attention_mask[0].sum().item())

        payload = {
            "audio_path": audio_path,
            "input_features": input_features[0, :, :feature_length]
                .to(dtype=_CACHE_DTYPE)
                .contiguous(),
            "feature_length": torch.tensor(feature_length, dtype=torch.long),
            "audio_token_length": torch.tensor(
                int(audio_token_lengths[0].item()), dtype=torch.long
            ),
        }
        if not _NO_SPATIAL_CACHE:
            payload["spatial_audio"] = (
                torch.from_numpy(foa).to(dtype=_CACHE_DTYPE).contiguous()
            )
            payload["spatial_audio_length"] = torch.tensor(foa_length, dtype=torch.long)

        # 先写临时文件再原子 rename，避免主进程崩掉后 cache 里出现半成品
        tmp_path = out_path + f".tmp.{os.getpid()}"
        torch.save(payload, tmp_path)
        os.replace(tmp_path, out_path)
        return audio_path, relpath, None
    except Exception as exc:  # noqa: BLE001
        return audio_path, "", f"{type(exc).__name__}: {exc}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-process (CPU) version of Qwen audio feature cache precompute.",
    )
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--so-repo", type=str, default=DEFAULT_SO_REPO)
    parser.add_argument("--qa-root", type=str, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    parser.add_argument("--max-samples-per-split", type=int, default=None)
    parser.add_argument("--cache-dir", type=str, required=True)
    parser.add_argument(
        "--cache-dtype",
        type=str,
        default="float16",
        choices=("float16", "float32", "bfloat16"),
    )
    parser.add_argument(
        "--no-spatial-cache",
        action="store_true",
        help="不缓存 spatial_audio 原始波形（推荐！原始 20s 4ch fp16 单条 2.5MB × 398K ≈ 1TB，"
             "不缓存后 cache 仅 ~100GB；训练时 collator 自动 fallback 到 sf.read）。",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="跳过磁盘上已存在的 .pt 文件（支持中断续传）。",
    )
    parser.add_argument(
        "--num-procs",
        type=int,
        default=32,
        help="并发 worker 进程数。机器 192 核时 32~64 是甜区；超过 64 可能被 NFS IO 卡住。",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=16,
        help="Pool.imap_unordered 的 chunksize，降低 IPC 频率。",
    )
    parser.add_argument(
        "--log-errors-every",
        type=int,
        default=50,
        help="每累积多少个错误打印一次汇总。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir_abs = os.path.abspath(args.cache_dir)
    os.makedirs(cache_dir_abs, exist_ok=True)
    manifest_path = os.path.join(cache_dir_abs, "manifest.json")

    # ---------- 1. 收集任务列表 ----------
    print(
        f"[{time.strftime('%H:%M:%S')}] Collecting unique audio paths from "
        f"{args.qa_root} splits={args.splits}"
    )
    all_audios = collect_unique_audio_paths(
        qa_root=args.qa_root,
        splits=list(args.splits),
        max_samples_per_split=args.max_samples_per_split,
    )
    print(f"Total unique audios: {len(all_audios):,}")

    # ---------- 2. 先过滤已完成的（skip-existing 模式） ----------
    done_entries: Dict[str, str] = {}
    if args.skip_existing:
        t_scan = time.time()
        todo: List[str] = []
        for ap in tqdm(all_audios, desc="scan existing", smoothing=0.1):
            relpath = build_cache_relpath(ap)
            if os.path.exists(os.path.join(cache_dir_abs, relpath)):
                done_entries[ap] = relpath
            else:
                todo.append(ap)
        print(
            f"[{time.strftime('%H:%M:%S')}] Scan done in {time.time()-t_scan:.1f}s: "
            f"already_done={len(done_entries):,}, todo={len(todo):,}"
        )
    else:
        todo = list(all_audios)

    # ---------- 2b. 合并已存在 manifest 的条目，避免覆盖丢失其它 split 的索引 ----------
    # 典型场景：先跑 valid+test 生成 manifest，再跑 train 时若直接 w 模式写 manifest，
    # 会丢失 valid+test 的条目。这里先读入旧 manifest（如果存在），把其中的条目合并进来。
    prior_entries: Dict[str, str] = {}
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as fh:
                prior_payload = json.load(fh)
            prior_cache_dir = os.path.abspath(prior_payload.get("cache_dir", cache_dir_abs))
            if prior_cache_dir != cache_dir_abs:
                print(
                    f"[warn] existing manifest cache_dir={prior_cache_dir} != current={cache_dir_abs}, "
                    "skipping prior-entries merge (relpaths may not align)."
                )
            else:
                raw_prior = prior_payload.get("entries", {})
                # 过滤掉磁盘上已不存在的 .pt（防陈旧 manifest 污染）
                kept = 0
                for ap, relpath in raw_prior.items():
                    if os.path.exists(os.path.join(cache_dir_abs, relpath)):
                        prior_entries[os.path.abspath(ap)] = relpath
                        kept += 1
                print(
                    f"[merge] Loaded {kept:,}/{len(raw_prior):,} valid entries from "
                    f"existing manifest.json (stale entries dropped)"
                )
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] failed to read existing manifest, ignoring: {exc}")

    entries: Dict[str, str] = dict(prior_entries)
    entries.update(done_entries)
    errors: List[Tuple[str, str]] = []

    # ---------- 3. 多进程处理 ----------
    if todo:
        print(
            f"[{time.strftime('%H:%M:%S')}] Spawning {args.num_procs} CPU workers "
            f"(chunksize={args.chunk_size}) to process {len(todo):,} audios..."
        )
        t_proc = time.time()
        with Pool(
            processes=args.num_procs,
            initializer=_init_worker,
            initargs=(
                args.model_id,
                args.so_repo,
                cache_dir_abs,
                args.cache_dtype,
                args.no_spatial_cache,
            ),
        ) as pool:
            it = pool.imap_unordered(_process_one, todo, chunksize=args.chunk_size)
            for audio_path, relpath, err in tqdm(
                it, total=len(todo), desc="precompute(mp)", smoothing=0.05
            ):
                if err is None:
                    entries[audio_path] = relpath
                else:
                    errors.append((audio_path, err))
                    if len(errors) % args.log_errors_every == 0:
                        tqdm.write(
                            f"[errors so far {len(errors)}] latest: {audio_path} → {err}"
                        )
        elapsed = time.time() - t_proc
        n = max(len(todo), 1)
        print(
            f"[{time.strftime('%H:%M:%S')}] Processed {len(todo):,} audios in {elapsed:.1f}s "
            f"({n/elapsed:.1f} audios/s, avg {elapsed/n*1000:.1f} ms/audio) "
            f"ok={len(todo)-len(errors):,} errors={len(errors):,}"
        )
    else:
        print("Nothing to do (all entries already cached).")

    # ---------- 4. 写 manifest ----------
    manifest_payload = {
        "cache_dir": cache_dir_abs,
        "entries": entries,
        "metadata": {
            "model_id": args.model_id,
            "qa_root": os.path.abspath(args.qa_root),
            "splits": list(args.splits),
            "num_unique_audio": len(entries),
            "sample_rate": SAMPLE_RATE,
            "max_audio_seconds": MAX_AUDIO_SECONDS,
            "cache_dtype": args.cache_dtype,
            "no_spatial_cache": args.no_spatial_cache,
            "num_errors": len(errors),
        },
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest_payload, handle, indent=2, sort_keys=True)
    print(
        f"[{time.strftime('%H:%M:%S')}] Wrote manifest to {manifest_path} "
        f"({len(entries):,} entries)"
    )

    # ---------- 5. 错误汇总 ----------
    if errors:
        err_path = os.path.join(cache_dir_abs, "precompute_errors.jsonl")
        with open(err_path, "w", encoding="utf-8") as fh:
            for ap, msg in errors:
                fh.write(json.dumps({"audio_path": ap, "error": msg}) + "\n")
        print(
            f"Logged {len(errors):,} errors to {err_path} (first 5 below):"
        )
        for ap, msg in errors[:5]:
            print(f"  {ap} → {msg}")


if __name__ == "__main__":
    main()
