#!/usr/bin/env python3
"""Build a stratified 20% subsample of the spatial QA difficulty splits.

Source layout:
  {SOURCE_ROOT}/{easy,medium,hard}/{train,valid}.jsonl

Output layout (single combined train/valid):
  {OUTPUT_ROOT}/train.jsonl
  {OUTPUT_ROOT}/valid.jsonl
  {OUTPUT_ROOT}/manifest.json

Strategy:
  - Stratify by (difficulty, task_name): within each (difficulty, task_name)
    bucket, randomly sample exactly `ratio * |bucket|` rows (rounded up so
    very small buckets still keep ≥1 row).
  - Tag every output row with `difficulty` (added field) so downstream
    training/bench can re-bucket by difficulty.
  - Preserve original record ordering inside each bucket via a fixed-seed
    RNG (so the subset is fully deterministic given seed).
  - Final train.jsonl is shuffled across (difficulty, task_name) so DDP
    sharding doesn't end up with one rank seeing only easy.

Usage:
    python scripts/prepare_spatial_subsample.py \
        --source-root /path/to/SO-Dataset/qa \
        --output-root /path/to/SO-Dataset/qa_subsample_20pct \
        --ratio 0.2 --seed 42

Designed for the mono-replay catastrophic-forgetting experiment. The
resulting root drops straight into SO-7B / IV training as
`--qa-root`.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

DIFFICULTIES = ("easy", "medium", "hard")
SPLITS = ("train", "valid")


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _stratified_sample(
    rows: list[dict],
    ratio: float,
    rng: random.Random,
    key: str = "task_name",
) -> tuple[list[dict], dict[str, dict[str, int]]]:
    """Sample `ratio` of rows per `key` bucket. Returns (sampled, stats)."""
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        buckets[str(r.get(key, "_unknown"))].append(r)

    sampled: list[dict] = []
    stats: dict[str, dict[str, int]] = {}
    for bk, bucket in buckets.items():
        n = len(bucket)
        # Round up so tiny buckets still contribute at least 1 row.
        k = max(1, math.ceil(n * ratio)) if n > 0 else 0
        idxs = list(range(n))
        rng.shuffle(idxs)
        chosen = [bucket[i] for i in idxs[:k]]
        sampled.extend(chosen)
        stats[bk] = {"original": n, "sampled": k}
    return sampled, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--source-root",
        required=True,
        type=Path,
        help="Root containing easy/, medium/, hard/ subdirs each with train.jsonl & valid.jsonl",
    )
    ap.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="Output root; train.jsonl + valid.jsonl + manifest.json will be written here.",
    )
    ap.add_argument("--ratio", type=float, default=0.20, help="Per-bucket sample ratio (default 0.20).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--difficulties",
        nargs="+",
        default=list(DIFFICULTIES),
        choices=list(DIFFICULTIES),
        help="Which difficulty splits to combine (default: all three).",
    )
    args = ap.parse_args()

    if not (0.0 < args.ratio <= 1.0):
        raise ValueError(f"--ratio must be in (0, 1], got {args.ratio}")

    args.output_root.mkdir(parents=True, exist_ok=True)

    manifest: dict = {
        "source_root": str(args.source_root),
        "output_root": str(args.output_root),
        "ratio": args.ratio,
        "seed": args.seed,
        "difficulties": args.difficulties,
        "stats": {},
    }

    # Use a per-(split, difficulty) RNG seeded deterministically so that
    # changing one difficulty/split doesn't perturb others.
    base_seed = args.seed

    for split in SPLITS:
        merged: list[dict] = []
        split_stats: dict[str, dict[str, dict[str, int]]] = {}
        for diff in args.difficulties:
            src = args.source_root / diff / f"{split}.jsonl"
            if not src.is_file():
                print(f"[skip] missing {src}")
                split_stats[diff] = {"_missing": {"original": 0, "sampled": 0}}
                continue
            rows = _read_jsonl(src)
            # Tag difficulty (source files don't carry it).
            for r in rows:
                r.setdefault("difficulty", diff)
            rng = random.Random(hash((base_seed, split, diff)) & 0xFFFFFFFF)
            sampled, stats = _stratified_sample(rows, args.ratio, rng)
            split_stats[diff] = stats
            merged.extend(sampled)
            tot_orig = sum(s["original"] for s in stats.values())
            tot_sampled = sum(s["sampled"] for s in stats.values())
            print(f"[{split}/{diff}] {tot_orig} -> {tot_sampled}")

        # Final cross-difficulty shuffle (deterministic).
        final_rng = random.Random(hash((base_seed, "final", split)) & 0xFFFFFFFF)
        final_rng.shuffle(merged)

        out = args.output_root / f"{split}.jsonl"
        with out.open("w") as fh:
            for r in merged:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[write] {out}: {len(merged)} rows")

        # Aggregate counts for manifest sanity.
        diff_counter = Counter(r["difficulty"] for r in merged)
        task_counter = Counter(r["task_name"] for r in merged)
        manifest["stats"][split] = {
            "total_sampled": len(merged),
            "by_difficulty": dict(diff_counter),
            "by_task_name": dict(task_counter),
            "per_difficulty_per_task": split_stats,
        }

    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"[write] {manifest_path}")


if __name__ == "__main__":
    main()
