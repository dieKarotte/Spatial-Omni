#!/usr/bin/env python3
"""Extract SO-Dataset HuggingFace tar shards into the layout expected by
Spatial-Omni trainers and bench scripts.

The HF release layout is::

    <SRC>/
      archives/audio/{train,valid,test}/audio-*-NNNNNN.tar
      archives/annotations/{train,valid,test}/annotations-*-NNNNNN.tar
      metadata/{train,valid,test}.jsonl
      qa/{train,valid,test}.jsonl
      label_mapping.json
      manifests/

After extraction, ``<DST>`` mirrors the paths the metadata expects::

    <DST>/
      audio/{train,valid,test}/foa_*.wav
      annotations/{train,valid,test}/foa_*.csv
      metadata/                  (symlink to <SRC>/metadata)
      qa/                        (symlink to <SRC>/qa)
      label_mapping.json         (symlink)
      manifests/                 (symlink)

Examples
--------
Smoke (1 train shard + full valid + full test + all annotations):
    python scripts/data/extract_so_dataset.py \\
        --src /path/to/hf_release --dst /path/to/SO-Dataset \\
        --splits train valid test --n-shards 1

Full extraction:
    python scripts/data/extract_so_dataset.py \\
        --src /path/to/hf_release --dst /path/to/SO-Dataset \\
        --splits train valid test
"""
import argparse
import os
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable


def _list_shards(src_archive_dir: Path, n_shards: int | None) -> list[Path]:
    if not src_archive_dir.exists():
        return []
    shards = sorted(src_archive_dir.glob("*.tar"))
    return shards[:n_shards] if n_shards else shards


def _extract_one(shard: Path, dst: Path, dry_run: bool) -> tuple[Path, int, int]:
    if dry_run:
        return shard, 0, 0
    with tarfile.open(shard, "r") as tf:
        members = tf.getmembers()
        n_existing = 0
        n_new = 0
        for m in members:
            target_path = dst / m.name
            if target_path.exists() and target_path.is_file() and m.isfile():
                # Skip already-extracted files for idempotency.
                n_existing += 1
                continue
            tf.extract(m, dst)
            n_new += 1
        return shard, n_existing, n_new


def _ensure_symlink(src: Path, dst: Path) -> None:
    if not src.exists():
        print(f"[symlink] WARN: source missing: {src}", file=sys.stderr)
        return
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() and dst.resolve() == src.resolve():
            return
        print(f"[symlink] {dst} already exists; leaving it alone")
        return
    os.symlink(src, dst)
    print(f"[symlink] {dst} -> {src}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="HF release root (contains archives/, metadata/, qa/, ...)")
    p.add_argument("--dst", required=True, help="Destination root (will be created)")
    p.add_argument("--splits", nargs="+", default=["train", "valid", "test"],
                   choices=["train", "valid", "test"])
    p.add_argument("--n-shards", type=int, default=None,
                   help="If set, only extract first N audio shards per split. Annotations are always fully extracted.")
    p.add_argument("--audio-only", action="store_true")
    p.add_argument("--annotations-only", action="store_true")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.audio_only and args.annotations_only:
        sys.exit("--audio-only and --annotations-only are mutually exclusive")

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    dst.mkdir(parents=True, exist_ok=True)

    do_audio = not args.annotations_only
    do_anno = not args.audio_only

    jobs: list[Path] = []
    for split in args.splits:
        if do_audio:
            jobs.extend(_list_shards(src / "archives" / "audio" / split, args.n_shards))
        if do_anno:
            # Always extract all annotation shards (small, ~1 tar per split).
            jobs.extend(_list_shards(src / "archives" / "annotations" / split, None))

    print(f"[extract] {len(jobs)} tar shards to extract -> {dst}")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_extract_one, j, dst, args.dry_run): j for j in jobs}
        for fut in as_completed(futs):
            shard, n_existing, n_new = fut.result()
            print(f"[extract] {shard.name}  new={n_new}  skipped_existing={n_existing}")

    # Symlink the small assets so metadata / qa / vocabulary live under <dst>.
    for name in ("metadata", "qa", "manifests", "label_mapping.json"):
        _ensure_symlink(src / name, dst / name)

    print(f"[extract] done. Layout under: {dst}")
    print("  audio/{train,valid,test}/*.wav")
    print("  annotations/{train,valid,test}/*.csv")
    print("  metadata/, qa/, manifests/, label_mapping.json   (symlinks to release)")


if __name__ == "__main__":
    main()
