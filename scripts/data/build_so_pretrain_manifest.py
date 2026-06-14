#!/usr/bin/env python3
"""Build an SO-Encoder pretraining manifest from SO-Dataset's metadata JSONL.

The bundled SO-Encoder loader (`spatial_omni/encoders/beats/so_dataset.py`)
natively supports the SO-Dataset release schema (``entry["audio"]["foa_path"]``,
``sources[*]["source_trajectory_csv_path"]``, etc.). This script just:

1. Resolves the ``audio.foa_path`` and ``source_trajectory_csv_path`` to
   absolute paths under ``--data-root`` so the loader doesn't depend on cwd.
2. Optionally filters to entries whose audio file is on disk (lets you run
   smoke trainings against a partially extracted release).
3. Optionally caps to ``--max-records`` for tiny smoke runs.

Example
-------
    python scripts/data/build_so_pretrain_manifest.py \\
        --metadata-jsonl /path/to/SO-Dataset/metadata/train.jsonl \\
        --data-root /path/to/SO-Dataset \\
        --output /path/to/SO-Dataset/pretrain-train.jsonl \\
        --filter-missing
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterator


def _stream_records(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _absolutize(p: str | None, data_root: Path) -> str | None:
    if p is None:
        return None
    if Path(p).is_absolute():
        return p
    return str((data_root / p).resolve())


def _convert_record(rec: Dict[str, Any], data_root: Path) -> Dict[str, Any]:
    out = dict(rec)
    audio = out.get("audio")
    if isinstance(audio, dict):
        new_audio = dict(audio)
        if "foa_path" in new_audio:
            new_audio["foa_path"] = _absolutize(new_audio["foa_path"], data_root)
        out["audio"] = new_audio

    if "scene_annotation_csv_path" in out:
        out["scene_annotation_csv_path"] = _absolutize(out["scene_annotation_csv_path"], data_root)

    sources = out.get("sources")
    if isinstance(sources, list):
        new_sources = []
        for src in sources:
            if not isinstance(src, dict):
                new_sources.append(src)
                continue
            new_src = dict(src)
            if "source_trajectory_csv_path" in new_src:
                new_src["source_trajectory_csv_path"] = _absolutize(
                    new_src["source_trajectory_csv_path"], data_root
                )
            new_sources.append(new_src)
        out["sources"] = new_sources
    return out


def _audio_path_of(rec: Dict[str, Any]) -> str | None:
    audio = rec.get("audio")
    if isinstance(audio, dict):
        return audio.get("foa_path")
    return rec.get("output_foa_path") or rec.get("audio_path")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metadata-jsonl", required=True, type=Path)
    p.add_argument("--data-root", required=True, type=Path,
                   help="SO-Dataset extraction root; relative paths resolve under here.")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--filter-missing", action="store_true",
                   help="Drop records whose FOA wav is not on disk (useful when only some shards are extracted).")
    p.add_argument("--max-records", type=int, default=None)
    args = p.parse_args()

    if not args.metadata_jsonl.exists():
        sys.exit(f"missing metadata jsonl: {args.metadata_jsonl}")
    args.data_root = args.data_root.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    n_in = n_out = n_missing = 0
    with args.output.open("w", encoding="utf-8") as out_fh:
        for rec in _stream_records(args.metadata_jsonl):
            n_in += 1
            converted = _convert_record(rec, args.data_root)
            if args.filter_missing:
                ap = _audio_path_of(converted)
                if ap is None or not Path(ap).exists():
                    n_missing += 1
                    continue
            out_fh.write(json.dumps(converted, ensure_ascii=False) + "\n")
            n_out += 1
            if args.max_records is not None and n_out >= args.max_records:
                break

    print(f"[manifest] read={n_in}  written={n_out}  filtered_missing={n_missing}")
    print(f"[manifest] -> {args.output}")


if __name__ == "__main__":
    main()
