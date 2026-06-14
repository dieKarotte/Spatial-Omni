#!/usr/bin/env python3
"""Generate the SO-Dataset 63-class source-vocabulary CSV.

The SO-Encoder cls head's output dimensions are ordered by *FSD50K class
frequency* (the order used during pretraining), NOT alphabetically. To run
the released SO-Encoder checkpoint on the SO-Dataset HuggingFace release,
the vocabulary must follow this same ordering.

Two ways to build it:

1) Copy the canonical FSD50K vocabulary directly. This is the recommended
   path. Pass ``--fsd50k-vocab path/to/FSD50K.ground_truth/final_vocabulary.csv``
   and the script will copy/normalise it. The generated CSV's row index is the
   model's cls-head output dimension.

2) Build from the SO-Dataset release ``label_mapping.json`` (alphabetical).
   This is ONLY for backwards compat with code that needs the alphabetical
   ordering — DO NOT use it with a pretrained SO-Encoder checkpoint.

The dataset loader (``so_dataset.py``) joins source records to vocabulary
rows by **label name** (the ``label`` string), so as long as the row order
matches the cls head, everything lines up.

Examples:
    # Recommended: vocab aligned with released SO-Encoder ckpt
    python scripts/data/build_so_vocab.py \\
        --fsd50k-vocab /path/to/FSD50K.ground_truth/final_vocabulary.csv \\
        --output       /path/to/SO-Dataset/so_vocab.csv

    # Alphabetical (legacy, NOT compatible with released ckpts)
    python scripts/data/build_so_vocab.py \\
        --label-mapping /path/to/SO-Dataset/label_mapping.json \\
        --output        /path/to/SO-Dataset/so_vocab.csv \\
        --alphabetical
"""
import argparse
import csv
import json
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--fsd50k-vocab",
        help="Path to FSD50K.ground_truth/final_vocabulary.csv (recommended). "
             "Its row order = the SO-Encoder cls-head output dimension order.",
    )
    p.add_argument(
        "--label-mapping",
        help="Path to label_mapping.json from the SO-Dataset release "
             "(only used together with --alphabetical).",
    )
    p.add_argument(
        "--alphabetical",
        action="store_true",
        help="Emit alphabetically-sorted vocab (legacy; NOT compatible with "
             "released SO-Encoder checkpoints).",
    )
    p.add_argument("--output", required=True, help="Path to write the vocabulary CSV.")
    args = p.parse_args()

    if args.alphabetical:
        if not args.label_mapping:
            sys.exit("--alphabetical requires --label-mapping")
        mapping = json.loads(Path(args.label_mapping).read_text())
        cid_to_name = mapping.get("class_id_to_name") or {}
        if not cid_to_name:
            sys.exit(f"label_mapping.json missing 'class_id_to_name': {args.label_mapping}")
        rows = sorted(
            ((int(k), v) for k, v in cid_to_name.items()),
            key=lambda x: x[0],
        )
        out_rows = [(cid, name, name) for cid, name in rows]
        header = ["label_id", "final_label", "clean_label"]
    else:
        if not args.fsd50k_vocab:
            sys.exit(
                "--fsd50k-vocab is required (or pass --alphabetical to emit "
                "the legacy alphabetical ordering)."
            )
        with open(args.fsd50k_vocab, "r", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            header = next(reader)
            out_rows = [tuple(row) for row in reader]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for row in out_rows:
            writer.writerow(row)

    print(f"[vocab] wrote {len(out_rows)} rows -> {out_path}")
    print(f"[vocab] expected num_classes for SO-Encoder pretrain: {len(out_rows)}")


if __name__ == "__main__":
    main()
