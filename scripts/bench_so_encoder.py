#!/usr/bin/env python3
"""Evaluate an SO-Encoder checkpoint on a test split.

Reuses the trainer's `evaluate_one_epoch` so the metric computation is
identical to the running validation during training (Official DCASE
SELD metrics for `local_spatial_track` supervision).

Usage:
    PYTHONPATH=. python scripts/bench_so_encoder.py \\
        --checkpoint /path/to/best.pt \\
        --test-manifest /path/to/SO-Dataset/pretrain-test.jsonl \\
        --source-vocab /path/to/SO-Dataset/so_vocab.csv \\
        --pretrained-beats-ckpt /path/to/BEATs_iter3_plus_AS2M.pt \\
        --batch-size 4 --num-workers 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Optional bundled BEATs repo (set SO_BEATS_REPO env if needed)
_so_beats_repo = os.environ.get("SO_BEATS_REPO", "")
if _so_beats_repo and _so_beats_repo not in sys.path:
    sys.path.insert(0, _so_beats_repo)

from spatial_omni.encoders.beats.so_dataset import SpatialDataset, collate_spatial_batch
from spatial_omni.encoders.beats.train_so_pretrain import (
    build_dataset_config,
    build_model,
    evaluate_one_epoch,
    make_so_encoder_config,
    _legacy_safe_torch_load,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to a SO-Encoder checkpoint (.pt) to evaluate.")
    parser.add_argument("--test-manifest", required=True,
                        help="Path to the test-split manifest (jsonl).")
    parser.add_argument("--source-vocab", required=True,
                        help="Path to source-class vocabulary CSV.")
    parser.add_argument("--source-num-classes", type=int, default=63)
    parser.add_argument("--pretrained-beats-ckpt", default=os.environ.get(
        "SO_BEATS_TRUNK_CKPT",
        "pretrain_ckpt/BEATs_iter3_plus_AS2M.pt/BEATs_iter3_plus_AS2M.pt",
    ))
    parser.add_argument("--class-finetuned-ckpt", default="")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-json", default=None,
                        help="Where to write the metrics dict (default: alongside the ckpt).")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Build the canonical SO-Encoder cfg + model so the architecture matches
    # the checkpoint exactly (the trainer's load path does this for us).
    cfg = make_so_encoder_config(
        train_manifest_path=args.test_manifest,  # placeholder; not used
        valid_manifest_path=args.test_manifest,
        pretrained_beats_ckpt=args.pretrained_beats_ckpt,
        class_finetuned_ckpt=args.class_finetuned_ckpt,
        source_vocab_path=args.source_vocab,
        source_num_classes=args.source_num_classes,
    )
    cfg.distributed = False
    cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers
    # Mirror training-side wiring: build_dataset_config() syncs from cfg.model,
    # so the vocab path/num-classes must live on the model cfg too.
    cfg.model.source_vocab_path = args.source_vocab
    cfg.model.source_num_classes = args.source_num_classes

    # Build the model and load the checkpoint state dict.
    model = build_model(cfg).to(device)
    print(f"[Bench] Loading checkpoint: {args.checkpoint}")
    ckpt = _legacy_safe_torch_load(args.checkpoint)
    state_dict = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[Bench] state_dict load: matched={len(state_dict) - len(unexpected)}  "
          f"missing={len(missing)}  unexpected={len(unexpected)}")
    if missing:
        print(f"[Bench]   missing sample: {missing[:5]}")
    if unexpected:
        print(f"[Bench]   unexpected sample: {unexpected[:5]}")

    # Build a single-shard test DataLoader using the same collator as training.
    dataset_cfg = build_dataset_config(cfg)
    test_dataset_cfg = type(dataset_cfg)(**dataset_cfg.__dict__) if not hasattr(dataset_cfg, "__dataclass_fields__") else dataset_cfg
    # Use the cfg as-is — build_dataset_config already pulls splits from the train cfg
    import copy
    test_dataset_cfg = copy.deepcopy(dataset_cfg)
    test_dataset_cfg.allowed_splits = cfg.test_splits or cfg.val_splits

    print(f"[Bench] Building test dataset from {args.test_manifest}")
    dataset = SpatialDataset(manifest_path=args.test_manifest, config=test_dataset_cfg)
    print(f"[Bench] Test set: {len(dataset)} samples")

    import functools
    collate_fn = functools.partial(collate_spatial_batch, config=test_dataset_cfg)
    test_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    # Reuse the trainer's eval loop verbatim.
    metrics, _examples, _csv = evaluate_one_epoch(model, test_loader, cfg)

    # Pretty-print results.
    print()
    print("=" * 70)
    print(f"  SO-Encoder evaluation on TEST set ({len(dataset)} samples)")
    print(f"  ckpt: {args.checkpoint}")
    print("=" * 70)
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            print(f"  {k:30s}: {v:.4f}")
        else:
            print(f"  {k:30s}: {v}")
    print("=" * 70)

    # Persist.
    out_path = Path(args.output_json) if args.output_json else Path(args.checkpoint).with_suffix(".test_metrics.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2, default=float))
    print(f"[Bench] metrics saved to {out_path}")


if __name__ == "__main__":
    main()
