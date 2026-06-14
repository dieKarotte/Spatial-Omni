"""Strip internal paths / identifiers from an SO-Encoder training checkpoint
before public release.

The released SO-Encoder checkpoint is a full training snapshot
(`model_state_dict` + `optimizer_state_dict` + `train_metrics` + `train_cfg`).
This is intentional: downstream users can resume pretraining from the exact
optimizer state. But the `train_cfg` saved during training records absolute
filesystem paths (manifest paths, output_dir, fsd50k vocab path, etc.) that
are specific to the original training environment. This script clears those
fields without touching the model or optimizer state, so the release can stay
self-contained.

What is preserved
-----------------
- ``model_state_dict``        — full SO-Encoder weights (~648 MB)
- ``optimizer_state_dict``    — Adam moments for resume (~740 MB)
- ``epoch``                   — last epoch index
- ``best_metric_name`` / ``best_metric_value``
- ``train_metrics`` / ``val_metrics``  — per-epoch metric arrays
- ``train_cfg``               — non-path hyperparameters (lr, weight_decay,
                                EMA settings, loss lambdas, etc.)

What is scrubbed
----------------
- ``train_manifest_paths`` / ``val_manifest_paths`` / ``test_manifest_paths``
- ``train_manifest_path`` / ``val_manifest_path`` / ``test_manifest_path``
- ``output_dir``
- ``resume_from_checkpoint``
- ``pretrained_beats_ckpt`` / ``class_finetuned_ckpt`` / ``init_from_spatial_ckpt``
- ``dataset.source_vocab.vocab_path``

The legacy training pipeline pickled a ``spatial_beats.SpatialBEATsConfig``
object inside ``train_cfg["model"]``. We register a stub alias so the legacy
pickle resolves to the renamed ``SOBackboneConfig``; the resulting bytes are
re-pickled under the new name so future loads don't need the stub.

Usage
-----
::

    python scripts/data/release_so_encoder_ckpt.py \\
        --input  /path/to/SO-Encoder_finetuned.pt \\
        --output /path/to/SO-Encoder_finetuned_release.pt
"""

from __future__ import annotations

import argparse
import os
import sys
import types
from typing import Any


SCRUB_STRING_FIELDS = (
    "train_manifest_path",
    "val_manifest_path",
    "test_manifest_path",
    "output_dir",
    "resume_from_checkpoint",
    "pretrained_beats_ckpt",
    "class_finetuned_ckpt",
    "init_from_spatial_ckpt",
)

# Tuple-valued path fields (manifest collections).
SCRUB_TUPLE_FIELDS = (
    "train_manifest_paths",
    "val_manifest_paths",
    "test_manifest_paths",
)


def install_legacy_pickle_stub(repo_root: str) -> None:
    """Register a ``spatial_beats`` module so legacy pickled configs resolve."""
    sys.path.insert(0, repo_root)
    from spatial_omni.encoders.beats.so_backbone import SOBackboneConfig  # noqa: E402

    stub = types.ModuleType("spatial_beats")
    stub.SpatialBEATsConfig = SOBackboneConfig
    sys.modules["spatial_beats"] = stub


def scrub_train_cfg(cfg: dict, *, verbose: bool = True) -> int:
    """Clear path-bearing fields. Returns number of fields scrubbed."""
    n_scrubbed = 0
    if not isinstance(cfg, dict):
        return 0

    for field in SCRUB_STRING_FIELDS:
        if field in cfg and cfg[field]:
            if verbose:
                print(f"  [scrub str ] {field}: {cfg[field]!r} -> ''")
            cfg[field] = ""
            n_scrubbed += 1

    for field in SCRUB_TUPLE_FIELDS:
        if field in cfg and cfg[field]:
            old = cfg[field]
            if verbose:
                preview = repr(old)
                if len(preview) > 90:
                    preview = preview[:87] + "..."
                print(f"  [scrub tup ] {field}: {preview} -> ()")
            # Preserve original type (tuple vs list).
            cfg[field] = type(old)() if isinstance(old, (tuple, list)) else ()
            n_scrubbed += 1

    dataset = cfg.get("dataset")
    if isinstance(dataset, dict):
        sv = dataset.get("source_vocab")
        if isinstance(sv, dict) and sv.get("vocab_path"):
            if verbose:
                print(
                    f"  [scrub dset] dataset.source_vocab.vocab_path: "
                    f"{sv['vocab_path']!r} -> ''"
                )
            sv["vocab_path"] = ""
            n_scrubbed += 1

    return n_scrubbed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Path to the source ckpt.")
    parser.add_argument("--output", required=True, help="Where to write the cleaned ckpt.")
    parser.add_argument(
        "--repo-root",
        default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        ),
        help="Spatial-Omni repo root used for sys.path injection.",
    )
    parser.add_argument(
        "--keep-optimizer",
        action="store_true",
        default=True,
        help="(default) Keep optimizer_state_dict so users can resume training.",
    )
    parser.add_argument(
        "--strip-optimizer",
        dest="keep_optimizer",
        action="store_false",
        help="Drop optimizer_state_dict for an inference-only release (~half the size).",
    )
    args = parser.parse_args()

    if os.path.abspath(args.input) == os.path.abspath(args.output):
        raise SystemExit("--input and --output must differ; refusing to overwrite in place.")

    print(f"[release] repo root  : {args.repo_root}")
    install_legacy_pickle_stub(args.repo_root)

    import torch  # noqa: E402

    print(f"[release] loading    : {args.input}")
    ckpt: dict[str, Any] = torch.load(args.input, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise SystemExit(
            f"Expected the checkpoint to be a dict, got {type(ckpt).__name__}"
        )

    print(f"[release] top keys   : {sorted(ckpt.keys())}")

    n_msd = sum(1 for v in ckpt.get("model_state_dict", {}).values() if hasattr(v, "numel"))
    n_osd_present = "optimizer_state_dict" in ckpt
    print(f"[release] model_state_dict   : {n_msd} tensors")
    print(f"[release] optimizer_state    : {'present' if n_osd_present else 'absent'}")

    cfg = ckpt.get("train_cfg")
    if cfg is None:
        print("[release] train_cfg missing; skipping scrub.")
    else:
        print(f"[release] scrubbing train_cfg ({len(cfg)} keys) ...")
        n = scrub_train_cfg(cfg, verbose=True)
        print(f"[release] scrubbed {n} field(s).")

    if not args.keep_optimizer and "optimizer_state_dict" in ckpt:
        del ckpt["optimizer_state_dict"]
        print("[release] dropped optimizer_state_dict (inference-only release).")

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    print(f"[release] writing    : {args.output}")
    torch.save(ckpt, args.output)

    in_size = os.path.getsize(args.input) / 1e6
    out_size = os.path.getsize(args.output) / 1e6
    print(f"[release] sizes      : input={in_size:.1f} MB  output={out_size:.1f} MB")
    print("[release] done.")


if __name__ == "__main__":
    main()
