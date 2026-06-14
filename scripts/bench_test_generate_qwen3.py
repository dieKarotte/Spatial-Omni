#!/usr/bin/env python
"""Qwen3-Omni-MoE bench entrypoint (predictions only).

Thin wrapper around ``scripts/bench_test_generate.py`` that swaps in the Qwen3
spatial processor and model. The rest of the bench loop (DDP sharding, collator,
ablation hooks, predictions.jsonl format) is unchanged so that
``scripts/score_test_predictions.py`` works identically.

Usage:
    torchrun --nproc_per_node=8 scripts/bench_test_generate_qwen3.py \\
        --checkpoint-paths runs/so_30b/stage2_encoder_lora/checkpoints/best_trainable.pt \\
        --qa-root /path/to/SO-Dataset/qa \\
        --split test --batch-size 1 --num-workers 4 \\
        --output-dir runs/so_30b/stage2_encoder_lora/bench/test
"""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Inject the local transformers fork so qwen3_omni_moe is importable.
_FORK = os.environ.get(
    "QWEN3_OMNI_FORK",
    "${QWEN3_TRANSFORMERS_FORK}",
)
if os.path.isdir(_FORK) and _FORK not in sys.path:
    sys.path.insert(0, _FORK)

# Apply the same monkey-patches the Qwen3 trainer wrapper uses.
import train_so_qa as _trainer  # noqa: E402
from train_so_qa_qwen3 import (  # noqa: E402
    _build_model_qwen3,
    _build_processor_qwen3,
)

# Patch BEFORE bench module imports `build_processor / build_model`.
_trainer.build_processor = _build_processor_qwen3
_trainer.build_model = _build_model_qwen3

# scripts.batch_bench imports `build_processor / build_model` from
# `train_so_qa` at module-import time. To make the patch stick we
# import batch_bench AFTER patching, then re-bind its module-level names.
from scripts import batch_bench_so_qa as _bb  # noqa: E402

_bb.build_processor = _build_processor_qwen3
_bb.build_model = _build_model_qwen3

# bench_test_generate also re-imports the same names from batch_bench at the
# top, so re-import after patching.
from scripts import bench_test_generate as _bench  # noqa: E402

_bench.build_processor = _build_processor_qwen3
_bench.build_model = _build_model_qwen3


def main() -> int:
    return _bench.main()


if __name__ == "__main__":
    sys.exit(main())
