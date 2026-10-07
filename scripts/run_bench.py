"""Unified bench driver: one entrypoint to dispatch test-split generation
across all Spatial-Omni baselines.

Today the repo has three parallel bench scripts that each know how to build
ONE backbone's model:

    scripts/bench_test_generate.py       -> SO-7B (SO-Encoder + Qwen2.5-Omni)
    scripts/bench_test_generate_iv.py    -> IV / Neural-IV baseline
    scripts/bench_test_generate_af3.py   -> AF3 baseline

Each has its own parse_args() + setup_distributed() + main(). This driver is
a thin dispatcher that chooses one of them based on --baseline, rewrites
sys.argv to the sub-script's CLI, and calls its main() IN-PROCESS. Because
we call main() in-process (not subprocess), torchrun wraps run_bench.py just
like any sub-script -- DDP state is set up once, no nested launches.

## Baselines

    so-7b                (SO-Encoder + Qwen2.5-Omni)
    so-30b               (SO-Encoder + Qwen3-Omni-30B)
    zero-spatial         (same backbone as so-7b, but --spatial-ablation zero)
    zero-spatial-30b     (same backbone as so-30b, but --spatial-ablation zero)
    iv                   (IV baseline + Qwen2.5-Omni)
    neural-iv            (Neural-IV baseline + Qwen2.5-Omni)
    af3                  (AF3 baseline)

Note on iv vs neural-iv: both dispatch to bench_test_generate_iv.py. The
choice between IV and Neural-IV is encoded inside the checkpoint's
train_args.json (spatial_encoder_type field), NOT set by this driver. The
outer --baseline is a labelling / sanity-check knob so you notice if you
point at the wrong run-dir.

Note on zero-spatial: this is an ablation, not a separate model. It uses
the so-7b backbone and injects --spatial-ablation zero to the sub-
script, which zeros the spatial-audio content before generate() while
keeping attention masks / lengths. Only so-7b supports ablation today;
--spatial-ablation is NOT exposed on the outer CLI to keep a single
baseline axis -- to run the ablation, pass --baseline zero-spatial.

## Usage

    torchrun --nproc_per_node=8 scripts/run_bench.py \\
        --baseline so-7b \\
        --qa-root /path/to/SO-Dataset/qa \\
        --split test \\
        --checkpoint-paths runs/so_7b/stage2_encoder_lora/checkpoints/best_trainable.pt \\
        --output-dir runs/so_7b/stage2_encoder_lora/bench/test

    torchrun --nproc_per_node=8 scripts/run_bench.py \\
        --baseline iv \\
        --qa-root /path/to/SO-Dataset/qa \\
        --checkpoint-paths runs/iv/stage2_encoder_lora/checkpoints/best_trainable.pt

    torchrun --nproc_per_node=8 scripts/run_bench.py \\
        --baseline zero-spatial \\
        --qa-root /path/to/SO-Dataset/qa \\
        --checkpoint-paths runs/so_7b/stage2_encoder_lora/checkpoints/best_trainable.pt

After predictions.jsonl is emitted, score with (same as today):

    python scripts/score_test_predictions.py \\
        --predictions-jsonl <output_dir>/<ckpt>/predictions.jsonl \\
        --azimuth-threshold-deg 20 --elevation-threshold-deg 10
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


# --baseline -> (sub-script module name, extra argv to inject)
BASELINE_TO_MODULE = {
    "so-7b":              ("scripts.bench_test_generate",       []),
    "so-30b":             ("scripts.bench_test_generate_qwen3", []),
    "zero-spatial":       ("scripts.bench_test_generate",       ["--spatial-ablation", "zero"]),
    "zero-spatial-30b":   ("scripts.bench_test_generate_qwen3", ["--spatial-ablation", "zero"]),
    "iv":                 ("scripts.bench_test_generate_iv",    []),
    "neural-iv":          ("scripts.bench_test_generate_iv",    []),
    "af3":                ("scripts.bench_test_generate_af3",   []),
}


def parse_outer_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # --- REQUIRED: which baseline ---
    p.add_argument(
        "--baseline",
        required=True,
        choices=sorted(BASELINE_TO_MODULE.keys()),
        help="Which baseline to bench. 'zero-spatial' == so-7b backbone with "
             "spatial audio zeroed out (ablation). 'iv' vs 'neural-iv' both go to the IV "
             "bench script; the actual branch comes from the ckpt's train_args.json.",
    )

    # --- Checkpoint selection (three forms, any combo; one of them required) ---
    p.add_argument("--run-dir", type=str, default=None,
                    help="Base dir; used as prefix for --checkpoint-tags / --checkpoint-glob.")
    p.add_argument("--checkpoint-tags", nargs="+", default=None,
                    help="Tags (without '_trainable.pt' suffix) under <run-dir>/checkpoints/.")
    p.add_argument("--checkpoint-paths", nargs="+", default=None,
                    help="Explicit .pt paths; overrides --run-dir selection.")
    p.add_argument("--checkpoint-glob", type=str, default=None,
                    help="Glob under <run-dir>/checkpoints/, e.g. 'step_0*_trainable.pt'.")

    # --- Data ---
    p.add_argument("--qa-root", type=str, required=True,
                    help="QA root containing <split>.jsonl.")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--max-samples", type=int, default=None,
                    help="Cap on QA records (smoke-test knob).")
    p.add_argument("--task-names", nargs="+", default=None,
                    help="Filter to these task_name values. Default: all.")
    p.add_argument("--question-classes", nargs="+", default=None)

    # --- Caching ---
    p.add_argument("--audio-feature-cache-manifest", type=str, default=None)
    p.add_argument("--audio-feature-cache-max-entries", type=int, default=256)

    # --- Output ---
    p.add_argument("--output-dir", type=str, default=None,
                    help="Where to write <ckpt>/predictions.jsonl. "
                         "Defaults to <run-dir>/bench/<split>/ inside the sub-script.")
    p.add_argument("--skip-existing", action="store_true",
                    help="Skip checkpoints whose predictions.jsonl already exists.")

    # --- Inference config (common subset across all three sub-scripts) ---
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--persistent-workers", action="store_true")
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--device-map", type=str, default=None,
                    help="HF device_map (e.g. 'auto') for the BEATs path. Ignored by IV/AF3.")
    p.add_argument("--dtype", type=str, default="bfloat16",
                    choices=("float32", "bfloat16", "float16"))
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--num-beams", type=int, default=4)
    p.add_argument("--do-sample", action="store_true")
    p.add_argument("--mono-audio-zero-spatial-tokens", action="store_true",
                    help="[Spatial-Omni only] Compatibility mode for mono "
                         "benchmarks such as MMAU: feed real mono audio to "
                         "the original audio encoder and pass all-zero "
                         "spatial_tokens with the correct length.")
    p.add_argument("--mono-audio-w-channel-spatial-encoder", action="store_true",
                    help="[Spatial-Omni only] Diagnostic mode for mono/stereo "
                         "benchmarks such as MMAU: downmix input audio to mono, "
                         "place it in the FOA W channel, set X/Y/Z to zero, "
                         "and run the normal spatial encoder/projector path.")

    # --- IV-only but we mirror permissively; BEATs/AF3 will reject if passed ---
    p.add_argument("--attn-impl", type=str, default=None,
                    choices=("auto", "flash_attention_2", "sdpa", "eager"),
                    help="[IV / Neural-IV only] Attention implementation. "
                         "Ignored by so-7b / af3.")

    # --- DDP ---
    p.add_argument("--local-rank", type=int, default=-1)
    return p.parse_args()


# Subset of outer args that are universally supported by all three sub-scripts.
# The special-case keys are handled in build_sub_argv().
_UNIVERSAL_KEYS = {
    "run_dir", "checkpoint_tags", "checkpoint_paths", "checkpoint_glob",
    "qa_root", "split", "max_samples", "task_names", "question_classes",
    "audio_feature_cache_manifest", "audio_feature_cache_max_entries",
    "output_dir", "skip_existing",
    "batch_size", "num_workers", "persistent_workers", "prefetch_factor",
    "device", "dtype", "max_new_tokens", "num_beams", "do_sample",
    "local_rank",
}
# Sub-script-specific keys: only forwarded when the target script supports them.
_SCRIPT_ACCEPTS = {
    "scripts.bench_test_generate":     {
        "device_map",
        "mono_audio_zero_spatial_tokens",
        "mono_audio_w_channel_spatial_encoder",
    },  # BEATs
    "scripts.bench_test_generate_iv":  {"attn_impl", "device_map"},    # IV
    "scripts.bench_test_generate_af3": {
        "mono_audio_zero_spatial_tokens",
        "mono_audio_w_channel_spatial_encoder",
    },
}


def _flag_of(key: str) -> str:
    return "--" + key.replace("_", "-")


def build_sub_argv(outer: argparse.Namespace, module_name: str, extra: list) -> list:
    """Translate outer CLI -> sub-script CLI. Drops None / False flags; forwards
    sub-script-specific keys only when the target script accepts them so that
    the sub-script's argparse does not choke on unknown flags.
    """
    accepted_extras = _SCRIPT_ACCEPTS.get(module_name, set())
    argv: list = []
    for key, value in vars(outer).items():
        if key == "baseline":
            continue
        if value is None or value is False:
            continue
        if key not in _UNIVERSAL_KEYS and key not in accepted_extras:
            # User passed a sub-script-specific flag to the "wrong" baseline;
            # silently drop instead of forwarding. Warn rank-0 so it's visible.
            if value not in (None, False, [], (), ""):
                local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
                if local_rank == 0:
                    print(
                        f"[run_bench] warning: --{key.replace('_','-')} is not "
                        f"applicable to baseline via {module_name}; ignoring.",
                        file=sys.stderr,
                        flush=True,
                    )
            continue
        flag = _flag_of(key)
        if value is True:
            argv.append(flag)
        elif isinstance(value, (list, tuple)):
            argv.append(flag)
            argv.extend(str(x) for x in value)
        else:
            argv.extend([flag, str(value)])
    argv.extend(extra)
    return argv


def main() -> int:
    outer = parse_outer_args()

    # At least one form of checkpoint selection must be present (delegated to the
    # sub-script's own resolver, but fail fast here with a clearer message).
    if not (outer.checkpoint_tags or outer.checkpoint_paths or outer.checkpoint_glob):
        print(
            "[run_bench] ERROR: provide at least one of --checkpoint-paths, "
            "--checkpoint-tags, or --checkpoint-glob.",
            file=sys.stderr,
        )
        return 2

    module_name, extra = BASELINE_TO_MODULE[outer.baseline]
    sub_argv = build_sub_argv(outer, module_name, extra)

    # Rewrite sys.argv so the sub-script's parse_args() sees the translated CLI.
    sub = importlib.import_module(module_name)
    # argv[0] is conventionally the program name; use the sub-script's filename
    # so any error messages reference the actual script the user would have run.
    sys.argv = [module_name.rsplit(".", 1)[-1] + ".py", *sub_argv]

    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        print(f"[run_bench] baseline={outer.baseline} -> {module_name}.main()",
              file=sys.stderr, flush=True)
        print(f"[run_bench] argv = {' '.join(sys.argv[1:])}",
              file=sys.stderr, flush=True)

    return sub.main()


if __name__ == "__main__":
    sys.exit(main())
