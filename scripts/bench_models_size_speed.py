#!/usr/bin/env python3
"""Benchmark parameter count, model size and inference speed for:

    base          : original Qwen2.5-Omni-7B (no spatial branch)
    neural_iv     : Spatial-Omni with Neural-IV spatial encoder
    so_backbone : Spatial-Omni with Spatial-BEATs spatial encoder (default)

Usage
-----
    # all three, single GPU, bf16
    python scripts/bench_models_size_speed.py \
        --targets base neural_iv so_backbone \
        --model-id /path/to/Qwen2.5-Omni-7B \
        --beats-checkpoint /path/to/so_backbone/best.pt \
        --beats-repo /path/to/unilm/beats \
        --output-json bench_size_speed.json

    # only one target
    python scripts/bench_models_size_speed.py --targets so_backbone ...

Numbers reported per target
---------------------------
* params : total / trainable / per-submodule (audio_tower, spatial encoder,
           projector, thinker LLM, talker if present, vision)
* size_mb: dtype-aware bytes summed over .parameters() and .buffers()
* speed  : prefill latency (ms), decode tokens/s, total generate latency
           (mean ± std over `--n-trials`, after `--n-warmup` warmups)

The synthetic input is a 20 s zero-FOA waveform + a fixed text prompt. We do
NOT use real audio because we only care about the per-token compute pattern,
not output quality.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

SAMPLE_RATE = 16_000
MAX_SECONDS = 20
MAX_SAMPLES = SAMPLE_RATE * MAX_SECONDS


# ---------------------------------------------------------------------------
# Param / size accounting
# ---------------------------------------------------------------------------

_SUBMODULE_PROBES = [
    # name → list of attribute paths to try (first hit wins)
    ("thinker_llm", ["thinker.model", "model"]),
    ("audio_tower", ["thinker.audio_tower", "audio_tower"]),
    ("visual",      ["thinker.visual", "visual"]),
    ("talker",      ["talker"]),
    ("token2wav",   ["token2wav"]),
    ("so_encoder",   ["thinker.so_encoder"]),
    ("so_projector", ["thinker.so_projector"]),
    ("spatial_iv_adapter",      ["thinker.spatial_iv_adapter"]),
    ("spatial_iv_projector",    ["thinker.spatial_iv_projector"]),
    ("spatial_neural_iv_adapter",   ["thinker.spatial_neural_iv_adapter"]),
    ("spatial_neural_iv_projector", ["thinker.spatial_neural_iv_projector"]),
    ("seld_backbone",          ["thinker.seld_spatial_backbone"]),
    ("seld_feature_bridge",    ["thinker.seld_feature_bridge"]),
    ("seld_spatial_adapter",   ["thinker.seld_spatial_adapter"]),
    ("seld_spatial_projector", ["thinker.seld_spatial_projector"]),
]


def _resolve_attr(model, dotted: str):
    obj = model
    for part in dotted.split("."):
        if not hasattr(obj, part):
            return None
        obj = getattr(obj, part)
    return obj


def count_module_params(module: torch.nn.Module) -> Dict[str, int]:
    if module is None:
        return {"total": 0, "trainable": 0}
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}


def module_size_bytes(module: torch.nn.Module) -> int:
    """Sum bytes of parameters + buffers, dtype-aware."""
    if module is None:
        return 0
    total = 0
    for t in list(module.parameters()) + list(module.buffers()):
        total += t.numel() * t.element_size()
    return int(total)


def gather_param_breakdown(model: torch.nn.Module) -> Dict[str, Dict[str, int]]:
    out: "OrderedDict[str, Dict[str, int]]" = OrderedDict()
    out["model_total"] = count_module_params(model)
    out["model_total"]["size_bytes"] = module_size_bytes(model)
    seen_ids = set()  # avoid double-counting if probes alias each other

    for name, candidates in _SUBMODULE_PROBES:
        sub = None
        for path in candidates:
            sub = _resolve_attr(model, path)
            if sub is not None:
                break
        if sub is None:
            continue
        sub_id = id(sub)
        if sub_id in seen_ids:
            continue
        seen_ids.add(sub_id)
        info = count_module_params(sub)
        info["size_bytes"] = module_size_bytes(sub)
        out[name] = info
    return out


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def load_base_qwen(args) -> "tuple[Any, Any, Dict[str, Any]]":
    """Load stock Qwen2.5-Omni without the spatial branch."""
    from transformers import Qwen2_5OmniProcessor
    # Use the project-local subclass so our processor codepath is exercised.
    from spatial_omni.model.modeling_qwen2_5_omni import (
        Qwen2_5OmniForConditionalGeneration,
    )
    dtype = getattr(torch, args.dtype)
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_id)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        attn_implementation=args.attn_impl,
        low_cpu_mem_usage=True,
    )
    # Match the spatial branches: disable the Talker (token2wav speech head)
    # so generate() only runs the Thinker LLM. Without this the base path
    # also runs the speech synthesis pipeline and end-to-end latency is
    # dominated by token2wav, not by the LLM itself.
    if hasattr(model, "disable_talker"):
        model.disable_talker()
    return model, processor, {"target": "base"}


def _make_iv_args(args, encoder_type: str) -> argparse.Namespace:
    return argparse.Namespace(
        model_id=args.model_id,
        so_repo=REPO_ROOT,
        spatial_encoder_type=encoder_type,
        iv_token_dim=256,
        iv_projector_hidden_dim=512,
        iv_num_mel_bins=64,
        iv_band_pool=0,
        iv_output_scale=0.02,
        iv_feature_to_seld_ratio=5,
        iv_downsample_factor=4,
        neural_iv_hidden_channels=64,
        baseline_repo_path=args.baseline_repo,
        seld_feature_stats_dir=args.seld_feature_stats_dir,
        train_mode="projector_only",
        lora_r=16, lora_alpha=32, lora_dropout=0.05,
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_target_prefixes=["thinker.model"],
        dtype=args.dtype, attn_impl=args.attn_impl,
        device=args.device, device_map=None,
        gradient_checkpointing=False,
        iv_modules_fp32=False,
        projector_fp32=False,
    )


def load_neural_iv(args):
    from train_spatial_iv_qa import build_model, build_processor  # type: ignore
    iv_args = _make_iv_args(args, encoder_type="neural_iv")
    processor = build_processor(args.model_id, REPO_ROOT)
    processor.tokenizer.padding_side = "left"
    model = build_model(iv_args, processor)
    return model, processor, {"target": "neural_iv"}


def load_so_backbone(args):
    from train_so_qa import build_model, build_processor  # type: ignore
    beats_args = argparse.Namespace(
        model_id=args.model_id,
        so_repo=REPO_ROOT,
        beats_checkpoint=args.beats_checkpoint,
        beats_repo=args.beats_repo,
        train_mode="encoder_lora",
        lora_r=16, lora_alpha=32, lora_dropout=0.05,
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_target_prefixes=["thinker.model"],
        dtype=args.dtype, attn_impl=args.attn_impl,
        device=args.device, device_map=None,
        gradient_checkpointing=False,
        projector_fp32=False,
        projector_type="pixel_shuffle",
        projector_shuffle_factor=4,
        encoder_token_rate=10.0,
        mixed_spatial_replay=False,
    )
    processor = build_processor(args.model_id, REPO_ROOT)
    processor.tokenizer.padding_side = "left"
    model = build_model(beats_args, processor)
    return model, processor, {"target": "so_backbone"}


LOADERS = {
    "base": load_base_qwen,
    "neural_iv": load_neural_iv,
    "so_backbone": load_so_backbone,
}


# ---------------------------------------------------------------------------
# Synthetic input + speed benchmark
# ---------------------------------------------------------------------------

def build_synthetic_inputs(target: str, processor, device, dtype) -> Dict[str, Any]:
    """Build a single-sample batch with a fixed prompt + 20 s zero FOA wave.

    All three processors expose the same `audio=` kwarg (a list of arrays).
    The spatial processor requires shape[0] in {1, 4} per sample (mono vs FOA);
    the stock Qwen processor accepts 1-D mono arrays.
    """
    n = MAX_SAMPLES  # 20 s × 16 kHz
    if target == "base":
        text = "<|audio_bos|><|AUDIO|><|audio_eos|>\nWhat is happening in this audio?\n"
        mono = np.zeros((n,), dtype=np.float32)
        batch = processor(
            text=[text], audio=[mono], sampling_rate=SAMPLE_RATE,
            return_tensors="pt", padding=True,
        )
    else:
        # Spatial-aware processor: feed 4-channel FOA via the same `audio=`
        # kwarg. Shape per sample must be [C, T] with C=4.
        text = "<|audio|><|spatial|>\nWhere is the source located?\n"
        foa = np.zeros((4, n), dtype=np.float32)
        batch = processor(
            text=[text], audio=[foa], sampling_rate=SAMPLE_RATE,
            return_tensors="pt", padding=True,
        )

    # Move tensors to device. `spatial_audio` is the raw FOA waveform and
    # gets STFT'd inside the spatial encoder; cuFFT does not support bf16,
    # so we keep waveform tensors in fp32. Other float tensors (e.g.
    # `input_features` mel) are cast to the model's compute dtype.
    waveform_keys = {"spatial_audio"}
    moved: Dict[str, Any] = {}
    for k, v in batch.items():
        if not torch.is_tensor(v):
            moved[k] = v
            continue
        if v.is_floating_point() and k not in waveform_keys:
            moved[k] = v.to(device=device, dtype=dtype)
        else:
            moved[k] = v.to(device=device)
    return moved


@torch.inference_mode()
def time_generate(model, batch, max_new_tokens: int,
                  n_warmup: int, n_trials: int) -> Dict[str, float]:
    """Measure end-to-end thinker.generate latency (s) and peak GPU memory (GB).

    We deliberately call `model.thinker.generate(...)` (NOT the top-level
    `Qwen2_5OmniForConditionalGeneration.generate`). The top-level generate
    runs Thinker → Talker → token2wav for speech synthesis; even with
    `disable_talker()` and `return_audio=False`, some HF builds still invoke
    chunked token2wav and the latency is dominated by speech synthesis, not
    by the LLM. Spatial-Omni branches train and bench against `thinker.generate`
    directly, so this is also what we want to compare for the LLM benchmark.

    Peak memory is measured *only* during the timed trials (not warmup), with
    `torch.cuda.reset_peak_memory_stats()` reset before each trial and the max
    over trials reported. Latency is mean ± std over trials.
    """
    # Resolve the actual generation target. Both the stock Qwen2.5-Omni and
    # the spatial subclasses expose `.thinker`; fall back to top-level
    # `.generate` only if it's missing.
    gen_target = getattr(model, "thinker", model)

    use_cuda = next(model.parameters()).is_cuda
    device = next(model.parameters()).device

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        # Force every trial to actually decode `max_new_tokens` steps so we
        # benchmark the same compute volume across targets. With fresh-init
        # spatial projectors (untrained), the model often emits EOS at step 1-2
        # and the timed `.generate()` returns early — making neural_iv look
        # 5x faster than base purely because it stopped sooner. `min_new_tokens
        # == max_new_tokens` plus `eos_token_id=None` pin the loop length.
        min_new_tokens=max_new_tokens,
        eos_token_id=None,
        do_sample=False,
        num_beams=1,
        use_cache=True,
    )

    if use_cuda:
        torch.cuda.synchronize()

    # Warmup (CUDA kernels, autotune, allocator caches).
    for _ in range(n_warmup):
        _ = gen_target.generate(**batch, **gen_kwargs)
        if use_cuda:
            torch.cuda.synchronize()

    latencies_s: List[float] = []
    peak_bytes_per_trial: List[int] = []
    for _ in range(n_trials):
        if use_cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        _ = gen_target.generate(**batch, **gen_kwargs)
        if use_cuda:
            torch.cuda.synchronize()
        latencies_s.append(time.perf_counter() - t0)
        if use_cuda:
            peak_bytes_per_trial.append(int(torch.cuda.max_memory_allocated(device)))

    lat = np.asarray(latencies_s)
    peak_gb = (max(peak_bytes_per_trial) / (1024 ** 3)) if peak_bytes_per_trial else 0.0

    return {
        "latency_s_mean":   float(lat.mean()),
        "latency_s_std":    float(lat.std()),
        "latency_s_min":    float(lat.min()),
        "peak_gpu_mem_gb":  float(peak_gb),
        "max_new_tokens":   int(max_new_tokens),
        "n_trials":         int(n_trials),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--targets", nargs="+",
                   default=["base", "neural_iv", "so_backbone"],
                   choices=list(LOADERS))
    p.add_argument("--model-id", required=True,
                   help="Qwen2.5-Omni-7B HF dir.")
    p.add_argument("--beats-checkpoint", default=None,
                   help="Required for --targets includes so_backbone.")
    p.add_argument("--beats-repo", default=None,
                   help="Path to unilm/beats; required for so_backbone.")
    p.add_argument("--baseline-repo",
                   default="${DCASE_BASELINE_REPO}")
    p.add_argument("--seld-feature-stats-dir",
                   default="${SELD_FEATURE_STATS_DIR}")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16",
                   choices=("float32", "bfloat16", "float16"))
    p.add_argument("--attn-impl", default="sdpa",
                   choices=("sdpa", "flash_attention_2", "eager"))
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--n-warmup", type=int, default=2)
    p.add_argument("--n-trials", type=int, default=5)
    p.add_argument("--skip-speed", action="store_true",
                   help="Only measure params + size, skip generate timing.")
    p.add_argument("--output-json", default="bench_size_speed.json")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if "so_backbone" in args.targets and (
        args.beats_checkpoint is None or args.beats_repo is None
    ):
        print("[ERROR] --beats-checkpoint and --beats-repo are required when "
              "so_backbone is in --targets.", file=sys.stderr)
        return 1

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    results: Dict[str, Any] = {
        "config": {
            "model_id": args.model_id,
            "device": str(device),
            "dtype": args.dtype,
            "attn_impl": args.attn_impl,
            "max_new_tokens": args.max_new_tokens,
            "n_warmup": args.n_warmup,
            "n_trials": args.n_trials,
        },
        "targets": {},
    }

    for target in args.targets:
        print(f"\n========== {target} ==========", flush=True)
        loader = LOADERS[target]
        model, processor, meta = loader(args)
        model.to(device=device)
        model.eval()

        breakdown = gather_param_breakdown(model)
        total_params = breakdown["model_total"]["total"]
        # Pretty-print sub-module breakdown
        for name, info in breakdown.items():
            mb = info["size_bytes"] / (1024 ** 2)
            print(f"  {name:32s}  total={info['total']/1e6:8.2f} M  "
                  f"trainable={info['trainable']/1e6:8.2f} M  size={mb:8.1f} MB")
        print(f"  >>> total_params = {total_params/1e6:.2f} M "
              f"({total_params/1e9:.3f} B)")

        speed: Optional[Dict[str, float]] = None
        if not args.skip_speed:
            try:
                batch = build_synthetic_inputs(target, processor, device, dtype)
                # Diagnostic: print prompt + audio shapes so we can verify
                # that all targets actually feed comparable workloads to the
                # LLM (the ~500 audio_tower output tokens dominate prefill).
                shape_summary = {
                    k: tuple(v.shape) for k, v in batch.items()
                    if torch.is_tensor(v)
                }
                print(f"  [batch shapes] {shape_summary}")
                speed = time_generate(model, batch,
                                      max_new_tokens=args.max_new_tokens,
                                      n_warmup=args.n_warmup,
                                      n_trials=args.n_trials)
                print(f"  >>> end_to_end_latency = "
                      f"{speed['latency_s_mean']:.3f} ± "
                      f"{speed['latency_s_std']:.3f} s "
                      f"(min {speed['latency_s_min']:.3f} s, "
                      f"{args.max_new_tokens} new tokens)")
                print(f"  >>> peak_inference_gpu_mem = "
                      f"{speed['peak_gpu_mem_gb']:.2f} GB")
            except Exception as exc:  # pragma: no cover
                print(f"  [speed] failed: {exc}")
                speed = {"error": repr(exc)}

        results["targets"][target] = {
            "meta": meta,
            "total_params": total_params,
            "params_breakdown": breakdown,
            "speed": speed,
        }

        # Free for next target
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[ok] wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
