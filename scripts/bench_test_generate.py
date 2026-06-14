"""Generate test-split predictions for Spatial-Omni checkpoints.

This script is **generation-only**. It loads one or more trained checkpoints,
runs inference on a QA split (default: `test`), and writes `predictions.jsonl`
per checkpoint. It does NOT compute task-aware metrics — use
`scripts/score_test_predictions.py` separately on the emitted
`predictions.jsonl`.

Why split into two scripts:
    * Inference requires GPUs, heavy dependencies (transformers + PEFT),
      DDP, model-specific collators, and is expensive.
    * Scoring is CPU-only, deterministic, fast, and can optionally call
      the OpenAI-compatible LLM judge. Separating it lets you re-score
      the same predictions with different thresholds / with/without LLM
      judge / on different metric subsets, at zero GPU cost.

Usage (single checkpoint):
    torchrun --nproc_per_node=8 scripts/bench_test_generate.py \\
        --checkpoint-paths runs/so_7b/stage2_encoder_lora/checkpoints/best_trainable.pt \\
        --qa-root /path/to/SO-Dataset/qa \\
        --split test \\
        --batch-size 1 --num-workers 4 \\
        --output-dir runs/so_7b/stage2_encoder_lora/bench/test

Usage (multiple checkpoints):
    torchrun --nproc_per_node=8 scripts/bench_test_generate.py \\
        --run-dir runs/so_7b/stage2_encoder_lora \\
        --checkpoint-glob 'step_01[0-9]000_trainable.pt' \\
        --qa-root /path/to/SO-Dataset/qa --split test

After this emits `predictions.jsonl`, score with:
    python scripts/score_test_predictions.py \\
        --predictions-jsonl .../predictions.jsonl \\
        --qa-root /path/to/SO-Dataset/qa --split test \\
        --llm-judge --llm-concurrency 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tqdm.auto import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Reuse the batch_bench module for model loading, collator, and generation
# loop — those pieces already work and match the train-time pipeline exactly.
from scripts.batch_bench_so_qa import (  # type: ignore  # noqa: E402
    SpatialBeatsEvalCollator,
    clean_generated_answer,
    filter_dataset,
    finalize_distributed_prediction_file,
    get_model_device,
    instantiate_model_for_checkpoint,
    resolve_checkpoint_paths,
    to_generation_inputs,
)
from train_so_qa import (  # type: ignore  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    DEFAULT_QA_ROOT,
    QwenAudioFeatureCache,
    build_qa_dataset,
    cleanup_distributed,
    distributed_barrier,
    get_rank,
    is_distributed,
    is_main_process,
    make_loader,
    normalize_answer,
    rank0_print,
    setup_distributed,
    shard_dataset_for_rank,
    unwrap_model,
)


# --------------------------------------------------------------------------- #
# Inference loop with optional ablation hooks
# --------------------------------------------------------------------------- #

SPATIAL_KEYS = ("spatial_audio",
                "spatial_audio_attention_mask",
                "spatial_audio_lengths",
                "seld_features",
                "seld_feature_attention_mask",
                "seld_feature_lengths",
                "seld_hidden_states",
                "seld_hidden_attention_mask",
                "seld_hidden_lengths",
                "spatial_tokens")
SPATIAL_KEYS = SPATIAL_KEYS + ("projected_spatial_tokens",)


def _apply_spatial_ablation(inputs: Dict[str, torch.Tensor], mode: str) -> Dict[str, torch.Tensor]:
    """Mutate the generation inputs to ablate the spatial branch.

    mode:
        "none"   - no change.
        "zero"   - keep all spatial-audio shape info (lengths / attn mask /
                   feature presence) so the thinker still dispatches to the
                   spatial pathway, but zero out the raw waveform / feature
                   content. This measures "does the spatial encoder content
                   matter".
        "noise"  - replace spatial_audio with unit-std Gaussian noise of the
                   same shape/dtype/device. Same attention mask / length.
                   This tells you "does the model use the spatial signal
                   or any plausible 4-channel input".
    """
    if mode == "none":
        return inputs
    if "spatial_audio" in inputs and isinstance(inputs["spatial_audio"], torch.Tensor):
        t = inputs["spatial_audio"]
        if mode == "zero":
            inputs["spatial_audio"] = torch.zeros_like(t)
        elif mode == "noise":
            inputs["spatial_audio"] = torch.randn_like(t)
        else:
            raise ValueError(f"Unknown ablation mode: {mode}")
    # If cached seld features are provided, zero / randomize them too.
    if "seld_features" in inputs and isinstance(inputs["seld_features"], torch.Tensor):
        t = inputs["seld_features"]
        if mode == "zero":
            inputs["seld_features"] = torch.zeros_like(t)
        else:
            inputs["seld_features"] = torch.randn_like(t)
    if "seld_hidden_states" in inputs and isinstance(inputs["seld_hidden_states"], torch.Tensor):
        t = inputs["seld_hidden_states"]
        if mode == "zero":
            inputs["seld_hidden_states"] = torch.zeros_like(t)
        else:
            inputs["seld_hidden_states"] = torch.randn_like(t)
    if "spatial_tokens" in inputs and isinstance(inputs["spatial_tokens"], torch.Tensor):
        t = inputs["spatial_tokens"]
        if mode == "zero":
            inputs["spatial_tokens"] = torch.zeros_like(t)
        else:
            inputs["spatial_tokens"] = torch.randn_like(t)
    if "projected_spatial_tokens" in inputs and isinstance(inputs["projected_spatial_tokens"], torch.Tensor):
        t = inputs["projected_spatial_tokens"]
        if mode == "zero":
            inputs["projected_spatial_tokens"] = torch.zeros_like(t)
        else:
            inputs["projected_spatial_tokens"] = torch.randn_like(t)
    return inputs


def run_generation_bench_with_ablation(
    model,
    processor,
    loader,
    output_jsonl_path: str,
    max_new_tokens: int,
    num_beams: int,
    do_sample: bool,
    bench_name: str,
    spatial_ablation: str = "none",
) -> Dict[str, Any]:
    """Inference loop matching batch_bench's `run_generation_bench` but with
    an optional per-batch spatial-ablation hook applied right before
    `model.generate(...)`.
    """
    model.eval()
    local_records: List[Dict[str, Any]] = []
    rank = get_rank()
    shard_output_path = f"{output_jsonl_path}.rank{rank}.jsonl"
    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)
    eval_model = unwrap_model(model)
    input_device = get_model_device(eval_model)

    with open(shard_output_path, "w", encoding="utf-8") as handle:
        with torch.no_grad():
            progress = tqdm(loader, desc=bench_name, leave=False,
                            disable=not is_main_process())
            for step_i, batch in enumerate(progress):
                generation_inputs = to_generation_inputs(batch, input_device)
                generation_inputs = _apply_spatial_ablation(generation_inputs, spatial_ablation)
                generated = eval_model.generate(
                    **generation_inputs,
                    return_audio=False,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    do_sample=do_sample,
                )
                ml = generation_inputs["input_ids"].shape[1]
                generated = generated.detach().cpu()
                for index in range(len(batch["meta"])):
                    prediction_ids = generated[index, ml:]
                    prediction_text = processor.tokenizer.decode(
                        prediction_ids, skip_special_tokens=True).strip()
                    cleaned_prediction = clean_generated_answer(prediction_text)
                    meta = batch["meta"][index]
                    answer_text = str(meta["answer"]).strip()
                    cleaned_answer = clean_generated_answer(answer_text)
                    raw_em = int(normalize_answer(prediction_text) == normalize_answer(answer_text))
                    cln_em = int(normalize_answer(cleaned_prediction) == normalize_answer(cleaned_answer))
                    record = {
                        "pair_id": meta.get("pair_id"),
                        "task_name": meta.get("task_name"),
                        "question": meta.get("question"),
                        "prompt": meta.get("prompt"),
                        "answer": answer_text,
                        "audio_path": meta.get("audio_path"),
                        "scene_id": meta.get("scene_id"),
                        "segment_stem": meta.get("segment_stem"),
                        "canonical_answer": meta.get("canonical_answer"),
                        "question_class": meta.get("question_class"),
                        "answer_meta": meta.get("answer_meta"),
                        "prediction": prediction_text,
                        "prediction_cleaned": cleaned_prediction,
                        "raw_exact_match": raw_em,
                        "cleaned_exact_match": cln_em,
                        "spatial_ablation": spatial_ablation,
                    }
                    local_records.append(record)
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                # Free generated + generation_inputs before the next batch
                # to reduce fragmentation. Empty cache every 50 batches so
                # long-running bench jobs don't accumulate KV-cache leftovers
                # that later trigger "Failed to CUDA calloc N bytes" in NCCL.
                del generated, generation_inputs
                if (step_i + 1) % 50 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

    distributed_barrier()
    if not is_main_process():
        return {}
    merged = finalize_distributed_prediction_file(output_jsonl_path)
    total = max(len(merged), 1)
    raw_em = sum(float(r["raw_exact_match"]) for r in merged) / total
    cln_em = sum(float(r["cleaned_exact_match"]) for r in merged) / total
    return {
        "examples": len(merged),
        "raw_exact_match": raw_em,
        "cleaned_exact_match": cln_em,
        "spatial_ablation": spatial_ablation,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Checkpoint selection (same options as batch_bench, one of them required).
    p.add_argument("--run-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                    help="Only used as prefix for --checkpoint-tags / --checkpoint-glob.")
    p.add_argument("--checkpoint-tags", nargs="+", default=None,
                    help="Tags (without the _trainable.pt suffix) under <run-dir>/checkpoints/.")
    p.add_argument("--checkpoint-paths", nargs="+", default=None,
                    help="Explicit .pt paths; overrides --run-dir selection.")
    p.add_argument("--checkpoint-glob", type=str, default=None,
                    help="Glob under <run-dir>/checkpoints/, e.g. 'step_0*_trainable.pt'.")

    # Data.
    p.add_argument("--qa-root", type=str, default=DEFAULT_QA_ROOT,
                    help="QA root containing <split>.jsonl.")
    p.add_argument("--audio-root", type=str, default=None,
                    help="Optional audio root prefix. Use when audio_path in "
                         "the QA jsonl is relative to a different root than "
                         "qa-root (e.g. SO-Dataset HF release).")
    p.add_argument("--audio-roots", nargs="+", default=None,
                    help="Multiple audio search roots; takes precedence over --audio-root.")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--max-samples", type=int, default=None,
                    help="Cap on QA records loaded (smoke-test knob).")
    p.add_argument("--task-names", nargs="+", default=None,
                    help="Filter to these task_name values. Default: all.")
    p.add_argument("--question-classes", nargs="+", default=None)

    # Caching (speeds up dataloader).
    p.add_argument("--audio-feature-cache-manifest", type=str, default=None)
    p.add_argument("--audio-feature-cache-max-entries", type=int, default=256)

    # Output.
    p.add_argument("--output-dir", type=str, default=None,
                    help="Where to write <ckpt>/predictions.jsonl. "
                         "Defaults to <run-dir>/bench/<split>/.")
    p.add_argument("--skip-existing", action="store_true",
                    help="Skip checkpoints whose predictions.jsonl already exists.")

    # Inference config.
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--persistent-workers", action="store_true")
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--device-map", type=str, default=None,
                    help="HF device_map (e.g. 'auto') to shard one big model across GPUs. "
                         "When set, --device is still used for the input tensors.")
    p.add_argument("--dtype", type=str, default="bfloat16",
                    choices=("float32", "bfloat16", "float16"))
    p.add_argument("--max-new-tokens", type=int, default=96,
                    help="Detect-source answers can be long; bump this if you see truncation.")
    p.add_argument("--num-beams", type=int, default=1)
    p.add_argument("--do-sample", action="store_true")

    # Spatial ablation (diagnostic).
    p.add_argument("--spatial-ablation", type=str, default="none",
                    choices=("none", "zero", "noise"),
                    help="Diagnostic: override the spatial input before "
                         "generate(). 'zero' replaces spatial_audio with zeros "
                         "(keeping attention mask / lengths), 'noise' replaces "
                         "with unit-std Gaussian. Use this to test whether the "
                         "model actually uses the spatial branch. "
                         "Output dir auto-suffixes with __<mode> to keep "
                         "predictions separate from the baseline run.")
    p.add_argument("--mono-audio-zero-spatial-tokens", action="store_true",
                    help="MMAU/mono compatibility mode: feed real mono audio "
                         "to the original audio encoder, keep <|spatial|> "
                         "placeholders, and pass correctly-sized all-zero "
                         "spatial_tokens directly instead of requiring FOA.")
    p.add_argument("--mono-audio-w-channel-spatial-encoder", action="store_true",
                    help="MMAU/mono diagnostic mode: downmix mono/stereo audio "
                         "to mono, put it in FOA W, set X/Y/Z to zero, and run "
                         "the normal spatial encoder/projector path.")
    p.add_argument("--drop-mono-audio", action="store_true",
                    help="Decoder-only baseline: drop the Qwen mono <|AUDIO|> "
                         "branch entirely (no audio_token in the prompt, "
                         "no input_features sent to the processor). Spatial "
                         "encoder still runs and emits real <|spatial|> tokens. "
                         "Quantifies how much QA performance comes from the "
                         "(untrained) Qwen audio_tower vs the spatial encoder. "
                         "Output dir auto-suffixes with __drop_mono_audio.")

    # DDP.
    p.add_argument("--local-rank", type=int, default=-1)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.mono_audio_zero_spatial_tokens and args.mono_audio_w_channel_spatial_encoder:
        raise ValueError(
            "--mono-audio-zero-spatial-tokens and "
            "--mono-audio-w-channel-spatial-encoder are mutually exclusive."
        )
    if args.drop_mono_audio and (
        args.mono_audio_zero_spatial_tokens
        or args.mono_audio_w_channel_spatial_encoder
    ):
        raise ValueError(
            "--drop-mono-audio is mutually exclusive with the "
            "--mono-audio-* compat modes."
        )
    args = setup_distributed(args)

    checkpoint_paths = resolve_checkpoint_paths(args)
    audio_feature_cache: Optional[QwenAudioFeatureCache] = None
    if args.audio_feature_cache_manifest and args.mono_audio_w_channel_spatial_encoder:
        rank0_print(
            "[bench] audio feature cache ignored for "
            "--mono-audio-w-channel-spatial-encoder because W-only spatial "
            "audio is synthesized online."
        )
    elif args.audio_feature_cache_manifest:
        audio_feature_cache = QwenAudioFeatureCache(
            manifest_path=args.audio_feature_cache_manifest,
            max_entries=args.audio_feature_cache_max_entries,
        )
        rank0_print(
            f"[bench] audio feature cache: {audio_feature_cache.manifest_path} "
            f"(entries={len(audio_feature_cache):,})"
        )

    # Build the dataset once; it will be reused for every checkpoint.
    audio_search_roots = []
    if args.audio_roots:
        audio_search_roots = [os.path.abspath(r) for r in args.audio_roots]
    elif args.audio_root:
        audio_search_roots = [os.path.abspath(args.audio_root)]
    dataset, _, _ = build_qa_dataset(
        [args.qa_root], args.split, args.max_samples,
        audio_search_roots=audio_search_roots or None,
    )
    dataset = filter_dataset(dataset, args.task_names, args.question_classes)
    dataset = shard_dataset_for_rank(dataset)
    if len(dataset) == 0:
        raise RuntimeError("Empty dataset after filtering.")

    # If --output-dir was passed, trust it (caller handled all suffixing).
    # Otherwise build default and apply ablation suffixes.
    if args.output_dir:
        output_dir = os.path.abspath(args.output_dir)
    else:
        output_dir = os.path.abspath(os.path.join(args.run_dir, "bench", args.split))
        if args.spatial_ablation != "none":
            output_dir = output_dir + f"__ablation_{args.spatial_ablation}"
        if args.drop_mono_audio:
            output_dir = output_dir + "__drop_mono_audio"
    os.makedirs(output_dir, exist_ok=True)
    rank0_print(f"[bench] output_dir={output_dir}")
    rank0_print(f"[bench] spatial_ablation={args.spatial_ablation}")
    rank0_print(f"[bench] drop_mono_audio={args.drop_mono_audio}")
    rank0_print(f"[bench] mono_audio_zero_spatial_tokens={args.mono_audio_zero_spatial_tokens}")
    rank0_print(
        "[bench] mono_audio_w_channel_spatial_encoder="
        f"{args.mono_audio_w_channel_spatial_encoder}"
    )
    rank0_print(f"[bench] {len(checkpoint_paths)} checkpoint(s) to run")

    summary: List[Dict[str, Any]] = []

    for checkpoint_path in checkpoint_paths:
        ckpt_name = Path(checkpoint_path).stem.replace("_trainable", "")
        ckpt_out_dir = os.path.join(output_dir, ckpt_name)
        predictions_jsonl = os.path.join(ckpt_out_dir, "predictions.jsonl")

        if args.skip_existing and os.path.exists(predictions_jsonl):
            rank0_print(f"[bench] {ckpt_name}: skip (predictions.jsonl exists)")
            distributed_barrier()
            continue

        rank0_print(f"\n[bench] === {ckpt_name} ===")
        model, processor, train_args, checkpoint, load_result = \
            instantiate_model_for_checkpoint(args, checkpoint_path)
        rank0_print(
            f"[bench] {ckpt_name}: loaded "
            f"missing={len(load_result.missing_keys)} "
            f"unexpected={len(load_result.unexpected_keys)}"
        )
        thinker_config = getattr(getattr(unwrap_model(model), "thinker"), "config")
        text_config = getattr(thinker_config, "text_config", None)
        zero_projected_spatial_dim = int(
            getattr(text_config, "hidden_size", getattr(thinker_config, "hidden_size", 3584))
        )

        loader = make_loader(
            dataset=dataset,
            collator=SpatialBeatsEvalCollator(
                processor=processor,
                audio_feature_cache=audio_feature_cache,
                mono_audio_zero_spatial_tokens=args.mono_audio_zero_spatial_tokens,
                mono_audio_w_channel_spatial_encoder=args.mono_audio_w_channel_spatial_encoder,
                zero_projected_spatial_dim=zero_projected_spatial_dim,
                drop_mono_audio=args.drop_mono_audio,
            ),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            sampler=None,
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor,
        )
        quick_metrics = run_generation_bench_with_ablation(
            model=model,
            processor=processor,
            loader=loader,
            output_jsonl_path=predictions_jsonl,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            do_sample=args.do_sample,
            bench_name=f"bench:{ckpt_name}[{args.spatial_ablation}]",
            spatial_ablation=args.spatial_ablation,
        )
        distributed_barrier()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if is_main_process():
            payload: Dict[str, Any] = {
                "checkpoint": os.path.abspath(checkpoint_path),
                "checkpoint_epoch": checkpoint.get("epoch"),
                "predictions_jsonl": os.path.abspath(predictions_jsonl),
                "quick_metrics": quick_metrics,  # raw_em / cleaned_em (token-level sanity)
                "train_mode": train_args.get("train_mode"),
                "task_filter": args.task_names,
                "question_class_filter": args.question_classes,
                "mono_audio_zero_spatial_tokens": bool(args.mono_audio_zero_spatial_tokens),
                "mono_audio_w_channel_spatial_encoder": bool(args.mono_audio_w_channel_spatial_encoder),
                "zero_projected_spatial_dim": zero_projected_spatial_dim,
            }
            with open(os.path.join(ckpt_out_dir, "bench_summary.json"),
                        "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True,
                          ensure_ascii=False)
            summary.append(payload)
            rank0_print(
                f"[bench] {ckpt_name}: predictions={quick_metrics.get('examples', 0)} "
                f"raw_em={quick_metrics.get('raw_exact_match', 0.0):.4f} "
                f"cleaned_em={quick_metrics.get('cleaned_exact_match', 0.0):.4f}  "
                f"→ {predictions_jsonl}"
            )
        distributed_barrier()

    if is_main_process() and summary:
        summary_path = os.path.join(output_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True, ensure_ascii=False)
        rank0_print(f"[bench] wrote {summary_path}")
        rank0_print(
            "[bench] Next step: run `scripts/score_test_predictions.py` "
            "on each predictions.jsonl to get task-aware metrics + LLM judge."
        )

    cleanup_distributed()
    return 0


if __name__ == "__main__":
    sys.exit(main())
