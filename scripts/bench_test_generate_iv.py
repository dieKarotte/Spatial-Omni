"""Generate test-split predictions for Spatial-Omni IV / Neural-IV checkpoints.

Why this file exists (vs bench_test_generate.py):
    bench_test_generate.py reuses `scripts/batch_bench_so_qa.py`,
    which imports `build_model` from `train_so_qa.py`. That path
    is the **BEATs** encoder and the model class/config fields are different
    from the IV / Neural-IV baselines, so it cannot load IV checkpoints
    (spatial_encoder_type=iv|neural_iv). This script is the IV sibling:
    same output schema (`predictions.jsonl`), same CLI, but it builds the
    model through `train_spatial_iv_qa.py`.

Generation-only; scoring stays in scripts/score_test_predictions.py.

Usage (single checkpoint):
    torchrun --nproc_per_node=8 scripts/bench_test_generate_iv.py \\
        --checkpoint-paths runs/iv/stage2_encoder_lora/checkpoints/best_trainable.pt \\
        --qa-root /path/to/SO-Dataset/qa --split test \\
        --output-dir runs/iv/stage2_encoder_lora/bench/test

Multiple checkpoints / globbing behaves the same as bench_test_generate.py.
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

# Generic helpers — these do NOT depend on the BEATs vs IV path.
from scripts.batch_bench_so_qa import (  # type: ignore  # noqa: E402
    clean_generated_answer,
    filter_dataset,
    finalize_distributed_prediction_file,
    get_model_device,
    resolve_checkpoint_paths,
    to_generation_inputs,
)

# IV-path model builder. Importing the train module as a whole runs its
# argparse at import time if you `python train_spatial_iv_qa.py ...` but
# as a library import it's fine.
from train_spatial_iv_qa import (  # type: ignore  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    DEFAULT_QA_ROOT,
    DEFAULT_SO_REPO,
    QwenAudioFeatureCache,
    SpatialBeatsQACollator,
    apply_llm_lora,
    build_model,
    build_processor,
    build_qa_dataset,
    cleanup_distributed,
    configure_encoder_lora_training,
    distributed_barrier,
    freeze_all_but_projector,
    get_rank,
    is_main_process,
    make_loader,
    normalize_answer,
    rank0_print,
    setup_distributed,
    shard_dataset_for_rank,
    unwrap_model,
)


# --------------------------------------------------------------------------- #
# IV-specific model instantiation                                             #
# --------------------------------------------------------------------------- #

def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def infer_train_args_path(checkpoint_path: str) -> str:
    """`<run_dir>/checkpoints/foo_trainable.pt` → `<run_dir>/train_args.json`."""
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(checkpoint_path)))
    path = os.path.join(run_dir, "train_args.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"train_args.json not found for checkpoint: {checkpoint_path} "
            f"(expected at {path}). The IV training script writes this at "
            "the start of training; make sure you're pointing at a completed "
            "run dir, not just a bare checkpoint."
        )
    return path


# IV-specific fields we expect in train_args.json. If the user's runs were
# done with an older training script that didn't dump all of these, we fall
# back to sensible defaults that match the launch shell.
_IV_DEFAULTS: Dict[str, Any] = {
    "spatial_encoder_type": "iv",
    "iv_token_dim": 256,
    "iv_projector_hidden_dim": 512,
    "iv_num_mel_bins": 64,
    "iv_band_pool": 0,
    "iv_output_scale": 0.02,
    "iv_feature_to_seld_ratio": 5,
    "iv_downsample_factor": 4,
    "neural_iv_hidden_channels": 64,
    "baseline_repo_path": "${DCASE_BASELINE_REPO}",
    "seld_feature_stats_dir": "${SELD_FEATURE_STATS_DIR}",
    "train_mode": "projector_only",
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "lora_target_prefixes": ["thinker.model"],
    "dtype": "bfloat16",
    "attn_impl": "auto",
    "iv_modules_fp32": False,
    "so_repo": DEFAULT_SO_REPO,
}


def build_eval_model_args(runtime_args: argparse.Namespace,
                          train_args: Dict[str, Any]) -> argparse.Namespace:
    merged = dict(_IV_DEFAULTS)
    merged.update({k: v for k, v in train_args.items() if v is not None})
    # Runtime-only knobs (device / attn impl) may be overridden via CLI.
    # CRITICAL: always forward runtime_args.device so DDP ranks each land on
    # their own cuda:{LOCAL_RANK}. train_args.json stores the training-time
    # device (usually "cuda:0"), which would otherwise pile all 8 ranks onto
    # cuda:0 and OOM.
    merged["device"] = runtime_args.device
    if getattr(runtime_args, "device_map", None):
        merged["device_map"] = runtime_args.device_map
    if getattr(runtime_args, "attn_impl", None):
        merged["attn_impl"] = runtime_args.attn_impl
    merged["dtype"] = runtime_args.dtype
    merged.setdefault("model_id", train_args.get("model_id")
                      or "Qwen/Qwen2.5-Omni-7B")
    return argparse.Namespace(**merged)


def instantiate_iv_model_for_checkpoint(runtime_args: argparse.Namespace,
                                        checkpoint_path: str):
    train_args = _load_json(infer_train_args_path(checkpoint_path))
    model_args = build_eval_model_args(runtime_args, train_args)
    processor = build_processor(model_args.model_id, model_args.so_repo)
    processor.tokenizer.padding_side = "left"
    model = build_model(model_args, processor)

    train_mode = str(model_args.train_mode)
    if train_mode == "projector_only":
        freeze_all_but_projector(model)
    elif train_mode == "encoder_lora":
        model, _ = apply_llm_lora(model, model_args)
        configure_encoder_lora_training(model, model_args)
    else:
        raise ValueError(
            f"Unsupported train_mode for IV baseline: {train_mode!r} "
            f"(expected projector_only or encoder_lora)"
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    from spatial_omni.utils.ckpt_compat import remap_legacy_state_dict
    state_dict = checkpoint.get("trainable_state_dict", checkpoint)
    state_dict = remap_legacy_state_dict(state_dict)
    load_result = model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, processor, train_args, checkpoint, load_result


# --------------------------------------------------------------------------- #
# Generation loop (same schema as bench_test_generate.py, no ablation)         #
# --------------------------------------------------------------------------- #

def run_generation_bench(
    model,
    processor,
    loader,
    output_jsonl_path: str,
    max_new_tokens: int,
    num_beams: int,
    do_sample: bool,
    bench_name: str,
) -> Dict[str, Any]:
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
                        "prediction": prediction_text,
                        "prediction_cleaned": cleaned_prediction,
                        "raw_exact_match": raw_em,
                        "cleaned_exact_match": cln_em,
                    }
                    local_records.append(record)
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
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
    }


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Checkpoint selection
    p.add_argument("--run-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--checkpoint-tags", nargs="+", default=None)
    p.add_argument("--checkpoint-paths", nargs="+", default=None)
    p.add_argument("--checkpoint-glob", type=str, default=None)
    # Data
    p.add_argument("--qa-root", type=str, default=DEFAULT_QA_ROOT)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--task-names", nargs="+", default=None)
    p.add_argument("--question-classes", nargs="+", default=None)
    # Caching
    p.add_argument("--audio-feature-cache-manifest", type=str, default=None)
    p.add_argument("--audio-feature-cache-max-entries", type=int, default=256)
    # Output
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--skip-existing", action="store_true")
    # Inference config
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--persistent-workers", action="store_true")
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--device-map", type=str, default=None)
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=("float32", "bfloat16", "float16"))
    p.add_argument("--attn-impl", type=str, default="auto",
                   choices=("auto", "flash_attention_2", "sdpa", "eager"))
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--num-beams", type=int, default=1)
    p.add_argument("--do-sample", action="store_true")
    # Decoder-only ablation: drop the Qwen mono <|AUDIO|> branch entirely.
    # Spatial encoder still runs and emits real <|spatial|> tokens. Useful
    # to measure how much QA accuracy comes from Qwen's untrained audio_tower
    # vs the trained spatial encoder. Output dir auto-suffixes with
    # __drop_mono_audio so it doesn't overwrite the joint baseline.
    p.add_argument("--drop-mono-audio", action="store_true",
                   help="Decoder-only baseline: only spatial tokens drive "
                        "the LLM, no <|AUDIO|> placeholder, no Qwen mono "
                        "audio_tower forward.")
    # DDP
    p.add_argument("--local-rank", type=int, default=-1)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args = setup_distributed(args)

    checkpoint_paths = resolve_checkpoint_paths(args)
    audio_feature_cache: Optional[QwenAudioFeatureCache] = None
    if args.audio_feature_cache_manifest:
        audio_feature_cache = QwenAudioFeatureCache(
            manifest_path=args.audio_feature_cache_manifest,
            max_entries=args.audio_feature_cache_max_entries,
        )
        rank0_print(
            f"[bench-iv] audio feature cache: {audio_feature_cache.manifest_path} "
            f"(entries={len(audio_feature_cache):,})"
        )

    dataset, _, _ = build_qa_dataset([args.qa_root], args.split, args.max_samples)
    dataset = filter_dataset(dataset, args.task_names, args.question_classes)
    dataset = shard_dataset_for_rank(dataset)
    if len(dataset) == 0:
        raise RuntimeError("Empty dataset after filtering.")

    # If the user passed an explicit --output-dir, trust it as final.
    # Otherwise build a default of <run_dir>/bench/<split> and append
    # __drop_mono_audio suffix in ablation mode.
    if args.output_dir:
        output_dir = os.path.abspath(args.output_dir)
    else:
        output_dir = os.path.abspath(os.path.join(args.run_dir, "bench", args.split))
        if args.drop_mono_audio:
            output_dir = output_dir + "__drop_mono_audio"
    os.makedirs(output_dir, exist_ok=True)
    rank0_print(f"[bench-iv] output_dir={output_dir}")
    rank0_print(f"[bench-iv] drop_mono_audio={args.drop_mono_audio}")
    rank0_print(f"[bench-iv] {len(checkpoint_paths)} checkpoint(s) to run")

    summary: List[Dict[str, Any]] = []

    for checkpoint_path in checkpoint_paths:
        ckpt_name = Path(checkpoint_path).stem.replace("_trainable", "")
        ckpt_out_dir = os.path.join(output_dir, ckpt_name)
        predictions_jsonl = os.path.join(ckpt_out_dir, "predictions.jsonl")

        if args.skip_existing and os.path.exists(predictions_jsonl):
            rank0_print(f"[bench-iv] {ckpt_name}: skip (predictions.jsonl exists)")
            distributed_barrier()
            continue

        rank0_print(f"\n[bench-iv] === {ckpt_name} ===")
        model, processor, train_args, checkpoint, load_result = \
            instantiate_iv_model_for_checkpoint(args, checkpoint_path)
        rank0_print(
            f"[bench-iv] {ckpt_name}: loaded "
            f"missing={len(load_result.missing_keys)} "
            f"unexpected={len(load_result.unexpected_keys)}"
        )

        # Build an eval-only collator by reusing SpatialBeatsQACollator with
        # include_generation_inputs=True. We don't need labels here.
        collator = SpatialBeatsQACollator(
            processor=processor,
            audio_feature_cache=audio_feature_cache,
            include_generation_inputs=True,
            drop_mono_audio=args.drop_mono_audio,
        )
        loader = make_loader(
            dataset=dataset,
            collator=collator,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            sampler=None,
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor,
        )
        quick_metrics = run_generation_bench(
            model=model,
            processor=processor,
            loader=loader,
            output_jsonl_path=predictions_jsonl,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            do_sample=args.do_sample,
            bench_name=f"bench-iv:{ckpt_name}",
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
                "quick_metrics": quick_metrics,
                "train_mode": train_args.get("train_mode"),
                "spatial_encoder_type": train_args.get("spatial_encoder_type"),
                "task_filter": args.task_names,
                "question_class_filter": args.question_classes,
            }
            with open(os.path.join(ckpt_out_dir, "bench_summary.json"),
                      "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True,
                          ensure_ascii=False)
            summary.append(payload)
            rank0_print(
                f"[bench-iv] {ckpt_name}: predictions={quick_metrics.get('examples', 0)} "
                f"raw_em={quick_metrics.get('raw_exact_match', 0.0):.4f} "
                f"cleaned_em={quick_metrics.get('cleaned_exact_match', 0.0):.4f}  "
                f"→ {predictions_jsonl}"
            )
        distributed_barrier()

    if is_main_process() and summary:
        summary_path = os.path.join(output_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True, ensure_ascii=False)
        rank0_print(f"[bench-iv] wrote {summary_path}")
        rank0_print(
            "[bench-iv] Next step: run `scripts/score_test_predictions.py` "
            "on each predictions.jsonl to get task-aware metrics + LLM judge."
        )

    cleanup_distributed()
    return 0


if __name__ == "__main__":
    sys.exit(main())
