import argparse
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Subset
from tqdm.auto import tqdm

from train_spatial_qa import (
    DEFAULT_MODEL_ID,
    DEFAULT_QA_ROOT,
    DEFAULT_QA_VERSION,
    DEFAULT_LEGACY_REPO,
    MAX_AUDIO_SAMPLES,
    SAMPLE_RATE,
    QA_VERSION_TO_SUBDIR,
    QAAudioJsonlDataset,
    SpatialQACollator,
    add_legacy_repo_to_path,
    apply_llm_lora,
    build_left_padded_text_batch,
    build_generation_inputs,
    cleanup_distributed,
    build_model,
    build_processor,
    configure_spatial_lora_training,
    configure_adapter_lora_training,
    distributed_barrier,
    freeze_all_but_spatial_modules,
    get_rank,
    is_main_process,
    make_loader,
    normalize_answer,
    rank0_print,
    resolve_qa_root,
    resolve_qa_split_path,
    setup_distributed,
    unwrap_model,
)


DEFAULT_RUN_DIR = (
    "${DCASE_BASELINE_REPO}/"
    "spatial_qa_runs/spatial_lora_4gpu_full"
)

DCASE_CORE_CATEGORY_TASKS = {
    "azimuth": [
        "estimate_azimuth",
        "classify_azimuth_bin_text",
        "classify_azimuth_bin_choice",
        "compare_azimuth",
        "same_azimuth",
    ],
    "elevation": [
        "estimate_elevation",
        "classify_elevation_bin_text",
        "classify_elevation_bin_choice",
        "compare_elevation",
    ],
    "distance": [
        "estimate_distance",
        "compare_distance",
    ],
    "motion": [
        "classify_motion",
        "detect_motion",
    ],
    "detect": [
        "detect_time",
        "count_sources",
        "identify_source_by_location",
        "identify_source_by_doa",
        "identify_source_by_doa_distance",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark base Qwen and adapted spatial checkpoints on the QA test split."
    )
    parser.add_argument("--legacy-repo-path", type=str, default=DEFAULT_LEGACY_REPO)
    parser.add_argument("--run-dir", type=str, default=DEFAULT_RUN_DIR)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--checkpoint-tag", type=str, default="best")
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument(
        "--qa-root",
        type=str,
        default=None,
        help="Explicit QA directory containing train/valid/test json or jsonl files. Overrides --qa-version.",
    )
    parser.add_argument(
        "--qa-version",
        type=str,
        default=DEFAULT_QA_VERSION,
        choices=sorted(QA_VERSION_TO_SUBDIR.keys()),
        help="Named QA dataset version under the prepared_datasets root.",
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--category-groups",
        nargs="+",
        default=None,
        choices=sorted(DCASE_CORE_CATEGORY_TASKS.keys()),
        help="Run separate benchmarks on DCASE task groups aligned to the OV1 category split.",
    )
    parser.add_argument(
        "--task-names",
        nargs="+",
        default=None,
        help="Optional explicit DCASE task_name filter. Overrides --category-groups if set.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=("float32", "bfloat16", "float16"),
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--local-rank", type=int, default=-1)
    parser.add_argument("--seld-feature-cache-manifest", type=str, default=None)
    parser.add_argument("--seld-hidden-cache-manifest", type=str, default=None)
    parser.add_argument("--skip-base-qwen", action="store_true")
    parser.add_argument("--skip-adapted", action="store_true")
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=("both", "base", "adapted"),
        help="Run base Qwen, adapted checkpoint, or both in separate invocations.",
    )
    parser.add_argument(
        "--question-classes",
        nargs="+",
        default=None,
        help="Filter by question_class in the QA records.",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[name]


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_checkpoint_path(args: argparse.Namespace) -> str:
    if args.checkpoint_path is not None:
        return os.path.abspath(args.checkpoint_path)
    checkpoint_name = f"{args.checkpoint_tag}_trainable.pt"
    return os.path.abspath(os.path.join(args.run_dir, "checkpoints", checkpoint_name))


def build_eval_model_args(args: argparse.Namespace, train_args: Dict[str, Any]) -> argparse.Namespace:
    defaults = {
        "legacy_repo_path": DEFAULT_LEGACY_REPO,
        "model_id": DEFAULT_MODEL_ID,
        "baseline_repo_path": os.path.abspath(os.getcwd()),
        "seld_task_id": "233",
        "seld_checkpoint_path": train_args.get("seld_checkpoint_path"),
        "seld_feature_stats_dir": train_args.get("seld_feature_stats_dir"),
        "gradient_checkpointing": False,
        "spatial_fp32": bool(train_args.get("spatial_fp32", False)),
        "spatial_backbone_fp32": bool(train_args.get("spatial_backbone_fp32", False)),
        "lora_r": int(train_args.get("lora_r", 16)),
        "lora_alpha": int(train_args.get("lora_alpha", 32)),
        "lora_dropout": float(train_args.get("lora_dropout", 0.05)),
        "lora_target_modules": list(train_args.get("lora_target_modules", [])),
        "lora_target_prefixes": list(train_args.get("lora_target_prefixes", ["thinker.model"])),
        "train_mode": train_args.get("train_mode", "spatial_lora"),
        "device": args.device,
        "dtype": args.dtype,
    }
    merged = dict(defaults)
    merged.update(train_args)
    # Runtime launch context must override the saved training args.
    merged["device"] = args.device
    merged["dtype"] = args.dtype
    merged["legacy_repo_path"] = args.legacy_repo_path
    if args.model_id is not None:
        merged["model_id"] = args.model_id
    if "lora_target_modules" not in merged or not merged["lora_target_modules"]:
        merged["lora_target_modules"] = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    if "lora_target_prefixes" not in merged or not merged["lora_target_prefixes"]:
        merged["lora_target_prefixes"] = ["thinker.model"]
    return argparse.Namespace(**merged)


def load_adapted_model(
    args: argparse.Namespace,
    checkpoint_path: str,
    train_args: Dict[str, Any],
):
    model_args = build_eval_model_args(args, train_args)
    processor = build_processor(model_args.model_id)
    processor.tokenizer.padding_side = "left"
    model = build_model(model_args, processor)

    if model_args.train_mode == "spatial_only":
        freeze_all_but_spatial_modules(model)
    elif model_args.train_mode == "adapter_lora":
        model, _ = apply_llm_lora(model, model_args)
        configure_adapter_lora_training(model, model_args)
    elif model_args.train_mode == "spatial_lora":
        model, _ = apply_llm_lora(model, model_args)
        configure_spatial_lora_training(model, model_args)
    else:
        pass

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    from spatial_omni.utils.ckpt_compat import remap_legacy_state_dict
    state_dict = checkpoint.get("trainable_state_dict", checkpoint)
    state_dict = remap_legacy_state_dict(state_dict)
    load_result = model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, processor, checkpoint, load_result


def load_base_qwen_model(args: argparse.Namespace, model_id: str):
    add_legacy_repo_to_path(args.legacy_repo_path)
    from spatial_omni.model.modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration
    from spatial_omni.model.processing_qwen2_5_omni import Qwen2_5OmniProcessor

    processor = Qwen2_5OmniProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype_from_name(args.dtype),
        low_cpu_mem_usage=True,
    )
    if hasattr(model, "disable_talker"):
        model.disable_talker()
    model.to(args.device)
    model.eval()
    return model, processor


def clean_generated_answer(text: str) -> str:
    value = str(text).replace("\r\n", "\n").strip()
    for marker in ("Human:", "Question:", "\nHuman:", "\nQuestion:"):
        if marker in value:
            value = value.split(marker, 1)[0].strip()
    value = next((line.strip() for line in value.splitlines() if line.strip()), "")
    if re.fullmatch(r"[-+]?\d+\.0+", value):
        value = value.split(".", 1)[0]
    return value.strip()


@dataclass
class BaseAudioEvalCollator:
    processor: Any
    sample_rate: int = SAMPLE_RATE
    max_audio_samples: int = MAX_AUDIO_SAMPLES

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        audio_arrays: List[np.ndarray] = []
        texts: List[str] = []
        meta: List[Dict[str, Any]] = []

        for feature in features:
            wav, sr = sf.read(feature["audio_path"], dtype="float32", always_2d=True)
            if sr != self.sample_rate:
                raise ValueError(
                    f"Expected {self.sample_rate} Hz audio, got {sr} for {feature['audio_path']}"
                )
            wav = wav.T
            if wav.ndim != 2 or wav.shape[0] < 1:
                raise ValueError(f"Expected audio shape [C, T], got {tuple(wav.shape)}")
            if wav.shape[1] > self.max_audio_samples:
                wav = wav[:, : self.max_audio_samples]

            mono_audio = wav[0].astype(np.float32, copy=False)
            prompt_text = f"{self.processor.audio_token}\n{feature['prompt'].rstrip()}\n"
            audio_arrays.append(mono_audio)
            texts.append(prompt_text)
            meta.append(
                {
                    "pair_id": feature.get("pair_id"),
                    "task_name": feature.get("task_name"),
                    "question": feature.get("question"),
                    "prompt": feature.get("prompt"),
                    "answer": feature.get("answer"),
                    "audio_path": feature.get("audio_path"),
                }
            )

        batch = self.processor(
            text=texts,
            audio=audio_arrays,
            padding=True,
            return_tensors="pt",
        )
        batch["meta"] = meta
        prefix_lengths = batch["attention_mask"].sum(dim=1).to(dtype=torch.long)
        batch["prefix_lengths"] = prefix_lengths
        pad_token_id = int(self.processor.tokenizer.pad_token_id or 0)
        generation_input_ids, generation_attention_mask = build_left_padded_text_batch(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            prefix_lengths=prefix_lengths,
            pad_token_id=pad_token_id,
        )
        batch["gen_input_ids"] = generation_input_ids
        batch["gen_attention_mask"] = generation_attention_mask
        for key, value in list(batch.items()):
            if key in {"input_ids", "attention_mask", "prefix_lengths", "gen_input_ids", "gen_attention_mask"}:
                continue
            if isinstance(value, torch.Tensor):
                batch[f"gen_{key}"] = value
        return batch


def compute_metrics_from_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    per_task: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"count": 0.0, "raw_exact_match": 0.0, "cleaned_exact_match": 0.0}
    )
    total_raw = 0.0
    total_cleaned = 0.0
    for record in records:
        task_name = str(record.get("task_name") or "unknown")
        raw_em = float(record["raw_exact_match"])
        cleaned_em = float(record["cleaned_exact_match"])
        total_raw += raw_em
        total_cleaned += cleaned_em
        per_task[task_name]["count"] += 1.0
        per_task[task_name]["raw_exact_match"] += raw_em
        per_task[task_name]["cleaned_exact_match"] += cleaned_em

    total_count = float(len(records))
    summary = {
        "examples": int(total_count),
        "raw_exact_match": total_raw / max(total_count, 1.0),
        "cleaned_exact_match": total_cleaned / max(total_count, 1.0),
        "per_task": {},
    }
    for task_name, stats in sorted(per_task.items()):
        count = max(float(stats["count"]), 1.0)
        summary["per_task"][task_name] = {
            "count": int(stats["count"]),
            "raw_exact_match": float(stats["raw_exact_match"]) / count,
            "cleaned_exact_match": float(stats["cleaned_exact_match"]) / count,
        }
    return summary


def filter_dataset_by_task_names(dataset, task_names: List[str]):
    allowed = set(task_names)
    indices = [
        index
        for index, record in enumerate(dataset.records)
        if str(record.get("task_name")) in allowed
    ]
    return Subset(dataset, indices)


def filter_dataset_by_field(dataset, field_name: str, allowed_values: List[str]):
    allowed = set(allowed_values)
    indices = [
        index
        for index, record in enumerate(dataset.records)
        if str(record.get(field_name)) in allowed
    ]
    return Subset(dataset, indices)


def finalize_distributed_prediction_file(output_jsonl_path: str) -> List[Dict[str, Any]]:
    shard_paths = [
        f"{output_jsonl_path}.rank{rank}.jsonl"
        for rank in range(int(os.environ.get("WORLD_SIZE", "1")))
    ] if dist_is_enabled() else [f"{output_jsonl_path}.rank0.jsonl"]
    merged_records: List[Dict[str, Any]] = []
    with open(output_jsonl_path, "w", encoding="utf-8") as merged_handle:
        for shard_path in shard_paths:
            if not os.path.exists(shard_path):
                continue
            with open(shard_path, "r", encoding="utf-8") as shard_handle:
                for line in shard_handle:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    merged_records.append(record)
                    merged_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            os.remove(shard_path)
    return merged_records


def dist_is_enabled() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def run_generation_bench(
    model,
    processor,
    loader,
    device: str,
    output_jsonl_path: str,
    max_new_tokens: int,
    num_beams: int,
    do_sample: bool,
    bench_name: str,
) -> Dict[str, Any]:
    model.eval()
    records: List[Dict[str, Any]] = []
    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)
    rank = get_rank()
    shard_output_path = f"{output_jsonl_path}.rank{rank}.jsonl"

    with open(shard_output_path, "w", encoding="utf-8") as handle:
        with torch.no_grad():
            progress = tqdm(loader, desc=bench_name, leave=False, disable=not is_main_process())
            for batch in progress:
                generation_inputs = build_generation_inputs(batch, device)
                generation_inputs.pop("prefix_lengths", None)
                generated = unwrap_model(model).generate(
                    **generation_inputs,
                    return_audio=False,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    do_sample=do_sample,
                )
                prompt_lengths = generation_inputs["attention_mask"].sum(dim=1).tolist()
                generated = generated.detach().cpu()

                for index, prompt_length in enumerate(prompt_lengths):
                    prediction_ids = generated[index, int(prompt_length):]
                    prediction_text = processor.tokenizer.decode(
                        prediction_ids,
                        skip_special_tokens=True,
                    ).strip()
                    cleaned_prediction = clean_generated_answer(prediction_text)
                    meta = batch["meta"][index]
                    answer_text = str(meta["answer"]).strip()
                    cleaned_answer = clean_generated_answer(answer_text)
                    raw_exact_match = int(
                        normalize_answer(prediction_text) == normalize_answer(answer_text)
                    )
                    cleaned_exact_match = int(
                        normalize_answer(cleaned_prediction) == normalize_answer(cleaned_answer)
                    )
                    record = {
                        "pair_id": meta.get("pair_id"),
                        "task_name": meta.get("task_name"),
                        "question": meta.get("question"),
                        "prompt": meta.get("prompt"),
                        "answer": answer_text,
                        "prediction": prediction_text,
                        "prediction_cleaned": cleaned_prediction,
                        "raw_exact_match": raw_exact_match,
                        "cleaned_exact_match": cleaned_exact_match,
                    }
                    records.append(record)
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")

                if is_main_process():
                    metrics = compute_metrics_from_records(records)
                    progress.set_postfix(
                        raw_em=f"{metrics['raw_exact_match']:.4f}",
                        clean_em=f"{metrics['cleaned_exact_match']:.4f}",
                        n=int(metrics["examples"]),
                    )

    distributed_barrier()
    if not is_main_process():
        return {}
    merged_records = finalize_distributed_prediction_file(output_jsonl_path)
    return compute_metrics_from_records(merged_records)


def print_summary(name: str, metrics: Dict[str, Any]) -> None:
    print(
        f"[{name}] examples={metrics['examples']} "
        f"raw_em={metrics['raw_exact_match']:.4f} "
        f"clean_em={metrics['cleaned_exact_match']:.4f}"
    )
    for task_name, stats in metrics["per_task"].items():
        print(
            f"  - {task_name}: count={stats['count']} "
            f"raw_em={stats['raw_exact_match']:.4f} "
            f"clean_em={stats['cleaned_exact_match']:.4f}"
        )


def run_benchmark_for_dataset(
    dataset,
    dataset_name: str,
    args: argparse.Namespace,
    output_dir: str,
    model_id: str,
    train_args: Dict[str, Any],
    checkpoint_path: str,
    combined_summary: Dict[str, Any],
) -> None:
    if len(dataset) == 0:
        raise RuntimeError(f"Filtered dataset for {dataset_name} is empty.")

    dataset_output_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_output_dir, exist_ok=True)
    dataset_summary: Dict[str, Any] = {
        "examples": len(dataset),
    }

    if not args.skip_base_qwen:
        base_model, base_processor = load_base_qwen_model(args, model_id)
        base_sampler = DistributedSampler(dataset, shuffle=False) if args.distributed else None
        base_loader = make_loader(
            dataset=dataset,
            collator=BaseAudioEvalCollator(processor=base_processor),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            sampler=base_sampler,
        )
        base_predictions_path = os.path.join(dataset_output_dir, "base_qwen_audio_only_predictions.jsonl")
        base_metrics = run_generation_bench(
            model=base_model,
            processor=base_processor,
            loader=base_loader,
            device=args.device,
            output_jsonl_path=base_predictions_path,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            do_sample=args.do_sample,
            bench_name=f"{dataset_name}:base_qwen",
        )
        if is_main_process():
            dataset_summary["base_qwen_audio_only"] = base_metrics
            print_summary(f"{dataset_name}/base_qwen_audio_only", base_metrics)
        del base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        distributed_barrier()

    if not args.skip_adapted:
        adapted_model, adapted_processor, checkpoint_payload, load_result = load_adapted_model(
            args=args,
            checkpoint_path=checkpoint_path,
            train_args=train_args,
        )
        spatial_collator = SpatialQACollator(
            processor=adapted_processor,
            include_generation_inputs=True,
        )
        adapted_sampler = DistributedSampler(dataset, shuffle=False) if args.distributed else None
        adapted_loader = make_loader(
            dataset=dataset,
            collator=spatial_collator,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            sampler=adapted_sampler,
        )
        adapted_predictions_path = os.path.join(dataset_output_dir, "adapted_spatial_predictions.jsonl")
        adapted_metrics = run_generation_bench(
            model=adapted_model,
            processor=adapted_processor,
            loader=adapted_loader,
            device=args.device,
            output_jsonl_path=adapted_predictions_path,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            do_sample=args.do_sample,
            bench_name=f"{dataset_name}:adapted_spatial",
        )
        if is_main_process():
            dataset_summary["adapted_spatial_checkpoint"] = {
                **adapted_metrics,
                "checkpoint_epoch": checkpoint_payload.get("epoch"),
                "checkpoint_step": checkpoint_payload.get("step"),
                "missing_keys": len(load_result.missing_keys),
                "unexpected_keys": len(load_result.unexpected_keys),
            }
            print_summary(f"{dataset_name}/adapted_spatial_checkpoint", adapted_metrics)
        del adapted_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        distributed_barrier()

    if is_main_process():
        summary_path = os.path.join(dataset_output_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(dataset_summary, handle, indent=2, sort_keys=True, ensure_ascii=False)
        combined_summary[dataset_name] = dataset_summary


def main() -> None:
    args = parse_args()
    args = setup_distributed(args)
    if args.mode == "base":
        args.skip_adapted = True
    elif args.mode == "adapted":
        args.skip_base_qwen = True

    try:
        add_legacy_repo_to_path(args.legacy_repo_path)

        train_args_path = os.path.join(args.run_dir, "train_args.json")
        if os.path.exists(train_args_path):
            train_args = load_json(train_args_path)
        elif args.skip_adapted:
            train_args = {}
            rank0_print(
                f"train_args.json not found under {args.run_dir}; continuing in base-only mode "
                "with CLI/default arguments."
            )
        else:
            raise FileNotFoundError(f"Missing train_args.json: {train_args_path}")

        model_id = args.model_id or train_args.get("model_id") or DEFAULT_MODEL_ID
        # 优先级：--qa-root > --qa-version（非默认值）> train_args 中保存的 qa_root > 默认值
        if args.qa_root:
            # 用户显式传了 --qa-root，直接使用
            qa_root = resolve_qa_root(args.qa_root, args.qa_version)
        elif args.qa_version != DEFAULT_QA_VERSION:
            # 用户显式传了 --qa-version（非默认值），按版本解析，忽略 train_args 中的 qa_root
            qa_root = resolve_qa_root(None, args.qa_version)
        elif train_args.get("qa_root"):
            # 使用训练时保存的 qa_root
            qa_root = resolve_qa_root(train_args["qa_root"], args.qa_version)
        else:
            qa_root = DEFAULT_QA_ROOT
        split_path = resolve_qa_split_path(qa_root, args.split)

        if args.output_dir is None:
            if args.mode == "both":
                output_dir = os.path.join(args.run_dir, "bench_results", args.split)
            else:
                output_dir = os.path.join(args.run_dir, "bench_results", args.split, f"{args.mode}_only")
        else:
            output_dir = os.path.abspath(args.output_dir)
        if is_main_process():
            os.makedirs(output_dir, exist_ok=True)

        checkpoint_path = None
        if not args.skip_adapted:
            checkpoint_path = resolve_checkpoint_path(args)
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Missing adapted checkpoint: {checkpoint_path}")

        dataset = QAAudioJsonlDataset(
            split_path,
            max_samples=args.max_samples,
            feature_cache_manifest_path=args.seld_feature_cache_manifest,
            hidden_cache_manifest_path=args.seld_hidden_cache_manifest,
        )
        if len(dataset) == 0:
            raise RuntimeError(f"Empty evaluation split: {split_path}")

        rank0_print(
            f"Loaded benchmark dataset {split_path}: examples={len(dataset)} "
            f"mode={args.mode} world_size={args.world_size} batch_size={args.batch_size}"
        )

        combined_summary: Dict[str, Any] = {
            "split": args.split,
            "examples": len(dataset),
            "qa_root": qa_root,
            "split_path": split_path,
            "model_id": model_id,
            "checkpoint_path": checkpoint_path,
            "mode": args.mode,
            "world_size": args.world_size,
            "batch_size_per_rank": args.batch_size,
        }

        if args.question_classes is not None:
            filtered_dataset = filter_dataset_by_field(dataset, "question_class", args.question_classes)
            run_benchmark_for_dataset(
                dataset=filtered_dataset,
                dataset_name="question_class_filter",
                args=args,
                output_dir=output_dir,
                model_id=model_id,
                train_args=train_args,
                checkpoint_path=checkpoint_path,
                combined_summary=combined_summary,
            )
        elif args.task_names is not None:
            filtered_dataset = filter_dataset_by_task_names(dataset, args.task_names)
            run_benchmark_for_dataset(
                dataset=filtered_dataset,
                dataset_name="task_filter",
                args=args,
                output_dir=output_dir,
                model_id=model_id,
                train_args=train_args,
                checkpoint_path=checkpoint_path,
                combined_summary=combined_summary,
            )
        elif args.category_groups is not None:
            has_question_class = any(record.get("question_class") for record in dataset.records)
            for category_name in args.category_groups:
                if has_question_class:
                    filtered_dataset = filter_dataset_by_field(dataset, "question_class", [category_name])
                else:
                    filtered_dataset = filter_dataset_by_task_names(
                        dataset,
                        DCASE_CORE_CATEGORY_TASKS[category_name],
                    )
                run_benchmark_for_dataset(
                    dataset=filtered_dataset,
                    dataset_name=category_name,
                    args=args,
                    output_dir=output_dir,
                    model_id=model_id,
                    train_args=train_args,
                    checkpoint_path=checkpoint_path,
                    combined_summary=combined_summary,
                )
        else:
            run_benchmark_for_dataset(
                dataset=dataset,
                dataset_name="full_split",
                args=args,
                output_dir=output_dir,
                model_id=model_id,
                train_args=train_args,
                checkpoint_path=checkpoint_path,
                combined_summary=combined_summary,
            )

        if is_main_process():
            summary_path = os.path.join(output_dir, "summary.json")
            with open(summary_path, "w", encoding="utf-8") as handle:
                json.dump(combined_summary, handle, indent=2, sort_keys=True, ensure_ascii=False)
            print(f"Saved benchmark summary to {summary_path}")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
