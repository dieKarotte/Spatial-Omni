import argparse
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm.auto import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from train_so_qa import (
    DEFAULT_MODEL_ID,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_QA_ROOT,
    DEFAULT_SO_REPO,
    MAX_AUDIO_SAMPLES,
    QwenAudioFeatureCache,
    SAMPLE_RATE,
    SpatialBeatsQACollator,
    apply_llm_lora,
    build_left_padded_batch,
    build_model,
    build_processor,
    build_qa_dataset,
    cleanup_distributed,
    configure_beats_lora_training,
    configure_encoder_lora_training,
    distributed_barrier,
    dtype_from_name,
    freeze_all_but_projector,
    get_rank,
    get_world_size,
    is_distributed,
    is_main_process,
    make_loader,
    normalize_answer,
    rank0_print,
    resolve_qa_split_path,
    setup_distributed,
    shard_dataset_for_rank,
    unwrap_model,
)


DEFAULT_SCORE_SCRIPT = os.environ.get(
    "SO_SCORE_SCRIPT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "score_test_predictions.py"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch benchmark Spatial-BEATs Spatial-Omni checkpoints on a QA split."
    )
    parser.add_argument("--run-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint-tags", nargs="+", default=None,
                        help="Checkpoint tags without the _trainable.pt suffix, e.g. best epoch_001 step_0007000")
    parser.add_argument("--checkpoint-paths", nargs="+", default=None,
                        help="Explicit checkpoint .pt paths.")
    parser.add_argument("--checkpoint-glob", type=str, default=None,
                        help="Glob under <run-dir>/checkpoints, e.g. 'step_000[7-9]000_trainable.pt'")
    parser.add_argument("--qa-root", type=str, default=DEFAULT_QA_ROOT)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--task-names", nargs="+", default=None)
    parser.add_argument("--question-classes", nargs="+", default=None)
    parser.add_argument("--audio-feature-cache-manifest", type=str, default=None)
    parser.add_argument("--audio-feature-cache-max-entries", type=int, default=256)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--device-map", type=str, default=None,
                        help="HF device_map (e.g. 'auto') to shard model across GPUs. "
                             "When set, --device is still used for inference tensors.")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=("float32", "bfloat16", "float16"))
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--score-script-path", type=str, default=DEFAULT_SCORE_SCRIPT)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--local-rank", type=int, default=-1)
    return parser.parse_args()


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def clean_generated_answer(text: str) -> str:
    value = str(text).replace("\r\n", "\n").strip()
    for marker in ("Human:", "Question:", "\nHuman:", "\nQuestion:"):
        if marker in value:
            value = value.split(marker, 1)[0].strip()
    value = next((line.strip() for line in value.splitlines() if line.strip()), "")
    if re.fullmatch(r"[-+]?\d+\.0+", value):
        value = value.split(".", 1)[0]
    return value.strip()


def resolve_checkpoint_paths(args: argparse.Namespace) -> List[str]:
    run_dir = os.path.abspath(args.run_dir)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    paths: List[str] = []
    if args.checkpoint_tags:
        for tag in args.checkpoint_tags:
            paths.append(os.path.join(checkpoint_dir, f"{tag}_trainable.pt"))
    if args.checkpoint_paths:
        paths.extend(os.path.abspath(path) for path in args.checkpoint_paths)
    if args.checkpoint_glob:
        paths.extend(str(path) for path in sorted(Path(checkpoint_dir).glob(args.checkpoint_glob)))
    if not paths:
        raise ValueError("Provide at least one of --checkpoint-tags, --checkpoint-paths, or --checkpoint-glob.")
    deduped: List[str] = []
    seen = set()
    for path in paths:
        ap = os.path.abspath(path)
        if ap in seen:
            continue
        if not os.path.exists(ap):
            raise FileNotFoundError(f"Checkpoint not found: {ap}")
        seen.add(ap)
        deduped.append(ap)
    return deduped


def infer_train_args_path(checkpoint_path: str) -> str:
    """Locate the train_args.json that describes how to rebuild the model.

    Two layouts are supported:
      1. Training run layout — ``<run>/checkpoints/<ckpt>.pt`` with
         ``<run>/train_args.json`` two levels up (how the trainer writes it).
      2. Flat release layout — ``<dir>/<ckpt>.pt`` with ``<dir>/train_args.json``
         sitting right next to the checkpoint (convenient for distributing a
         single checkpoint file).
    """
    abs_ckpt = os.path.abspath(checkpoint_path)
    candidates = [
        # Layout 1: <run>/train_args.json (ckpt under <run>/checkpoints/)
        os.path.join(os.path.dirname(os.path.dirname(abs_ckpt)), "train_args.json"),
        # Layout 2: train_args.json next to the checkpoint
        os.path.join(os.path.dirname(abs_ckpt), "train_args.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "train_args.json not found for checkpoint "
        f"{checkpoint_path}. Looked in: {candidates}"
    )


def build_eval_model_args(runtime_args: argparse.Namespace, train_args: Dict[str, Any]) -> argparse.Namespace:
    merged = dict(train_args)
    merged.setdefault("model_id", DEFAULT_MODEL_ID)
    merged.setdefault("beats_checkpoint", train_args.get("beats_checkpoint"))
    merged.setdefault("beats_repo", train_args.get("beats_repo"))
    merged.setdefault("so_repo", train_args.get("so_repo", DEFAULT_SO_REPO))
    merged.setdefault("train_mode", train_args.get("train_mode", "projector_only"))
    merged.setdefault("lora_r", int(train_args.get("lora_r", 16)))
    merged.setdefault("lora_alpha", int(train_args.get("lora_alpha", 32)))
    merged.setdefault("lora_dropout", float(train_args.get("lora_dropout", 0.05)))
    merged.setdefault("lora_target_modules", list(train_args.get("lora_target_modules", [])))
    merged.setdefault("lora_target_prefixes", list(train_args.get("lora_target_prefixes", ["thinker.model"])))
    merged.setdefault("projector_type", train_args.get("projector_type", "mlp"))
    merged.setdefault("projector_shuffle_factor", int(train_args.get("projector_shuffle_factor", 1)))
    merged["device"] = runtime_args.device
    merged["device_map"] = getattr(runtime_args, "device_map", None)
    merged["dtype"] = runtime_args.dtype
    merged["gradient_checkpointing"] = False
    merged["projector_fp32"] = bool(train_args.get("projector_fp32", False))
    return argparse.Namespace(**merged)


def instantiate_model_for_checkpoint(runtime_args: argparse.Namespace, checkpoint_path: str):
    train_args = load_json(infer_train_args_path(checkpoint_path))
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
    elif train_mode == "beats_lora":
        model, _ = apply_llm_lora(model, model_args)
        configure_beats_lora_training(model, model_args)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    from spatial_omni.utils.ckpt_compat import remap_legacy_state_dict
    state_dict = checkpoint.get("trainable_state_dict", checkpoint)
    state_dict = remap_legacy_state_dict(state_dict)
    load_result = model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, processor, train_args, checkpoint, load_result


@dataclass
class SpatialBeatsEvalCollator:
    processor: Any
    audio_feature_cache: Optional[QwenAudioFeatureCache] = None
    sample_rate: int = SAMPLE_RATE
    max_audio_samples: int = MAX_AUDIO_SAMPLES
    mono_audio_zero_spatial_tokens: bool = False
    mono_audio_w_channel_spatial_encoder: bool = False
    zero_projected_spatial_dim: int = 3584
    # NEW: "decoder-only" ablation — drop the Qwen mono <|AUDIO|> branch
    # entirely; spatial encoder keeps producing real <|spatial|> tokens.
    # Use this to measure how much of the model's QA performance comes
    # from Qwen's untrained audio_tower vs the spatial encoder. The model
    # was trained with both branches active, so this is OOD for the LoRA;
    # results below the joint baseline are expected and meaningful.
    drop_mono_audio: bool = False

    def __post_init__(self) -> None:
        if self.mono_audio_zero_spatial_tokens and self.mono_audio_w_channel_spatial_encoder:
            raise ValueError(
                "mono_audio_zero_spatial_tokens and "
                "mono_audio_w_channel_spatial_encoder are mutually exclusive."
            )
        if self.drop_mono_audio and (
            self.mono_audio_zero_spatial_tokens
            or self.mono_audio_w_channel_spatial_encoder
        ):
            raise ValueError(
                "drop_mono_audio is mutually exclusive with the "
                "mono_audio_* compat modes."
            )

    @property
    def _mono_compat_mode(self) -> bool:
        return self.mono_audio_zero_spatial_tokens or self.mono_audio_w_channel_spatial_encoder

    def _resample_audio_if_needed(self, wav: np.ndarray, sr: int, audio_path: str) -> np.ndarray:
        if sr == self.sample_rate:
            return wav
        if not self._mono_compat_mode:
            raise ValueError(f"Expected {self.sample_rate}Hz got {sr} for {audio_path}")
        try:
            from scipy.signal import resample_poly
        except ImportError as exc:
            raise ImportError(
                "scipy is required to resample mono MMAU audio in "
                "mono compatibility modes."
            ) from exc
        gcd = math.gcd(int(sr), int(self.sample_rate))
        up = int(self.sample_rate) // gcd
        down = int(sr) // gcd
        return resample_poly(wav, up, down, axis=0).astype(np.float32, copy=False)

    def _read_audio(self, audio_path: str) -> tuple[np.ndarray, int]:
        import soundfile as sf

        if not self._mono_compat_mode:
            return sf.read(audio_path, dtype="float32", always_2d=True)

        info = sf.info(audio_path)
        sr = int(info.samplerate)
        source_max_samples = int(math.ceil(self.max_audio_samples * sr / self.sample_rate))
        return sf.read(
            audio_path,
            frames=source_max_samples,
            dtype="float32",
            always_2d=True,
        )

    def _build_w_channel_foa(self, wav: np.ndarray) -> np.ndarray:
        mono = wav.mean(axis=1).astype(np.float32, copy=False)
        if mono.shape[0] > self.max_audio_samples:
            mono = mono[: self.max_audio_samples]
        foa = np.zeros((4, mono.shape[0]), dtype=np.float32)
        foa[0, :] = mono
        return foa

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        audio_arrs: List[np.ndarray] = []
        # In drop_mono_audio mode we keep `audio_arrs` (= what goes into the
        # processor → Qwen audio_tower) zero-filled, but spatial_audio_arrs
        # holds the REAL FOA waveform for the spatial encoder branch.
        # In normal mode they're identical (same list). We only diverge them
        # when drop_mono_audio is on.
        spatial_audio_arrs: List[np.ndarray] = []
        prompts: List[str] = []
        meta: List[Dict[str, Any]] = []
        cached_input_features: List[torch.Tensor] = []
        cached_feature_lengths: List[int] = []
        sa_lens: List[int] = []

        for feat in features:
            cache_item = None
            if self.audio_feature_cache is not None and not self.mono_audio_w_channel_spatial_encoder:
                try:
                    cache_item = self.audio_feature_cache.load(feat["audio_path"])
                except KeyError:
                    cache_item = None
            if (
                not self._mono_compat_mode
                and cache_item is not None
                and "spatial_audio" in cache_item
                and "spatial_audio_length" in cache_item
            ):
                wav = cache_item["spatial_audio"].to(dtype=torch.float32).cpu().numpy()
                if wav.ndim != 2 or wav.shape[0] != 4:
                    raise ValueError(f"Cached spatial_audio must have shape [4, T], got {tuple(wav.shape)}")
                T = int(cache_item["spatial_audio_length"].item())
                wav = wav[:, :T]
            else:
                wav, sr = self._read_audio(feat["audio_path"])
                wav = self._resample_audio_if_needed(wav, sr, feat["audio_path"])
                if self.mono_audio_zero_spatial_tokens:
                    wav = wav.mean(axis=1, keepdims=True).T
                elif self.mono_audio_w_channel_spatial_encoder:
                    wav = self._build_w_channel_foa(wav)
                else:
                    wav = wav.T
                    if wav.shape[0] != 4:
                        raise ValueError(f"Expected 4ch FOA, got {wav.shape}")
                if wav.shape[1] > self.max_audio_samples:
                    wav = wav[:, : self.max_audio_samples]
                T = wav.shape[1]

            sa_lens.append(T)
            real_wav = wav.astype(np.float32, copy=False)
            spatial_audio_arrs.append(real_wav)  # always real FOA for spatial encoder
            if self.drop_mono_audio:
                # Hand the Qwen mono audio_tower a zero waveform; the spatial
                # branch keeps the real one (via spatial_audio_arrs above).
                audio_arrs.append(np.zeros_like(real_wav, dtype=np.float32))
            else:
                audio_arrs.append(real_wav)
            # Prompt structure is unchanged: <|AUDIO|><|spatial|>\n<prompt>\n
            prompt_prefix = (
                self.processor.audio_token
                + self.processor.spatial_token
                + f"\n{str(feat['prompt']).rstrip()}\n"
            )
            prompts.append(prompt_prefix)
            pid = feat.get("pair_id")
            if pid is None or pid == "":
                import hashlib
                key = "|".join(
                    str(feat.get(k, ""))
                    for k in ("scene_id", "segment_stem", "task_name", "question", "audio_path")
                )
                pid = "auto_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
            meta.append(
                {
                    "pair_id": pid,
                    "task_name": feat.get("task_name"),
                    "question": feat.get("question"),
                    "prompt": feat.get("prompt"),
                    "answer": feat.get("answer"),
                    "audio_path": feat.get("audio_path"),
                    "scene_id": feat.get("scene_id"),
                    "segment_stem": feat.get("segment_stem"),
                    "canonical_answer": feat.get("canonical_answer"),
                    "question_class": feat.get("question_class"),
                    "answer_meta": feat.get("answer_meta"),
                }
            )
            if cache_item is not None:
                cached_input_features.append(cache_item["input_features"])
                cached_feature_lengths.append(int(cache_item["feature_length"].item()))

        lens_t = torch.tensor(sa_lens, dtype=torch.long)
        batch_size = len(audio_arrs)
        processor_kwargs: Dict[str, Any] = {}
        # In drop_mono_audio mode skip the cached input_features — they were
        # computed from the real audio and would re-introduce information into
        # the Qwen audio_tower. Letting the processor recompute features from
        # the zeroed waveform gives true zero-information mono input.
        if (
            not self.drop_mono_audio
            and self.audio_feature_cache is not None
            and cached_input_features
        ):
            feature_dim = int(cached_input_features[0].shape[0])
            max_feature_length = max(cached_feature_lengths)
            input_features = torch.zeros(
                batch_size, feature_dim, max_feature_length, dtype=cached_input_features[0].dtype
            )
            feature_attention_mask = torch.zeros(batch_size, max_feature_length, dtype=torch.long)
            for index, (feature_tensor, feature_length) in enumerate(zip(cached_input_features, cached_feature_lengths)):
                input_features[index, :, :feature_length] = feature_tensor[:, :feature_length]
                feature_attention_mask[index, :feature_length] = 1
            processor_kwargs["input_features"] = input_features
            processor_kwargs["feature_attention_mask"] = feature_attention_mask

        if self.mono_audio_zero_spatial_tokens:
            if not hasattr(self.processor, "_samples_to_so_backbone_tokens"):
                raise AttributeError("Processor does not expose _samples_to_so_backbone_tokens().")
            spatial_token_lengths = self.processor._samples_to_so_backbone_tokens(lens_t)
            t_spatial_max = int(spatial_token_lengths.max().item())
            processor_kwargs["projected_spatial_tokens"] = torch.zeros(
                batch_size,
                t_spatial_max,
                int(self.zero_projected_spatial_dim),
                dtype=torch.float32,
            )
            processor_kwargs["spatial_token_lengths"] = spatial_token_lengths
            processor_kwargs["allow_mono_spatial_tokens"] = True

        # 强制右填充：build_left_padded_batch 需要 input_ids[i, :pl_i] 是真实前缀，
        # 若 tokenizer 左填充则 input_ids[i, :pl_i] 包含 padding token 而非完整前缀，
        # 导致 <|spatial|> token 丢失并引发 RoPE modal_order 验证错误。
        # 注意：Qwen2_5OmniProcessorKwargs._defaults 里 text_kwargs.padding_side="left" 是
        # 硬编码在处理器内部的，仅修改 tokenizer.padding_side 不会覆盖它；
        # 必须通过在 __call__ 中传入 padding_side='right' kwarg 才能覆盖。
        # In drop_mono_audio mode `audio_arrs` already contains zero-filled
        # waveforms (see above), so the processor will compute zero
        # input_features and the audio_tower forward will produce ~zero
        # embeddings. Spatial branch is unaffected.
        batch = self.processor(
            text=prompts,
            audio=audio_arrs,
            padding=True,
            padding_side="right",
            return_tensors="pt",
            **processor_kwargs,
        )
        if not self.mono_audio_zero_spatial_tokens:
            t_max = int(lens_t.max())
            sa_t = torch.zeros(batch_size, t_max, 4, dtype=torch.float32)
            # Use spatial_audio_arrs (always real FOA) — distinct from
            # audio_arrs which may be zero-filled in drop_mono_audio mode.
            for index, wav in enumerate(spatial_audio_arrs):
                sa_t[index, : wav.shape[1]] = torch.from_numpy(wav.T)
            batch["spatial_audio"] = sa_t
            batch["spatial_audio_attention_mask"] = (
                torch.arange(t_max).unsqueeze(0) < lens_t.unsqueeze(1)
            )
            batch["spatial_audio_lengths"] = lens_t
        batch["meta"] = meta
        prefix_lengths = batch["attention_mask"].sum(1).long()
        batch["prefix_lengths"] = prefix_lengths
        pad_token_id = int(self.processor.tokenizer.pad_token_id or 0)
        generation_input_ids, generation_attention_mask = build_left_padded_batch(
            batch["input_ids"], batch["attention_mask"], prefix_lengths, pad_token_id
        )
        batch["gen_input_ids"] = generation_input_ids
        batch["gen_attention_mask"] = generation_attention_mask
        for key, value in list(batch.items()):
            if key in {"input_ids", "attention_mask", "prefix_lengths", "meta", "gen_input_ids", "gen_attention_mask"}:
                continue
            if isinstance(value, torch.Tensor):
                batch[f"gen_{key}"] = value
        return batch


def filter_dataset(dataset, task_names: Optional[List[str]], question_classes: Optional[List[str]]):
    if not task_names and not question_classes:
        return dataset
    allowed_tasks = set(task_names or [])
    allowed_classes = set(question_classes or [])
    indices = []
    records = dataset.records if hasattr(dataset, "records") else None
    if records is None:
        return dataset
    for index, record in enumerate(records):
        if allowed_tasks and str(record.get("task_name")) not in allowed_tasks:
            continue
        if allowed_classes and str(record.get("question_class")) not in allowed_classes:
            continue
        indices.append(index)
    return torch.utils.data.Subset(dataset, indices)


def to_generation_inputs(batch: Dict[str, Any], device: str) -> Dict[str, torch.Tensor]:
    inputs = {}
    for key, value in batch.items():
        if not key.startswith("gen_") or not isinstance(value, torch.Tensor):
            continue
        inputs[key[4:]] = value.to(device)
    return inputs


def get_model_device(model) -> str:
    """获取模型（或其第一层）所在的 device，兼容 device_map='auto' 多卡分布场景。"""
    m = unwrap_model(model)
    try:
        p = next(m.parameters())
        return str(p.device)
    except StopIteration:
        return "cpu"


def finalize_distributed_prediction_file(output_jsonl_path: str) -> List[Dict[str, Any]]:
    if is_distributed():
        shard_paths = [f"{output_jsonl_path}.rank{rank}.jsonl" for rank in range(get_world_size())]
    else:
        shard_paths = [f"{output_jsonl_path}.rank0.jsonl"]
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
    local_records: List[Dict[str, Any]] = []
    rank = get_rank()
    shard_output_path = f"{output_jsonl_path}.rank{rank}.jsonl"
    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)
    eval_model = unwrap_model(model)
    # device_map='auto' 时模型分布在多卡，取第一个参数的实际 device 作为输入 tensor 的目标
    input_device = get_model_device(eval_model)

    with open(shard_output_path, "w", encoding="utf-8") as handle:
        with torch.no_grad():
            progress = tqdm(loader, desc=bench_name, leave=False, disable=not is_main_process())
            for batch in progress:
                generation_inputs = to_generation_inputs(batch, input_device)
                generated = eval_model.generate(
                    **generation_inputs,
                    return_audio=False,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    do_sample=do_sample,
                )
                # ml = 左填充 batch 的统一序列长度；generate() 输出 [B, ml+k]，
                # 新 token 从 ml 开始，对所有样本统一，不能用各自的 prompt_length。
                ml = generation_inputs["input_ids"].shape[1]
                generated = generated.detach().cpu()
                for index in range(len(batch["meta"])):
                    prediction_ids = generated[index, ml:]
                    prediction_text = processor.tokenizer.decode(prediction_ids, skip_special_tokens=True).strip()
                    cleaned_prediction = clean_generated_answer(prediction_text)
                    meta = batch["meta"][index]
                    answer_text = str(meta["answer"]).strip()
                    cleaned_answer = clean_generated_answer(answer_text)
                    raw_exact_match = int(normalize_answer(prediction_text) == normalize_answer(answer_text))
                    cleaned_exact_match = int(normalize_answer(cleaned_prediction) == normalize_answer(cleaned_answer))
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
                    local_records.append(record)
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    distributed_barrier()
    if not is_main_process():
        return {}
    merged_records = finalize_distributed_prediction_file(output_jsonl_path)
    total = max(len(merged_records), 1)
    raw_em = sum(float(record["raw_exact_match"]) for record in merged_records) / total
    clean_em = sum(float(record["cleaned_exact_match"]) for record in merged_records) / total
    return {
        "examples": len(merged_records),
        "raw_exact_match": raw_em,
        "cleaned_exact_match": clean_em,
    }


def score_predictions(args: argparse.Namespace, predictions_jsonl: str) -> Dict[str, Any]:
    output_json = os.path.join(os.path.dirname(predictions_jsonl), "result.json")
    cmd = [
        sys.executable,
        os.path.abspath(args.score_script_path),
        "--predictions-jsonl", os.path.abspath(predictions_jsonl),
        "--qa-root", os.path.abspath(args.qa_root),
        "--split", args.split,
        "--output-json", os.path.abspath(output_json),
    ]
    subprocess.run(cmd, check=True)
    return load_json(output_json)


def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    ordered = sorted(results, key=lambda item: item["score_summary"]["task_aware_accuracy"], reverse=True)
    return {
        "checkpoints": ordered,
        "best_checkpoint": ordered[0]["checkpoint"] if ordered else None,
    }


def main() -> None:
    args = parse_args()
    args = setup_distributed(args)

    checkpoint_paths = resolve_checkpoint_paths(args)
    audio_feature_cache = None
    if args.audio_feature_cache_manifest:
        audio_feature_cache = QwenAudioFeatureCache(
            manifest_path=args.audio_feature_cache_manifest,
            max_entries=args.audio_feature_cache_max_entries,
        )
        rank0_print(
            f"Using audio feature cache: {audio_feature_cache.manifest_path} "
            f"(entries={len(audio_feature_cache):,}, in_memory_max={audio_feature_cache.max_entries})"
        )

    dataset, _, _ = build_qa_dataset([args.qa_root], args.split, args.max_samples)
    dataset = filter_dataset(dataset, args.task_names, args.question_classes)
    dataset = shard_dataset_for_rank(dataset)
    if len(dataset) == 0:
        raise RuntimeError("Benchmark dataset is empty after filtering.")

    output_dir = os.path.abspath(args.output_dir or os.path.join(args.run_dir, "bench", args.split))
    os.makedirs(output_dir, exist_ok=True)

    all_results: List[Dict[str, Any]] = []
    for checkpoint_path in checkpoint_paths:
        checkpoint_name = Path(checkpoint_path).stem.replace("_trainable", "")
        checkpoint_output_dir = os.path.join(output_dir, checkpoint_name)
        predictions_jsonl = os.path.join(checkpoint_output_dir, "predictions.jsonl")
        result_json = os.path.join(checkpoint_output_dir, "result.json")

        if args.skip_existing and os.path.exists(predictions_jsonl) and os.path.exists(result_json):
            rank0_print(f"Skipping existing benchmark for {checkpoint_name}")
            distributed_barrier()
            score_summary = load_json(result_json) if is_main_process() else {}
            if is_main_process():
                all_results.append({
                    "checkpoint": os.path.abspath(checkpoint_path),
                    "predictions_jsonl": os.path.abspath(predictions_jsonl),
                    "score_summary": score_summary,
                })
            continue

        model, processor, train_args, checkpoint, load_result = instantiate_model_for_checkpoint(args, checkpoint_path)
        rank0_print(
            f"[{checkpoint_name}] loaded missing={len(load_result.missing_keys)} "
            f"unexpected={len(load_result.unexpected_keys)}"
        )
        loader = make_loader(
            dataset=dataset,
            collator=SpatialBeatsEvalCollator(
                processor=processor,
                audio_feature_cache=audio_feature_cache,
            ),
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
            device=args.device,
            output_jsonl_path=predictions_jsonl,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            do_sample=args.do_sample,
            bench_name=f"bench:{checkpoint_name}",
        )
        distributed_barrier()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if is_main_process():
            score_summary = score_predictions(args, predictions_jsonl)
            payload = {
                "checkpoint": os.path.abspath(checkpoint_path),
                "checkpoint_epoch": checkpoint.get("epoch"),
                "predictions_jsonl": os.path.abspath(predictions_jsonl),
                "quick_metrics": quick_metrics,
                "score_summary": score_summary,
                "train_args_path": infer_train_args_path(checkpoint_path),
                "train_mode": train_args.get("train_mode"),
            }
            with open(os.path.join(checkpoint_output_dir, "bench_summary.json"), "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            all_results.append(payload)
            rank0_print(
                f"[{checkpoint_name}] task_aware_accuracy={score_summary['task_aware_accuracy']:.4f} "
                f"examples={score_summary['examples']}"
            )
        distributed_barrier()

    if is_main_process():
        summary = summarize_results(all_results)
        summary_path = os.path.join(output_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True, ensure_ascii=False)
        rank0_print(f"Saved batch benchmark summary to {summary_path}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
