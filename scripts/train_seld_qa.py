import argparse
import inspect
import json
import math
import os
import sys
import time
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

try:
    from peft import LoraConfig, TaskType, get_peft_model
except ImportError:
    LoraConfig = None
    TaskType = None
    get_peft_model = None


DEFAULT_LEGACY_REPO = None  # No longer needed; spatial_omni package is used directly
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-Omni-7B"
# Repo root (…/Spatial-Omni) and the vendored DCASE SELD baseline shipped inside
# the package. The vendored copy provides parameters.py + the SELD model defs so
# the SELD path needs no external DCASE checkout.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDORED_SELDNET_REPO = os.path.join(REPO_ROOT, "spatial_omni", "encoders", "seldnet")
DEFAULT_QA_DATASET_ROOT = (
    "${DCASE_BASELINE_REPO}/"
    "prepared_datasets/starss23_foa_plus_29cls_20s"
)
DEFAULT_QA_ROOT = (
    f"{DEFAULT_QA_DATASET_ROOT}/qa_pairs"
)
DEFAULT_QA_VERSION = "default"
QA_VERSION_TO_SUBDIR = {
    "default": "qa_pairs",
}
DEFAULT_SELD233_CKPT = os.environ.get(
    "SELD233_CKPT",
    "${DCASE_BASELINE_REPO}/3_1_dev_split0_multiaccdoa_foa_model.h5",
)
DEFAULT_SELD233_STATS_DIR = os.environ.get("SELD_FEATURE_STATS_DIR", "")
DEFAULT_OUTPUT_DIR = (
    "${DCASE_BASELINE_REPO}/"
    "spatial_qa_runs/default_run"
)
SAMPLE_RATE = 16000
MAX_AUDIO_SECONDS = 20
MAX_AUDIO_SAMPLES = SAMPLE_RATE * MAX_AUDIO_SECONDS
DEFAULT_LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Qwen spatial QA model on FOA + QA json/jsonl data.")
    parser.add_argument("--legacy-repo-path", type=str, default=DEFAULT_LEGACY_REPO)
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--qa-root",
        type=str,
        default=None,
        help="Explicit QA directory containing train/valid/test json or jsonl files. Overrides --qa-version.",
    )
    parser.add_argument(
        "--qa-roots",
        nargs="+",
        default=None,
        help="Multiple QA directories to concatenate for joint training/validation. Overrides --qa-root and --qa-version.",
    )
    parser.add_argument(
        "--audio-root",
        type=str,
        default=None,
        help="Optional root prepended to relative audio_path values in the QA "
             "jsonl (matches the SO-Dataset release layout where audio_path is "
             "e.g. 'audio/train/foo.wav'). Absolute audio_path values are used as-is.",
    )
    parser.add_argument(
        "--audio-roots",
        nargs="+",
        default=None,
        help="Multiple candidate audio roots, tried in order for relative audio_path. Overrides --audio-root.",
    )
    parser.add_argument(
        "--qa-version",
        type=str,
        default=DEFAULT_QA_VERSION,
        choices=sorted(QA_VERSION_TO_SUBDIR.keys()),
        help="Named QA dataset version under the prepared_datasets root.",
    )
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--valid-split", type=str, default="valid")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-valid-samples", type=int, default=None)
    parser.add_argument("--valid-subset-ratio", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--persistent-workers",
        action="store_true",
        help="Keep DataLoader workers alive across epochs when num_workers > 0.",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="DataLoader prefetch_factor used when num_workers > 0.",
    )
    parser.add_argument(
        "--feature-cache-max-entries",
        type=int,
        default=32,
        help="Maximum number of cached seld feature tensors kept in memory per collator process. Set <= 0 to disable in-memory caching.",
    )
    parser.add_argument(
        "--hidden-cache-max-entries",
        type=int,
        default=128,
        help="Maximum number of cached seld hidden tensors kept in memory per collator process. Set <= 0 to disable in-memory caching.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--local-rank", type=int, default=-1)
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=("float32", "bfloat16", "float16"),
    )
    parser.add_argument(
        "--baseline-repo-path",
        type=str,
        default=os.environ.get("DCASE_BASELINE_REPO", VENDORED_SELDNET_REPO),
        help="DCASE SELD baseline repo root (provides parameters.py). "
             "Defaults to the vendored copy under spatial_omni/encoders/seldnet.",
    )
    parser.add_argument("--seld-task-id", type=str, default="233")
    parser.add_argument("--seld-checkpoint-path", type=str, default=DEFAULT_SELD233_CKPT)
    parser.add_argument("--seld-feature-stats-dir", type=str, default=DEFAULT_SELD233_STATS_DIR)
    parser.add_argument(
        "--seld-feature-cache-manifest",
        type=str,
        default=None,
        help="Optional manifest.json produced by precompute_seld_feature_cache.py. "
        "When set, the collator loads cached feature_bridge outputs and still runs the SELD233 backbone online.",
    )
    parser.add_argument(
        "--seld-hidden-cache-manifest",
        type=str,
        default=None,
        help="Optional manifest.json produced by precompute_seld_hidden_cache.py. "
        "When set, the collator loads cached SELD233 hidden states instead of running "
        "feature_bridge + backbone online.",
    )
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--resume-checkpoint-path",
        type=str,
        default=None,
        help="Path to a saved *_trainable.pt checkpoint to resume from.",
    )
    parser.add_argument(
        "--resume-tag",
        type=str,
        default=None,
        help="Checkpoint tag under <output-dir>/checkpoints, e.g. 'last', 'best', or 'epoch_001'. Ignored when --resume-checkpoint-path is set.",
    )
    parser.add_argument(
        "--resume-model-only",
        action="store_true",
        help="Resume model weights only and reset optimizer/scheduler state.",
    )
    parser.add_argument("--save-full-model", action="store_true")
    parser.add_argument("--save-every-epoch", action="store_true")
    parser.add_argument("--save-every-n-optimizer-steps", type=int, default=1000)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--spatial-fp32", action="store_true")
    parser.add_argument("--optimizer-step-per-batch", action="store_true")
    parser.add_argument("--spatial-backbone-fp32", action="store_true")
    parser.add_argument("--valid-generate-batch-size", type=int, default=1)
    parser.add_argument("--valid-generate-max-samples", type=int, default=32)
    parser.add_argument("--valid-max-new-tokens", type=int, default=32)
    parser.add_argument("--valid-num-beams", type=int, default=1)
    parser.add_argument("--valid-do-sample", action="store_true")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--freeze-spatial-only",
        dest="train_mode",
        action="store_const",
        const="spatial_only",
        help="Only train the spatial adapter and projector.",
    )
    mode_group.add_argument(
        "--train-spatial-lora",
        dest="train_mode",
        action="store_const",
        const="spatial_lora",
        help="Train spatial encoder + adapter + projector + LLM LoRA.",
    )
    mode_group.add_argument(
        "--train-adapter-lora",
        dest="train_mode",
        action="store_const",
        const="adapter_lora",
        help="Freeze the spatial encoder backbone and train adapter + projector + LLM LoRA.",
    )
    mode_group.add_argument(
        "--train-all",
        dest="train_mode",
        action="store_const",
        const="all",
        help="Train all model parameters.",
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=list(DEFAULT_LORA_TARGET_MODULES),
    )
    parser.add_argument(
        "--lora-target-prefixes",
        nargs="+",
        default=["thinker.model"],
        help="Only apply LoRA to modules under these prefixes.",
    )
    parser.set_defaults(train_mode="all", save_every_epoch=True)
    args = parser.parse_args()
    args.qa_roots = resolve_qa_roots(args.qa_roots, args.qa_root, args.qa_version)
    args.qa_root = args.qa_roots[0]
    # Normalize audio roots: --audio-roots overrides --audio-root; both optional.
    if args.audio_roots:
        args.audio_roots = [os.path.abspath(r) for r in args.audio_roots]
    elif args.audio_root:
        args.audio_roots = [os.path.abspath(args.audio_root)]
    else:
        args.audio_roots = []
    args.freeze_spatial_only = args.train_mode == "spatial_only"
    return args


def add_legacy_repo_to_path(legacy_repo_path: str) -> None:
    # Guard against None/empty: inserting None into sys.path poisons later import
    # machinery (importlib find_spec / os.stat(None) -> TypeError). The vendored
    # spatial_omni.encoders.seldnet package is used directly, so this is normally
    # a no-op unless an explicit external legacy repo is requested.
    if not legacy_repo_path:
        return
    if legacy_repo_path not in sys.path:
        sys.path.insert(0, legacy_repo_path)


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[name]


def enable_gradient_checkpointing(model) -> None:
    if not hasattr(model, "gradient_checkpointing_enable"):
        return
    signature = inspect.signature(model.gradient_checkpointing_enable)
    if "gradient_checkpointing_kwargs" in signature.parameters:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    else:
        model.gradient_checkpointing_enable()


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def rank0_print(*args, **kwargs) -> None:
    if is_main_process():
        print(*args, **kwargs)


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def distributed_barrier() -> None:
    if is_distributed():
        dist.barrier()


def setup_distributed(args: argparse.Namespace) -> argparse.Namespace:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    args.distributed = world_size > 1
    if not args.distributed:
        args.rank = 0
        args.world_size = 1
        return args

    local_rank = args.local_rank
    if local_rank < 0:
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank < 0:
        raise RuntimeError("Distributed launch detected but LOCAL_RANK is missing.")
    args.local_rank = local_rank
    args.rank = int(os.environ["RANK"])
    args.world_size = world_size

    if not torch.cuda.is_available():
        raise RuntimeError("Distributed training currently requires CUDA.")

    torch.cuda.set_device(local_rank)
    args.device = f"cuda:{local_rank}"
    backend = "nccl"
    dist.init_process_group(backend=backend, init_method="env://")
    return args


def cleanup_distributed() -> None:
    if is_distributed():
        dist.destroy_process_group()


def reduce_scalar_sum(value: float, device: str) -> float:
    if not is_distributed():
        return float(value)
    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def sample_subset_indices(dataset_size: int, subset_ratio: float, seed: int, epoch: int) -> List[int]:
    if dataset_size <= 0:
        return []
    ratio = float(max(0.0, min(1.0, subset_ratio)))
    subset_size = dataset_size if ratio >= 1.0 else max(1, int(round(dataset_size * ratio)))
    subset_size = min(subset_size, dataset_size)
    rng = np.random.default_rng(seed + epoch)
    indices = rng.choice(dataset_size, size=subset_size, replace=False)
    return sorted(int(index) for index in indices.tolist())


def resolve_qa_split_path(qa_root: str, split_name: str) -> str:
    candidates = [
        os.path.join(qa_root, f"{split_name}.jsonl"),
        os.path.join(qa_root, f"{split_name}.json"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    raise FileNotFoundError(
        f"Missing QA split '{split_name}' under {qa_root}. Checked: {candidates}"
    )


def resolve_qa_root(qa_root: Optional[str], qa_version: str) -> str:
    if qa_root is not None and str(qa_root).strip():
        return os.path.abspath(str(qa_root))
    subdir = QA_VERSION_TO_SUBDIR[qa_version]
    return os.path.abspath(os.path.join(DEFAULT_QA_DATASET_ROOT, subdir))


def resolve_qa_roots(
    qa_roots: Optional[List[str]],
    qa_root: Optional[str],
    qa_version: str,
) -> List[str]:
    if qa_roots:
        roots = [os.path.abspath(str(root)) for root in qa_roots if str(root).strip()]
        if not roots:
            raise ValueError("--qa-roots was provided but no valid roots were found.")
        return roots
    return [resolve_qa_root(qa_root, qa_version)]


def load_qa_records(qa_path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    qa_path = os.path.abspath(qa_path)
    records: List[Dict[str, Any]] = []
    if qa_path.endswith(".jsonl"):
        with open(qa_path, "r", encoding="utf-8") as handle:
            for line_index, line in enumerate(handle):
                if max_samples is not None and len(records) >= max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"Expected JSON object at line {line_index} in {qa_path}")
                records.append(record)
        return records

    if qa_path.endswith(".json"):
        with open(qa_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            iterable = payload
        elif isinstance(payload, dict) and isinstance(payload.get("records"), list):
            iterable = payload["records"]
        elif isinstance(payload, dict) and isinstance(payload.get("data"), list):
            iterable = payload["data"]
        else:
            raise ValueError(
                f"Unsupported QA JSON structure in {qa_path}. Expected a list or a dict with 'records'/'data'."
            )
        for record_index, record in enumerate(iterable):
            if max_samples is not None and len(records) >= max_samples:
                break
            if not isinstance(record, dict):
                raise ValueError(f"Expected JSON object at index {record_index} in {qa_path}")
            records.append(record)
        return records

    raise ValueError(f"Unsupported QA file format: {qa_path}")


def build_left_padded_text_batch(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prefix_lengths: torch.LongTensor,
    pad_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    max_length = int(prefix_lengths.max().item())
    batch_size = int(input_ids.shape[0])
    generation_input_ids = torch.full(
        (batch_size, max_length),
        fill_value=pad_token_id,
        dtype=input_ids.dtype,
    )
    generation_attention_mask = torch.zeros(
        (batch_size, max_length),
        dtype=attention_mask.dtype,
    )
    for index, prefix_length in enumerate(prefix_lengths.tolist()):
        start = max_length - prefix_length
        generation_input_ids[index, start:] = input_ids[index, :prefix_length]
        generation_attention_mask[index, start:] = 1
    return generation_input_ids, generation_attention_mask


class QAAudioJsonlDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        max_samples: Optional[int] = None,
        feature_cache_manifest_path: Optional[str] = None,
        hidden_cache_manifest_path: Optional[str] = None,
        audio_roots: Optional[List[str]] = None,
    ) -> None:
        self.records: List[Dict[str, Any]] = []
        feature_cache_manifest = load_tensor_cache_manifest(feature_cache_manifest_path)
        hidden_cache_manifest = load_hidden_cache_manifest(hidden_cache_manifest_path)
        raw_records = load_qa_records(jsonl_path, max_samples=max_samples)
        qa_dir = os.path.dirname(os.path.abspath(jsonl_path))
        for record_index, record in enumerate(raw_records):
            audio_path = record.get("audio_path")
            # Schema compat: the SO-Dataset release uses `question`; older SELD
            # QA used `prompt`. Accept either, normalizing to `prompt` so the
            # collator (which reads `prompt`) works for both.
            prompt = record.get("prompt")
            if prompt is None:
                prompt = record.get("question")
                if prompt is not None:
                    record["prompt"] = prompt
            answer = record.get("answer")
            # Resolve audio_path: absolute as-is; relative against --audio-root(s),
            # then the QA dir and its parent (release layout puts audio/ beside qa/).
            resolved = self._resolve_audio_path(audio_path, audio_roots, qa_dir)
            if resolved is None:
                raise FileNotFoundError(
                    f"Missing audio_path for record {record_index} in {jsonl_path}: "
                    f"{audio_path!r} (searched audio_roots={audio_roots}, qa_dir={qa_dir})"
                )
            record["audio_path"] = resolved
            audio_path = resolved
            if prompt is None or answer is None:
                raise ValueError(
                    f"Each record must contain prompt/question and answer. "
                    f"Broken record {record_index} in {jsonl_path}"
                )
            resolved_audio_path = os.path.abspath(audio_path)
            if feature_cache_manifest is not None:
                cache_path = feature_cache_manifest.get(resolved_audio_path)
                if cache_path is None:
                    raise KeyError(
                        f"Audio path missing from feature cache manifest: {resolved_audio_path}"
                    )
                record["seld_feature_cache_path"] = cache_path
            if hidden_cache_manifest is not None:
                cache_path = hidden_cache_manifest.get(resolved_audio_path)
                if cache_path is None:
                    raise KeyError(
                        f"Audio path missing from hidden cache manifest: {resolved_audio_path}"
                    )
                record["seld_hidden_cache_path"] = cache_path
            self.records.append(record)

    @staticmethod
    def _resolve_audio_path(
        audio_path: Optional[str],
        audio_roots: Optional[List[str]],
        qa_dir: str,
    ) -> Optional[str]:
        if not audio_path:
            return None
        if os.path.isabs(audio_path):
            return audio_path if os.path.exists(audio_path) else None
        candidates: List[str] = []
        for root in (audio_roots or []):
            candidates.append(os.path.join(root, audio_path))
        candidates.append(os.path.join(qa_dir, audio_path))
        candidates.append(os.path.join(os.path.dirname(qa_dir), audio_path))
        for cand in candidates:
            if os.path.exists(cand):
                return os.path.abspath(cand)
        return None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.records[index]


def build_qa_dataset(
    qa_roots: List[str],
    split_name: str,
    max_samples: Optional[int],
    feature_cache_manifest_path: Optional[str],
    hidden_cache_manifest_path: Optional[str],
    audio_roots: Optional[List[str]] = None,
) -> Tuple[Dataset, List[str], List[int]]:
    split_paths: List[str] = []
    datasets: List[Dataset] = []
    dataset_sizes: List[int] = []

    for qa_root in qa_roots:
        split_path = resolve_qa_split_path(qa_root, split_name)
        dataset = QAAudioJsonlDataset(
            split_path,
            max_samples=max_samples,
            feature_cache_manifest_path=feature_cache_manifest_path,
            hidden_cache_manifest_path=hidden_cache_manifest_path,
            audio_roots=audio_roots,
        )
        split_paths.append(split_path)
        datasets.append(dataset)
        dataset_sizes.append(len(dataset))

    if len(datasets) == 1:
        return datasets[0], split_paths, dataset_sizes
    return ConcatDataset(datasets), split_paths, dataset_sizes


@dataclass
class SpatialQACollator:
    processor: Any
    ignore_index: int = -100
    sample_rate: int = SAMPLE_RATE
    max_audio_samples: int = MAX_AUDIO_SAMPLES
    include_generation_inputs: bool = False
    feature_cache_store: Optional[OrderedDict[str, Dict[str, torch.Tensor]]] = None
    hidden_cache_store: Optional[OrderedDict[str, Dict[str, torch.Tensor]]] = None
    feature_cache_max_entries: int = 32
    hidden_cache_max_entries: int = 128

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        audio_arrays: List[np.ndarray] = []
        full_texts: List[str] = []
        prefix_texts: List[str] = []
        answer_suffix_texts: List[str] = []
        meta: List[Dict[str, Any]] = []
        eos_token = getattr(self.processor.tokenizer, "eos_token", None)
        if not eos_token:
            raise ValueError("Tokenizer is missing eos_token; cannot append EOS to supervised answers.")

        for feature in features:
            wav, sr = sf.read(feature["audio_path"], dtype="float32", always_2d=True)
            if sr != self.sample_rate:
                raise ValueError(
                    f"Expected {self.sample_rate} Hz audio, got {sr} for {feature['audio_path']}"
                )
            wav = wav.T
            if wav.shape[0] != 4:
                raise ValueError(
                    f"Expected 4-channel FOA audio, got shape {wav.shape} for {feature['audio_path']}"
                )
            if wav.shape[1] > self.max_audio_samples:
                wav = wav[:, : self.max_audio_samples]

            prefix_text = (
                f"{self.processor.audio_token}{self.processor.spatial_token}\n"
                f"{feature['prompt'].rstrip()}\n"
            )
            answer_text = str(feature["answer"]).strip()
            answer_suffix_text = answer_text + eos_token
            full_text = prefix_text + answer_suffix_text

            audio_arrays.append(wav.astype(np.float32, copy=False))
            prefix_texts.append(prefix_text)
            answer_suffix_texts.append(answer_suffix_text)
            full_texts.append(full_text)
            meta.append(
                {
                    "pair_id": feature.get("pair_id"),
                    "task_name": feature.get("task_name"),
                    "answer": feature.get("answer"),
                    "audio_path": feature.get("audio_path"),
                    "prompt": feature.get("prompt"),
                    "question": feature.get("question"),
                }
            )

        feature_payload = self._build_feature_cache_payload(features)
        hidden_payload = self._build_hidden_cache_payload(features)
        if feature_payload and hidden_payload:
            raise ValueError("Feature cache and hidden cache cannot be enabled in the same batch.")

        processor_kwargs = dict(
            text=full_texts,
            audio=audio_arrays,
            padding=True,
            return_tensors="pt",
        )
        processor_kwargs.update(feature_payload)
        processor_kwargs.update(hidden_payload)
        batch = self.processor(**processor_kwargs)

        labels = batch["input_ids"].clone()
        if "attention_mask" in batch:
            labels = labels.masked_fill(batch["attention_mask"] == 0, self.ignore_index)

        prefix_lengths = self._compute_prefix_lengths(
            batch=batch,
            answer_suffix_texts=answer_suffix_texts,
        )
        for index, prefix_length in enumerate(prefix_lengths.tolist()):
            labels[index, :prefix_length] = self.ignore_index

        if (labels != self.ignore_index).sum(dim=1).min().item() <= 0:
            raise ValueError("At least one sample has no supervised answer tokens left.")

        batch["labels"] = labels
        batch["meta"] = meta
        batch["prefix_lengths"] = prefix_lengths
        if self.include_generation_inputs:
            generation_batch = self._build_generation_batch(
                batch=batch,
                prefix_lengths=prefix_lengths,
            )
            for key, value in generation_batch.items():
                if isinstance(value, torch.Tensor):
                    batch[f"gen_{key}"] = value
        return batch

    def _build_hidden_cache_payload(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        cache_paths = [feature.get("seld_hidden_cache_path") for feature in features]
        if not any(cache_paths):
            return {}
        if not all(cache_paths):
            raise ValueError("Either every sample in the batch must provide seld_hidden_cache_path, or none may.")

        cache_entries = [self._load_hidden_cache(cache_path) for cache_path in cache_paths]
        hidden_lengths = torch.tensor(
            [int(entry["seld_hidden_lengths"]) for entry in cache_entries],
            dtype=torch.long,
        )
        max_hidden_steps = int(hidden_lengths.max().item())
        hidden_dim = int(cache_entries[0]["seld_hidden_states"].shape[-1])
        hidden_states = torch.zeros(
            (len(cache_entries), max_hidden_steps, hidden_dim),
            dtype=cache_entries[0]["seld_hidden_states"].dtype,
        )
        for index, entry in enumerate(cache_entries):
            current_hidden = entry["seld_hidden_states"]
            current_length = int(hidden_lengths[index].item())
            if current_hidden.ndim != 2:
                raise ValueError(
                    "Cached seld_hidden_states must have shape [T_seld, D_seld], "
                    f"got {tuple(current_hidden.shape)} for {cache_paths[index]}"
                )
            if current_hidden.shape[0] != current_length:
                raise ValueError(
                    f"Cached hidden length mismatch for {cache_paths[index]}: "
                    f"{current_hidden.shape[0]} vs {current_length}"
                )
            hidden_states[index, :current_length] = current_hidden

        hidden_attention_mask = (
            torch.arange(max_hidden_steps).unsqueeze(0) < hidden_lengths.unsqueeze(1)
        )
        return {
            "seld_hidden_states": hidden_states,
            "seld_hidden_attention_mask": hidden_attention_mask,
            "seld_hidden_lengths": hidden_lengths,
        }

    def _build_feature_cache_payload(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        cache_paths = [feature.get("seld_feature_cache_path") for feature in features]
        if not any(cache_paths):
            return {}
        if not all(cache_paths):
            raise ValueError("Either every sample in the batch must provide seld_feature_cache_path, or none may.")

        cache_entries = [self._load_feature_cache(cache_path) for cache_path in cache_paths]
        feature_lengths = torch.tensor(
            [int(entry["seld_feature_lengths"]) for entry in cache_entries],
            dtype=torch.long,
        )
        max_feature_steps = int(feature_lengths.max().item())
        num_channels = int(cache_entries[0]["seld_features"].shape[0])
        num_mels = int(cache_entries[0]["seld_features"].shape[-1])
        feature_tensor = torch.zeros(
            (len(cache_entries), num_channels, max_feature_steps, num_mels),
            dtype=cache_entries[0]["seld_features"].dtype,
        )
        for index, entry in enumerate(cache_entries):
            current_features = entry["seld_features"]
            current_length = int(feature_lengths[index].item())
            if current_features.ndim != 3:
                raise ValueError(
                    "Cached seld_features must have shape [7, T_feat, M], "
                    f"got {tuple(current_features.shape)} for {cache_paths[index]}"
                )
            if current_features.shape[1] != current_length:
                raise ValueError(
                    f"Cached feature length mismatch for {cache_paths[index]}: "
                    f"{current_features.shape[1]} vs {current_length}"
                )
            feature_tensor[index, :, :current_length, :] = current_features

        feature_attention_mask = (
            torch.arange(max_feature_steps).unsqueeze(0) < feature_lengths.unsqueeze(1)
        )
        return {
            "seld_features": feature_tensor,
            "seld_feature_attention_mask": feature_attention_mask,
            "seld_feature_lengths": feature_lengths,
        }

    def _load_feature_cache(self, cache_path: str) -> Dict[str, torch.Tensor]:
        if self.feature_cache_max_entries > 0:
            if self.feature_cache_store is None:
                self.feature_cache_store = OrderedDict()
            cached = self.feature_cache_store.get(cache_path)
            if cached is not None:
                self.feature_cache_store.move_to_end(cache_path)
                return cached

        payload = torch.load(cache_path, map_location="cpu")
        if "seld_features" not in payload:
            raise KeyError(f"Feature cache missing 'seld_features': {cache_path}")
        feature_tensor = torch.as_tensor(payload["seld_features"])
        if feature_tensor.ndim != 3:
            raise ValueError(
                f"Cached seld_features must have shape [7, T_feat, M], got {tuple(feature_tensor.shape)}"
            )
        if "seld_feature_lengths" in payload:
            feature_length = int(torch.as_tensor(payload["seld_feature_lengths"]).item())
        else:
            feature_length = int(feature_tensor.shape[1])
        cached = {
            "seld_features": feature_tensor.to(dtype=feature_tensor.dtype).contiguous(),
            "seld_feature_lengths": torch.tensor(feature_length, dtype=torch.long),
        }
        if self.feature_cache_max_entries > 0 and self.feature_cache_store is not None:
            self.feature_cache_store[cache_path] = cached
            self.feature_cache_store.move_to_end(cache_path)
            while len(self.feature_cache_store) > int(self.feature_cache_max_entries):
                self.feature_cache_store.popitem(last=False)
        return cached

    def _load_hidden_cache(self, cache_path: str) -> Dict[str, torch.Tensor]:
        if self.hidden_cache_max_entries > 0:
            if self.hidden_cache_store is None:
                self.hidden_cache_store = OrderedDict()
            cached = self.hidden_cache_store.get(cache_path)
            if cached is not None:
                self.hidden_cache_store.move_to_end(cache_path)
                return cached

        payload = torch.load(cache_path, map_location="cpu")
        if "seld_hidden_states" not in payload:
            raise KeyError(f"Hidden cache missing 'seld_hidden_states': {cache_path}")
        hidden_states = torch.as_tensor(payload["seld_hidden_states"])
        if hidden_states.ndim != 2:
            raise ValueError(
                f"Cached seld_hidden_states must have shape [T_seld, D_seld], got {tuple(hidden_states.shape)}"
            )
        if "seld_hidden_lengths" in payload:
            hidden_length = int(torch.as_tensor(payload["seld_hidden_lengths"]).item())
        else:
            hidden_length = int(hidden_states.shape[0])
        cached = {
            "seld_hidden_states": hidden_states.to(dtype=hidden_states.dtype).contiguous(),
            "seld_hidden_lengths": torch.tensor(hidden_length, dtype=torch.long),
        }
        if self.hidden_cache_max_entries > 0 and self.hidden_cache_store is not None:
            self.hidden_cache_store[cache_path] = cached
            self.hidden_cache_store.move_to_end(cache_path)
            while len(self.hidden_cache_store) > int(self.hidden_cache_max_entries):
                self.hidden_cache_store.popitem(last=False)
        return cached

    def _compute_prefix_lengths(
        self,
        batch: Dict[str, Any],
        answer_suffix_texts: List[str],
    ) -> torch.LongTensor:
        answer_batch = self.processor.tokenizer(
            answer_suffix_texts,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        answer_lengths = answer_batch["attention_mask"].sum(dim=1).to(dtype=torch.long)
        valid_lengths = batch["attention_mask"].sum(dim=1).to(dtype=torch.long)
        prefix_lengths = valid_lengths - answer_lengths
        if (prefix_lengths < 0).any():
            raise ValueError(
                "Computed negative prefix length; answer tokenization exceeds full input length."
            )
        return prefix_lengths

    def _expand_modal_placeholders(
        self,
        prefix_text: str,
        audio_count: int,
        spatial_count: int,
        video_count: int,
    ) -> str:
        if self.processor.audio_token in prefix_text:
            prefix_text = prefix_text.replace(self.processor.audio_token, self.processor.audio_token * audio_count, 1)
        if self.processor.spatial_token in prefix_text:
            prefix_text = prefix_text.replace(
                self.processor.spatial_token,
                self.processor.spatial_token * spatial_count,
                1,
            )
        if self.processor.video_token in prefix_text:
            prefix_text = prefix_text.replace(self.processor.video_token, self.processor.video_token * video_count, 1)
        return prefix_text

    def _build_generation_batch(
        self,
        batch: Dict[str, Any],
        prefix_lengths: torch.LongTensor,
    ) -> Dict[str, torch.Tensor]:
        pad_token_id = int(self.processor.tokenizer.pad_token_id or 0)
        generation_input_ids, generation_attention_mask = build_left_padded_text_batch(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            prefix_lengths=prefix_lengths,
            pad_token_id=pad_token_id,
        )

        generation_batch: Dict[str, torch.Tensor] = {
            "input_ids": generation_input_ids,
            "attention_mask": generation_attention_mask,
        }
        for key, value in list(batch.items()):
            if key in {"input_ids", "attention_mask", "labels", "meta", "prefix_lengths"}:
                continue
            if isinstance(value, torch.Tensor):
                generation_batch[key] = value
        return generation_batch


def make_loader(
    dataset: Dataset,
    collator: SpatialQACollator,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    sampler=None,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
) -> DataLoader:
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle if sampler is None else False,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": collator,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
        loader_kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(**loader_kwargs)


def load_hidden_cache_manifest(hidden_cache_manifest_path: Optional[str]) -> Optional[Dict[str, str]]:
    return load_tensor_cache_manifest(hidden_cache_manifest_path)


def load_tensor_cache_manifest(cache_manifest_path: Optional[str]) -> Optional[Dict[str, str]]:
    if cache_manifest_path is None:
        return None
    with open(cache_manifest_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    manifest_root = os.path.dirname(os.path.abspath(cache_manifest_path))
    if isinstance(payload, dict) and "entries" in payload:
        entries = payload["entries"]
        cache_root = payload.get("cache_dir", manifest_root)
    else:
        entries = payload
        cache_root = manifest_root

    if not isinstance(entries, dict):
        raise ValueError("Cache manifest must be a JSON object or contain an 'entries' object.")

    resolved: Dict[str, str] = {}
    for audio_path, cache_path in entries.items():
        absolute_audio_path = os.path.abspath(audio_path)
        if not os.path.isabs(cache_path):
            cache_path = os.path.join(cache_root, cache_path)
        resolved[absolute_audio_path] = os.path.abspath(cache_path)
    return resolved


def build_processor(model_id: str):
    from spatial_omni.model.processing_qwen2_5_omni import Qwen2_5OmniProcessor
    from spatial_omni.model.processing_so import Qwen2_5OmniSpatialProcessor

    base_processor = Qwen2_5OmniProcessor.from_pretrained(model_id)
    return Qwen2_5OmniSpatialProcessor(
        image_processor=base_processor.image_processor,
        feature_extractor=base_processor.feature_extractor,
        tokenizer=base_processor.tokenizer,
        chat_template=base_processor.chat_template,
    )


def infer_feature_to_seld_ratio(task_params: Dict[str, Any]) -> int:
    feature_sequence_length = task_params.get("feature_sequence_length")
    label_sequence_length = task_params.get("label_sequence_length")
    if feature_sequence_length is not None and label_sequence_length:
        ratio = int(round(float(feature_sequence_length) / float(label_sequence_length)))
        if ratio > 0:
            return ratio
    t_pool_size = task_params.get("t_pool_size")
    if isinstance(t_pool_size, (list, tuple)) and len(t_pool_size) > 0:
        ratio = int(t_pool_size[0])
        if ratio > 0:
            return ratio
    return 5


def load_seld_task_params(baseline_repo_path: str, task_id: str) -> Dict[str, Any]:
    # Prefer the vendored copy as a *proper package* import. This avoids polluting
    # sys.path[0] with a directory full of bare modules (parameters.py, etc.),
    # which would otherwise sit ahead of site-packages for the rest of the process
    # and corrupt later importlib.find_spec/metadata lookups (e.g. transformers'
    # flash_attn detection crashing on an os.stat(None) during a runtime lazy import).
    try:
        from spatial_omni.encoders.seldnet import parameters as _params
        return _params.get_params(str(task_id))
    except ImportError:
        pass
    # External baseline checkout: insert on sys.path only long enough to import,
    # then remove it so the search path is left clean.
    baseline_repo_abs = os.path.abspath(baseline_repo_path)
    inserted = False
    if baseline_repo_abs not in sys.path:
        sys.path.insert(0, baseline_repo_abs)
        inserted = True
    try:
        import parameters
        return parameters.get_params(str(task_id))
    finally:
        if inserted:
            try:
                sys.path.remove(baseline_repo_abs)
            except ValueError:
                pass


def build_model(args: argparse.Namespace, processor):
    from spatial_omni.model.configuration import Qwen2_5OmniConfig
    from spatial_omni.model.modeling_so_thinker import Qwen2_5OmniSpatialForConditionalGeneration

    config = Qwen2_5OmniConfig.from_pretrained(args.model_id)
    config.loss_type = "ForCausalLMLoss"
    thinker_config = config.thinker_config
    task_params = load_seld_task_params(args.baseline_repo_path, args.seld_task_id)
    thinker_config.loss_type = "ForCausalLMLoss"
    thinker_config.use_seld_spatial_modality = True
    thinker_config.seld_checkpoint_path = args.seld_checkpoint_path
    thinker_config.seld_baseline_repo_path = args.baseline_repo_path
    thinker_config.seld_task_id = str(args.seld_task_id)
    if args.seld_feature_stats_dir == DEFAULT_SELD233_STATS_DIR and str(args.seld_task_id) != "233":
        thinker_config.seld_feature_stats_dir = task_params["feat_label_dir"]
    else:
        thinker_config.seld_feature_stats_dir = args.seld_feature_stats_dir
    thinker_config.seld_hop_length = int(task_params.get("hop_len", 320))
    thinker_config.seld_feature_to_seld_ratio = infer_feature_to_seld_ratio(task_params)
    thinker_config.seld_num_mel_bins = int(task_params.get("nb_mel_bins", thinker_config.seld_num_mel_bins))
    thinker_config.seld_max_audio_seconds = float(MAX_AUDIO_SECONDS)
    thinker_config.seld_freeze_backbone = args.train_mode in {"spatial_only", "adapter_lora"}

    model = Qwen2_5OmniSpatialForConditionalGeneration.from_pretrained(
        args.model_id,
        config=config,
        torch_dtype=dtype_from_name(args.dtype),
        low_cpu_mem_usage=True,
    )
    processor.sync_spatial_tokenizer_with_model(model)
    model.disable_talker()
    if args.gradient_checkpointing:
        enable_gradient_checkpointing(model)
        model.config.use_cache = False
        model.thinker.config.use_cache = False
    model.to(args.device)
    if args.train_mode == "spatial_lora":
        materialize_spatial_backbone(model, args.device)
    if args.spatial_fp32:
        cast_spatial_modules_to_fp32(model)
    if args.spatial_backbone_fp32 and args.train_mode == "spatial_lora":
        cast_spatial_backbone_to_fp32(model)
    return model


def cast_spatial_modules_to_fp32(model) -> None:
    for module_name in ("seld_spatial_adapter", "seld_spatial_projector"):
        module = getattr(model.thinker, module_name, None)
        if module is not None:
            module.to(dtype=torch.float32)


def materialize_spatial_backbone(model, device: str) -> None:
    backbone = getattr(model.thinker, "seld_backbone", None)
    if backbone is None:
        raise RuntimeError("Model thinker is missing seld_backbone.")
    backbone.freeze_backbone = False
    baseline_model = backbone._get_or_create_model(torch.device(device))
    for parameter in baseline_model.parameters():
        parameter.requires_grad_(True)
    baseline_model.train()


def cast_spatial_backbone_to_fp32(model) -> None:
    backbone = getattr(model.thinker, "seld_backbone", None)
    if backbone is None or getattr(backbone, "_baseline_model", None) is None:
        return
    backbone._baseline_model.to(dtype=torch.float32)


def freeze_all_but_spatial_modules(model) -> List[str]:
    enabled: List[str] = []
    for _, parameter in model.named_parameters():
        parameter.requires_grad_(False)
    for name, parameter in model.named_parameters():
        if "seld_spatial_adapter" in name or "seld_spatial_projector" in name:
            parameter.requires_grad_(True)
            enabled.append(name)
    return enabled


def resolve_lora_target_modules(
    model,
    prefixes: List[str],
    target_suffixes: List[str],
) -> List[str]:
    resolved: List[str] = []
    for module_name, _module in model.named_modules():
        if not any(module_name.startswith(prefix) for prefix in prefixes):
            continue
        if module_name.rsplit(".", 1)[-1] in target_suffixes:
            resolved.append(module_name)
    if not resolved:
        raise ValueError(
            f"No LoRA target modules found under prefixes={prefixes} with suffixes={target_suffixes}."
        )
    return sorted(set(resolved))


def apply_llm_lora(model, args: argparse.Namespace):
    if get_peft_model is None or LoraConfig is None or TaskType is None:
        raise ImportError("peft is required for --train-spatial-lora.")

    target_modules = resolve_lora_target_modules(
        model,
        prefixes=list(args.lora_target_prefixes),
        target_suffixes=list(args.lora_target_modules),
    )
    lora_config = LoraConfig(
        r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    if args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        enable_gradient_checkpointing(model)
    return model, target_modules


def configure_spatial_lora_training(model, args: argparse.Namespace) -> List[str]:
    enabled: List[str] = []
    for _, parameter in model.named_parameters():
        parameter.requires_grad_(False)

    for name, parameter in model.named_parameters():
        if "seld_backbone" in name and (".fnn_list." in name or ".sed_head." in name):
            continue
        if (
            "seld_backbone" in name
            or "seld_spatial_adapter" in name
            or "seld_spatial_projector" in name
            or "lora_" in name
        ):
            parameter.requires_grad_(True)
            enabled.append(name)
    return enabled


def configure_adapter_lora_training(model, args: argparse.Namespace) -> List[str]:
    enabled: List[str] = []
    for _, parameter in model.named_parameters():
        parameter.requires_grad_(False)

    for name, parameter in model.named_parameters():
        if (
            "seld_spatial_adapter" in name
            or "seld_spatial_projector" in name
            or "lora_" in name
        ):
            parameter.requires_grad_(True)
            enabled.append(name)
    return enabled


def build_optimizer(model, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    decay_params = []
    no_decay_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim == 1 or name.endswith(".bias") or "norm" in name.lower():
            no_decay_params.append(parameter)
        else:
            decay_params.append(parameter)
    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(param_groups, lr=lr)


def resolve_resume_checkpoint_path(args: argparse.Namespace) -> Optional[str]:
    if args.resume_checkpoint_path is not None:
        return os.path.abspath(args.resume_checkpoint_path)
    if args.resume_tag:
        return os.path.abspath(
            os.path.join(args.output_dir, "checkpoints", f"{args.resume_tag}_trainable.pt")
        )
    return None


def infer_best_valid_from_output_dir(output_dir: str) -> Tuple[float, int]:
    best_path = os.path.join(output_dir, "checkpoints", "best_trainable.pt")
    if not os.path.exists(best_path):
        return float("inf"), -1
    payload = torch.load(best_path, map_location="cpu")
    metrics = payload.get("metrics") or {}
    valid_loss = float(metrics.get("valid_loss", float("inf")))
    epoch = int(payload.get("epoch", -1))
    return valid_loss, epoch


def resume_training_state(
    model,
    optimizer,
    scheduler,
    resume_checkpoint_path: str,
    resume_model_only: bool,
    device: str,
) -> Dict[str, Any]:
    checkpoint = torch.load(resume_checkpoint_path, map_location="cpu")
    from spatial_omni.utils.ckpt_compat import remap_legacy_state_dict
    trainable_state = checkpoint.get("trainable_state_dict", checkpoint)
    trainable_state = remap_legacy_state_dict(trainable_state)
    load_result = unwrap_model(model).load_state_dict(trainable_state, strict=False)

    if not resume_model_only:
        optimizer_state = checkpoint.get("optimizer")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        scheduler_state = checkpoint.get("scheduler")
        if scheduler is not None and scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)

    metrics = checkpoint.get("metrics") or {}
    if resume_model_only:
        resumed_epoch = 0
        global_optimizer_step = 0
    else:
        resumed_epoch = int(checkpoint.get("epoch", 0))
        scheduler_step = 0
        if scheduler is not None:
            scheduler_step = int(getattr(scheduler, "last_epoch", 0))
        global_optimizer_step = int(metrics.get("global_optimizer_step", 0))
        if global_optimizer_step <= 0:
            global_optimizer_step = max(
                scheduler_step,
                int(checkpoint.get("step", 0)),
            )
    return {
        "checkpoint": checkpoint,
        "load_result": load_result,
        "start_epoch": resumed_epoch + 1,
        "global_optimizer_step": global_optimizer_step,
        "global_micro_step": 0,
    }


def save_trainable_checkpoint(
    model,
    optimizer,
    scheduler,
    output_path: str,
    epoch: int,
    step: int,
    metrics: Dict[str, Any],
) -> None:
    model = unwrap_model(model)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    trainable_state = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            trainable_state[name] = parameter.detach().cpu()
    payload = {
        "epoch": epoch,
        "step": step,
        "metrics": metrics,
        "trainable_state_dict": trainable_state,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }
    torch.save(payload, output_path)


def save_artifacts(
    model,
    processor,
    optimizer,
    scheduler,
    args: argparse.Namespace,
    epoch: int,
    step: int,
    metrics: Dict[str, Any],
    tag: str,
) -> None:
    if not is_main_process():
        return
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    trainable_path = os.path.join(ckpt_dir, f"{tag}_trainable.pt")
    save_trainable_checkpoint(model, optimizer, scheduler, trainable_path, epoch, step, metrics)
    if args.save_full_model:
        full_dir = os.path.join(ckpt_dir, f"{tag}_full")
        unwrap_model(model).save_pretrained(full_dir)
        processor.save_pretrained(full_dir)


def to_device(batch: Dict[str, Any], device: str) -> Dict[str, torch.Tensor]:
    model_inputs: Dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if key in {"meta", "prefix_lengths"}:
            continue
        if isinstance(value, torch.Tensor):
            model_inputs[key] = value.to(device)
    return model_inputs


def count_supervised_tokens(labels: torch.Tensor, ignore_index: int = -100) -> int:
    return int((labels != ignore_index).sum().item())


def normalize_answer(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def compute_batch_loss(model, batch: Dict[str, Any], device: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
    model_inputs = to_device(batch, device)
    outputs = model(**model_inputs, return_dict=True)
    if outputs.loss is None:
        raise RuntimeError("Model returned loss=None.")
    if not torch.isfinite(outputs.loss):
        raise RuntimeError(f"Non-finite loss encountered: {float(outputs.loss.detach().item())}")
    batch_stats = {
        "loss": float(outputs.loss.detach().item()),
        "logits_shape": tuple(outputs.logits.shape) if outputs.logits is not None else None,
        "supervised_tokens": count_supervised_tokens(batch["labels"]),
    }
    return outputs.loss, batch_stats


def evaluate(model, loader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    total_weighted_loss = 0.0
    total_batches = 0
    total_supervised = 0
    progress = tqdm(
        loader,
        desc="valid",
        leave=False,
        disable=not is_main_process(),
    )
    with torch.no_grad():
        for batch in progress:
            loss, stats = compute_batch_loss(model, batch, device)
            total_weighted_loss += float(loss.detach().item()) * int(stats["supervised_tokens"])
            total_batches += 1
            total_supervised += int(stats["supervised_tokens"])
            if is_main_process():
                progress.set_postfix(
                    loss=f"{stats['loss']:.4f}",
                    supervised=int(stats["supervised_tokens"]),
                )
    if total_batches == 0:
        raise RuntimeError("Validation loader is empty.")
    total_weighted_loss = reduce_scalar_sum(total_weighted_loss, device)
    total_batches = int(reduce_scalar_sum(float(total_batches), device))
    total_supervised = int(reduce_scalar_sum(float(total_supervised), device))
    valid_loss = total_weighted_loss / max(total_supervised, 1)
    return {
        "valid_loss": valid_loss,
        "valid_batches": float(total_batches),
        "valid_supervised_tokens": float(total_supervised),
    }


def train_one_epoch(
    model,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: str,
    grad_accum_steps: int,
    max_grad_norm: float,
    log_every: int,
    epoch: int,
    optimizer_step_per_batch: bool,
    writer=None,
    global_step_start: int = 0,
    global_optimizer_step_start: int = 0,
    on_optimizer_step=None,
) -> Dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_weighted_loss = 0.0
    total_batches = 0
    total_supervised = 0
    optimizer_steps = 0
    epoch_start = time.time()
    progress = tqdm(
        loader,
        desc=f"epoch {epoch}",
        leave=False,
        disable=not is_main_process(),
    )

    for step, batch in enumerate(progress, start=1):
        loss, stats = compute_batch_loss(model, batch, device)
        total_weighted_loss += stats["loss"] * int(stats["supervised_tokens"])
        total_batches += 1
        total_supervised += int(stats["supervised_tokens"])

        should_step = optimizer_step_per_batch or (step % grad_accum_steps == 0) or (step == len(loader))
        sync_context = nullcontext()
        if is_distributed() and isinstance(model, DDP) and not should_step:
            sync_context = model.no_sync()
        scaled_loss = loss / grad_accum_steps
        with sync_context:
            scaled_loss.backward()
        if should_step:
            if max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    max_grad_norm,
                )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            if on_optimizer_step is not None:
                on_optimizer_step(
                    global_optimizer_step_start + optimizer_steps,
                    {
                        "epoch": epoch,
                        "micro_step": step,
                        "loss": float(stats["loss"]),
                        "supervised_tokens": int(stats["supervised_tokens"]),
                    },
                )

        current_lr = optimizer.param_groups[0]["lr"]
        if is_main_process():
            progress.set_postfix(
                step=f"{step}/{len(loader)}",
                loss=f"{stats['loss']:.4f}",
                lr=f"{current_lr:.2e}",
            )
        if writer is not None:
            global_micro_step = global_step_start + step
            writer.add_scalar("train/batch_loss", float(stats["loss"]), global_micro_step)
            writer.add_scalar("train/batch_supervised_tokens", float(stats["supervised_tokens"]), global_micro_step)
            writer.add_scalar("train/lr", float(current_lr), global_micro_step)

    elapsed = time.time() - epoch_start
    if total_batches == 0:
        raise RuntimeError("Training loader is empty.")
    total_weighted_loss = reduce_scalar_sum(total_weighted_loss, device)
    total_batches = int(reduce_scalar_sum(float(total_batches), device))
    total_supervised = int(reduce_scalar_sum(float(total_supervised), device))
    optimizer_steps = int(reduce_scalar_sum(float(optimizer_steps), device)) // max(get_world_size(), 1)
    train_loss = total_weighted_loss / max(total_supervised, 1)
    return {
        "train_loss": train_loss,
        "train_batches": float(total_batches),
        "train_supervised_tokens": float(total_supervised),
        "optimizer_steps": float(optimizer_steps),
        "epoch_seconds": elapsed,
        "micro_steps": float(total_batches),
    }


def dump_args(args: argparse.Namespace) -> None:
    if not is_main_process():
        return
    os.makedirs(args.output_dir, exist_ok=True)
    args_path = os.path.join(args.output_dir, "train_args.json")
    with open(args_path, "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, sort_keys=True)


def save_processor_assets(processor, output_dir: str) -> None:
    if not is_main_process():
        return
    processor_dir = os.path.join(output_dir, "processor")
    os.makedirs(processor_dir, exist_ok=True)
    processor.save_pretrained(processor_dir)


def build_tensorboard_writer(output_dir: str):
    if not is_main_process():
        return None
    tb_dir = os.path.join(output_dir, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)
    return SummaryWriter(log_dir=tb_dir)


def format_prediction_preview(records: List[Dict[str, Any]]) -> str:
    if not records:
        return "No validation generation samples."
    lines: List[str] = []
    for index, record in enumerate(records[:10], start=1):
        lines.append(f"[{index}] pair_id={record.get('pair_id')} task={record.get('task_name')}")
        lines.append(f"Q: {record.get('prompt', '')}")
        lines.append(f"GT: {record.get('answer', '')}")
        lines.append(f"Pred: {record.get('prediction', '')}")
        lines.append(f"EM: {record.get('exact_match', 0)}")
        lines.append("")
    return "\n".join(lines).strip()


def build_generation_inputs(batch: Dict[str, Any], device: str) -> Dict[str, torch.Tensor]:
    generation_inputs: Dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if not key.startswith("gen_") or not isinstance(value, torch.Tensor):
            continue
        generation_inputs[key[4:]] = value.to(device)
    return generation_inputs


def run_validation_generation(
    model,
    processor,
    loader: DataLoader,
    device: str,
    epoch: int,
    output_dir: str,
    max_new_tokens: int,
    num_beams: int,
    do_sample: bool,
) -> Dict[str, float]:
    if not is_main_process():
        distributed_barrier()
        return {
            "valid_generate_examples": 0.0,
            "valid_exact_match": 0.0,
            "preview_records": [],
        }

    model.eval()
    predictions_dir = os.path.join(output_dir, "valid_predictions")
    os.makedirs(predictions_dir, exist_ok=True)
    prediction_path = os.path.join(predictions_dir, f"epoch_{epoch:03d}.jsonl")

    total_examples = 0
    exact_matches = 0
    preview_records: List[Dict[str, Any]] = []

    with open(prediction_path, "w", encoding="utf-8") as handle:
        with torch.no_grad():
            progress = tqdm(
                loader,
                desc=f"valid_gen e{epoch}",
                leave=False,
                disable=not is_main_process(),
            )
            for batch in progress:
                generation_inputs = build_generation_inputs(batch, device)
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
                    meta = batch["meta"][index]
                    answer_text = str(meta["answer"]).strip()
                    exact_match = int(normalize_answer(prediction_text) == normalize_answer(answer_text))
                    exact_matches += exact_match
                    total_examples += 1

                    record = {
                        "epoch": epoch,
                        "pair_id": meta.get("pair_id"),
                        "task_name": meta.get("task_name"),
                        "question": meta.get("question"),
                        "prompt": meta.get("prompt"),
                        "answer": answer_text,
                        "prediction": prediction_text,
                        "exact_match": exact_match,
                    }
                    if len(preview_records) < 10:
                        preview_records.append(record)
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                if total_examples > 0:
                    progress.set_postfix(
                        examples=total_examples,
                        em=f"{exact_matches / total_examples:.4f}",
                    )
    distributed_barrier()
    return {
        "valid_generate_examples": float(total_examples),
        "valid_exact_match": float(exact_matches / max(total_examples, 1)),
        "preview_records": preview_records,
    }


def print_dataset_preview(processor, batch: Dict[str, Any], split_name: str) -> None:
    audio_token_id = int(processor.tokenizer.convert_tokens_to_ids(processor.audio_token))
    spatial_token_id = int(processor.tokenizer.convert_tokens_to_ids(processor.spatial_token))
    print(f"== {split_name} Batch Preview ==")
    for key in (
        "input_ids",
        "attention_mask",
        "input_features",
        "feature_attention_mask",
        "spatial_audio",
        "spatial_audio_attention_mask",
        "spatial_audio_lengths",
        "seld_features",
        "seld_feature_attention_mask",
        "seld_feature_lengths",
        "seld_hidden_states",
        "seld_hidden_attention_mask",
        "seld_hidden_lengths",
        "spatial_token_lengths",
        "labels",
    ):
        if key not in batch:
            continue
        value = batch[key]
        if isinstance(value, torch.Tensor):
            print(f"{key}: shape={tuple(value.shape)} dtype={value.dtype}")
    print("prefix_lengths:", batch["prefix_lengths"].tolist())
    for index, meta in enumerate(batch["meta"]):
        audio_count = int((batch["input_ids"][index] == audio_token_id).sum().item())
        spatial_count = int((batch["input_ids"][index] == spatial_token_id).sum().item())
        supervised_count = count_supervised_tokens(batch["labels"][index])
        print(
            f"[{split_name} sample {index}] pair_id={meta['pair_id']} task={meta['task_name']} "
            f"audio_tokens={audio_count} spatial_tokens={spatial_count} supervised_answer_tokens={supervised_count}"
        )


def save_epoch_metrics(output_dir: str, epoch: int, metrics: Dict[str, Any]) -> None:
    if not is_main_process():
        return
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, f"epoch_{epoch:03d}.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)


def build_epoch_valid_loaders(
    valid_dataset: Dataset,
    processor,
    args: argparse.Namespace,
    epoch: int,
) -> Tuple[DataLoader, DataLoader, int, int]:
    collator = SpatialQACollator(
        processor=processor,
        include_generation_inputs=False,
        feature_cache_max_entries=args.feature_cache_max_entries,
        hidden_cache_max_entries=args.hidden_cache_max_entries,
    )
    valid_generation_collator = SpatialQACollator(
        processor=processor,
        include_generation_inputs=True,
        feature_cache_max_entries=args.feature_cache_max_entries,
        hidden_cache_max_entries=args.hidden_cache_max_entries,
    )

    valid_indices = sample_subset_indices(
        dataset_size=len(valid_dataset),
        subset_ratio=args.valid_subset_ratio,
        seed=args.seed + 10000,
        epoch=epoch,
    )
    epoch_valid_dataset = Subset(valid_dataset, valid_indices)

    valid_sampler = DistributedSampler(epoch_valid_dataset, shuffle=False) if args.distributed else None
    valid_loader = make_loader(
        epoch_valid_dataset,
        collator,
        args.batch_size,
        args.num_workers,
        shuffle=False,
        sampler=valid_sampler,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )

    generation_indices = valid_indices
    if args.valid_generate_max_samples is not None and len(generation_indices) > args.valid_generate_max_samples:
        generation_indices = generation_indices[: args.valid_generate_max_samples]
    epoch_generation_dataset = Subset(valid_dataset, generation_indices)
    valid_generation_loader = make_loader(
        epoch_generation_dataset,
        valid_generation_collator,
        args.valid_generate_batch_size,
        args.num_workers,
        shuffle=False,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )
    return valid_loader, valid_generation_loader, len(valid_indices), len(generation_indices)


def main() -> None:
    args = parse_args()
    args = setup_distributed(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed + get_rank())

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device {args.device}, but CUDA is not available.")
    if args.device == "cpu" and args.dtype != "float32":
        raise ValueError("CPU training should use --dtype float32.")

    add_legacy_repo_to_path(args.legacy_repo_path)
    dump_args(args)
    writer = build_tensorboard_writer(args.output_dir)
    if writer is not None:
        writer.add_text("config/args", json.dumps(vars(args), indent=2, sort_keys=True), 0)

    task_params = load_seld_task_params(args.baseline_repo_path, args.seld_task_id)
    processor = build_processor(args.model_id)
    processor.seld_num_feature_channels = int(task_params.get("num_feature_channels", 7))
    processor.seld_num_mel_bins = int(task_params.get("nb_mel_bins", 64))
    processor.seld_hop_length = int(task_params.get("hop_len", 320))
    processor.seld_feature_to_seld_ratio = infer_feature_to_seld_ratio(task_params)
    processor.seld_downsample_factor = 4
    save_processor_assets(processor, args.output_dir)
    if args.seld_feature_cache_manifest is not None and args.seld_hidden_cache_manifest is not None:
        raise ValueError("Use either --seld-feature-cache-manifest or --seld-hidden-cache-manifest, not both.")
    if args.seld_hidden_cache_manifest is not None and args.train_mode not in {"spatial_only", "adapter_lora"}:
        raise ValueError(
            "Cached seld hidden states bypass feature_bridge + backbone, so they can only be used with "
            "--freeze-spatial-only or --train-adapter-lora. Disable --seld-hidden-cache-manifest if you want to train the spatial encoder."
        )
    collator = SpatialQACollator(
        processor=processor,
        include_generation_inputs=False,
        feature_cache_max_entries=args.feature_cache_max_entries,
        hidden_cache_max_entries=args.hidden_cache_max_entries,
    )
    train_dataset, train_paths, train_sizes = build_qa_dataset(
        args.qa_roots,
        args.train_split,
        args.max_train_samples,
        args.seld_feature_cache_manifest,
        args.seld_hidden_cache_manifest,
        audio_roots=args.audio_roots,
    )
    valid_dataset, valid_paths, valid_sizes = build_qa_dataset(
        args.qa_roots,
        args.valid_split,
        args.max_valid_samples,
        args.seld_feature_cache_manifest,
        args.seld_hidden_cache_manifest,
        audio_roots=args.audio_roots,
    )
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if args.distributed else None
    train_loader = make_loader(
        train_dataset,
        collator,
        args.batch_size,
        args.num_workers,
        shuffle=True,
        sampler=train_sampler,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )
    preview_valid_loader, _, preview_valid_size, preview_generate_size = build_epoch_valid_loaders(
        valid_dataset=valid_dataset,
        processor=processor,
        args=args,
        epoch=1,
    )

    qa_root_summary = ", ".join(args.qa_roots)
    train_split_summary = ", ".join(
        f"{os.path.dirname(path)}:{args.train_split}={size}" for path, size in zip(train_paths, train_sizes)
    )
    valid_split_summary = ", ".join(
        f"{os.path.dirname(path)}:{args.valid_split}={size}" for path, size in zip(valid_paths, valid_sizes)
    )
    rank0_print(
        f"Loaded QA roots [{qa_root_summary}]: train={len(train_dataset)} valid={len(valid_dataset)} "
        f"train_components=[{train_split_summary}] valid_components=[{valid_split_summary}] "
        f"valid_subset_ratio={args.valid_subset_ratio:.3f} "
        f"preview_valid_subset={preview_valid_size} preview_valid_generate={preview_generate_size} "
        f"world_size={get_world_size()} "
        f"batch_size={args.batch_size} grad_accum_steps={args.grad_accum_steps}"
    )
    if is_main_process():
        print_dataset_preview(processor, next(iter(train_loader)), "train")
        print_dataset_preview(processor, next(iter(preview_valid_loader)), "valid")

    model = build_model(args, processor)
    lora_targets: List[str] = []
    if args.train_mode == "spatial_only":
        trainable_names = freeze_all_but_spatial_modules(model)
        rank0_print(f"Training spatial modules only. Trainable parameter tensors: {len(trainable_names)}")
    elif args.train_mode == "adapter_lora":
        model, lora_targets = apply_llm_lora(model, args)
        trainable_names = configure_adapter_lora_training(model, args)
        rank0_print(
            "Training spatial adapter + projector + LLM LoRA with frozen spatial encoder. "
            f"LoRA targets: {len(lora_targets)} modules. Trainable parameter tensors: {len(trainable_names)}"
        )
    elif args.train_mode == "spatial_lora":
        model, lora_targets = apply_llm_lora(model, args)
        trainable_names = configure_spatial_lora_training(model, args)
        rank0_print(
            "Training spatial encoder + adapter + projector + LLM LoRA. "
            f"LoRA targets: {len(lora_targets)} modules. Trainable parameter tensors: {len(trainable_names)}"
        )
    else:
        trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        rank0_print(f"Training all model parameters. Trainable parameter tensors: {len(trainable_names)}")
    if not trainable_names:
        raise RuntimeError("No trainable parameters were enabled.")
    if lora_targets:
        rank0_print(f"Resolved LoRA target modules: {', '.join(lora_targets[:20])}")
    if writer is not None:
        writer.add_text("model/trainable_parameters", "\n".join(trainable_names[:500]), 0)
        if lora_targets:
            writer.add_text("model/lora_targets", "\n".join(lora_targets), 0)
    if args.distributed:
        model = DDP(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=False,
            broadcast_buffers=False,
        )
        if args.train_mode in {"spatial_lora", "adapter_lora"} and args.gradient_checkpointing and hasattr(model, "_set_static_graph"):
            model._set_static_graph()

    optimizer = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    total_optimizer_steps = (
        math.ceil(len(train_loader) / (1 if args.optimizer_step_per_batch else args.grad_accum_steps)) * args.epochs
    )
    warmup_steps = int(total_optimizer_steps * args.warmup_ratio)
    scheduler = None
    if total_optimizer_steps > 0:
        from transformers.optimization import get_cosine_schedule_with_warmup

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_optimizer_steps,
        )

    best_valid_loss, best_epoch = infer_best_valid_from_output_dir(args.output_dir)
    if not math.isfinite(best_valid_loss):
        best_valid_loss = float("inf")
        best_epoch = -1
    global_micro_step = 0
    global_optimizer_step = 0
    start_epoch = 1

    resume_checkpoint_path = resolve_resume_checkpoint_path(args)
    if resume_checkpoint_path is not None:
        if not os.path.exists(resume_checkpoint_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_checkpoint_path}")
        resume_state = resume_training_state(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            resume_checkpoint_path=resume_checkpoint_path,
            resume_model_only=args.resume_model_only,
            device=args.device,
        )
        start_epoch = int(resume_state["start_epoch"])
        global_optimizer_step = int(resume_state["global_optimizer_step"])
        global_micro_step = int(resume_state["global_micro_step"])
        load_result = resume_state["load_result"]
        rank0_print(
            f"Resumed from {resume_checkpoint_path}: next_epoch={start_epoch} "
            f"global_optimizer_step={global_optimizer_step} "
            f"missing_keys={len(load_result.missing_keys)} unexpected_keys={len(load_result.unexpected_keys)}"
        )
        if start_epoch > args.epochs:
            raise ValueError(
                f"Resume checkpoint is already at epoch {start_epoch - 1}, but --epochs={args.epochs}. "
                "Set --epochs to the total target epoch count after resuming."
            )

    def maybe_save_intermediate_checkpoint(current_global_optimizer_step: int, step_stats: Dict[str, Any]) -> None:
        if args.save_every_n_optimizer_steps <= 0:
            return
        if current_global_optimizer_step % int(args.save_every_n_optimizer_steps) != 0:
            return
        save_artifacts(
            model=model,
            processor=processor,
            optimizer=optimizer,
            scheduler=scheduler,
            args=args,
            epoch=int(step_stats["epoch"]),
            step=int(current_global_optimizer_step),
            metrics={
                "save_type": "optimizer_step",
                "epoch": int(step_stats["epoch"]),
                "micro_step": int(step_stats["micro_step"]),
                "loss": float(step_stats["loss"]),
                "supervised_tokens": int(step_stats["supervised_tokens"]),
                "global_optimizer_step": int(current_global_optimizer_step),
            },
            tag=f"step_{int(current_global_optimizer_step):07d}",
        )

    for epoch in range(start_epoch, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        valid_loader, valid_generation_loader, epoch_valid_size, epoch_generate_size = build_epoch_valid_loaders(
            valid_dataset=valid_dataset,
            processor=processor,
            args=args,
            epoch=epoch,
        )
        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=args.device,
            grad_accum_steps=max(1, args.grad_accum_steps),
            max_grad_norm=args.max_grad_norm,
            log_every=args.log_every,
            epoch=epoch,
            optimizer_step_per_batch=args.optimizer_step_per_batch,
            writer=writer,
            global_step_start=global_micro_step,
            global_optimizer_step_start=global_optimizer_step,
            on_optimizer_step=maybe_save_intermediate_checkpoint,
        )
        global_micro_step += int(train_stats["micro_steps"])
        global_optimizer_step += int(train_stats["optimizer_steps"])
        valid_stats = evaluate(model, valid_loader, args.device)
        distributed_barrier()
        generation_stats = run_validation_generation(
            model=model,
            processor=processor,
            loader=valid_generation_loader,
            device=args.device,
            epoch=epoch,
            output_dir=args.output_dir,
            max_new_tokens=args.valid_max_new_tokens,
            num_beams=args.valid_num_beams,
            do_sample=args.valid_do_sample,
        )
        preview_records = generation_stats.pop("preview_records", [])
        summary = {
            **train_stats,
            **valid_stats,
            **generation_stats,
            "valid_subset_size": float(epoch_valid_size),
            "valid_generation_size": float(epoch_generate_size),
        }
        if writer is not None:
            writer.add_scalar("epoch/train_loss", float(summary["train_loss"]), epoch)
            writer.add_scalar("epoch/valid_loss", float(summary["valid_loss"]), epoch)
            writer.add_scalar("epoch/valid_exact_match", float(summary["valid_exact_match"]), epoch)
            writer.add_scalar("epoch/optimizer_steps", float(summary["optimizer_steps"]), epoch)
            writer.add_scalar("epoch/epoch_seconds", float(summary["epoch_seconds"]), epoch)
            writer.add_scalar("epoch/train_supervised_tokens", float(summary["train_supervised_tokens"]), epoch)
            writer.add_scalar("epoch/valid_supervised_tokens", float(summary["valid_supervised_tokens"]), epoch)
            writer.add_scalar("epoch/valid_subset_size", float(summary["valid_subset_size"]), epoch)
            writer.add_scalar("epoch/valid_generation_size", float(summary["valid_generation_size"]), epoch)
            writer.add_text(
                f"valid_predictions/epoch_{epoch:03d}",
                format_prediction_preview(preview_records),
                epoch,
            )
            writer.flush()
        rank0_print(
            f"[epoch {epoch}] train_loss={summary['train_loss']:.6f} "
            f"valid_loss={summary['valid_loss']:.6f} "
            f"valid_exact_match={summary['valid_exact_match']:.4f} "
            f"valid_subset={int(summary['valid_subset_size'])} "
            f"optimizer_steps={int(summary['optimizer_steps'])} "
            f"epoch_seconds={summary['epoch_seconds']:.2f}"
        )
        save_epoch_metrics(args.output_dir, epoch, summary)

        if args.save_every_epoch:
            save_artifacts(
                model=model,
                processor=processor,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                epoch=epoch,
                step=int(summary["optimizer_steps"]),
                metrics=summary,
                tag=f"epoch_{epoch:03d}",
            )

        save_artifacts(
            model=model,
            processor=processor,
            optimizer=optimizer,
            scheduler=scheduler,
            args=args,
            epoch=epoch,
            step=int(summary["optimizer_steps"]),
            metrics=summary,
            tag="last",
        )

        if summary["valid_loss"] < best_valid_loss:
            best_valid_loss = summary["valid_loss"]
            best_epoch = epoch
            save_artifacts(
                model=model,
                processor=processor,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                epoch=epoch,
                step=int(summary["optimizer_steps"]),
                metrics=summary,
                tag="best",
            )
            rank0_print(f"Updated best checkpoint at epoch {epoch} with valid_loss={best_valid_loss:.6f}")

    rank0_print(
        f"Training complete. best_epoch={best_epoch} best_valid_loss={best_valid_loss:.6f} "
        f"output_dir={args.output_dir}"
    )
    if writer is not None:
        writer.close()
    cleanup_distributed()


if __name__ == "__main__":
    main()
