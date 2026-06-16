"""Training script for IV / Neural-IV spatial baselines + Qwen2.5-Omni spatial QA.

Two training stages (IV / Neural-IV baselines have no pretrained backbone to unfreeze):
Stage 1 (projector_only)  - train IV/Neural-IV adapter + projector only
Stage 2 (encoder_lora)    - train IV/Neural-IV adapter + projector + LLM LoRA

Single-stage example (Stage 1, IV baseline):
    torchrun --nproc_per_node=4 train_spatial_iv_qa.py \\
        --model-id /path/to/Qwen2.5-Omni-7B \\
        --spatial-encoder-type iv \\
        --qa-root /path/to/easy_qa_root \\
        --output-dir ./runs/iv_stage1 \\
        --projector-only --epochs 3 --lr 1e-4

Resume into Stage 2:
    torchrun --nproc_per_node=4 train_spatial_iv_qa.py \\
        --resume-checkpoint-path ./runs/iv_stage1/checkpoints/best_trainable.pt \\
        --resume-model-only --encoder-lora \\
        --spatial-encoder-type iv \\
        --output-dir ./runs/iv_stage2 --epochs 3 --lr 3e-5

Use --spatial-encoder-type neural_iv to switch to the Neural-IV variant.
"""
import argparse
from collections import OrderedDict
from datetime import timedelta
import inspect
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    from peft import LoraConfig, TaskType, get_peft_model
except ImportError:
    LoraConfig = TaskType = get_peft_model = None

DEFAULT_MODEL_ID = os.environ.get("SO_BASE_MODEL", "Qwen/Qwen2.5-Omni-7B")
DEFAULT_BEATS_CKPT = os.environ.get("SO_ENCODER_CKPT", "")
DEFAULT_BEATS_REPO = os.environ.get("SO_BEATS_REPO", "")
DEFAULT_QA_ROOT = os.environ.get("SO_DATASET_ROOT", "")
DEFAULT_OUTPUT_DIR = "./so_runs/default_run_iv"
DEFAULT_SO_REPO = os.environ.get("SO_REPO", "")
SAMPLE_RATE = 16000
MAX_AUDIO_SECONDS = 20
MAX_AUDIO_SAMPLES = SAMPLE_RATE * MAX_AUDIO_SECONDS
# LLM-side spatial token rate (tokens consumed per second). Encoder-native rate
# is controlled by --encoder-token-rate (10 Hz native); the projector does
# k-way temporal pooling to land at this LLM rate.
TARGET_TOKEN_RATE = 2.5
DEFAULT_ENCODER_TOKEN_RATE = 10.0
DEFAULT_LORA_TARGET_MODULES = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    # IV / Neural-IV baselines do not need an external encoder ckpt, but
    # they share the SELD233 FeatureBridge for STFT / log-mel / IV. Provide
    # the DCASE baseline repo path and feature stats dir so the FeatureBridge
    # can load `parameters.py` and normalization stats.
    p.add_argument(
        "--baseline-repo-path",
        type=str,
        default=os.environ.get("DCASE_BASELINE_REPO", ""),
        help="DCASE2024_seld_baseline repo root (used by FeatureBridge to load parameters.py).",
    )
    p.add_argument(
        "--seld-feature-stats-dir",
        type=str,
        default=os.environ.get("SELD_FEATURE_STATS_DIR", ""),
        help="Directory containing `foa_wts` normalization stats file.",
    )
    p.add_argument(
        "--spatial-encoder-type",
        type=str,
        required=True,
        choices=("iv", "neural_iv"),
        help="Which IV baseline to use.",
    )
    p.add_argument("--iv-token-dim", type=int, default=256)
    p.add_argument("--iv-projector-hidden-dim", type=int, default=512)
    p.add_argument("--iv-num-mel-bins", type=int, default=64)
    p.add_argument("--iv-band-pool", type=int, default=0,
                    help="Extra frequency pooling; 0 means use full num_mel_bins.")
    p.add_argument("--iv-output-scale", type=float, default=0.02,
                    help="Initial output scale for spatial tokens (DCASE default).")
    p.add_argument("--iv-feature-to-seld-ratio", type=int, default=5)
    p.add_argument("--iv-downsample-factor", type=int, default=4)
    p.add_argument("--neural-iv-hidden-channels", type=int, default=64)
    p.add_argument("--so-repo", default=os.environ.get("SO_REPO", os.path.dirname(os.path.abspath(__file__))))
    p.add_argument("--qa-root", default=None)
    p.add_argument("--qa-roots", nargs="+", default=None)
    p.add_argument("--audio-root", default=None,
                    help="Extra root searched when QA records have a relative "
                         "audio_path (e.g. SO-Dataset HF release uses paths "
                         "like 'audio/train/foo.wav' relative to the dataset "
                         "root, not the qa/ subdir). Pass the dataset root "
                         "here. Multiple roots: --audio-roots a b c.")
    p.add_argument("--audio-roots", nargs="+", default=None,
                    help="Multiple audio search roots (overrides --audio-root).")
    p.add_argument("--audio-feature-cache-manifest", default=None)
    p.add_argument("--audio-feature-cache-max-entries", type=int, default=256)
    p.add_argument("--train-split", default="train")
    p.add_argument("--valid-split", default="valid")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-valid-samples", type=int, default=None)
    p.add_argument("--valid-subset-ratio", type=float, default=0.1)
    p.add_argument("--step-valid-subset-ratio", type=float, default=0.05)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--persistent-workers", action="store_true")
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--projector-lr", type=float, default=None,
                    help="Override lr for spatial IV adapter + projector params. Default: --lr.")
    p.add_argument("--lora-lr", type=float, default=None,
                    help="Override lr for LoRA params. Default: --lr.")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--projector-weight-decay", type=float, default=None,
                    help="Override weight decay for projector. Default: --weight-decay.")
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--local-rank", type=int, default=-1)
    p.add_argument("--dtype", default="bfloat16", choices=("float32", "bfloat16", "float16"))
    mg = p.add_mutually_exclusive_group()
    mg.add_argument("--projector-only", dest="train_mode", action="store_const", const="projector_only",
                    help="Stage 1: train IV adapter + projector only.")
    mg.add_argument("--encoder-lora", dest="train_mode", action="store_const", const="encoder_lora",
                    help="Stage 2: train IV adapter + projector + LLM LoRA.")
    mg.add_argument("--train-all", dest="train_mode", action="store_const", const="all",
                    help="Train all parameters.")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-target-modules", nargs="+", default=list(DEFAULT_LORA_TARGET_MODULES))
    p.add_argument("--lora-target-prefixes", nargs="+", default=["thinker.model"])
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--attn-impl", default="auto",
                   choices=("flash_attention_2", "sdpa", "eager", "auto"),
                   help="Attention implementation passed to from_pretrained(). "
                        "Default 'auto' auto-detects flash_attn and falls back to 'sdpa' when not installed.")
    p.add_argument("--projector-fp32", action="store_true",
                    help="Cast IV projector to fp32 for stability.")
    p.add_argument("--iv-modules-fp32", action="store_true",
                    help="Cast the IV / Neural-IV adapter (conv_encoder / "
                         "token_norm / token_head) to fp32 for numerical "
                         "stability on near-silent frames. The shared "
                         "feature_bridge is always run in fp32 + no_grad "
                         "regardless of this flag (it has no trainable "
                         "parameters). Mirror of spatial-beats --spatial-fp32.")
    p.add_argument("--projector-type", default="mlp",
                    choices=("mlp", "mlp_ln", "pixel_shuffle"),
                    help="(UNUSED for IV path; kept for CLI compatibility.)")
    p.add_argument("--projector-shuffle-factor", type=int, default=1,
                    help="(UNUSED for IV path; IV uses SOTokenProjector directly.)")
    p.add_argument("--encoder-token-rate", type=float, default=DEFAULT_ENCODER_TOKEN_RATE,
                    help="(UNUSED for IV path; 2.5Hz hardcoded via hop_length/ratio/downsample.)")
    p.add_argument("--optimizer-step-per-batch", action="store_true")
    p.add_argument("--valid-generate-batch-size", type=int, default=1)
    p.add_argument("--valid-generate-max-samples", type=int, default=32,
                    help="Cap on generation samples per epoch for valid_em. "
                         "Ignored when --valid-generate-full is set.")
    p.add_argument("--valid-generate-full", action="store_true",
                    help="Generate predictions on the full valid split every "
                         "epoch (overrides --valid-generate-max-samples). "
                         "All predictions are written to "
                         "valid_predictions/epoch_NNN.jsonl with task_name "
                         "for offline per-task analysis. Expect a large "
                         "wall-time increase proportional to valid size.")
    p.add_argument("--valid-max-new-tokens", type=int, default=48)
    p.add_argument("--valid-num-beams", type=int, default=1)
    p.add_argument("--valid-do-sample", action="store_true")
    p.add_argument("--resume-checkpoint-path", default=None)
    p.add_argument("--resume-tag", default=None, help="e.g. 'last', 'best', 'epoch_003'")
    p.add_argument("--resume-model-only", action="store_true")
    p.add_argument("--save-every-epoch", action="store_true")
    p.add_argument("--save-every-n-optimizer-steps", type=int, default=1000)
    p.add_argument("--valid-every-n-optimizer-steps", type=int, default=1000)
    p.add_argument("--save-full-model", action="store_true")
    p.set_defaults(train_mode="projector_only", save_every_epoch=True)
    args = p.parse_args()
    if args.qa_roots:
        args.qa_roots = [os.path.abspath(r) for r in args.qa_roots]
    else:
        args.qa_roots = [os.path.abspath(args.qa_root or DEFAULT_QA_ROOT)]
    args.qa_root = args.qa_roots[0]
    # Normalise audio search roots (mirror train_so_qa.py). Order: --audio-roots,
    # --audio-root, then qa-dir-parent (auto, handled inside QAAudioDataset).
    if args.audio_roots:
        args.audio_roots = [os.path.abspath(r) for r in args.audio_roots]
    elif args.audio_root:
        args.audio_roots = [os.path.abspath(args.audio_root)]
    else:
        args.audio_roots = []
    return args


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def is_distributed(): return dist.is_available() and dist.is_initialized()
def get_rank(): return dist.get_rank() if is_distributed() else 0
def get_world_size(): return dist.get_world_size() if is_distributed() else 1
def is_main_process(): return get_rank() == 0

def rank0_print(*args, **kwargs):
    if is_main_process(): print(*args, **kwargs)

def debug_rank_print(*args, **kwargs):
    if os.environ.get("SO_DEBUG_SYNC", "0") == "1":
        print(f"[rank{get_rank()} {time.strftime('%H:%M:%S')}]", *args, **kwargs, flush=True)

def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model

def distributed_barrier():
    if not is_distributed(): return
    # Pin barrier to the current CUDA device so NCCL can use the correct stream
    # and the collective does not accidentally fall back to CPU/gloo.
    if torch.cuda.is_available():
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        dist.barrier()

def reduce_scalar_sum(value, device):
    if not is_distributed(): return float(value)
    t = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item())

def setup_distributed(args):
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    args.distributed = ws > 1
    if not args.distributed:
        args.rank = 0; args.world_size = 1; return args
    lr = args.local_rank
    if lr < 0: lr = int(os.environ.get("LOCAL_RANK", "-1"))
    if lr < 0: raise RuntimeError("LOCAL_RANK missing.")
    args.local_rank = lr; args.rank = int(os.environ["RANK"]); args.world_size = ws
    torch.cuda.set_device(lr); args.device = f"cuda:{lr}"
    # Default NCCL watchdog timeout is 10 min, which is too tight when NFS I/O
    # jitters or when rank-0 checkpoint saving to shared storage is slow.
    # Expose via env so it can be tuned per job.
    timeout_min = int(os.environ.get("SO_NCCL_TIMEOUT_MIN", "60"))
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(minutes=max(10, timeout_min)),
    )
    return args

def cleanup_distributed():
    if is_distributed(): dist.destroy_process_group()

def dtype_from_name(n):
    return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[n]


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def load_qa_records(qa_path, max_samples=None):
    qa_path = os.path.abspath(qa_path)
    records = []
    if qa_path.endswith(".jsonl"):
        with open(qa_path, encoding="utf-8") as f:
            for line in f:
                if max_samples is not None and len(records) >= max_samples: break
                line = line.strip()
                if line: records.append(json.loads(line))
        return records
    if qa_path.endswith(".json"):
        with open(qa_path, encoding="utf-8") as f: payload = json.load(f)
        it = payload if isinstance(payload, list) else payload.get("records", payload.get("data", []))
        for r in it:
            if max_samples is not None and len(records) >= max_samples: break
            records.append(r)
        return records
    raise ValueError(f"Unsupported format: {qa_path}")

def resolve_qa_split_path(qa_root, split_name):
    for ext in (".jsonl", ".json"):
        p = os.path.join(qa_root, f"{split_name}{ext}")
        if os.path.exists(p): return os.path.abspath(p)
    raise FileNotFoundError(f"Missing '{split_name}' under {qa_root}")

class QAAudioDataset(Dataset):
    def __init__(self, path, max_samples=None, audio_search_roots=None):
        self.records = []
        # Audio-path resolution roots (in priority order). The first root is
        # always the directory of the JSONL itself (legacy behavior); extra
        # roots come from ``--audio-root`` / ``--audio-roots`` (e.g. SO-Dataset
        # HF release where ``audio_path`` is relative to the dataset root,
        # not the qa/ subdir). We resolve once at __init__ time so subsequent
        # ``sf.read(record["audio_path"])`` calls use absolute paths and never
        # stat() the filesystem mid-training.
        qa_dir = os.path.dirname(os.path.abspath(path))
        roots = [qa_dir]
        if audio_search_roots:
            for r in audio_search_roots:
                if r and r not in roots:
                    roots.append(os.path.abspath(r))
        # Auto-fallback: also try the parent of qa_dir (covers
        # ``<root>/qa/{split}.jsonl`` referencing ``audio/.../foo.wav`` at
        # ``<root>/audio/...``).
        parent = os.path.dirname(qa_dir)
        if parent and parent != qa_dir and parent not in roots:
            roots.append(parent)
        self._audio_search_roots = roots

        for i, r in enumerate(load_qa_records(path, max_samples)):
            # NOTE: do NOT call os.path.exists() here on absolute paths — on
            # NFS with 700K+ records that becomes O(N) stat() calls taking
            # 10+ minutes before training starts. Missing files will raise a
            # clear error in the collator. We only probe the search roots
            # when the path is relative.
            ap = r.get("audio_path")
            if ap is None:
                raise ValueError(f"Record {i} missing audio_path")
            if not os.path.isabs(ap):
                resolved = None
                for root in self._audio_search_roots:
                    cand = os.path.join(root, ap)
                    if os.path.exists(cand):
                        resolved = cand
                        break
                # If none of the roots have it, fall back to qa-dir-relative
                # so the missing-file error message points somewhere stable.
                r["audio_path"] = resolved or os.path.join(qa_dir, ap)
            # 兼容两种数据格式：
            #   - 旧版 QA: 同时有 `prompt`（带 "Question: ...\nAnswer with only..."
            #     包装）和 `question`（裸问题）
            #   - 新版 QA: 只有 `question` / `answer`
            # 统一以 `prompt` 为准；若缺失，则回退到 `question`。
            if r.get("prompt") is None:
                q = r.get("question")
                if q is None:
                    raise ValueError(f"Record {i} missing both 'prompt' and 'question'")
                r["prompt"] = str(q)
            if r.get("answer") is None:
                raise ValueError(f"Record {i} missing answer")
            self.records.append(r)
    def __len__(self): return len(self.records)
    def __getitem__(self, i): return self.records[i]

def build_qa_dataset(qa_roots, split, max_samples, audio_search_roots=None):
    ps, ds, ss = [], [], []
    for root in qa_roots:
        p = resolve_qa_split_path(root, split)
        rank0_print(f"[{time.strftime('%H:%M:%S')}] Loading {split} split: {p} ...")
        d = QAAudioDataset(p, max_samples, audio_search_roots=audio_search_roots)
        rank0_print(f"[{time.strftime('%H:%M:%S')}] Loaded {len(d):,} records.")
        ps.append(p); ds.append(d); ss.append(len(d))
    if len(ds) == 1: return ds[0], ps, ss
    return ConcatDataset(ds), ps, ss


class QwenAudioFeatureCache:
    def __init__(self, manifest_path: str, max_entries: int = 256):
        manifest_path = os.path.abspath(manifest_path)
        with open(manifest_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self.manifest_path = manifest_path
        self.cache_dir = os.path.abspath(payload["cache_dir"])
        self.entries = {
            os.path.abspath(audio_path): relpath
            for audio_path, relpath in payload["entries"].items()
        }
        self.max_entries = max(0, int(max_entries))
        self._memory_cache: "OrderedDict[str, Dict[str, torch.Tensor]]" = OrderedDict()

    def __len__(self) -> int:
        return len(self.entries)

    def load(self, audio_path: str) -> Dict[str, torch.Tensor]:
        audio_path = os.path.abspath(audio_path)
        cached = self._memory_cache.get(audio_path)
        if cached is not None:
            self._memory_cache.move_to_end(audio_path)
            return cached
        relpath = self.entries.get(audio_path)
        if relpath is None:
            raise KeyError(f"Audio feature cache missing entry for {audio_path}")
        payload = torch.load(
            os.path.join(self.cache_dir, relpath),
            map_location="cpu",
            weights_only=False,
        )
        result = {
            "input_features": torch.as_tensor(payload["input_features"]),
            "feature_length": torch.as_tensor(payload["feature_length"]).long(),
        }
        if "audio_token_length" in payload:
            result["audio_token_length"] = torch.as_tensor(payload["audio_token_length"]).long()
        if "spatial_audio" in payload:
            result["spatial_audio"] = torch.as_tensor(payload["spatial_audio"])
        if "spatial_audio_length" in payload:
            result["spatial_audio_length"] = torch.as_tensor(payload["spatial_audio_length"]).long()
        if self.max_entries > 0:
            self._memory_cache[audio_path] = result
            self._memory_cache.move_to_end(audio_path)
            while len(self._memory_cache) > self.max_entries:
                self._memory_cache.popitem(last=False)
        return result

def sample_subset_indices(n, ratio, seed, epoch):
    if n <= 0: return []
    ratio = float(max(0.0, min(1.0, ratio)))
    size = n if ratio >= 1.0 else max(1, int(round(n * ratio)))
    rng = np.random.default_rng(seed + epoch)
    return sorted(int(i) for i in rng.choice(n, size=min(size, n), replace=False))

def shard_dataset_for_rank(dataset):
    """Evenly shard a dataset across ranks using a DistributedSampler-style
    schedule with padding.

    Important: unlike a naive strided shard, this guarantees every rank gets
    the SAME number of samples.  Uneven per-rank batch counts will cause some
    ranks to reach the trailing all_reduce much earlier than others, and over
    a long run the drift (plus NFS I/O jitter) eventually crosses the NCCL
    watchdog timeout.
    """
    if not is_distributed():
        return dataset
    n = len(dataset)
    if n <= 0:
        return dataset
    world = get_world_size()
    rank = get_rank()
    # Ceil-divide so every rank gets the same count; pad by repeating samples
    # from the front (identical to DistributedSampler(drop_last=False)).
    per_rank = (n + world - 1) // world
    total = per_rank * world
    base = list(range(n))
    if total > n:
        base = base + base[: total - n]
    idx = base[rank * per_rank : (rank + 1) * per_rank]
    return Subset(dataset, idx)

def build_left_padded_batch(input_ids, attn, prefix_lengths, pad_id):
    ml = int(prefix_lengths.max()); B = input_ids.shape[0]
    gi = torch.full((B, ml), fill_value=pad_id, dtype=input_ids.dtype)
    gm = torch.zeros((B, ml), dtype=attn.dtype)
    for i, pl in enumerate(prefix_lengths.tolist()):
        s = ml - pl; gi[i, s:] = input_ids[i, :pl]; gm[i, s:] = 1
    return gi, gm


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

@dataclass
class SpatialBeatsQACollator:
    """Collate FOA audio + QA text pairs into training/generation batches.

    Prompt: <|audio|><|spatial|>\\n{prompt}\\n{answer}<eos>
    Labels: -100 for prefix, token IDs for answer only.
    Adds spatial_audio [B,T_max,4] + attention_mask + lengths.
    """
    processor: Any
    audio_feature_cache: Optional[QwenAudioFeatureCache] = None
    ignore_index: int = -100
    sample_rate: int = SAMPLE_RATE
    max_audio_samples: int = MAX_AUDIO_SAMPLES
    include_generation_inputs: bool = False
    target_token_rate: float = TARGET_TOKEN_RATE
    # NEW (eval-only ablation): drop the Qwen mono <|AUDIO|> branch entirely
    # so only <|spatial|> tokens drive the LLM. Used by bench_test_generate_iv
    # for the "decoder-only" baseline. Default False: training is unaffected.
    drop_mono_audio: bool = False

    def __call__(self, features):
        audio_arrs, full_texts, ans_sfxs, meta, sa_lens = [], [], [], [], []
        spatial_audio_arrs = []  # always real FOA, used for sa_t reconstruction
        cached_input_features, cached_feature_lengths = [], []
        eos = getattr(self.processor.tokenizer, "eos_token", None) or ""

        for feat in features:
            cache_item = None
            if self.audio_feature_cache is not None:
                cache_item = self.audio_feature_cache.load(feat["audio_path"])
            if cache_item is not None and "spatial_audio" in cache_item and "spatial_audio_length" in cache_item:
                wav = cache_item["spatial_audio"].to(dtype=torch.float32).cpu().numpy()
                if wav.ndim != 2 or wav.shape[0] != 4:
                    raise ValueError(f"Cached spatial_audio must have shape [4, T], got {tuple(wav.shape)}")
                T = int(cache_item["spatial_audio_length"].item())
                wav = wav[:, :T]
            else:
                wav, sr = sf.read(feat["audio_path"], dtype="float32", always_2d=True)
                if sr != self.sample_rate:
                    raise ValueError(f"Expected {self.sample_rate}Hz got {sr} for {feat['audio_path']}")
                wav = wav.T  # [4, T]
                if wav.shape[0] != 4:
                    raise ValueError(f"Expected 4ch FOA, got {wav.shape}")
                if wav.shape[1] > self.max_audio_samples:
                    wav = wav[:, :self.max_audio_samples]
                T = wav.shape[1]
            sa_lens.append(T)
            # Prompt structure unchanged (processor requires <|AUDIO|> +
            # <|spatial|>). drop_mono_audio works by zeroing the waveform fed
            # to the processor / Qwen audio_tower, while keeping the real FOA
            # for the spatial encoder.
            prefix = (
                self.processor.audio_token
                + self.processor.spatial_token
                + f"\n{feat['prompt'].rstrip()}\n"
            )
            ans = str(feat["answer"]).strip()
            ans_sfx = ans + eos
            full_texts.append(prefix + ans_sfx)
            ans_sfxs.append(ans_sfx)
            real_wav = wav.astype(np.float32, copy=False)
            spatial_audio_arrs.append(real_wav)
            if self.drop_mono_audio:
                audio_arrs.append(np.zeros_like(real_wav, dtype=np.float32))
            else:
                audio_arrs.append(real_wav)
            if cache_item is not None:
                cached_input_features.append(cache_item["input_features"])
                cached_feature_lengths.append(int(cache_item["feature_length"].item()))
            meta.append({
                "pair_id": feat.get("pair_id"), "task_name": feat.get("task_name"),
                "answer": feat.get("answer"), "audio_path": feat.get("audio_path"),
                "prompt": feat.get("prompt"), "question": feat.get("question"),
                "canonical_answer": feat.get("canonical_answer"),
                "scene_id": feat.get("scene_id"),
                "segment_stem": feat.get("segment_stem"),
            })

        # Build padded spatial_audio [B, T_max, 4]
        lens_t = torch.tensor(sa_lens, dtype=torch.long)
        T_max = int(lens_t.max()); B = len(audio_arrs)
        sa_t = torch.zeros(B, T_max, 4, dtype=torch.float32)
        # Build spatial_audio from spatial_audio_arrs (always real FOA);
        # audio_arrs may be zero-filled in drop_mono_audio mode.
        for i, wav in enumerate(spatial_audio_arrs):
            T = wav.shape[1]; sa_t[i, :T] = torch.from_numpy(wav.T)

        processor_kwargs = {}
        # In drop_mono_audio mode we must NOT pass cached input_features;
        # they would re-feed the Qwen audio_tower we're trying to bypass.
        if not self.drop_mono_audio and self.audio_feature_cache is not None:
            feature_dim = int(cached_input_features[0].shape[0]) if cached_input_features else 0
            max_feature_length = max(cached_feature_lengths) if cached_feature_lengths else 0
            input_features = torch.zeros(B, feature_dim, max_feature_length, dtype=cached_input_features[0].dtype)
            feature_attention_mask = torch.zeros(B, max_feature_length, dtype=torch.long)
            for index, (feature_tensor, feature_length) in enumerate(zip(cached_input_features, cached_feature_lengths)):
                input_features[index, :, :feature_length] = feature_tensor[:, :feature_length]
                feature_attention_mask[index, :feature_length] = 1
            processor_kwargs["input_features"] = input_features
            processor_kwargs["feature_attention_mask"] = feature_attention_mask

        # Base processor: mel features + tokenization + padding.
        # IMPORTANT: the Qwen2.5-Omni processor defaults to left-padding for text,
        # but our label/prefix math in this collator assumes right-padding
        # (labels[:, :pl] = -100, prefix = input_ids[:, :pl]). Temporarily flip
        # the tokenizer to right-pad so the base processor honours it; the
        # generation branch below explicitly left-pads via build_left_padded_batch.
        # In drop_mono_audio mode, audio_arrs is already zero-filled (see above).
        tok = self.processor.tokenizer
        prev_padding_side = getattr(tok, "padding_side", "left")
        tok.padding_side = "right"
        try:
            batch = self.processor(
                text=full_texts,
                audio=audio_arrs,
                padding=True,
                padding_side="right",
                return_tensors="pt",
                **processor_kwargs,
            )
        finally:
            tok.padding_side = prev_padding_side

        # Attach spatial branch tensors
        batch["spatial_audio"] = sa_t
        batch["spatial_audio_attention_mask"] = (
            torch.arange(T_max).unsqueeze(0) < lens_t.unsqueeze(1)
        )
        batch["spatial_audio_lengths"] = lens_t

        # Build labels: -100 for prefix, token IDs for answer
        labels = batch["input_ids"].clone()
        if "attention_mask" in batch:
            labels = labels.masked_fill(batch["attention_mask"] == 0, self.ignore_index)
        ab = self.processor.tokenizer(
            ans_sfxs, padding=True, return_tensors="pt", add_special_tokens=False
        )
        al = ab["attention_mask"].sum(1).long()
        vl = batch["attention_mask"].sum(1).long()
        pl = vl - al
        if (pl < 0).any(): raise ValueError("Negative prefix length.")
        for i, p in enumerate(pl.tolist()): labels[i, :p] = self.ignore_index
        if (labels != self.ignore_index).sum(1).min().item() <= 0:
            raise ValueError("Sample with no supervised answer tokens.")
        batch["labels"] = labels; batch["meta"] = meta; batch["prefix_lengths"] = pl

        # Defensive: assert every FOA sample still carries the expected number of
        # <|spatial|> placeholders after prefix masking. If right-padding ever
        # regresses (e.g. processor default flips back to left), this surfaces
        # the bug here instead of inside generate() with a cryptic modal-order
        # error.
        try:
            spatial_token_str = self.processor.spatial_token
            spatial_token_id = int(
                self.processor.tokenizer.convert_tokens_to_ids(spatial_token_str)
            )
        except Exception:
            spatial_token_id = None
        if spatial_token_id is not None and spatial_token_id >= 0:
            spatial_counts = (batch["input_ids"] == spatial_token_id).sum(dim=1).tolist()
            # Use the processor's own expansion routine so the expected count
            # matches the placeholder count exactly. IV / Neural-IV / SELD233
            # paths use integer frame-math (hop_length / ratio / downsample),
            # while so_backbone uses banker-rounded 2.5Hz.
            try:
                enc_type = getattr(self.processor, "spatial_encoder_type", "seld")
                sa_t = torch.as_tensor(sa_lens, dtype=torch.long)
                if enc_type == "so_backbone":
                    exp_t = self.processor._samples_to_so_backbone_tokens(sa_t)
                else:
                    # seld / iv / neural_iv: use the same frame-math the
                    # processor uses to expand placeholders.
                    try:
                        from spatial_omni.utils.spatial_seld_utils import (
                            samples_to_spatial_length_bundle,
                        )
                    except ImportError:
                        from spatial_omni.model.processing_so import (
                            samples_to_spatial_length_bundle,  # type: ignore
                        )
                    bundle = samples_to_spatial_length_bundle(
                        sa_t,
                        hop_length=int(getattr(self.processor, "seld_hop_length", 320)),
                        feature_to_seld_ratio=int(getattr(self.processor, "seld_feature_to_seld_ratio", 5)),
                        downsample_factor=int(getattr(self.processor, "seld_downsample_factor", 4)),
                    )
                    exp_t = bundle.spatial_token_lengths
                expected_per_sample = exp_t.tolist()
            except Exception:
                expected_per_sample = [
                    int(round(self.target_token_rate * (l / self.sample_rate)))
                    for l in sa_lens
                ]
            for idx, (got, exp) in enumerate(zip(spatial_counts, expected_per_sample)):
                if exp > 0 and got != exp:
                    raise ValueError(
                        f"[SpatialBeatsQACollator] sample {idx} has {got} spatial "
                        f"placeholders in input_ids, expected {exp} (audio_len="
                        f"{sa_lens[idx]}, target_rate={self.target_token_rate}). "
                        "Check padding_side / placeholder expansion."
                    )

        if self.include_generation_inputs:
            pid = int(self.processor.tokenizer.pad_token_id or 0)
            gi, gm = build_left_padded_batch(batch["input_ids"], batch["attention_mask"], pl, pid)
            gb = {"input_ids": gi, "attention_mask": gm}
            for k, v in batch.items():
                if k in {"input_ids", "attention_mask", "labels", "meta", "prefix_lengths"}: continue
                if isinstance(v, torch.Tensor): gb[k] = v
            for k, v in gb.items():
                if isinstance(v, torch.Tensor): batch[f"gen_{k}"] = v
        return batch


def make_loader(dataset, collator, batch_size, num_workers, shuffle,
                sampler=None, persistent_workers=False, prefetch_factor=2):
    kw = {
        "dataset": dataset, "batch_size": batch_size,
        "shuffle": shuffle if sampler is None else False, "sampler": sampler,
        "num_workers": num_workers, "pin_memory": torch.cuda.is_available(),
        "collate_fn": collator,
    }
    if num_workers > 0:
        kw["persistent_workers"] = persistent_workers                                                                                                        
        kw["prefetch_factor"] = prefetch_factor
    return DataLoader(**kw)                                                                                                                                  
                                                        
def build_epoch_valid_loaders(valid_ds, processor, args, epoch, audio_feature_cache=None):
    step_ratio = float(max(0.0, min(1.0, getattr(args, "step_valid_subset_ratio", args.valid_subset_ratio))))
    vi = sample_subset_indices(len(valid_ds), step_ratio, args.seed, epoch)
    if getattr(args, "valid_generate_full", False):
        # Run generation on the ENTIRE valid split; stable across epochs.
        gi = list(range(len(valid_ds)))
    else:
        gs = min(args.valid_generate_max_samples, len(valid_ds))
        rng = np.random.default_rng(args.seed + epoch + 10000)
        if len(valid_ds) > 0 and gs > 0:
            gi = sorted(int(i) for i in rng.choice(len(valid_ds), size=gs, replace=False))
        else:
            gi = []
    step_ds = shard_dataset_for_rank(Subset(valid_ds, vi))
    full_ds = shard_dataset_for_rank(valid_ds)
    gen_ds = shard_dataset_for_rank(Subset(valid_ds, gi))
    step_vl = make_loader(
        step_ds,
        SpatialBeatsQACollator(
            processor=processor,
            audio_feature_cache=audio_feature_cache,
            include_generation_inputs=False,
        ),
        args.batch_size, args.num_workers, False, None,
        args.persistent_workers, args.prefetch_factor,
    )
    full_vl = make_loader(
        full_ds,
        SpatialBeatsQACollator(
            processor=processor,
            audio_feature_cache=audio_feature_cache,
            include_generation_inputs=False,
        ),
        args.batch_size, args.num_workers, False, None,
        args.persistent_workers, args.prefetch_factor,
    )
    gl = make_loader(
        gen_ds,
        SpatialBeatsQACollator(
            processor=processor,
            audio_feature_cache=audio_feature_cache,
            include_generation_inputs=True,
        ),
        args.valid_generate_batch_size, 0, False,
    )
    return step_vl, full_vl, gl, len(vi), len(valid_ds), len(gi)


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def enable_gradient_checkpointing(model):
    if not hasattr(model, "gradient_checkpointing_enable"): return
    sig = inspect.signature(model.gradient_checkpointing_enable)
    if "gradient_checkpointing_kwargs" in sig.parameters:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.gradient_checkpointing_enable()

def build_processor(model_id, sqr):
    if sqr not in sys.path: sys.path.insert(0, sqr)
    from spatial_omni.model.processing_so import Qwen2_5OmniSpatialProcessor  # type: ignore
    from spatial_omni.model.processing_qwen2_5_omni import Qwen2_5OmniProcessor  # type: ignore
    base = Qwen2_5OmniProcessor.from_pretrained(model_id)
    return Qwen2_5OmniSpatialProcessor(
        image_processor=base.image_processor, feature_extractor=base.feature_extractor,
        tokenizer=base.tokenizer, chat_template=base.chat_template,
    )

def build_model(args, processor):
    if args.so_repo not in sys.path: sys.path.insert(0, args.so_repo)
    from spatial_omni.model.modeling_so_thinker import (  # type: ignore
        Qwen2_5OmniSpatialForConditionalGeneration,
    )
    try:
        from spatial_omni.model.configuration_qwen2_5_omni import Qwen2_5OmniConfig  # type: ignore
    except ImportError:
        from spatial_omni.model.configuration import Qwen2_5OmniConfig  # type: ignore

    cfg = Qwen2_5OmniConfig.from_pretrained(args.model_id)
    cfg.loss_type = "ForCausalLMLoss"
    tc = cfg.thinker_config; tc.loss_type = "ForCausalLMLoss"
    # --- Switch to IV / Neural-IV baseline ---
    tc.spatial_encoder_type = args.spatial_encoder_type
    tc.spatial_iv_token_dim = int(args.iv_token_dim)
    tc.spatial_iv_projector_hidden_dim = int(args.iv_projector_hidden_dim)
    tc.spatial_iv_num_mel_bins = int(args.iv_num_mel_bins)
    tc.spatial_iv_band_pool = int(args.iv_band_pool)
    tc.spatial_iv_output_scale = float(args.iv_output_scale)
    tc.spatial_iv_feature_to_seld_ratio = int(args.iv_feature_to_seld_ratio)
    tc.spatial_iv_downsample_factor = int(args.iv_downsample_factor)
    tc.spatial_iv_max_audio_seconds = float(MAX_AUDIO_SECONDS)
    tc.spatial_neural_iv_hidden_channels = int(args.neural_iv_hidden_channels)
    # Shared FeatureBridge uses the SELD233 config fields.
    tc.seld_baseline_repo_path = os.path.abspath(args.baseline_repo_path) if args.baseline_repo_path else ""
    tc.seld_feature_stats_dir = os.path.abspath(args.seld_feature_stats_dir) if args.seld_feature_stats_dir else ""
    tc.seld_task_id = "233"
    tc.seld_num_feature_channels = 7
    tc.seld_hop_length = 320  # 2.5Hz target after ratio/downsample
    # LLM-side spatial token rate: 16kHz / hop_length / feature_to_seld_ratio / downsample_factor
    # = 16000/320/5/4 = 2.5 Hz → 20s clip = 50 placeholders (matches SO-Encoder)
    placeholders_per_clip = int(round(MAX_AUDIO_SECONDS * TARGET_TOKEN_RATE))
    rank0_print(
        f"[build_model] encoder_type={args.spatial_encoder_type}, "
        f"token_dim={tc.spatial_iv_token_dim}, band_pool={tc.spatial_iv_band_pool}, "
        f"placeholders per {MAX_AUDIO_SECONDS}s clip={placeholders_per_clip}"
    )

    device_map = getattr(args, "device_map", None)
    # 启用 flash-attn v2（如果可用）：对 long-context Qwen2.5-Omni 提速显著
    # "auto"（默认）：自动检测 flash_attn 包是否可导入，没装则降级到 sdpa
    attn_impl = getattr(args, "attn_impl", "auto")
    if attn_impl == "auto":
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            attn_impl = "sdpa"
        rank0_print(f"[build_model] attn_impl='auto' resolved to '{attn_impl}'")
    from_pretrained_kwargs = {
        "config": cfg,
        "torch_dtype": dtype_from_name(args.dtype),
        "low_cpu_mem_usage": True,
    }
    if attn_impl and attn_impl != "auto":
        from_pretrained_kwargs["attn_implementation"] = attn_impl
    if device_map is not None:
        from_pretrained_kwargs["device_map"] = device_map
    model = Qwen2_5OmniSpatialForConditionalGeneration.from_pretrained(
        args.model_id, **from_pretrained_kwargs,
    )
    rank0_print(f"[build_model] attn_implementation={attn_impl}")
    processor.sync_spatial_tokenizer_with_model(model)
    model.disable_talker()
    # ALWAYS disable KV cache during training. Rationale:
    #   1. We do full-sequence forward + backward, no autoregressive generation,
    #      so DynamicCache just allocates memory we never read back.
    #   2. More importantly, Qwen2.5-Omni's flash_attention_2 branch in
    #      `_update_causal_mask` raises
    #        "You are attempting to perform batched generation with
    #         padding_side='right'..."
    #      whenever `past_key_values is not None` *and* the last-column of the
    #      attention mask is not all 1s (i.e. right-padding). Our collator
    #      deliberately right-pads for label/prefix bookkeeping, so this check
    #      trips every step with flash-attn + use_cache=True (= the default
    #      when gradient_checkpointing is OFF).
    # Both the top-level config and the thinker text config must be disabled;
    # `_update_causal_mask` reads `self.config.use_cache` on the *thinker text*
    # module, but `from_pretrained` copies use_cache into both.
    model.config.use_cache = False
    if hasattr(model, "thinker"):
        model.thinker.config.use_cache = False
        # Text backbone reads its own config; chase it down explicitly.
        text_cfg = getattr(getattr(model.thinker, "config", None), "text_config", None)
        if text_cfg is not None:
            text_cfg.use_cache = False
        text_model = getattr(model.thinker, "model", None)
        if text_model is not None and hasattr(text_model, "config"):
            text_model.config.use_cache = False
    if args.gradient_checkpointing:
        enable_gradient_checkpointing(model)
    if device_map is None:
        model.to(args.device)
    # -----------------------------------------------------------------
    # CRITICAL: re-initialize the IV/Neural-IV adapter + projector weights
    # *after* from_pretrained + model.to(device).
    #
    # Root cause: from_pretrained(low_cpu_mem_usage=True) instantiates the
    # spatial adapter submodules on the meta device first, then materializes
    # their tensors onto CUDA as uninitialized memory when loading the Qwen
    # base checkpoint (those keys are "missing" from the checkpoint, so HF
    # leaves them at whatever random garbage shows up in the fresh CUDA
    # allocation). The `_init_conv_encoder` / `_init_linear_stack` calls
    # inside the adapter's __init__ ran on meta tensors and were silent
    # no-ops. The result: Conv2d weights can contain sNaN / 1e+30 / -inf
    # values, so even with fp32 inputs the conv output is 100% NaN at
    # step 1 and every trainable gradient becomes NaN.
    #
    # Empirically observed:
    #   [iv-probe] current_iv (finite) -> after conv_encoder (100% NaN)
    # with weights in fp32 after `IV_MODULES_FP32=1`. Re-running the
    # init methods now that the tensors live on CUDA fixes it.
    # -----------------------------------------------------------------
    for attr in ("spatial_iv_adapter", "spatial_neural_iv_adapter"):
        adapter = getattr(model.thinker, attr, None)
        if adapter is None:
            continue
        if hasattr(adapter, "_init_conv_encoder"):
            adapter._init_conv_encoder()
        if hasattr(adapter, "_init_linear_stack") and hasattr(adapter, "token_head"):
            adapter._init_linear_stack(adapter.token_head)
        # Ensure LayerNorm starts at weight=1, bias=0 (not garbage).
        if hasattr(adapter, "token_norm"):
            with torch.no_grad():
                adapter.token_norm.weight.fill_(1.0)
                adapter.token_norm.bias.zero_()
        rank0_print(f"[build_model] re-initialized {attr} weights after from_pretrained.")
    for attr in ("spatial_iv_projector", "spatial_neural_iv_projector"):
        proj = getattr(model.thinker, attr, None)
        if proj is None:
            continue
        # SOTokenProjector is a Sequential(Linear, GELU, Linear). Use
        # standard Xavier init for non-final, small-std for final.
        linear_layers = [m for m in proj.modules() if isinstance(m, torch.nn.Linear)]
        with torch.no_grad():
            for i, layer in enumerate(linear_layers):
                if i == len(linear_layers) - 1:
                    torch.nn.init.normal_(layer.weight, mean=0.0, std=1e-3)
                else:
                    torch.nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    layer.bias.zero_()
        rank0_print(f"[build_model] re-initialized {attr} weights after from_pretrained.")

    # Sanity check: confirm weights are finite.
    for attr in ("spatial_iv_adapter", "spatial_neural_iv_adapter",
                  "spatial_iv_projector", "spatial_neural_iv_projector"):
        mod = getattr(model.thinker, attr, None)
        if mod is None:
            continue
        for name, p in mod.named_parameters():
            if not torch.isfinite(p).all():
                n_nan = int(torch.isnan(p).sum().item())
                n_inf = int(torch.isinf(p).sum().item())
                raise RuntimeError(
                    f"[build_model] {attr}.{name} still contains non-finite "
                    f"values after re-init (nan={n_nan}, inf={n_inf}). "
                    f"Adapter init logic is broken.")
    if args.projector_fp32:
        # Cast whichever projector is active to fp32.
        for attr in ("spatial_iv_projector", "spatial_neural_iv_projector"):
            proj = getattr(model.thinker, attr, None)
            if proj is not None:
                proj.to(dtype=torch.float32)
                rank0_print(f"Cast {attr} to fp32.")
    if getattr(args, "iv_modules_fp32", False):
        # Cast the IV / Neural-IV adapter + projector to fp32, matching the
        # old DCASE2024_seld_baseline launch_train_neural_iv_baseline.sh
        # behaviour (the `--spatial-fp32` flag in train_legacy_spatial_qa.py →
        # cast_spatial_modules_to_fp32). That configuration ran neural_iv
        # stage1/stage2 without NaN grads on the same FOA data & same Qwen
        # 2.5-Omni trunk; bf16 on the adapter's backward is what triggers
        # bfloat16 underflow through LayerNorm / the final `output_scale *
        # token_head(...)` multiplication.
        #
        # We cast the whole adapter module (not just conv/norm/head), which
        # is how the old script did it — that guarantees all learnable
        # submodules share the same dtype and no mid-graph cast is required.
        # feature_bridge is non-trainable (STFT+mel+IV normalization on
        # frozen buffers) and is separately forced to fp32 via
        # autocast(enabled=False) + explicit `.to(torch.float32)` on the
        # input inside `_resolve_feature_output`.
        for attr in ("spatial_iv_adapter", "spatial_neural_iv_adapter",
                      "spatial_iv_projector", "spatial_neural_iv_projector"):
            mod = getattr(model.thinker, attr, None)
            if mod is not None:
                mod.to(dtype=torch.float32)
                rank0_print(f"Cast {attr} (whole module) to fp32.")
    return model


# ---------------------------------------------------------------------------
# Training mode configuration
# ---------------------------------------------------------------------------

def _iv_param_match(name: str) -> bool:
    """Match IV adapter / projector params (works for both iv and neural_iv).

    Matches: spatial_iv_adapter.*, spatial_iv_projector.*,
             spatial_neural_iv_adapter.*, spatial_neural_iv_projector.*
    Does NOT match: seld_feature_bridge.* (shared FeatureBridge is a pure
    operator and should stay frozen alongside everything else).
    """
    return "spatial_iv_adapter" in name \
        or "spatial_iv_projector" in name \
        or "spatial_neural_iv_adapter" in name \
        or "spatial_neural_iv_projector" in name


def freeze_all_but_projector(model):
    """Stage 1: only IV adapter + IV projector are trainable."""
    enabled = []
    for _, p in model.named_parameters(): p.requires_grad_(False)
    for n, p in model.named_parameters():
        if _iv_param_match(n): p.requires_grad_(True); enabled.append(n)
    return enabled

def resolve_lora_target_modules(model, prefixes, suffixes):
    res = []
    for mn, _ in model.named_modules():
        if not any(mn.startswith(p) for p in prefixes): continue
        if mn.rsplit(".", 1)[-1] in suffixes: res.append(mn)
    if not res: raise ValueError(f"No LoRA targets under {prefixes} with {suffixes}.")
    return sorted(set(res))

def apply_llm_lora(model, args):
    if get_peft_model is None: raise ImportError("pip install peft")
    tm = resolve_lora_target_modules(
        model, list(args.lora_target_prefixes), list(args.lora_target_modules)
    )
    lc = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                    bias="none", task_type=TaskType.CAUSAL_LM, target_modules=tm)
    model = get_peft_model(model, lc)
    if args.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    return model, tm

def configure_encoder_lora_training(model, args):
    """Stage 2: IV adapter + projector + LoRA trainable."""
    enabled = []
    for _, p in model.named_parameters(): p.requires_grad_(False)
    for n, p in model.named_parameters():
        if _iv_param_match(n) or "lora_" in n:
            p.requires_grad_(True); enabled.append(n)
    return enabled

def configure_beats_lora_training(model, args):
    """Not supported by IV / Neural-IV baselines (no pretrained backbone)."""
    raise NotImplementedError(
        "IV / Neural-IV baselines do not support --beats-lora (no pretrained backbone "
        "to unfreeze). Use --projector-only or --encoder-lora instead."
    )


# ---------------------------------------------------------------------------
# Optimizer / scheduler
# ---------------------------------------------------------------------------

def build_optimizer(model, args):
    """Build AdamW with separate param-groups for projector / LoRA / other.

    The 'projector' bucket covers BOTH the IV adapter and the IV projector
    (they are sized similarly and trained jointly). Each group can override
    `--lr` via `--projector-lr` / `--lora-lr`.
    """
    buckets: Dict[str, List[torch.nn.Parameter]] = {
        "projector_decay": [], "projector_nodecay": [],
        "lora_decay": [],      "lora_nodecay": [],
        "other_decay": [],     "other_nodecay": [],
    }
    counts = {"projector": 0, "lora": 0, "other": 0}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        nodecay = p.ndim == 1 or n.endswith(".bias") or "norm" in n.lower()
        if _iv_param_match(n):
            key = "projector"
        elif "lora_" in n:
            key = "lora"
        else:
            key = "other"
        counts[key] += 1
        buckets[f"{key}_{'nodecay' if nodecay else 'decay'}"].append(p)

    base_lr = args.lr
    proj_lr  = args.projector_lr if args.projector_lr is not None else base_lr
    lora_lr  = args.lora_lr      if args.lora_lr      is not None else base_lr
    proj_wd  = args.projector_weight_decay if args.projector_weight_decay is not None else args.weight_decay

    param_groups = [
        {"params": buckets["projector_decay"],   "lr": proj_lr,  "weight_decay": proj_wd,            "name": "projector_decay"},
        {"params": buckets["projector_nodecay"], "lr": proj_lr,  "weight_decay": 0.0,                "name": "projector_nodecay"},
        {"params": buckets["lora_decay"],        "lr": lora_lr,  "weight_decay": args.weight_decay,  "name": "lora_decay"},
        {"params": buckets["lora_nodecay"],      "lr": lora_lr,  "weight_decay": 0.0,                "name": "lora_nodecay"},
        {"params": buckets["other_decay"],       "lr": base_lr,  "weight_decay": args.weight_decay,  "name": "other_decay"},
        {"params": buckets["other_nodecay"],     "lr": base_lr,  "weight_decay": 0.0,                "name": "other_nodecay"},
    ]
    param_groups = [g for g in param_groups if len(g["params"]) > 0]
    rank0_print(
        f"Optimizer groups: projector={counts['projector']}(lr={proj_lr:.2e},wd={proj_wd}) "
        f"lora={counts['lora']}(lr={lora_lr:.2e}) "
        f"other={counts['other']}(lr={base_lr:.2e})"
    )
    return torch.optim.AdamW(param_groups)


# ---------------------------------------------------------------------------
# Checkpointing helpers
# ---------------------------------------------------------------------------

def resolve_resume_path(args):
    if args.resume_checkpoint_path: return os.path.abspath(args.resume_checkpoint_path)
    if args.resume_tag:
        return os.path.abspath(
            os.path.join(args.output_dir, "checkpoints", f"{args.resume_tag}_trainable.pt")
        )
    return None

def save_trainable_checkpoint(model, opt, sched, path, epoch, step, metrics):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ts = {n: p.detach().cpu() for n, p in unwrap_model(model).named_parameters() if p.requires_grad}
    torch.save({"epoch": epoch, "step": step, "metrics": metrics,
                "trainable_state_dict": ts, "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict() if sched else None}, path)

def save_artifacts(model, processor, opt, sched, args, epoch, step, metrics, tag):
    if not is_main_process(): return
    cd = os.path.join(args.output_dir, "checkpoints"); os.makedirs(cd, exist_ok=True)
    save_trainable_checkpoint(model, opt, sched, os.path.join(cd, f"{tag}_trainable.pt"),
                            epoch, step, metrics)
    if args.save_full_model:
        fd = os.path.join(cd, f"{tag}_full")
        unwrap_model(model).save_pretrained(fd); processor.save_pretrained(fd)

def resume_training_state(model, opt, sched, path, model_only, device):
    from spatial_omni.utils.ckpt_compat import remap_legacy_state_dict
    ckpt = torch.load(path, map_location="cpu")
    sd = ckpt.get("trainable_state_dict", ckpt)
    sd = remap_legacy_state_dict(sd)
    res = unwrap_model(model).load_state_dict(sd, strict=False)
    if not model_only:
        os_ = ckpt.get("optimizer")
        if os_ is not None: opt.load_state_dict(os_)
        ss = ckpt.get("scheduler")
        if sched is not None and ss is not None: sched.load_state_dict(ss)
    m = ckpt.get("metrics") or {}
    if model_only: return {"start_epoch": 1, "global_optimizer_step": 0, "load_result": res}
    return {"start_epoch": int(ckpt.get("epoch", 0)) + 1,
            "global_optimizer_step": int(m.get("global_optimizer_step", 0)),
            "load_result": res}

def infer_best_valid_from_output_dir(od):
    bp = os.path.join(od, "checkpoints", "best_trainable.pt")
    if not os.path.exists(bp): return float("inf"), -1
    pl = torch.load(bp, map_location="cpu"); m = pl.get("metrics") or {}
    return float(m.get("valid_loss", float("inf"))), int(pl.get("epoch", -1))


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------

def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()
            if isinstance(v, torch.Tensor) and k not in {"meta", "prefix_lengths"}}

def count_supervised_tokens(labels, ignore_index=-100):
    return int((labels != ignore_index).sum().item())

def normalize_answer(text):
    return " ".join(str(text).strip().lower().split())

def compute_batch_loss(model, batch, device):
    """Forward + loss. DOES NOT raise on non-finite loss; returns the raw
    loss tensor and lets the caller handle NaN/Inf cases with DDP-synchronized
    skip logic. Raising here would kill one rank and hang the others on the
    next all_reduce collective.
    """
    out = model(**to_device(batch, device), return_dict=True)
    if out.loss is None:
        raise RuntimeError("loss=None")
    return out.loss, {"loss": float(out.loss.detach()),
                    "supervised_tokens": count_supervised_tokens(batch["labels"])}

def evaluate(model, loader, device):
    debug_rank_print(f"enter evaluate batches={len(loader)}")
    # Use the unwrapped module so we don't go through DDP's forward hook during
    # validation.  Under no_grad DDP.forward typically short-circuits, but this
    # eliminates any chance of a stray reducer/broadcast collective firing
    # asynchronously and desyncing ranks.
    eval_model = unwrap_model(model)
    eval_model.eval()
    tw, ts = 0.0, 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="valid", leave=False, disable=not is_main_process()):
            loss, stats = compute_batch_loss(eval_model, batch, device)
            tw += float(loss) * stats["supervised_tokens"]
            ts += stats["supervised_tokens"]
    # IMPORTANT: do NOT raise on ts==0 before the all_reduce below, otherwise an
    # empty shard on a single rank would exit early and hang every other rank on
    # the all_reduce.  Always contribute to the collective first, then check the
    # global total.
    tw = reduce_scalar_sum(tw, device); ts = int(reduce_scalar_sum(float(ts), device))
    if ts == 0:
        raise RuntimeError("Validation loader produced 0 supervised tokens on every rank.")
    debug_rank_print(f"leave evaluate tokens={ts}")
    return {"valid_loss": tw / max(ts, 1), "valid_supervised_tokens": float(ts)}

def _sync_skip_flag(local_skip: bool, device) -> bool:
    """Across-rank OR of a boolean skip flag. If any rank's forward produced
    a NaN/Inf loss, all ranks must take the SAME skip branch, otherwise the
    DDP reducer / grad accumulation pipeline desyncs.
    """
    if not is_distributed():
        return local_skip
    flag = torch.tensor([1.0 if local_skip else 0.0], dtype=torch.float32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item() > 0.5)


def train_one_epoch(model, loader, opt, sched, device, grad_accum_steps, max_grad_norm,
                    log_every, epoch, optimizer_step_per_batch, writer=None,
                    global_step_start=0, global_optimizer_step_start=0, on_optimizer_step=None):
    model.train(); opt.zero_grad(set_to_none=True)
    tw, ts, os_ = 0.0, 0, 0; t0 = time.time()
    n_skipped_nonfinite_loss = 0
    n_skipped_nonfinite_grad = 0
    prog = tqdm(loader, desc=f"epoch {epoch}", leave=False, disable=not is_main_process())
    for step, batch in enumerate(prog, start=1):
        # 1. Forward: ALWAYS run on all ranks; DDP reducer 会在 backward 里做 allreduce
        loss, stats = compute_batch_loss(model, batch, device)

        # 2. Cross-rank OR of non-finite-loss flag: 任何一卡 NaN/Inf 都要全局 skip,
        #    否则只有部分 rank 跳过 backward/step, 下一步 all_reduce 就 hang/报错
        local_nonfinite = not torch.isfinite(loss).item()
        nonfinite_loss = _sync_skip_flag(local_nonfinite, device)

        if nonfinite_loss:
            # 所有 rank 同步 skip 掉这一 batch。
            # 关键：不能走 backward 也不能 zero_grad(其他 rank 的 accumulated grad)，
            # 只推进 scheduler/计数器，保持 rank 间状态一致。
            n_skipped_nonfinite_loss += 1
            if is_main_process() and n_skipped_nonfinite_loss <= 5:
                rank0_print(f"[nonfinite-loss skip] epoch={epoch} step={step} "
                            f"local_loss={stats['loss']} (all ranks synced skip)")
            # 在 step 边界也要清梯度，避免把旧的 accumulated grad 跨 batch 带过去
            should = optimizer_step_per_batch or step % grad_accum_steps == 0 or step == len(loader)
            if should:
                opt.zero_grad(set_to_none=True)
                if sched is not None: sched.step()
                os_ += 1
            continue

        tw += stats["loss"] * stats["supervised_tokens"]; ts += stats["supervised_tokens"]
        should = optimizer_step_per_batch or step % grad_accum_steps == 0 or step == len(loader)
        ctx = nullcontext()
        if is_distributed() and isinstance(model, DDP) and not should: ctx = model.no_sync()
        with ctx: (loss / grad_accum_steps).backward()
        if should:
            # 3. Clip grad norm; clip_grad_norm_ 会返回 total_norm, 可用它检测 inf/nan
            if max_grad_norm > 0:
                total_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_grad_norm)
            else:
                total_norm = torch.tensor(0.0, device=device)
            # 4. 跨 rank 同步 grad non-finite flag. clip_grad_norm_ 本身不 raise,
            #    但若 grad 里有 NaN/Inf, optimizer.step() 会把权重污染.
            grad_nonfinite_local = not torch.isfinite(total_norm).item()
            grad_nonfinite = _sync_skip_flag(grad_nonfinite_local, device)
            if grad_nonfinite:
                n_skipped_nonfinite_grad += 1
                if is_main_process() and n_skipped_nonfinite_grad <= 5:
                    rank0_print(f"[nonfinite-grad skip] epoch={epoch} step={step} "
                                f"local_grad_norm={total_norm.item()} (all ranks synced skip)")
                    # Diagnostic: full list of trainable params partitioned by
                    # (all-NaN, partial-NaN, finite). Helps distinguish
                    # "NaN originating inside this module" from "NaN flowing
                    # in from upstream".
                    all_nan, partial_nan, finite_with_grad, no_grad = [], [], [], []
                    for name, p in model.named_parameters():
                        if not p.requires_grad:
                            continue
                        if p.grad is None:
                            no_grad.append(name)
                            continue
                        g = p.grad
                        n_nan = int(torch.isnan(g).sum().item())
                        n_inf = int(torch.isinf(g).sum().item())
                        total = int(g.numel())
                        if n_nan == total:
                            all_nan.append((name, total, p.dtype))
                        elif n_nan + n_inf > 0:
                            partial_nan.append((name, n_nan, n_inf, total, p.dtype))
                        else:
                            g_abs = g.detach().abs()
                            finite_with_grad.append(
                                (name, float(g_abs.mean().item()),
                                 float(g_abs.max().item()), total, p.dtype))
                    rank0_print(f"[nonfinite-grad detail] all-NaN={len(all_nan)} "
                                f"partial-NaN={len(partial_nan)} "
                                f"finite={len(finite_with_grad)} "
                                f"grad=None={len(no_grad)}")
                    rank0_print("  -- all-NaN params (numel, dtype) --")
                    for name, total, dt in all_nan:
                        rank0_print(f"    {name:70s}  numel={total:8d}  {dt}")
                    if partial_nan:
                        rank0_print("  -- partial-NaN params (#nan, #inf, numel, dtype) --")
                        for name, n_nan, n_inf, total, dt in partial_nan:
                            rank0_print(f"    {name:70s}  nan={n_nan} inf={n_inf} numel={total} {dt}")
                    if finite_with_grad:
                        rank0_print("  -- FINITE grads (these params are training normally) --")
                        for name, g_mean, g_max, total, dt in finite_with_grad:
                            rank0_print(f"    {name:70s}  |g|_mean={g_mean:.3e}  "
                                        f"|g|_max={g_max:.3e}  numel={total}  {dt}")
                    if no_grad:
                        rank0_print(f"  -- trainable but grad=None (first 10) --")
                        for name in no_grad[:10]:
                            rank0_print(f"    {name}")
                opt.zero_grad(set_to_none=True)
                if sched is not None: sched.step()
                os_ += 1
            else:
                opt.step()
                if sched is not None: sched.step()
                opt.zero_grad(set_to_none=True); os_ += 1
                if on_optimizer_step is not None:
                    on_optimizer_step(global_optimizer_step_start + os_,
                                    {"epoch": epoch, "micro_step": step,
                                        "loss": stats["loss"],
                                        "supervised_tokens": stats["supervised_tokens"]})
        lr = opt.param_groups[0]["lr"]
        if is_main_process():
            prog.set_postfix(step=f"{step}/{len(loader)}", loss=f"{stats['loss']:.4f}",
                            lr=f"{lr:.2e}",
                            skip_l=n_skipped_nonfinite_loss, skip_g=n_skipped_nonfinite_grad)
        if writer is not None:
            g = global_step_start + step
            writer.add_scalar("train/batch_loss", stats["loss"], g)
            writer.add_scalar("train/lr", lr, g)
    elapsed = time.time() - t0
    # Barrier first so stragglers finish their last micro-batch before we
    # launch scalar all_reduces; otherwise the straggler eats into the NCCL
    # watchdog budget of the faster ranks.
    distributed_barrier()
    tw = reduce_scalar_sum(tw, device); ts = int(reduce_scalar_sum(float(ts), device))
    os_ = int(reduce_scalar_sum(float(os_), device)) // max(get_world_size(), 1)
    n_skipped_nonfinite_loss = int(reduce_scalar_sum(float(n_skipped_nonfinite_loss), device)) // max(get_world_size(), 1)
    n_skipped_nonfinite_grad = int(reduce_scalar_sum(float(n_skipped_nonfinite_grad), device)) // max(get_world_size(), 1)
    if is_main_process() and (n_skipped_nonfinite_loss or n_skipped_nonfinite_grad):
        rank0_print(f"[epoch {epoch}] skipped: nonfinite_loss={n_skipped_nonfinite_loss}, "
                    f"nonfinite_grad={n_skipped_nonfinite_grad}")
    if ts == 0:
        raise RuntimeError("Training epoch produced 0 supervised tokens across all ranks.")
    return {"train_loss": tw / max(ts, 1), "train_supervised_tokens": float(ts),
            "optimizer_steps": float(os_), "epoch_seconds": elapsed,
            "micro_steps": float(len(loader)),
            "nonfinite_loss_skipped": float(n_skipped_nonfinite_loss),
            "nonfinite_grad_skipped": float(n_skipped_nonfinite_grad)}

def run_validation_generation(model, processor, loader, device, epoch, output_dir,
                                max_new_tokens, num_beams, do_sample):
    debug_rank_print(f"enter generation batches={len(loader)}")
    # Generate on the unwrapped model to bypass DDP entirely; generate() is
    # incompatible with DDP's forward hook anyway.
    eval_model = unwrap_model(model)
    eval_model.eval()
    local_total, local_exact = 0, 0
    local_records = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"gen e{epoch}", leave=False, disable=not is_main_process()):
            gi = {k[4:]: v.to(device) for k, v in batch.items()
                if k.startswith("gen_") and isinstance(v, torch.Tensor)}
            if not gi: continue
            gen = eval_model.generate(
                **gi, return_audio=False,
                max_new_tokens=max_new_tokens, num_beams=num_beams, do_sample=do_sample)
            # ml = 左填充后 batch 的统一序列长度（所有样本共用此截取起点）。
            # generate() 输出 [B, ml + new_tokens]，新 token 从 ml 开始，
            # 对所有样本统一，与各自的 prefix_length 无关。
            # 不能用 pl_i（attention_mask.sum(1)）：当 pl_i < ml 时，
            # gen[i, pl_i:ml] 含有前缀末尾的文本 token，会被 decode 成 echo。
            ml = gi["input_ids"].shape[1]
            gen = gen.detach().cpu()
            for i in range(len(batch["meta"])):
                pt = processor.tokenizer.decode(
                    gen[i, ml:], skip_special_tokens=True).strip()
                m = batch["meta"][i]; gt = str(m["answer"]).strip()
                em = int(normalize_answer(pt) == normalize_answer(gt))
                local_exact += em; local_total += 1
                local_records.append({
                    "epoch": epoch, "pair_id": m.get("pair_id"),
                    "scene_id": m.get("scene_id"),
                    "segment_stem": m.get("segment_stem"),
                    "task_name": m.get("task_name"),
                    "question": m.get("question"),
                    "prompt": m.get("prompt"),
                    "canonical_answer": m.get("canonical_answer"),
                    "answer": gt, "prediction": pt, "exact_match": em,
                    "audio_path": m.get("audio_path"),
                })

    total = int(reduce_scalar_sum(float(local_total), device))
    exact = int(reduce_scalar_sum(float(local_exact), device))
    gathered_records = [local_records]
    if is_distributed():
        debug_rank_print(f"before all_gather_object local_records={len(local_records)}")
        gathered_records = [None for _ in range(get_world_size())]
        dist.all_gather_object(gathered_records, local_records)
        debug_rank_print("after all_gather_object")

    preview = []
    if is_main_process():
        pd_ = os.path.join(output_dir, "valid_predictions"); os.makedirs(pd_, exist_ok=True)
        all_records = []
        for records in gathered_records:
            if records:
                all_records.extend(records)
        all_records.sort(key=lambda r: (str(r.get("pair_id")), str(r.get("task_name"))))
        with open(os.path.join(pd_, f"epoch_{epoch:03d}.jsonl"), "w", encoding="utf-8") as fh:
            for rec in all_records:
                if len(preview) < 10:
                    preview.append(rec)
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    debug_rank_print("before generation barrier")
    distributed_barrier()
    debug_rank_print("after generation barrier")
    return {"valid_generate_examples": float(total),
            "valid_exact_match": float(exact / max(total, 1)),
            "preview_records": preview}

def run_step_validation(model, loader, device):
    # evaluate() switches the unwrapped module to eval(); restore to train()
    # so the outer training loop keeps DDP + dropout + grad state consistent.
    inner = unwrap_model(model)
    was_training = inner.training
    stats = evaluate(model, loader, device)
    if was_training:
        inner.train()
    return stats
                                                                                                                                                            
def format_preview(records):                           
    lines = []
    for i, r in enumerate(records[:8], 1):
        lines += [f"[{i}] pair_id={r.get('pair_id')} task={r.get('task_name')}",
                f"Q: {r.get('prompt', '')}", f"GT: {r.get('answer', '')}",
                f"Pred: {r.get('prediction', '')}", f"EM: {r.get('exact_match', 0)}", ""]
    return "\n".join(lines).strip()

def dump_args(args):
    if not is_main_process(): return
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "train_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

def save_epoch_metrics(output_dir, epoch, metrics):
    if not is_main_process(): return
    with open(os.path.join(output_dir, "epoch_metrics.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({"epoch": epoch, **metrics}, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    args = setup_distributed(args)
    np.random.seed(args.seed); torch.manual_seed(args.seed + get_rank())
    if args.so_repo not in sys.path: sys.path.insert(0, args.so_repo)
    dump_args(args)

    writer = None
    if is_main_process():
        td = os.path.join(args.output_dir, "tensorboard"); os.makedirs(td, exist_ok=True)
        writer = SummaryWriter(log_dir=td)
        writer.add_text("config/args", json.dumps(vars(args), indent=2, sort_keys=True), 0)

    processor = build_processor(args.model_id, args.so_repo)
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
    if is_main_process():
        pd_ = os.path.join(args.output_dir, "processor"); os.makedirs(pd_, exist_ok=True)
        processor.save_pretrained(pd_)

    train_ds, _, _ = build_qa_dataset(args.qa_roots, args.train_split, args.max_train_samples,
                                       audio_search_roots=args.audio_roots)
    valid_ds, _, _ = build_qa_dataset(args.qa_roots, args.valid_split, args.max_valid_samples,
                                       audio_search_roots=args.audio_roots)
    sampler = DistributedSampler(train_ds, shuffle=True) if args.distributed else None
    train_loader = make_loader(
        train_ds,
        SpatialBeatsQACollator(
            processor=processor,
            audio_feature_cache=audio_feature_cache,
            include_generation_inputs=False,
        ),
        args.batch_size, args.num_workers, True, sampler,
        args.persistent_workers, args.prefetch_factor,
    )
    rank0_print(f"Dataset train={len(train_ds):,} valid={len(valid_ds):,}"
                f" | batch={args.batch_size} accum={args.grad_accum_steps}"
                f" | world={get_world_size()} mode={args.train_mode}")

    model = build_model(args, processor)
    lora_targets = []
    if args.train_mode == "projector_only":
        trainable = freeze_all_but_projector(model)
        rank0_print(f"[Stage 1/projector_only] trainable={len(trainable)}")
    elif args.train_mode == "encoder_lora":
        model, lora_targets = apply_llm_lora(model, args)
        trainable = configure_encoder_lora_training(model, args)
        rank0_print(f"[Stage 2/encoder_lora] LoRA={len(lora_targets)} trainable={len(trainable)}")
    elif args.train_mode == "beats_lora":
        # 保留一个清晰的错误提示（IV baseline 不支持 stage3）
        raise ValueError(
            "IV / Neural-IV baselines do not support --beats-lora (no pretrained backbone). "
            "Use --projector-only or --encoder-lora."
        )
    else:
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        rank0_print(f"[all] trainable={len(trainable)}")
    if not trainable: raise RuntimeError("No trainable parameters.")
    if lora_targets: rank0_print(f"LoRA (first 20): {', '.join(lora_targets[:20])}")
    if writer: writer.add_text("model/trainable_parameters", "\n".join(trainable[:500]), 0)

    if args.distributed:
        # find_unused_parameters=True is defensively enabled for all stages.
        # Rationale: if any trainable parameter is skipped on some batch (e.g.
        # a branch in the model that depends on input flags, LoRA modules on
        # paths that aren't always exercised, etc.) DDP with the default
        # False would hang on the autograd reducer.  The small perf overhead
        # is worth the robustness.  Note: enabling this is incompatible with
        # _set_static_graph(), so we no longer call it.
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank,
                    find_unused_parameters=True,
                    broadcast_buffers=False)

    opt = build_optimizer(model, args)
    total_steps = (
        math.ceil(len(train_loader) / (1 if args.optimizer_step_per_batch else args.grad_accum_steps))
        * args.epochs
    )
    warmup = int(total_steps * args.warmup_ratio); sched = None
    if total_steps > 0:
        from transformers.optimization import get_cosine_schedule_with_warmup
        sched = get_cosine_schedule_with_warmup(
            opt, num_warmup_steps=warmup, num_training_steps=total_steps)
    rank0_print(f"Optimizer: steps={total_steps} warmup={warmup} lr={args.lr} wd={args.weight_decay}")

    best_loss, best_ep = infer_best_valid_from_output_dir(args.output_dir)
    if not math.isfinite(best_loss): best_loss, best_ep = float("inf"), -1
    start_ep, gms, gos = 1, 0, 0

    rp = resolve_resume_path(args)
    if rp:
        if not os.path.exists(rp): raise FileNotFoundError(f"Resume ckpt not found: {rp}")
        rs = resume_training_state(model, opt, sched, rp, args.resume_model_only, args.device)
        start_ep = int(rs["start_epoch"]); gos = int(rs["global_optimizer_step"])
        lr_ = rs["load_result"]
        rank0_print(f"Resumed {rp}: next_ep={start_ep} gos={gos}"
                    f" missing={len(lr_.missing_keys)} unexpected={len(lr_.unexpected_keys)}")
        if start_ep > args.epochs:
            raise ValueError(f"Checkpoint ep={start_ep-1} but --epochs={args.epochs}.")

    def maybe_save_step(cur, st):
        if args.save_every_n_optimizer_steps <= 0: return
        if cur % args.save_every_n_optimizer_steps != 0: return
        save_artifacts(model, processor, opt, sched, args, int(st["epoch"]), cur,
                        {"save_type": "step", **st}, f"step_{cur:07d}")

    for epoch in range(start_ep, args.epochs + 1):
        if sampler is not None: sampler.set_epoch(epoch)
        step_vl, full_vl, gl, nsv, nv, ng = build_epoch_valid_loaders(
            valid_ds,
            processor,
            args,
            epoch,
            audio_feature_cache=audio_feature_cache,
        )
        def on_optimizer_step(cur, st):
            # Checkpoint saving happens only on rank 0 and can take tens of
            # seconds on shared storage.  Surround it with a barrier so that
            # the other ranks wait here instead of racing ahead into the next
            # micro-batch's collective (which would otherwise stretch the gap
            # at the following all_reduce, eating into the NCCL watchdog
            # timeout).
            save_triggered = (
                args.save_every_n_optimizer_steps > 0
                and cur % args.save_every_n_optimizer_steps == 0
            )
            valid_triggered = (
                args.valid_every_n_optimizer_steps > 0
                and cur % args.valid_every_n_optimizer_steps == 0
            )
            if save_triggered:
                try:
                    maybe_save_step(cur, st)
                except Exception as exc:
                    # A rank-0-only crash here would deadlock the group on the
                    # barrier below.  Convert it to a warning and keep going.
                    if is_main_process():
                        print(f"[rank0] WARNING: step checkpoint save failed at step {cur}: {exc}",
                              flush=True)
                distributed_barrier()
            if not valid_triggered:
                return
            # Ensure every rank has finished its post-step bookkeeping before
            # we jump into validation forward passes.
            distributed_barrier()
            debug_rank_print(f"start step-valid optimizer_step={cur}")
            vs_step = run_step_validation(model, step_vl, args.device)
            debug_rank_print(f"end step-valid optimizer_step={cur} valid_loss={vs_step['valid_loss']:.6f}")
            rank0_print(f"[step-valid {cur}] valid_loss={vs_step['valid_loss']:.6f}")
            # Barrier BEFORE TensorBoard flush: release non-rank-0 ranks first
            # so a slow/hung NFS flush on rank 0 does not deadlock the group.
            # rank 0 will flush in the background while others proceed to the
            # next micro-batch.
            distributed_barrier()
            if writer is not None:
                writer.add_scalar("step/valid_loss", float(vs_step["valid_loss"]), cur)
                writer.add_scalar(
                    "step/valid_supervised_tokens",
                    float(vs_step["valid_supervised_tokens"]),
                    cur,
                )
                try:
                    writer.flush()
                except Exception as exc:
                    rank0_print(f"[rank0] WARNING: TensorBoard flush failed at step {cur}: {exc}")
        ts = train_one_epoch(
            model=model, loader=train_loader, opt=opt, sched=sched, device=args.device,
            grad_accum_steps=max(1, args.grad_accum_steps), max_grad_norm=args.max_grad_norm,
            log_every=args.log_every, epoch=epoch,
            optimizer_step_per_batch=args.optimizer_step_per_batch,
            writer=writer, global_step_start=gms, global_optimizer_step_start=gos,
            on_optimizer_step=on_optimizer_step,
        )
        gms += int(ts["micro_steps"]); gos += int(ts["optimizer_steps"])
        # Protective save BEFORE validation so a crash in valid / generation
        # (e.g. OOM, cuFFT errors) does not cost us a full epoch of training.
        # A final 'last' with valid metrics is rewritten below after a
        # successful valid pass.
        try:
            save_artifacts(model, processor, opt, sched, args, epoch, gos,
                            {"save_type": "pre_valid", "epoch": epoch,
                             "micro_step": gms, **ts},
                            "last")
            rank0_print(f"[epoch {epoch}] saved last (pre-valid snapshot)")
        except Exception as exc:
            if is_main_process():
                print(f"[rank0] WARNING: pre-valid save failed at epoch {epoch}: {exc}",
                      flush=True)
        distributed_barrier()
        debug_rank_print(f"start epoch-valid epoch={epoch}")
        vs = evaluate(model, full_vl, args.device)
        debug_rank_print(f"end epoch-valid epoch={epoch} valid_loss={vs['valid_loss']:.6f}")
        debug_rank_print(f"start epoch-generation epoch={epoch}")
        gs = run_validation_generation(model, processor, gl, args.device, epoch, args.output_dir,
                                        args.valid_max_new_tokens, args.valid_num_beams,
                                        args.valid_do_sample)
        debug_rank_print(f"end epoch-generation epoch={epoch}")
        pr = gs.pop("preview_records", [])
        summary = {**ts, **vs, **gs, "step_valid_subset_size": float(nsv),
                    "valid_subset_size": float(nv),
                    "valid_generation_size": float(ng),
                    "global_optimizer_step": float(gos)}
        if writer:
            for k in ("train_loss", "valid_loss", "valid_exact_match", "optimizer_steps",
                    "epoch_seconds", "train_supervised_tokens", "valid_supervised_tokens"):
                if k in summary: writer.add_scalar(f"epoch/{k}", float(summary[k]), epoch)
            writer.add_text(f"valid_predictions/epoch_{epoch:03d}", format_preview(pr), epoch)
            try:
                writer.flush()
            except Exception as exc:
                rank0_print(f"[rank0] WARNING: TensorBoard epoch flush failed (epoch {epoch}): {exc}")
        rank0_print(f"[epoch {epoch}] train_loss={summary['train_loss']:.6f}"
                    f" valid_loss={summary['valid_loss']:.6f}"
                    f" valid_em={summary['valid_exact_match']:.4f}"
                    f" opt_steps={int(summary['optimizer_steps'])}"
                    f" secs={summary['epoch_seconds']:.1f}")
        save_epoch_metrics(args.output_dir, epoch, summary)
        try:
            if args.save_every_epoch:
                save_artifacts(model, processor, opt, sched, args, epoch, gos, summary,
                                f"epoch_{epoch:03d}")
            save_artifacts(model, processor, opt, sched, args, epoch, gos, summary, "last")
            if summary["valid_loss"] < best_loss:
                best_loss = summary["valid_loss"]; best_ep = epoch
                save_artifacts(model, processor, opt, sched, args, epoch, gos, summary, "best")
                rank0_print(f"  -> New best epoch={epoch} valid_loss={best_loss:.6f}")
        except Exception as exc:
            # Only rank 0 ever writes.  Don't let a disk error deadlock the
            # collective on the next epoch's barrier.
            if is_main_process():
                print(f"[rank0] WARNING: epoch {epoch} save failed: {exc}", flush=True)
        # All ranks wait for rank-0 save to complete before starting the next
        # epoch; otherwise ranks 1..N race into the next forward pass while
        # rank 0 is still writing checkpoints to shared storage.
        distributed_barrier()

    rank0_print(f"Done. best_epoch={best_ep} best_valid_loss={best_loss:.6f}"
                f" output_dir={args.output_dir}")
    if writer: writer.close()
    cleanup_distributed()


if __name__ == "__main__":
    main()
