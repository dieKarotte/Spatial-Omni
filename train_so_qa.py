"""Training script for Spatial-BEATs + Qwen2.5-Omni spatial QA.                                                                                              
                                                                                           
Three training stages:                                                                                                                                       
Stage 1 (projector_only)  - freeze BEATs encoder, train projector only                                                                                     
Stage 2 (encoder_lora)    - freeze BEATs encoder, train projector + LLM LoRA                                                                               
Stage 3 (beats_lora)      - unfreeze BEATs encoder, train everything + LLM LoRA                                                                            
                                                                                                                                                            
Single-stage example (Stage 1):                        
    torchrun --nproc_per_node=4 train_so_qa.py \\
        --model-id /path/to/Qwen2.5-Omni-7B \\
        --beats-checkpoint /path/to/best.pt \\
        --qa-root /path/to/SO-Dataset/qa \\
        --output-dir ./runs/stage1 \\
        --projector-only --epochs 5 --lr 1e-4

Resume into Stage 2 (from stage1/best):
    torchrun --nproc_per_node=4 train_so_qa.py \\
        --resume-checkpoint-path ./runs/stage1/checkpoints/best_trainable.pt \\
        --resume-model-only --encoder-lora \\
        --output-dir ./runs/stage2 --epochs 3 --lr 3e-5

Resume into Stage 3 (from stage2/best):
    torchrun --nproc_per_node=4 train_so_qa.py \\
        --resume-checkpoint-path ./runs/stage2/checkpoints/best_trainable.pt \\
        --resume-model-only --beats-lora \\
        --output-dir ./runs/stage3 --epochs 3 --lr 1e-5
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
DEFAULT_OUTPUT_DIR = "./so_runs/default_run"
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
    p.add_argument("--beats-checkpoint", default=DEFAULT_BEATS_CKPT)
    p.add_argument("--beats-repo", default=DEFAULT_BEATS_REPO)
    p.add_argument("--so-repo", default=os.environ.get("SO_REPO", os.path.dirname(os.path.abspath(__file__))))
    p.add_argument("--qa-root", default=None)
    p.add_argument("--qa-roots", nargs="+", default=None)
    p.add_argument("--audio-root", default=None,
                    help="Optional audio root prefix. When the QA jsonl's "
                         "`audio_path` is relative to a different root than the "
                         "qa-root (e.g. SO-Dataset HF release where qa/*.jsonl "
                         "references audio/.. at the dataset root), pass that "
                         "root here. The qa-dir, --audio-root, and the qa-dir's "
                         "parent are all probed in order.")
    p.add_argument("--audio-roots", nargs="+", default=None,
                    help="Multiple audio search roots; takes precedence over "
                         "--audio-root.")
    p.add_argument("--replay-qa-root", default=None,
                    help="Optional mono replay QA root. Triggers --mixed-spatial-replay implicitly.")
    p.add_argument("--replay-qa-roots", nargs="+", default=None,
                    help="Optional mono replay QA roots (multi-source).")
    p.add_argument("--replay-train-split", default=None,
                    help="Replay train split name. Defaults to --train-split.")
    p.add_argument("--mixed-spatial-replay", action="store_true",
                    help="Enable mixed spatial+mono replay training. "
                         "Auto-enabled when --replay-qa-root(s) is provided.")
    p.add_argument("--spatial-replay-ratio", type=int, default=3,
                    help="Number of spatial samples per replay sample (default 3).")
    p.add_argument("--null-alignment-weight", type=float, default=0.05,
                    help="MSE weight for W-only null-alignment loss (default 0.05).")
    p.add_argument("--spatial-null-lr", type=float, default=None,
                    help="LR override for the spatial_null parameter. Default: --projector-lr or --lr.")
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
                    help="Override lr for so_projector. Default: --lr.")
    p.add_argument("--lora-lr", type=float, default=None,
                    help="Override lr for LoRA params. Default: --lr.")
    p.add_argument("--beats-lr", type=float, default=None,
                    help="Override lr for so_encoder (stage3). Default: --lr.")
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
                    help="Stage 1: freeze BEATs, train projector only.")
    mg.add_argument("--encoder-lora", dest="train_mode", action="store_const", const="encoder_lora",
                    help="Stage 2: freeze BEATs, train projector + LLM LoRA.")
    mg.add_argument("--beats-lora", dest="train_mode", action="store_const", const="beats_lora",
                    help="Stage 3: unfreeze BEATs, train encoder+projector+LLM LoRA.")
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
                    help="Cast so_projector to fp32 for stability.")
    p.add_argument("--projector-type", default="pixel_shuffle",
                    choices=("mlp", "mlp_ln", "pixel_shuffle"),
                    help="Spatial projector variant. Default 'pixel_shuffle' "
                         "with --projector-shuffle-factor=4 maps encoder 10Hz to "
                         "LLM 2.5Hz. 'mlp': LLaVA-1.5-style 2-layer MLP. "
                         "'mlp_ln': MLP + pre/post LayerNorm.")
    p.add_argument("--projector-shuffle-factor", type=int, default=4,
                    help="Temporal grouping factor for --projector-type=pixel_shuffle. "
                         "Reduces spatial token count by this factor; LLM-side rate "
                         "is encoder_token_rate / shuffle_factor.")
    p.add_argument("--encoder-token-rate", type=float, default=DEFAULT_ENCODER_TOKEN_RATE,
                    help="Native token rate (Hz) the SO-Encoder checkpoint emits "
                         "before projection. 10.0 for SO-Encoder; legacy 2.5Hz ckpts should "
                         "set this to 2.5 plus --projector-shuffle-factor=1.")
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
    # Build the audio search-roots list (used by QAAudioDataset for path
    # resolution). Order: --audio-roots, --audio-root, qa-dir-parent (auto).
    if args.audio_roots:
        args.audio_roots = [os.path.abspath(r) for r in args.audio_roots]
    elif args.audio_root:
        args.audio_roots = [os.path.abspath(args.audio_root)]
    else:
        args.audio_roots = []
    if args.replay_qa_roots:
        args.replay_qa_roots = [os.path.abspath(r) for r in args.replay_qa_roots]
    elif args.replay_qa_root:
        args.replay_qa_roots = [os.path.abspath(args.replay_qa_root)]
    else:
        args.replay_qa_roots = []
    args.mixed_spatial_replay = bool(args.mixed_spatial_replay or args.replay_qa_roots)
    if args.replay_train_split is None:
        args.replay_train_split = args.train_split
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
        # always the directory of the JSONL itself (legacy behavior). When
        # ``audio_search_roots`` is supplied (e.g. SO-Dataset HF release where
        # `audio_path` is relative to the dataset root, not the qa/ subdir),
        # they are checked in order. We resolve once at __init__ time so
        # subsequent ``sf.read(record["audio_path"])`` calls just use absolute
        # paths and never stat() the filesystem mid-training.
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
            ap = r.get("audio_path")
            if ap is None:
                raise ValueError(f"Record {i} missing audio_path")
            # Resolve audio_path once. Skip stat() for already-absolute paths
            # to avoid O(N) NFS stalls; only probe roots when the path is
            # relative.
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
    enable_mono_replay: bool = False

    @staticmethod
    def _downmix_to_mono(wav_2d: np.ndarray) -> np.ndarray:
        # wav_2d shape: [T, C] (always_2d output from soundfile).
        if wav_2d.shape[1] == 1:
            return wav_2d[:, 0].astype(np.float32, copy=False)
        return wav_2d.mean(axis=1).astype(np.float32, copy=False)

    def _resample_if_needed(self, wav: np.ndarray, sr: int, audio_path: str) -> np.ndarray:
        """Resample mono replay audio to self.sample_rate. Spatial samples must
        already be at the canonical rate."""
        if sr == self.sample_rate:
            return wav
        if not self.enable_mono_replay:
            raise ValueError(f"Expected {self.sample_rate}Hz got {sr} for {audio_path}")
        try:
            from scipy.signal import resample_poly
        except ImportError as exc:
            raise ImportError("scipy is required to resample mono replay audio.") from exc
        gcd = math.gcd(int(sr), int(self.sample_rate))
        up = int(self.sample_rate) // gcd
        down = int(sr) // gcd
        return resample_poly(wav, up, down, axis=0).astype(np.float32, copy=False)

    def __call__(self, features):
        if self.enable_mono_replay:
            return self._call_mixed_replay(features)
        audio_arrs, full_texts, ans_sfxs, meta, sa_lens = [], [], [], [], []
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
            # Keep a single placeholder in text. The processor expands it to
            # the required number of spatial tokens using spatial_token_lengths.
            prefix = (
                self.processor.audio_token
                + self.processor.spatial_token
                + f"\n{feat['prompt'].rstrip()}\n"
            )
            ans = str(feat["answer"]).strip()
            ans_sfx = ans + eos
            full_texts.append(prefix + ans_sfx)
            ans_sfxs.append(ans_sfx)
            audio_arrs.append(wav.astype(np.float32, copy=False))
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
        for i, wav in enumerate(audio_arrs):
            T = wav.shape[1]; sa_t[i, :T] = torch.from_numpy(wav.T)

        processor_kwargs = {}
        if self.audio_feature_cache is not None:
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
            # Use the processor's own banker-rounding routine so the expected
            # count matches the placeholder expansion exactly.
            try:
                exp_t = self.processor._samples_to_so_backbone_tokens(
                    torch.as_tensor(sa_lens, dtype=torch.long)
                )
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

    # ------------------------------------------------------------------ #
    # Mixed spatial+mono replay collation                                 #
    # ------------------------------------------------------------------ #
    def _call_mixed_replay(self, features):
        """Build a batch that may contain both FOA spatial and mono replay items.

        Per sample:
            - FOA: load 4ch wav as [4, T], wav.T → [T, 4]; mono = W channel.
            - Mono: read wav, downmix if needed, resample to 16 kHz; mono is the
              audio for both the LLM <|AUDIO|> path and the W-only spatial path.

        Prompt is identical to the spatial-only path:
            "<|AUDIO|><|spatial|>\n{prompt}\n{answer}{eos}"
        Mono samples still emit one <|spatial|> placeholder; the processor
        expands it to `spatial_token_lengths[i]` repeats and the model fills
        them with `spatial_null` at runtime.
        """
        eos = getattr(self.processor.tokenizer, "eos_token", None) or ""
        audio_arrs, full_texts, ans_sfxs, meta = [], [], [], []
        sa_lens: List[int] = []
        has_spatial_list: List[bool] = []
        foa_arrays: List[Optional[np.ndarray]] = []  # [4, T] or None
        mono_arrays: List[np.ndarray] = []           # [T] (always)

        for feat in features:
            is_spatial = bool(feat.get("_replay_has_spatial", feat.get("has_spatial", True)))
            wav, sr = sf.read(feat["audio_path"], dtype="float32", always_2d=True)
            wav = self._resample_if_needed(wav, sr, feat["audio_path"])
            if is_spatial:
                if wav.shape[1] != 4:
                    raise ValueError(
                        f"Expected 4ch FOA spatial sample, got shape {wav.shape} ({feat['audio_path']})"
                    )
                foa_ct = wav.T  # [4, T]
                if foa_ct.shape[1] > self.max_audio_samples:
                    foa_ct = foa_ct[:, :self.max_audio_samples]
                T = int(foa_ct.shape[1])
                mono = foa_ct[0].astype(np.float32, copy=False)
                foa_arrays.append(foa_ct.astype(np.float32, copy=False))
                # For the LLM <|AUDIO|> path the underlying processor still
                # accepts the 4ch FOA; pass it through unchanged.
                audio_arrs.append(foa_ct.astype(np.float32, copy=False))
            else:
                mono = self._downmix_to_mono(wav)
                if mono.shape[0] > self.max_audio_samples:
                    mono = mono[:self.max_audio_samples]
                T = int(mono.shape[0])
                foa_arrays.append(None)
                # Mono LLM input: shape [T] -> processor's _normalize_audio_list
                # treats it as mono; passes through as 1-channel.
                audio_arrs.append(mono.astype(np.float32, copy=False))
            sa_lens.append(T)
            mono_arrays.append(mono.astype(np.float32, copy=False))
            has_spatial_list.append(is_spatial)

            prefix = (
                self.processor.audio_token
                + self.processor.spatial_token
                + f"\n{feat['prompt'].rstrip()}\n"
            )
            ans = str(feat["answer"]).strip()
            ans_sfx = ans + eos
            full_texts.append(prefix + ans_sfx)
            ans_sfxs.append(ans_sfx)
            meta.append({
                "pair_id": feat.get("pair_id"), "task_name": feat.get("task_name"),
                "answer": feat.get("answer"), "audio_path": feat.get("audio_path"),
                "prompt": feat.get("prompt"), "question": feat.get("question"),
                "canonical_answer": feat.get("canonical_answer"),
                "scene_id": feat.get("scene_id"),
                "segment_stem": feat.get("segment_stem"),
                "has_spatial": is_spatial,
            })

        # Mixed FOA + mono cannot be sent through the underlying spatial processor
        # in one call (it requires same-channel batches). Compute per-sample
        # spatial_token_lengths via the canonical rounding helper, then call
        # the processor with `allow_mono_spatial_tokens=True` AND an explicit
        # spatial_token_lengths to force placeholder expansion uniformly.
        lens_t = torch.as_tensor(sa_lens, dtype=torch.long)
        try:
            token_lengths = self.processor._samples_to_so_backbone_tokens(lens_t)
        except Exception:
            token_lengths = torch.as_tensor(
                [int(round(self.target_token_rate * (l / self.sample_rate))) for l in sa_lens],
                dtype=torch.long,
            )

        # Mixed batches break the processor's "all-mono or all-FOA" invariant.
        # We avoid that by side-stepping the processor's audio/spatial path:
        # build the spatial tensors ourselves and ask the processor to only do
        # text+placeholder expansion + tokenization.
        # Strategy: convert ALL audio_arrs to mono shape [T] for the processor
        # (so it sees a uniform mono batch), expand placeholders via
        # explicit spatial_token_lengths and allow_mono_spatial_tokens=True.
        # Then we attach the real spatial_audio (4ch FOA only for spatial samples)
        # ourselves, plus mono_audio + has_spatial.
        mono_audio_list_for_proc = [m for m in mono_arrays]

        tok = self.processor.tokenizer
        prev_padding_side = getattr(tok, "padding_side", "left")
        tok.padding_side = "right"
        try:
            batch = self.processor(
                text=full_texts,
                audio=mono_audio_list_for_proc,
                padding=True,
                padding_side="right",
                return_tensors="pt",
                spatial_token_lengths=token_lengths,
                allow_mono_spatial_tokens=True,
            )
        finally:
            tok.padding_side = prev_padding_side

        # Build padded spatial_audio [B, T_max, 4] with FOA only for has_spatial.
        B = len(features)
        max_samples = self.max_audio_samples
        spatial_audio = torch.zeros((B, max_samples, 4), dtype=torch.float32)
        spatial_lengths = torch.zeros((B,), dtype=torch.long)
        mono_audio = torch.zeros((B, max_samples), dtype=torch.float32)
        mono_lengths = torch.as_tensor(sa_lens, dtype=torch.long)
        for idx, (mono, foa) in enumerate(zip(mono_arrays, foa_arrays)):
            valid = min(int(mono.shape[0]), max_samples)
            mono_audio[idx, :valid] = torch.from_numpy(mono[:valid])
            if foa is not None:
                fvalid = min(int(foa.shape[1]), max_samples)
                spatial_audio[idx, :fvalid, :] = torch.from_numpy(foa[:, :fvalid].T)
                spatial_lengths[idx] = fvalid
        # Override the processor-supplied spatial fields with our mixed-batch
        # versions (the processor only saw mono audio so it would not have
        # populated spatial_audio for FOA samples).
        batch["spatial_audio"] = spatial_audio
        batch["spatial_audio_attention_mask"] = (
            torch.arange(max_samples).unsqueeze(0) < spatial_lengths.unsqueeze(1)
        )
        batch["spatial_audio_lengths"] = spatial_lengths
        batch["spatial_token_lengths"] = token_lengths
        batch["has_spatial"] = torch.as_tensor(has_spatial_list, dtype=torch.bool)
        batch["mono_audio"] = mono_audio
        batch["mono_audio_lengths"] = mono_lengths

        # Labels: -100 for prefix + pad.
        labels = batch["input_ids"].clone()
        if "attention_mask" in batch:
            labels = labels.masked_fill(batch["attention_mask"] == 0, self.ignore_index)
        ab = self.processor.tokenizer(
            ans_sfxs, padding=True, return_tensors="pt", add_special_tokens=False
        )
        al = ab["attention_mask"].sum(1).long()
        vl = batch["attention_mask"].sum(1).long()
        pl = vl - al
        if (pl < 0).any():
            raise ValueError("Negative prefix length.")
        for i, p in enumerate(pl.tolist()):
            labels[i, :p] = self.ignore_index
        if (labels != self.ignore_index).sum(1).min().item() <= 0:
            raise ValueError("Sample with no supervised answer tokens.")
        batch["labels"] = labels
        batch["meta"] = meta
        batch["prefix_lengths"] = pl

        if self.include_generation_inputs:
            pid = int(self.processor.tokenizer.pad_token_id or 0)
            gi, gm = build_left_padded_batch(
                batch["input_ids"], batch["attention_mask"], pl, pid
            )
            gb = {"input_ids": gi, "attention_mask": gm}
            for k, v in batch.items():
                if k in {"input_ids", "attention_mask", "labels", "meta", "prefix_lengths"}:
                    continue
                if isinstance(v, torch.Tensor):
                    gb[k] = v
            for k, v in gb.items():
                if isinstance(v, torch.Tensor):
                    batch[f"gen_{k}"] = v
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
    # Switch to Spatial-BEATs encoder
    tc.spatial_encoder_type = "so_backbone"
    tc.so_backbone_checkpoint_path = os.path.abspath(args.beats_checkpoint)
    tc.so_backbone_repo_path = os.path.abspath(args.beats_repo)
    tc.so_encoder_dim = 768
    tc.so_projector_hidden_dim = 768
    # Apply projector variant + shuffle factor; LLM-side rate is
    # encoder_token_rate / shuffle_factor. For 10 Hz native + k=4 → 2.5 Hz.
    projector_type = getattr(args, "projector_type", "pixel_shuffle")
    shuffle_factor = int(getattr(args, "projector_shuffle_factor", 4))
    encoder_rate = float(getattr(args, "encoder_token_rate", DEFAULT_ENCODER_TOKEN_RATE))
    if shuffle_factor < 1:
        raise ValueError("--projector-shuffle-factor must be >= 1")
    if projector_type != "pixel_shuffle":
        # mlp / mlp_ln do not pool temporally; LLM sees the encoder native rate.
        shuffle_factor = 1
    effective_rate = encoder_rate / float(shuffle_factor)
    if abs(effective_rate - TARGET_TOKEN_RATE) > 1e-6:
        print(
            f"[build_model] WARNING: LLM-side spatial rate = encoder_token_rate "
            f"({encoder_rate}) / shuffle_factor ({shuffle_factor}) = "
            f"{effective_rate} Hz, not the conventional {TARGET_TOKEN_RATE} Hz. "
            f"Set shuffle_factor={int(round(encoder_rate / TARGET_TOKEN_RATE))} "
            f"for a {TARGET_TOKEN_RATE} Hz LLM feed."
        )
    tc.so_encoder_token_rate = encoder_rate
    tc.so_backbone_target_token_rate = effective_rate
    tc.so_projector_type = projector_type
    tc.so_projector_shuffle_factor = shuffle_factor
    placeholders_per_clip = int(round(MAX_AUDIO_SECONDS * effective_rate))
    rank0_print(
        f"[build_model] spatial rates: encoder={encoder_rate} Hz, "
        f"projector shuffle_factor={shuffle_factor}, LLM={effective_rate} Hz, "
        f"placeholders per {MAX_AUDIO_SECONDS}s clip={placeholders_per_clip}"
    )
    # Freeze BEATs backbone in stages 1 & 2; unfreeze in stage 3
    tc.so_backbone_freeze_backbone = args.train_mode in {"projector_only", "encoder_lora"}
    tc.so_backbone_max_audio_seconds = float(MAX_AUDIO_SECONDS)
    # Mono-replay knobs (gated; default OFF preserves old training byte-for-byte).
    tc.enable_spatial_replay = bool(getattr(args, "mixed_spatial_replay", False))
    if tc.enable_spatial_replay:
        tc.spatial_null_num_tokens = int(round(MAX_AUDIO_SECONDS * effective_rate))
        tc.spatial_null_alignment_weight = float(getattr(args, "null_alignment_weight", 0.05))

    device_map = getattr(args, "device_map", None)
    # 启用 flash-attn v2（如果可用）：对 long-context Qwen2.5-Omni 提速显著
    # （prompt+spatial_tokens ~1000+ tokens，attention 是主要瓶颈之一）。
    # "auto"（默认）：检测 flash_attn 是否可导入，没装则降级到 sdpa，避免上游兼容性问题
    # 未装 flash-attn 的环境直接崩掉。
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
    if args.gradient_checkpointing:
        enable_gradient_checkpointing(model)
        model.config.use_cache = False; model.thinker.config.use_cache = False
    if getattr(tc, "spatial_encoder_type", "seld") == "so_backbone":
          enc = getattr(model.thinker, "so_encoder", None)
          if enc is not None:
              rank0_print(f"[{time.strftime('%H:%M:%S')}] Building SOBackbone model on CPU ...")
              enc._build_model()
              rank0_print(f"[{time.strftime('%H:%M:%S')}] SOBackbone built.")
              if device_map is not None:
                  # device_map='auto' 已将各模块分配到不同 GPU；_build_model() 在 CPU 上
                  # 初始化 SOBackbone，需要将其移到 projector 所在的同一设备，
                  # 避免 accelerate hook 的 device 冲突。
                  proj = getattr(model.thinker, "so_projector", None)
                  if proj is not None:
                      try:
                          enc_target_device = next(proj.parameters()).device
                      except StopIteration:
                          enc_target_device = torch.device(args.device)
                  else:
                      enc_target_device = torch.device(args.device)
                  enc.to(enc_target_device)
                  rank0_print(f"[build_model] Moved so_encoder to {enc_target_device} (device_map mode).")
    if device_map is None:
        model.to(args.device)
    if args.projector_fp32:
        proj = getattr(model.thinker, "so_projector", None)
        if proj is not None:
            proj.to(dtype=torch.float32); rank0_print("Cast projector to fp32.")
    # Guard against HF from_pretrained(torch_dtype=...) materializing the new
    # `spatial_null` parameter from uninitialized memory (it isn't in the pretrained
    # checkpoint, so its __init__ randn fill never lands and the tensor is NaN).
    # Without this, every mono-replay batch injects NaN into inputs_embeds and CE
    # loss is NaN on step 1. Re-init only triggers when the param is meta or
    # non-finite; healthy resumes are bit-identical.
    thinker = getattr(model, "thinker", model)
    if hasattr(thinker, "reinit_spatial_null_if_needed"):
        if thinker.reinit_spatial_null_if_needed():
            sn = getattr(thinker, "spatial_null", None)
            if sn is not None:
                rank0_print(
                    f"[build_model] Re-initialized spatial_null (was meta/NaN/Inf): "
                    f"shape={tuple(sn.shape)} dtype={sn.dtype} "
                    f"|max|={float(sn.abs().max()):.3e} std={float(sn.float().std()):.3e}"
                )
    return model


# ---------------------------------------------------------------------------
# Training mode configuration
# ---------------------------------------------------------------------------

def freeze_all_but_projector(model):
    """Stage 1: only so_projector is trainable."""
    enabled = []
    for _, p in model.named_parameters(): p.requires_grad_(False)
    for n, p in model.named_parameters():
        if "so_projector" in n: p.requires_grad_(True); enabled.append(n)
    return enabled


# --------------------------------------------------------------------------- #
# Mono-replay datasets (only used when --mixed-spatial-replay is enabled)     #
# --------------------------------------------------------------------------- #
class TaggedDataset(Dataset):
    """Tag every record with `_replay_has_spatial` for the mixed collator."""

    def __init__(self, dataset, has_spatial: bool):
        self.dataset = dataset
        self.has_spatial = bool(has_spatial)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = dict(self.dataset[index])
        item["_replay_has_spatial"] = self.has_spatial
        return item


class RatioMixedDataset(Dataset):
    """Deterministic spatial:replay interleave (default 3:1).

    Index pattern: every (spatial_per_replay+1)-cycle slots is one replay
    sample, the rest are spatial samples. Robust under DistributedSampler
    shuffling because indices are mapped to underlying datasets statically.
    """

    def __init__(self, spatial_ds, replay_ds, spatial_per_replay: int = 3):
        self.spatial_ds = spatial_ds
        self.replay_ds = replay_ds
        self.spatial_per_replay = max(1, int(spatial_per_replay))
        if len(self.spatial_ds) == 0:
            raise ValueError("spatial dataset is empty")
        if len(self.replay_ds) == 0:
            raise ValueError("replay dataset is empty")
        replay_slots = max(1, math.ceil(len(self.spatial_ds) / self.spatial_per_replay))
        self._length = len(self.spatial_ds) + replay_slots

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        cycle = self.spatial_per_replay + 1
        if index % cycle == self.spatial_per_replay:
            replay_index = (index // cycle) % len(self.replay_ds)
            return self.replay_ds[replay_index]
        spatial_index = (index - index // cycle) % len(self.spatial_ds)
        return self.spatial_ds[spatial_index]


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
    """Stage 2: projector + LoRA trainable, BEATs frozen."""
    enabled = []
    for _, p in model.named_parameters(): p.requires_grad_(False)
    for n, p in model.named_parameters():
        if "so_projector" in n or "lora_" in n:
            p.requires_grad_(True); enabled.append(n)
    return enabled

def configure_beats_lora_training(model, args):
    """Stage 3: BEATs encoder + projector + LoRA all trainable."""
    enabled = []
    for _, p in model.named_parameters(): p.requires_grad_(False)
    enc = getattr(unwrap_model(model).thinker, "so_encoder", None)
    if enc is not None:
        enc._freeze_backbone = False
        for p in enc.model.parameters(): p.requires_grad_(True)
        enc.model.train(); rank0_print("Unfroze so_encoder (Stage 3).")
    else:
        rank0_print("WARNING: so_encoder not found!")
    for n, p in model.named_parameters():
        if "so_projector" in n or "so_encoder" in n or "lora_" in n:
            p.requires_grad_(True); enabled.append(n)
    return enabled


def configure_mixed_replay_training(model, args):
    """Replay mode (stage 3 default): unfreeze BEATs + projector + spatial_null + LoRA.

    Mirrors `configure_beats_lora_training` but additionally unfreezes the
    learned `spatial_null` parameter so the null token bank can drift toward
    the W-only encoder output.
    """
    enabled = []
    inner = unwrap_model(model)
    for _, p in inner.named_parameters(): p.requires_grad_(False)
    enc = getattr(inner.thinker, "so_encoder", None)
    if enc is not None and getattr(enc, "model", None) is not None:
        enc._freeze_backbone = False
        for p in enc.model.parameters(): p.requires_grad_(True)
        enc.model.train()
        rank0_print("Replay mode: unfroze so_encoder.")
    for n, p in inner.named_parameters():
        if (
            "so_projector" in n
            or "so_encoder" in n
            or "spatial_null" in n
            or "lora_" in n
        ):
            p.requires_grad_(True); enabled.append(n)
    return enabled


# ---------------------------------------------------------------------------
# Optimizer / scheduler
# ---------------------------------------------------------------------------

def build_optimizer(model, args):
    """Build AdamW with separate param-groups for projector / LoRA / BEATs / other.

    Each group can override the base `--lr` via `--projector-lr`, `--lora-lr`,
    `--beats-lr`.  The cosine scheduler scales each group's lr proportionally
    to its initial lr, so ratios are preserved throughout training.
    """
    buckets: Dict[str, List[torch.nn.Parameter]] = {
        "projector_decay": [], "projector_nodecay": [],
        "lora_decay": [],      "lora_nodecay": [],
        "beats_decay": [],     "beats_nodecay": [],
        "null_decay": [],      "null_nodecay": [],
        "other_decay": [],     "other_nodecay": [],
    }
    counts = {"projector": 0, "lora": 0, "beats": 0, "null": 0, "other": 0}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        nodecay = p.ndim == 1 or n.endswith(".bias") or "norm" in n.lower()
        if "so_projector" in n:
            key = "projector"
        elif "lora_" in n:
            key = "lora"
        elif "so_encoder" in n:
            key = "beats"
        elif "spatial_null" in n:
            key = "null"
        else:
            key = "other"
        counts[key] += 1
        buckets[f"{key}_{'nodecay' if nodecay else 'decay'}"].append(p)

    base_lr = args.lr
    proj_lr  = args.projector_lr if args.projector_lr is not None else base_lr
    lora_lr  = args.lora_lr      if args.lora_lr      is not None else base_lr
    beats_lr = args.beats_lr     if args.beats_lr     is not None else base_lr
    null_lr  = (args.spatial_null_lr if getattr(args, "spatial_null_lr", None) is not None
                else proj_lr)
    proj_wd  = args.projector_weight_decay if args.projector_weight_decay is not None else args.weight_decay

    param_groups = [
        {"params": buckets["projector_decay"],   "lr": proj_lr,  "weight_decay": proj_wd,            "name": "projector_decay"},
        {"params": buckets["projector_nodecay"], "lr": proj_lr,  "weight_decay": 0.0,                "name": "projector_nodecay"},
        {"params": buckets["lora_decay"],        "lr": lora_lr,  "weight_decay": args.weight_decay,  "name": "lora_decay"},
        {"params": buckets["lora_nodecay"],      "lr": lora_lr,  "weight_decay": 0.0,                "name": "lora_nodecay"},
        {"params": buckets["beats_decay"],       "lr": beats_lr, "weight_decay": args.weight_decay,  "name": "beats_decay"},
        {"params": buckets["beats_nodecay"],     "lr": beats_lr, "weight_decay": 0.0,                "name": "beats_nodecay"},
        {"params": buckets["null_decay"],        "lr": null_lr,  "weight_decay": args.weight_decay,  "name": "null_decay"},
        {"params": buckets["null_nodecay"],      "lr": null_lr,  "weight_decay": 0.0,                "name": "null_nodecay"},
        {"params": buckets["other_decay"],       "lr": base_lr,  "weight_decay": args.weight_decay,  "name": "other_decay"},
        {"params": buckets["other_nodecay"],     "lr": base_lr,  "weight_decay": 0.0,                "name": "other_nodecay"},
    ]
    param_groups = [g for g in param_groups if len(g["params"]) > 0]
    rank0_print(
        f"Optimizer groups: projector={counts['projector']}(lr={proj_lr:.2e},wd={proj_wd}) "
        f"lora={counts['lora']}(lr={lora_lr:.2e}) "
        f"beats={counts['beats']}(lr={beats_lr:.2e}) "
        f"spatial_null={counts['null']}(lr={null_lr:.2e}) "
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
    try:
        pl = torch.load(bp, map_location="cpu"); m = pl.get("metrics") or {}
    except Exception as exc:
        # Best ckpt corrupted (e.g. truncated by an out-of-disk save). Don't
        # fail the entire resume just to recover best_loss; treat it as
        # missing and let training overwrite it on the next valid pass.
        rank0_print(f"[resume] WARNING: best_trainable.pt unreadable ({exc}); "
                    f"treating as missing. Best-loss tracking restarts.")
        return float("inf"), -1
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
    out = model(**to_device(batch, device), return_dict=True)
    if out.loss is None: raise RuntimeError("loss=None")
    if not torch.isfinite(out.loss):
        # Hard-fail at the source so a NaN/Inf loss makes us read the bad
        # batch's meta and fix the underlying numeric instability rather
        # than silently zeroing it. The mono-replay design
        # explicitly requires gradients to flow through the W-only spatial
        # encoder path so it learns to map [W,0,0,0] toward spatial_null;
        # any post-hoc nan_to_num / grad-zero would defeat that.
        meta = batch.get("meta") or []
        sample_ids = []
        for m in meta[:8]:
            try:
                sample_ids.append(str(m.get("pair_id") or m.get("audio_path") or "?"))
            except Exception:
                sample_ids.append("?")
        raise RuntimeError(
            f"non-finite loss: {out.loss.item():.4g}  sample_ids={sample_ids}"
        )
    inner = unwrap_model(model)
    thinker = getattr(inner, "thinker", inner)
    replay_stats = dict(getattr(thinker, "_last_spatial_replay_stats", {}) or {})
    base_loss = float(out.loss.detach())
    stats = {
        "loss": base_loss,
        "supervised_tokens": count_supervised_tokens(batch["labels"]),
    }
    # Only emit the extended replay-stat keys when the model actually populated
    # them this step. Default training (no <has_spatial> / no replay) leaves
    # `_last_spatial_replay_stats` empty so we keep stats narrow and
    # tensorboard output bit-identical.
    if replay_stats:
        stats.update({
            "loss_total": float(replay_stats.get("loss_total", base_loss)),
            "loss_ce": float(replay_stats.get("loss_ce", base_loss)),
            "loss_null": float(replay_stats.get("loss_null", 0.0)),
            "spatial_samples": float(replay_stats.get("spatial_samples", 0.0)),
            "replay_samples": float(replay_stats.get("replay_samples", 0.0)),
            "spatial_null_norm": float(replay_stats.get("spatial_null_norm", 0.0)),
            "w_only_tokens_norm": float(replay_stats.get("w_only_tokens_norm", 0.0)),
            "w_only_null_cosine": float(replay_stats.get("w_only_null_cosine", 0.0)),
        })
    return out.loss, stats

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

def train_one_epoch(model, loader, opt, sched, device, grad_accum_steps, max_grad_norm,
                    log_every, epoch, optimizer_step_per_batch, writer=None,
                    global_step_start=0, global_optimizer_step_start=0, on_optimizer_step=None):
    model.train(); opt.zero_grad(set_to_none=True)
    # Stage-3 (BEATs unfrozen) calls torch.stft / torch.fft.rfft inside the
    # spatial preprocessor on every forward. After ~10k+ steps cuFFT's plan
    # cache + GPU allocator can fragment enough to throw CUFFT_INTERNAL_ERROR
    # mid-epoch and hang the rank (other ranks then hit NCCL watchdog).
    # Shrink the plan cache aggressively and periodically empty the allocator.
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.cufft_plan_cache.max_size = 8
        except Exception:
            pass
    tw, ts, os_ = 0.0, 0, 0; t0 = time.time()
    replay_totals = {
        "loss_ce": 0.0, "loss_null": 0.0,
        "spatial_samples": 0.0, "replay_samples": 0.0,
        "spatial_null_norm": 0.0, "w_only_tokens_norm": 0.0,
        "w_only_null_cosine": 0.0,
    }
    replay_metric_steps = 0
    prog = tqdm(loader, desc=f"epoch {epoch}", leave=False, disable=not is_main_process())
    for step, batch in enumerate(prog, start=1):
        loss, stats = compute_batch_loss(model, batch, device)
        tw += stats["loss"] * stats["supervised_tokens"]; ts += stats["supervised_tokens"]
        if "loss_ce" in stats or "loss_null" in stats:
            replay_totals["loss_ce"] += stats.get("loss_ce", stats["loss"]) * stats["supervised_tokens"]
            replay_totals["loss_null"] += stats.get("loss_null", 0.0) * stats["supervised_tokens"]
            for key in ("spatial_samples", "replay_samples"):
                replay_totals[key] += stats.get(key, 0.0)
            for key in ("spatial_null_norm", "w_only_tokens_norm", "w_only_null_cosine"):
                replay_totals[key] += stats.get(key, 0.0)
            replay_metric_steps += 1
        should = optimizer_step_per_batch or step % grad_accum_steps == 0 or step == len(loader)
        ctx = nullcontext()
        if is_distributed() and isinstance(model, DDP) and not should: ctx = model.no_sync()
        with ctx: (loss / grad_accum_steps).backward()
        if should:
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_grad_norm)
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
            postfix = {
                "step": f"{step}/{len(loader)}",
                "loss": f"{stats['loss']:.4f}",
                "lr": f"{lr:.2e}",
            }
            if stats.get("replay_samples", 0.0) or stats.get("spatial_samples", 0.0):
                postfix["ce"] = f"{stats.get('loss_ce', 0.0):.4f}"
                postfix["null"] = f"{stats.get('loss_null', 0.0):.4f}"
                postfix["sp"] = int(stats.get("spatial_samples", 0))
                postfix["rp"] = int(stats.get("replay_samples", 0))
            prog.set_postfix(**postfix)
        if writer is not None:
            g = global_step_start + step
            writer.add_scalar("train/batch_loss", stats["loss"], g)
            writer.add_scalar("train/lr", lr, g)
            for key in (
                "loss_ce", "loss_null", "spatial_samples", "replay_samples",
                "spatial_null_norm", "w_only_tokens_norm", "w_only_null_cosine",
            ):
                if key in stats:
                    writer.add_scalar(f"train/batch_{key}", stats[key], g)
        # Drain the allocator periodically so accumulated cuFFT workspaces
        # don't pile up across thousands of steps. 500 micro-steps (~6 min on
        # 8xA100 bs=2 accum=3) is light enough not to slow the run noticeably
        # but frequent enough to keep us out of CUFFT_INTERNAL_ERROR territory.
        if torch.cuda.is_available() and step % 500 == 0:
            torch.cuda.empty_cache()
    elapsed = time.time() - t0
    # Barrier first so stragglers finish their last micro-batch before we
    # launch scalar all_reduces; otherwise the straggler eats into the NCCL
    # watchdog budget of the faster ranks.
    distributed_barrier()
    tw = reduce_scalar_sum(tw, device); ts = int(reduce_scalar_sum(float(ts), device))
    os_ = int(reduce_scalar_sum(float(os_), device)) // max(get_world_size(), 1)
    if ts == 0:
        raise RuntimeError("Training epoch produced 0 supervised tokens across all ranks.")
    reduced_replay = {
        k: reduce_scalar_sum(v, device) for k, v in replay_totals.items()
    }
    metric_steps = int(reduce_scalar_sum(float(replay_metric_steps), device))
    result = {
        "train_loss": tw / max(ts, 1),
        "train_supervised_tokens": float(ts),
        "optimizer_steps": float(os_),
        "epoch_seconds": elapsed,
        "micro_steps": float(len(loader)),
    }
    if metric_steps > 0:
        denom_steps = max(metric_steps, 1)
        result.update({
            "train_loss_ce": reduced_replay["loss_ce"] / max(ts, 1),
            "train_loss_null": reduced_replay["loss_null"] / max(ts, 1),
            "train_spatial_samples": float(reduced_replay["spatial_samples"]),
            "train_replay_samples": float(reduced_replay["replay_samples"]),
            "train_spatial_null_norm": reduced_replay["spatial_null_norm"] / denom_steps,
            "train_w_only_tokens_norm": reduced_replay["w_only_tokens_norm"] / denom_steps,
            "train_w_only_null_cosine": reduced_replay["w_only_null_cosine"] / denom_steps,
        })
    return result

def run_validation_generation(model, processor, loader, device, epoch, output_dir,
                                max_new_tokens, num_beams, do_sample):
    debug_rank_print(f"enter generation batches={len(loader)}")
    # Generate on the unwrapped model to bypass DDP entirely; generate() is
    # incompatible with DDP's forward hook anyway.
    eval_model = unwrap_model(model)
    eval_model.eval()
    # Stage3 (BEATs unfrozen) + full-valid generation accumulates cuFFT plans
    # inside torch.stft across thousands of batches; if the training graph's
    # cached workspace isn't released, cuFFT hits CUFFT_INTERNAL_ERROR on the
    # N-th plan alloc. Clean slate before we start, and shrink the plan cache
    # so stale plans don't pile up during the loop.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.backends.cuda.cufft_plan_cache.max_size = 8
        except Exception:
            pass
    local_total, local_exact = 0, 0
    local_records = []
    with torch.no_grad():
        for step_i, batch in enumerate(tqdm(loader, desc=f"gen e{epoch}", leave=False, disable=not is_main_process())):
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
            # Free the batch's GPU tensors and periodically drain the allocator
            # / cuFFT plan cache so we don't OOM or trip CUFFT_INTERNAL_ERROR on
            # the N-th batch when generating against the full valid split with
            # a stage-3 (BEATs unfrozen) model.
            del gen, gi
            if torch.cuda.is_available() and (step_i + 1) % 100 == 0:
                torch.cuda.empty_cache()

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
    if args.mixed_spatial_replay:
        if not args.replay_qa_roots:
            raise ValueError("--mixed-spatial-replay requires --replay-qa-root(s).")
        replay_ds, _, _ = build_qa_dataset(
            args.replay_qa_roots, args.replay_train_split, args.max_train_samples,
            audio_search_roots=args.audio_roots,
        )
        train_ds = RatioMixedDataset(
            TaggedDataset(train_ds, has_spatial=True),
            TaggedDataset(replay_ds, has_spatial=False),
            spatial_per_replay=args.spatial_replay_ratio,
        )
        rank0_print(
            f"Mixed replay dataset enabled: spatial:replay={args.spatial_replay_ratio}:1 "
            f"mixed_train_len={len(train_ds):,}"
        )
    sampler = DistributedSampler(train_ds, shuffle=True) if args.distributed else None
    train_loader = make_loader(
        train_ds,
        SpatialBeatsQACollator(
            processor=processor,
            audio_feature_cache=None if args.mixed_spatial_replay else audio_feature_cache,
            include_generation_inputs=False,
            enable_mono_replay=args.mixed_spatial_replay,
        ),
        args.batch_size, args.num_workers, True, sampler,
        args.persistent_workers, args.prefetch_factor,
    )
    rank0_print(f"Dataset train={len(train_ds):,} valid={len(valid_ds):,}"
                f" | batch={args.batch_size} accum={args.grad_accum_steps}"
                f" | world={get_world_size()} mode={args.train_mode}")

    model = build_model(args, processor)
    lora_targets = []
    if args.mixed_spatial_replay:
        # Replay path runs on top of the user-selected train_mode (e.g. stage3
        # beats_lora). Apply LoRA + the unified replay freeze policy.
        model, lora_targets = apply_llm_lora(model, args)
        trainable = configure_mixed_replay_training(model, args)
        rank0_print(
            f"[mixed_spatial_replay] LoRA={len(lora_targets)} trainable={len(trainable)}"
        )
    elif args.train_mode == "projector_only":
        trainable = freeze_all_but_projector(model)
        rank0_print(f"[Stage 1/projector_only] trainable={len(trainable)}")
    elif args.train_mode == "encoder_lora":
        model, lora_targets = apply_llm_lora(model, args)
        trainable = configure_encoder_lora_training(model, args)
        rank0_print(f"[Stage 2/encoder_lora] LoRA={len(lora_targets)} trainable={len(trainable)}")
    elif args.train_mode == "beats_lora":
        model, lora_targets = apply_llm_lora(model, args)
        trainable = configure_beats_lora_training(model, args)
        rank0_print(f"[Stage 3/beats_lora] LoRA={len(lora_targets)} trainable={len(trainable)}")
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
        # Post-resume guard: if the resumed checkpoint did not contain
        # `spatial_null` (typical when resuming from a non-replay ckpt), the
        # parameter is left at whatever build_model produced. Re-run the guard
        # in case build_model's re-init was bypassed or the resume restored a
        # bad value. No-op when spatial_null is finite.
        thinker_post = getattr(unwrap_model(model), "thinker", unwrap_model(model))
        if hasattr(thinker_post, "reinit_spatial_null_if_needed"):
            if thinker_post.reinit_spatial_null_if_needed():
                sn = getattr(thinker_post, "spatial_null", None)
                if sn is not None:
                    rank0_print(
                        f"[resume] Re-initialized spatial_null post-resume "
                        f"(was meta/NaN/Inf): shape={tuple(sn.shape)} "
                        f"|max|={float(sn.abs().max()):.3e} "
                        f"std={float(sn.float().std()):.3e}"
                    )
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
        # ----------------------------------------------------------------
        # Protective save BEFORE validation / generation. If the following
        # valid loss eval or (stage-3) generation loop crashes (e.g. cuFFT
        # errors after tens of thousands of STFT calls), we still have a
        # recoverable `last_trainable.pt` at the end of this epoch's training
        # so the next resume does not have to redo a full epoch. We do NOT
        # update `best` or `epoch_XXX` here because those semantics require
        # a successful valid_loss / generation pass.
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
                    "epoch_seconds", "train_supervised_tokens", "valid_supervised_tokens",
                    "train_loss_ce", "train_loss_null", "train_spatial_samples",
                    "train_replay_samples", "train_spatial_null_norm",
                    "train_w_only_tokens_norm", "train_w_only_null_cosine"):
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
