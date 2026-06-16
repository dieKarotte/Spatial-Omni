#!/usr/bin/env python
"""Forward-only valid_loss probe (SO-30B / Qwen3-Omni-MoE path).

Verifies the loaded best ckpt + Qwen3 build path actually reproduces the
training-time valid_loss when given the SAME valid set the trainer saw. If
this matches, the ckpt loads correctly and the model build reproduces the
training architecture; a larger gap means resume / build is broken.

Paths are taken from environment variables / CLI flags so the script carries
no hard-coded absolute paths:

    QWEN3_TRANSFORMERS_FORK   transformers source tree containing qwen3_omni_moe
    SO_MODEL_ID               Qwen3-Omni-30B-A3B base model dir
    SO_ENCODER_CHECKPOINT     SO-Encoder (Spatial-BEATs) checkpoint .pt
    SO_ENCODER_REPO           BEATs repo (optional; vendored by default)

Usage:
    python scripts/probe_valid_loss_so_qwen3.py \\
        --ckpt runs/so_30b/stage3_beats_lora/checkpoints/best_trainable.pt \\
        --qa-root /path/to/SO-Dataset/qa \\
        --split valid --max-samples 64
"""
from __future__ import annotations

import argparse
import json  # noqa: F401  (kept for parity / ad-hoc debugging)
import os
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_FORK = os.environ.get("QWEN3_OMNI_FORK", os.environ.get("QWEN3_TRANSFORMERS_FORK", ""))
if _FORK and os.path.isdir(_FORK) and _FORK not in sys.path:
    sys.path.insert(0, _FORK)

# Same kaldi epsilon patch as train_so_qa_qwen3.py
import torchaudio.compliance.kaldi as _kaldi  # noqa: E402


def _safe_get_epsilon(device, dtype):
    return torch.tensor(torch.finfo(torch.float).eps, device=device, dtype=dtype)


_kaldi._get_epsilon = _safe_get_epsilon


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model-id", default=os.environ.get("SO_MODEL_ID", ""))
    ap.add_argument(
        "--beats-checkpoint",
        default=os.environ.get("SO_ENCODER_CHECKPOINT", ""),
    )
    ap.add_argument(
        "--beats-repo",
        default=os.environ.get("SO_ENCODER_REPO", ""),
    )
    ap.add_argument(
        "--qa-root",
        default=os.environ.get("SO_DATASET_QA_ROOT", ""),
        help="QA root containing {split}.jsonl (or set SO_DATASET_QA_ROOT).",
    )
    ap.add_argument("--audio-root", default=os.environ.get("SO_AUDIO_ROOT", None))
    ap.add_argument("--split", default="valid", help="train|valid|test")
    ap.add_argument("--max-samples", type=int, default=64)
    args = ap.parse_args()

    if not args.model_id:
        print("[probe] --model-id (or SO_MODEL_ID) is required", file=sys.stderr)
        return 2
    if not args.beats_checkpoint:
        print("[probe] --beats-checkpoint (or SO_ENCODER_CHECKPOINT) is required", file=sys.stderr)
        return 2
    if not args.qa_root:
        print("[probe] --qa-root (or SO_DATASET_QA_ROOT) is required", file=sys.stderr)
        return 2

    # Reuse the trainer's data + collator + model build helpers.
    import train_so_qa as _trainer
    import train_so_qa_qwen3  # noqa: F401  (registers monkey-patches)

    # Re-apply the Qwen3 build hooks (the train_so_qa_qwen3 module only patches
    # them when its main() runs).
    from train_so_qa_qwen3 import (
        _build_model_qwen3,
        _build_processor_qwen3,
    )

    _trainer.build_model = _build_model_qwen3
    _trainer.build_processor = _build_processor_qwen3

    # Build args namespace compatible with trainer.
    a = argparse.Namespace(
        model_id=args.model_id,
        so_repo=_ROOT,
        beats_checkpoint=args.beats_checkpoint,
        beats_repo=args.beats_repo,
        qa_root=args.qa_root,
        audio_root=args.audio_root,
        train_split="train",
        valid_split=args.split,
        device="cuda:0",
        device_map=None,
        dtype="bfloat16",
        attn_impl="flash_attention_2",
        gradient_checkpointing=True,
        train_mode="beats_lora",
        beats_lora=True,
        encoder_lora=False,
        projector_only=False,
        # LoRA matches training config
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_target_prefixes=["model.layers"],
        # Spatial config (must match training)
        projector_type="pixel_shuffle",
        projector_shuffle_factor=4,
        encoder_token_rate=10.0,
        # Misc args trainer reads
        max_train_samples=None,
        max_valid_samples=args.max_samples,
        batch_size=1,
        num_workers=2,
        prefetch_factor=2,
        persistent_workers=False,
        seed=0,
        spatial_fp32=False,
        spatial_backbone_fp32=False,
        sanitize_nonfinite_grads_patterns=[],
        audio_feature_cache_manifest=None,
        audio_feature_cache_max_entries=256,
        local_rank=0,
        distributed=False,
    )

    print("[probe] building processor + model ...")
    processor = _build_processor_qwen3(a.model_id, a.so_repo)
    model = _build_model_qwen3(a, processor)
    print(f"[probe] model built; spatial_token_id = {model.config.spatial_token_index}")

    # PEFT wrap (mirror trainer)
    from train_so_qa import apply_llm_lora

    model, target_modules = apply_llm_lora(model, a)
    print(f"[probe] PEFT wrap: {len(target_modules)} target modules")

    # Resume model weights ONLY (no opt/sched), exactly like trainer does.
    from train_so_qa import resume_training_state

    rs = resume_training_state(model, None, None, args.ckpt, model_only=True, device=a.device)
    lr = rs["load_result"]
    print(
        f"[probe] resume: missing={len(lr.missing_keys)} unexpected={len(lr.unexpected_keys)}"
    )
    if lr.unexpected_keys:
        print(f"[probe]   first unexpected: {lr.unexpected_keys[:3]}")

    # Sanity-check the PEFT runtime config: scaling, dropout, active adapter.
    # If LoRA was wrapped with a different config than the trainer used,
    # forward() will still load the weights but produce gibberish output.
    print("[probe] PEFT runtime config:")
    print(f"  active_adapters: {model.active_adapters}")
    print(f"  peft_config keys: {list(model.peft_config.keys())}")
    for adapter, cfg in model.peft_config.items():
        print(
            f"  adapter='{adapter}': r={cfg.r} alpha={cfg.lora_alpha} dropout={cfg.lora_dropout} "
            f"target_modules(first3)={list(cfg.target_modules)[:3]} bias={cfg.bias}"
        )
    # Inspect a real LoRA module's runtime scaling
    for n, m in model.named_modules():
        if "layers.0.self_attn.q_proj" in n and hasattr(m, "scaling"):
            print(f"  module {n}:")
            print(f"     scaling = {m.scaling}")
            print(f"     active_adapter = {m.active_adapter}")
            print(f"     base_layer.weight.abs_mean = {m.base_layer.weight.float().abs().mean().item():.6e}")
            if hasattr(m, "lora_A"):
                print(f"     lora_A keys: {list(m.lora_A.keys())}")
                if "default" in m.lora_A:
                    print(f"     lora_A.default.weight.abs_mean = {m.lora_A['default'].weight.float().abs().mean().item():.6e}")
                    print(f"     lora_B.default.weight.abs_mean = {m.lora_B['default'].weight.float().abs().mean().item():.6e}")
            break

    model.eval()

    # Build dataset + collator the trainer way.
    print(f"[probe] loading {a.qa_root}/{args.split}.jsonl (limit {args.max_samples}) ...")
    ds_kwargs = {"max_samples": args.max_samples}
    if a.audio_root:
        ds_kwargs["audio_search_roots"] = [os.path.abspath(a.audio_root)]
    valid_ds = _trainer.QAAudioDataset(
        os.path.join(a.qa_root, f"{args.split}.jsonl"),
        **ds_kwargs,
    )
    print(f"[probe] dataset size: {len(valid_ds)}")

    collator = _trainer.SpatialBeatsQACollator(
        processor=processor,
        target_token_rate=2.5,
        include_generation_inputs=False,
    )
    loader = torch.utils.data.DataLoader(
        valid_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collator
    )

    print(f"[probe] running forward on {len(valid_ds)} samples ...")
    n = len(valid_ds)
    total_loss_w = 0.0
    total_tokens = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            inputs = {
                k: (v.to(a.device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
                if k not in {"meta", "prefix_lengths"}
            }
            out = model(**inputs, return_dict=True)
            loss = float(out.loss.detach())
            n_tok = int((batch["labels"] != -100).sum().item())
            total_loss_w += loss * n_tok
            total_tokens += n_tok
            if i % 8 == 0 or i == n - 1:
                avg = total_loss_w / max(total_tokens, 1)
                print(f"  [{i+1:>3d}/{n}] loss={loss:.4f} tokens={n_tok}  running_avg={avg:.4f}")

    final = total_loss_w / max(total_tokens, 1)
    print(f"\n[probe] FINAL valid_loss on {a.qa_root}/{args.split}.jsonl[:{n}] = {final:.4f}")
    print()
    if final < 0.45:
        print("[probe] PASS: forward valid_loss matches training expectation; ckpt is "
              "loaded correctly and the model build reproduces the training architecture.")
    elif final < 0.55:
        print("[probe] PARTIAL: ckpt loads but loss is somewhat higher than training. "
              "Could be small-sample sampling noise vs the full training valid set.")
    else:
        print("[probe] FAIL: forward valid_loss is WAY higher than training. ckpt resume is "
              "not actually loading the trained weights, or model build is producing a "
              "different architecture than at train time.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
