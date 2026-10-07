#!/usr/bin/env python
"""SO-Phi4 training entrypoint: SO spatial modality on Phi-4-multimodal.

Stage curriculum mirrors SO-7B:
    projector_only : train so_projector only (base model + adapters frozen)
    encoder_lora   : + "so" LoRA adapter on the LLM (speech/vision adapters
                     stay frozen; base weights frozen)
    beats_lora     : + unfreeze the SO-Encoder (Spatial-BEATs) weights

Run under torchrun (DDP). Data: SO-Dataset qa/{split}.jsonl with FOA audio.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from train_so_qa import QAAudioDataset, RatioMixedDataset, TaggedDataset  # noqa: E402
from so_phi4 import (  # noqa: E402
    SO_ADAPTER_NAME,
    SoPhi4Processor,
    SoPhi4QACollator,
    get_so_phi4_class,
)


def rank0_print(*a, **kw):
    if int(os.environ.get("RANK", "0")) == 0:
        print(*a, **kw, flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", required=True)
    p.add_argument("--beats-checkpoint", required=True)
    p.add_argument("--beats-repo", default=None)
    p.add_argument("--qa-root", required=True)
    p.add_argument("--audio-root", default=None)
    p.add_argument("--train-split", default="train")
    p.add_argument("--valid-split", default="valid")
    p.add_argument("--train-mode", default="projector_only",
                   choices=["projector_only", "encoder_lora", "beats_lora"])
    p.add_argument("--output-dir", required=True)
    p.add_argument("--resume-checkpoint-path", default=None)
    p.add_argument("--replay-qa-roots", nargs="*", default=None,
                   help="mono replay manifests (flat jsonl: audio_path/question/answer); "
                        "enables mixed spatial+replay training with spatial_null")
    p.add_argument("--spatial-replay-ratio", type=int, default=3,
                   help="spatial samples per replay slot (3 = 3:1)")
    p.add_argument("--null-alignment-weight", type=float, default=0.05)
    p.add_argument("--replay-null-ratio", type=float, default=0.5,
                   help="P(replay sample filled with spatial_null); the rest "
                        "use the W-only encoder fill")
    p.add_argument("--max-replay-samples", type=int, default=None)
    p.add_argument("--merge-speech-lora", action="store_true",
                   help="merge the frozen speech LoRA into base weights "
                        "at load time; 'so' becomes the only adapter (slot-exclusive)")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum-steps", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4, help="projector lr")
    p.add_argument("--lora-lr", type=float, default=5e-5)
    p.add_argument("--beats-lr", type=float, default=1e-6)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-valid-samples", type=int, default=64)
    p.add_argument("--valid-every-n-optimizer-steps", type=int, default=500)
    p.add_argument("--save-every-epoch", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--attn-impl", default="sdpa")
    return p.parse_args()


def build_model_and_processor(args):
    from transformers import AutoProcessor

    base_processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    processor = SoPhi4Processor(base_processor)

    cls = get_so_phi4_class(args.model_id)
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    config._attn_implementation = args.attn_impl
    model = cls.from_pretrained(
        args.model_id,
        config=config,
        so_checkpoint_path=args.beats_checkpoint,
        so_beats_repo=args.beats_repo,
        so_freeze_backbone=True,
        so_enable_replay=bool(args.replay_qa_roots),
        so_null_alignment_weight=args.null_alignment_weight,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=False,
    )
    model.config.use_cache = False

    # from_pretrained builds modules under no_init_weights; checkpoint-absent
    # modules (our so_projector) keep garbage memory -> reset explicitly.
    for m in model.so_projector.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
    for n, p in model.so_projector.named_parameters():
        assert not p.isnan().any(), f"so_projector.{n} still has NaN after reset"
    if model.spatial_null is not None:
        torch.nn.init.normal_(model.spatial_null, mean=0.0, std=0.02)
        assert not model.spatial_null.isnan().any()

    # <|spatial|> reuses an existing unused embedding row when the checkpoint
    # vocab (200064) already covers the tokenizer id; only grow if needed.
    old_rows = model.model.embed_tokens.weight.shape[0]
    new_rows = len(processor.tokenizer)
    if new_rows > old_rows:
        model.resize_token_embeddings(new_rows)
        rank0_print(f"[model] resized embeddings {old_rows} -> {new_rows}")
    else:
        rank0_print(f"[model] embedding rows {old_rows} already cover spatial id "
                    f"{processor.spatial_token_id}; no resize")
    assert processor.spatial_token_id < model.model.embed_tokens.weight.shape[0]
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr(), \
        "tied embeddings broken after resize"
    model.set_spatial_token_id(processor.spatial_token_id)

    model.so_encoder._build_model()
    model.replay_null_ratio = float(getattr(args, "replay_null_ratio", 0.5))
    if getattr(args, "merge_speech_lora", False):
        model.merge_pretrained_adapters_into_base()  # Merge on CPU before moving to the device.
    return model, processor


def configure_trainable(model, args):
    for p in model.parameters():
        p.requires_grad_(False)

    groups = {"projector": [], "lora": [], "beats": []}
    for name, p in model.named_parameters():
        if name.startswith("so_projector.") or name == "spatial_null":
            p.requires_grad_(True)
            groups["projector"].append(p)

    if args.train_mode in ("encoder_lora", "beats_lora"):
        model.add_so_adapter(r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
        for name, p in model.named_parameters():
            if "lora_" in name and f".{SO_ADAPTER_NAME}." in name:
                p.requires_grad_(True)
                groups["lora"].append(p)

    if args.train_mode == "beats_lora":
        model.so_encoder._freeze_backbone = False
        for p in model.so_encoder.parameters():
            p.requires_grad_(True)
            groups["beats"].append(p)
        model.so_encoder.train()

    model._refreeze_pretrained_adapters()
    n_train = sum(p.numel() for g in groups.values() for p in g)
    rank0_print(f"[train] mode={args.train_mode} trainable params={n_train/1e6:.2f}M "
                f"(projector={sum(p.numel() for p in groups['projector'])/1e6:.2f}M "
                f"lora={sum(p.numel() for p in groups['lora'])/1e6:.2f}M "
                f"beats={sum(p.numel() for p in groups['beats'])/1e6:.1f}M)")
    return groups


def build_optimizer_and_scheduler(model, groups, args, total_opt_steps):
    param_groups = []
    if groups["projector"]:
        param_groups.append({"params": groups["projector"], "lr": args.lr,
                             "weight_decay": args.weight_decay, "name": "projector"})
    if groups["lora"]:
        param_groups.append({"params": groups["lora"], "lr": args.lora_lr,
                             "weight_decay": args.weight_decay, "name": "lora"})
    if groups["beats"]:
        param_groups.append({"params": groups["beats"], "lr": args.beats_lr,
                             "weight_decay": args.weight_decay, "name": "beats"})
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)

    warmup = max(1, int(total_opt_steps * args.warmup_ratio))

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_opt_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


def load_trainable_weights(model, state):
    """Load stage weights, allowing complete components newly enabled by a stage transition."""
    missing, unexpected = model.load_state_dict(state, strict=False)
    required = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    # Earlier stages intentionally omit components that are frozen there.
    # Once a component is present, require its complete trained state.
    newly_trainable = set()
    if not any(name.startswith("so_encoder.") for name in state):
        newly_trainable.update(name for name in required if name.startswith("so_encoder."))
    if not any("lora_" in name for name in state):
        newly_trainable.update(name for name in required if "lora_" in name)
    if "spatial_null" not in state:
        newly_trainable.add("spatial_null")
    missing_required = sorted(required.intersection(missing) - newly_trainable)
    if unexpected or missing_required:
        raise ValueError(f"Incompatible checkpoint: unexpected={unexpected[:10]}, "
                         f"missing trained keys={missing_required[:10]}")
    return missing, unexpected


def save_trainable(model, args, path):
    raw = model.module if isinstance(model, DDP) else model
    state = {n: p.detach().cpu() for n, p in raw.named_parameters() if p.requires_grad}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        raise FileExistsError(f"Checkpoint already exists: {path}")
    torch.save(state, path)
    rank0_print(f"[save] {path} ({len(state)} tensors)")


def move_batch(batch, device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


@torch.no_grad()
def evaluate(model, loader, device, max_batches=8):
    model.eval()
    total, count = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = move_batch(batch, device)
        out = model(**batch)
        total += float(out.loss.detach())
        count += 1
    model.train()
    return total / max(1, count)


def main():
    args = parse_args()
    if os.path.exists(os.path.join(args.output_dir, "train_args.json")):
        raise FileExistsError(f"Output directory contains an existing run: {args.output_dir}. Choose a new --output-dir.")
    torch.manual_seed(args.seed)

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if distributed:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world = int(os.environ["WORLD_SIZE"])
    else:
        rank, local_rank, world = 0, 0, 1
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    model, processor = build_model_and_processor(args)
    groups = configure_trainable(model, args)

    if args.resume_checkpoint_path:
        state = torch.load(args.resume_checkpoint_path, map_location="cpu", weights_only=True)
        missing, unexpected = load_trainable_weights(model, state)
        rank0_print(f"[resume] {args.resume_checkpoint_path}: "
                    f"missing_frozen_or_new={len(missing)} unexpected={len(unexpected)} "
                    "missing_trained=0")
    model.to(device)

    search_roots = [args.audio_root] if args.audio_root else None
    train_ds = QAAudioDataset(
        os.path.join(args.qa_root, f"{args.train_split}.jsonl"),
        max_samples=args.max_train_samples, audio_search_roots=search_roots)
    valid_ds = QAAudioDataset(
        os.path.join(args.qa_root, f"{args.valid_split}.jsonl"),
        max_samples=args.max_valid_samples, audio_search_roots=search_roots)
    if args.replay_qa_roots:
        replay_parts = [
            QAAudioDataset(path, max_samples=args.max_replay_samples,
                           audio_search_roots=search_roots)
            for path in args.replay_qa_roots
        ]
        replay_ds = (replay_parts[0] if len(replay_parts) == 1
                     else torch.utils.data.ConcatDataset(replay_parts))
        train_ds = RatioMixedDataset(
            TaggedDataset(train_ds, has_spatial=True),
            TaggedDataset(replay_ds, has_spatial=False),
            spatial_per_replay=args.spatial_replay_ratio,
        )
        rank0_print(f"[train] mixed replay: ratio={args.spatial_replay_ratio}:1 "
                    f"replay_rows={len(replay_ds)} mixed_len={len(train_ds)}")
    collator = SoPhi4QACollator(processor)

    sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True,
                                 seed=args.seed) if distributed else None
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=(sampler is None),
        sampler=sampler, collate_fn=collator, num_workers=args.num_workers,
        pin_memory=True, drop_last=True)
    valid_loader = DataLoader(
        valid_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator,
        num_workers=0, pin_memory=True)

    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    total_opt_steps = steps_per_epoch * args.epochs
    optimizer, scheduler = build_optimizer_and_scheduler(model, groups, args, total_opt_steps)
    rank0_print(f"[train] dataset={len(train_ds)} micro_steps/epoch={len(train_loader)} "
                f"opt_steps={total_opt_steps} global_batch="
                f"{args.batch_size * args.grad_accum_steps * world}")

    if distributed:
        # BEATs has branches that don't contribute to every batch's loss
        # (same defensive setting as the Qwen SO trainer).
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    if rank == 0:
        with open(os.path.join(args.output_dir, "train_args.json"), "x") as f:
            json.dump(vars(args), f, indent=2)
    writer = SummaryWriter(os.path.join(args.output_dir, "tensorboard")) if rank == 0 else None

    model.train()
    opt_step = 0
    running = []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        pbar = tqdm(train_loader, desc=f"epoch {epoch}", disable=rank != 0)
        for micro, batch in enumerate(pbar):
            batch = move_batch(batch, device)
            out = model(**batch)
            if not torch.isfinite(out.loss):
                raise FloatingPointError(f"Non-finite training loss at epoch={epoch}, micro={micro}")
            loss = out.loss / args.grad_accum_steps
            loss.backward()
            running.append(float(out.loss.detach()))

            if (micro + 1) % args.grad_accum_steps == 0 or micro + 1 == len(train_loader):
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for g in groups.values() for p in g if p.requires_grad],
                    args.max_grad_norm, error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                opt_step += 1
                if writer is not None:
                    writer.add_scalar("train/loss", float(out.loss.detach()), opt_step)
                    writer.add_scalar("train/learning_rate", scheduler.get_last_lr()[0], opt_step)
                    writer.add_scalar("train/grad_norm", float(grad_norm), opt_step)
                if rank == 0:
                    avg = sum(running[-50:]) / len(running[-50:])
                    pbar.set_postfix(loss=f"{avg:.4f}",
                                     lr=f"{scheduler.get_last_lr()[0]:.2e}",
                                     step=opt_step, grad_norm=f"{float(grad_norm):.4f}")
                if opt_step % args.valid_every_n_optimizer_steps == 0:
                    vl = evaluate(model, valid_loader, device)
                    if not math.isfinite(vl):
                        raise FloatingPointError(f"Non-finite validation loss at step={opt_step}")
                    if writer is not None:
                        writer.add_scalar("valid/loss", vl, opt_step)
                    rank0_print(f"[step-valid {opt_step}] valid_loss={vl:.6f}")

        if args.save_every_epoch and rank == 0:
            save_trainable(model, args, os.path.join(
                args.output_dir, "checkpoints", f"epoch_{epoch:03d}_trainable.pt"))

    if rank == 0:
        save_trainable(model, args, os.path.join(
            args.output_dir, "checkpoints", "last_trainable.pt"))
        rank0_print(f"[done] {time.time()-t0:.0f}s, final loss "
                    f"{sum(running[-50:])/len(running[-50:]):.4f}")
    if writer is not None:
        writer.close()
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
