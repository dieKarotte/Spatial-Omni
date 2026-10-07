#!/usr/bin/env python
"""Train the spatial branch and language adapters of SO-AF3."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from train_so_qa import QAAudioDataset, RatioMixedDataset, TaggedDataset  # noqa: E402


def rank0_print(*a, **kw):
    if int(os.environ.get("RANK", "0")) == 0:
        print(*a, **kw, flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--beats-checkpoint",
                   default=os.environ.get("SO_ENCODER_CKPT", os.path.join(_ROOT, "checkpoints/SO-Encoder/SO-Encoder.pt")))
    p.add_argument("--beats-repo", default=None)
    p.add_argument("--qa-root", default=os.path.join(_ROOT, "SO-Dataset/qa"))
    p.add_argument("--audio-root", default=os.path.join(_ROOT, "SO-Dataset"))
    p.add_argument("--train-split", default="train")
    p.add_argument("--valid-split", default="valid")
    p.add_argument("--train-mode", default="encoder_lora",
                   choices=["projector_only", "encoder_lora", "beats_lora"])
    p.add_argument("--output-dir", required=True)
    p.add_argument("--resume-checkpoint-path", default=None)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum-steps", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=2)
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
    p.add_argument("--max-valid-samples", type=int, default=16)
    p.add_argument("--valid-every-n-optimizer-steps", type=int, default=500)
    p.add_argument("--save-every-n-optimizer-steps", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--replay-qa-roots", nargs="*", default=None,
                   help="mono replay manifests (flat jsonl: audio_path/question/answer)")
    p.add_argument("--spatial-replay-ratio", type=int, default=3)
    p.add_argument("--null-alignment-weight", type=float, default=0.05)
    p.add_argument("--max-replay-samples", type=int, default=None)
    p.add_argument("--attn-impl", default="sdpa")
    return p.parse_args()


LORA_TARGET_RE = r"llm\.model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)"


def build_model_and_collator(args):
    from so_audiollm.af3 import SoAF3Collator, build_af3
    model, tokenizer = build_af3(
        args.model_dir, args.beats_checkpoint, args.beats_repo, args.attn_impl,
        enable_replay=bool(args.replay_qa_roots),
        null_alignment_weight=args.null_alignment_weight,
    )
    return model, tokenizer, SoAF3Collator(tokenizer)


def configure_trainable(model, args):
    for p in model.parameters():
        p.requires_grad_(False)

    groups = {"projector": [], "lora": [], "beats": []}
    # LoRA first: peft's inject_adapter_in_model re-freezes every non-LoRA
    # parameter when it finishes, so projector/BEATs must be unfrozen after it.
    if args.train_mode in ("encoder_lora", "beats_lora"):
        from peft import LoraConfig, inject_adapter_in_model
        cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                         lora_dropout=args.lora_dropout,
                         target_modules=LORA_TARGET_RE)
        inject_adapter_in_model(cfg, model, adapter_name="so")
        for name, p in model.named_parameters():
            if "lora_" in name:
                p.requires_grad_(True)
                groups["lora"].append(p)

    for name, p in model.named_parameters():
        if (".so_projector." in name or name.startswith("spatial.so_projector")
                or name.endswith("spatial.spatial_null")):
            p.requires_grad_(True)
            groups["projector"].append(p)

    if args.train_mode == "beats_lora":
        model.spatial.so_encoder._freeze_backbone = False
        for p in model.spatial.so_encoder.parameters():
            p.requires_grad_(True)
            groups["beats"].append(p)
        model.spatial.so_encoder.train()

    n = {k: sum(p.numel() for p in v) / 1e6 for k, v in groups.items()}
    rank0_print(f"[train] mode={args.train_mode} trainable: projector={n['projector']:.2f}M "
                f"lora={n['lora']:.2f}M beats={n['beats']:.1f}M")
    return groups


def build_optimizer_and_scheduler(groups, args, total_opt_steps):
    param_groups = []
    for key, lr in (("projector", args.lr), ("lora", args.lora_lr), ("beats", args.beats_lr)):
        if groups[key]:
            param_groups.append({"params": groups[key], "lr": lr,
                                 "weight_decay": args.weight_decay, "name": key})
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)
    warmup = max(1, int(total_opt_steps * args.warmup_ratio))

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_opt_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_trainable(model, path):
    raw = model.module if isinstance(model, DDP) else model
    state = {n: p.detach().cpu() for n, p in raw.named_parameters() if p.requires_grad}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        raise FileExistsError(f"Checkpoint already exists: {path}")
    torch.save(state, path)
    rank0_print(f"[save] {path} ({len(state)} tensors)")


def move_batch(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, loader, device, max_batches=8):
    model.eval()
    total, count = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        out = model(**move_batch(batch, device))
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
        rank = int(os.environ["RANK"]); local_rank = int(os.environ["LOCAL_RANK"])
        world = int(os.environ["WORLD_SIZE"])
    else:
        rank, local_rank, world = 0, 0, 1
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    model, tokenizer, collator = build_model_and_collator(args)
    groups = configure_trainable(model, args)
    if args.resume_checkpoint_path:
        state = torch.load(args.resume_checkpoint_path, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected:
            raise ValueError(f"Checkpoint contains unexpected model keys: {unexpected[:10]}")
        rank0_print(f"[resume] missing={len(missing)} unexpected={len(unexpected)}")
    model.to(device)

    roots = [args.audio_root] if args.audio_root else None
    train_ds = QAAudioDataset(os.path.join(args.qa_root, f"{args.train_split}.jsonl"),
                              max_samples=args.max_train_samples, audio_search_roots=roots)
    if args.replay_qa_roots:
        replay_parts = [QAAudioDataset(path, max_samples=args.max_replay_samples,
                                       audio_search_roots=roots)
                        for path in args.replay_qa_roots]
        replay_ds = (replay_parts[0] if len(replay_parts) == 1
                     else torch.utils.data.ConcatDataset(replay_parts))
        train_ds = RatioMixedDataset(
            TaggedDataset(train_ds, has_spatial=True),
            TaggedDataset(replay_ds, has_spatial=False),
            spatial_per_replay=args.spatial_replay_ratio)
        rank0_print(f"[train] mixed replay: ratio={args.spatial_replay_ratio}:1 "
                    f"replay_rows={len(replay_ds)} mixed_len={len(train_ds)}")
    valid_ds = QAAudioDataset(os.path.join(args.qa_root, f"{args.valid_split}.jsonl"),
                              max_samples=args.max_valid_samples, audio_search_roots=roots)

    sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True,
                                 seed=args.seed) if distributed else None
    num_workers = args.num_workers
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=(sampler is None), sampler=sampler,
                              collate_fn=collator, num_workers=num_workers,
                              pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False,
                              collate_fn=collator, num_workers=0, pin_memory=True)

    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    total_opt_steps = steps_per_epoch * args.epochs
    optimizer, scheduler = build_optimizer_and_scheduler(groups, args, total_opt_steps)
    rank0_print(f"[train] dataset={len(train_ds)} "
                f"micro/epoch={len(train_loader)} opt_steps={total_opt_steps} "
                f"global_batch={args.batch_size * args.grad_accum_steps * max(world,1)}")

    if distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    if rank == 0:
        with open(os.path.join(args.output_dir, "train_args.json"), "x") as f:
            json.dump(vars(args), f, indent=2)
    writer = SummaryWriter(os.path.join(args.output_dir, "tensorboard")) if rank == 0 else None

    model.train()
    opt_step, running, t0 = 0, [], time.time()
    for epoch in range(1, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        pbar = tqdm(train_loader, desc=f"epoch {epoch}", disable=rank != 0)
        for micro, batch in enumerate(pbar):
            out = model(**move_batch(batch, device))
            (out.loss / args.grad_accum_steps).backward()
            running.append(float(out.loss.detach()))
            if (micro + 1) % args.grad_accum_steps == 0 or micro + 1 == len(train_loader):
                torch.nn.utils.clip_grad_norm_(
                    [p for g in groups.values() for p in g], args.max_grad_norm)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
                opt_step += 1
                if writer is not None:
                    writer.add_scalar("train/loss", float(out.loss.detach()), opt_step)
                    writer.add_scalar("train/learning_rate", scheduler.get_last_lr()[0], opt_step)
                if rank == 0 and opt_step % 5 == 0:
                    avg = sum(running[-40:]) / len(running[-40:])
                    pbar.set_postfix(loss=f"{avg:.4f}",
                                     lr=f"{scheduler.get_last_lr()[0]:.2e}", step=opt_step)
                if opt_step % args.valid_every_n_optimizer_steps == 0:
                    vl = evaluate(model, valid_loader, device)
                    rank0_print(f"[step-valid {opt_step}] valid_loss={vl:.6f}")
                if rank == 0 and opt_step % args.save_every_n_optimizer_steps == 0:
                    save_trainable(model, os.path.join(
                        args.output_dir, "checkpoints",
                        f"step_{opt_step:07d}_trainable.pt"))
        if rank == 0:
            save_trainable(model, os.path.join(
                args.output_dir, "checkpoints", f"epoch_{epoch:03d}_trainable.pt"))
        vl = evaluate(model, valid_loader, device)
        rank0_print(f"[epoch {epoch}] valid_loss={vl:.6f}")

    if rank == 0:
        save_trainable(model, os.path.join(args.output_dir, "checkpoints", "last_trainable.pt"))
        rank0_print(f"[done] {time.time()-t0:.0f}s final_loss="
                    f"{sum(running[-40:])/len(running[-40:]):.4f}")
    if writer is not None:
        writer.close()
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
