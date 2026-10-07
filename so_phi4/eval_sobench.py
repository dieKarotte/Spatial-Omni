#!/usr/bin/env python
"""Generate rank-sharded SO-4B predictions from a complete stage-3 checkpoint.

Each input QA record is preserved with an added prediction field. Score
the resulting JSONL files with scripts/score_sobench.py after all ranks
finish. See docs/so-4b.md for single-GPU and distributed commands.
"""

from __future__ import annotations

import argparse
import json
import os

from tqdm import tqdm
import torch

import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import train_so_phi4 as T  # noqa: E402
from so_phi4.collator import _PROMPT_TEMPLATE, _load_foa, _load_mono  # noqa: E402
from so_phi4.processing import SAMPLE_RATE  # noqa: E402

EOS_IDS = [199999, 200020]  # <|endoftext|>, <|end|>


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model-id", required=True)
    p.add_argument("--beats-checkpoint", default=os.environ.get("SO_ENCODER_CKPT", os.path.join(_ROOT, "checkpoints/SO-Encoder/SO-Encoder.pt")))
    p.add_argument("--beats-repo", default=None)
    p.add_argument("--qa-root", required=True)
    p.add_argument("--audio-root", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--mono", action="store_true",
                   help="mono QA eval (MMAU): audio downmixed to mono, spatial "
                        "placeholders filled with the learned spatial_null")
    p.add_argument("--mono-fill", default="null", choices=["null", "foa_w"],
                   help="mono placeholder fill: null=trained spatial_null bank "
                        "(mix-training replay convention); foa_w=W-only FOA "
                        "through the SO encoder (no null bank)")
    p.add_argument("--adapter-mode", default="both",
                   choices=["both", "speech_only", "so_only", "legacy"],
                   help="Adapter activation policy (both=spatial and speech adapters; "
                        "other modes are ablations)")
    p.add_argument("--merge-speech-lora", action="store_true",
                   help="reconstruct the speech-LoRA-merged base used by "
                        "training run before loading trainable weights")
    p.add_argument("--subset-stride", type=int, default=1,
                   help="take records[::N] BEFORE rank sharding (MMAU manifest "
                        "is domain-grouped; stride keeps domain balance)")
    p.add_argument("--attn-impl", default="sdpa")
    return p.parse_args()


def main():
    args = parse_args()
    output_name = "predictions_rank" + os.environ.get("RANK", "0") + ".jsonl"
    if os.path.exists(os.path.join(args.output_dir, output_name)):
        raise FileExistsError(f"Predictions already exist in {args.output_dir}; choose a new --output-dir.")

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    has_null = "spatial_null" in state
    if args.mono and args.mono_fill == "null" and not has_null:
        raise ValueError("This checkpoint has no spatial_null. Use --mono-fill foa_w for mono evaluation.")

    margs = argparse.Namespace(
        model_id=args.model_id,
        beats_checkpoint=args.beats_checkpoint,
        beats_repo=args.beats_repo,
        attn_impl=args.attn_impl,
        train_mode="beats_lora",
        lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        # Reconstruct the optional replay parameter from checkpoint contents.
        replay_qa_roots=["enabled"] if has_null else None,
        null_alignment_weight=0.05,
        merge_speech_lora=args.merge_speech_lora,
    )
    model, proc = T.build_model_and_processor(margs)
    T.configure_trainable(model, margs)
    missing, unexpected = model.load_state_dict(state, strict=False)
    required = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    missing_required = sorted(required.intersection(missing))
    if unexpected or missing_required:
        raise ValueError(f"Incompatible checkpoint: unexpected={unexpected[:10]}, missing trainable keys={missing_required[:10]}")
    if rank == 0:
        print(f"[eval] resume {args.checkpoint}: missing={len(missing)} unexpected={len(unexpected)}")
    model.to(device).eval()
    model.config.use_cache = True
    model.set_adapter_mode(args.adapter_mode)
    if rank == 0:
        print(f"[eval] adapter_mode={args.adapter_mode}")

    records = [json.loads(l) for l in open(os.path.join(args.qa_root, f"{args.split}.jsonl"))]
    if args.subset_stride > 1:
        records = records[:: args.subset_stride]
    if args.max_samples:
        records = records[: args.max_samples]
    records = records[rank::world]
    if rank == 0:
        print(f"[eval] total={len(records) * world} per_rank~{len(records)}")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"predictions_rank{rank}.jsonl")
    n_done = 0
    with open(out_path, "x", encoding="utf-8") as fout:
        for start in tqdm(range(0, len(records), args.batch_size), desc="Generate"):
            chunk = records[start : start + args.batch_size]
            if args.mono:
                monos = [_load_mono(os.path.join(args.audio_root, r["audio_path"])) for r in chunk]
                import numpy as np
                wavs = []
                for m in monos:
                    w = np.zeros((m.shape[0], 4), dtype=np.float32)
                    w[:, 0] = m
                    wavs.append(w)
                audios = [(m.copy(), SAMPLE_RATE) for m in monos]
            else:
                wavs = [_load_foa(os.path.join(args.audio_root, r["audio_path"])) for r in chunk]
                audios = [(w[:, 0].copy(), SAMPLE_RATE) for w in wavs]
            texts = [
                _PROMPT_TEMPLATE.format(prompt=(r.get("prompt") or r["question"]).rstrip())
                for r in chunk
            ]
            inputs = proc(text=texts, audios=audios, spatial_audio=wavs)
            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            if args.mono and args.mono_fill == "null":
                inputs["has_spatial"] = torch.zeros(
                    len(chunk), dtype=torch.bool, device=device
                )
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    eos_token_id=EOS_IDS,
                )
            plen = inputs["input_ids"].shape[1]
            for i, r in enumerate(chunk):
                gen_text = proc.decode(out[i][plen:], skip_special_tokens=True).strip()
                row = dict(r)
                row["prediction"] = gen_text
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_done += len(chunk)
            if rank == 0 and (n_done % (args.batch_size * 20) < args.batch_size):
                print(f"[eval] rank0 {n_done}/{len(records)}", flush=True)

    if rank == 0:
        print(f"[eval] rank files written; merge after all ranks finish.")
    else:
        print(f"[eval] rank{rank} wrote {out_path} ({n_done})", flush=True)


if __name__ == "__main__":
    main()
