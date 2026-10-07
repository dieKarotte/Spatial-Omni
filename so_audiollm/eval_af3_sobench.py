#!/usr/bin/env python
"""SO-Bench / MMAU generation eval for SO-AF3 (batched, left-padded).

Prompt (qwen2 chat template, AF3 convention):
    <|im_start|>user\n<sound>*N_native <|spatial|>*T_sp \n{q}<|im_end|>\n
    <|im_start|>assistant\n
Batched prefill injects native sound-tower embeddings at <sound> rows and SO
spatial embeddings at <|spatial|> rows, then greedy-decodes with KV cache,
tracking per-row EOS (<|im_end|>). Left padding keeps row ends aligned.

"""

from __future__ import annotations

import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402
from tqdm import tqdm
import torch  # noqa: E402

from so_audiollm.af3 import (  # noqa: E402
    SOUND_TOKEN, af3_native_token_count, build_af3,
)
from so_audiollm.common import (  # noqa: E402
    MAX_AUDIO_SAMPLES, SPATIAL_TOKEN, load_foa, load_mono, pack_spatial_batch,
    spatial_token_count,
)
from transformers import WhisperFeatureExtractor  # noqa: E402

IM_END = "<|im_end|>"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--model-dir", required=True)
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
                   help="mono QA eval (MMAU): mono audio, spatial placeholders "
                        "null-filled when the ckpt has a trained spatial_null")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--no-spatial", action="store_true",
                   help="raw base-model eval: no <|spatial|> tokens at all")
    return p.parse_args()


@torch.no_grad()
def generate_batch(model, tokenizer, mel_extractor, records, audio_root,
                   max_new, mono_mode, no_spatial=False):
    device = next(model.parameters()).device
    B = len(records)
    wavs, monos = [], []
    ids_list = []
    for r in records:
        if mono_mode:
            mono = load_mono(os.path.join(audio_root, r["audio_path"]))
            wav = np.zeros((mono.shape[0], 4), dtype=np.float32)
            wav[:, 0] = mono
        else:
            wav = load_foa(os.path.join(audio_root, r["audio_path"]))
            mono = wav[:, 0].copy()
        wavs.append(wav)
        monos.append(mono)
        n_native = af3_native_token_count(mono.shape[0])
        t_sp = spatial_token_count(min(wav.shape[0], MAX_AUDIO_SAMPLES))
        q = str(r.get("prompt") or r["question"]).rstrip()
        text = ("<|im_start|>user\n" + SOUND_TOKEN * n_native
                + ("" if no_spatial else SPATIAL_TOKEN * t_sp)
                + f"\n{q}{IM_END}\n<|im_start|>assistant\n")
        ids_list.append(tokenizer(text, add_special_tokens=False).input_ids)

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    max_len = max(len(x) for x in ids_list)
    ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    attn = torch.zeros(B, max_len, dtype=torch.long)
    for i, row in enumerate(ids_list):  # left pad so all rows end at right edge
        ids[i, max_len - len(row):] = torch.tensor(row)
        attn[i, max_len - len(row):] = 1
    ids, attn = ids.to(device), attn.to(device)

    mel = mel_extractor(monos, sampling_rate=16000, return_tensors="pt",
                        padding="max_length").input_features.to(device)
    native_lengths = torch.tensor(
        [af3_native_token_count(m.shape[0]) for m in monos], device=device)
    spa, spa_mask, spa_lens, _ = pack_spatial_batch(wavs)
    spa, spa_mask, spa_lens = spa.to(device), spa_mask.to(device), spa_lens.to(device)

    embeds = model.llm.get_input_embeddings()(ids)
    native_flat = model.encode_native(mel, native_lengths)
    pos = torch.nonzero(ids == model.sound_token_id, as_tuple=True)
    if native_flat.shape[0] != pos[0].shape[0]:
        raise RuntimeError("native <sound> injection mismatch")
    embeds = embeds.index_put(pos, native_flat.to(embeds.dtype))

    if not no_spatial:
        hs = None
        if mono_mode and model.spatial.spatial_null is not None:
            hs = torch.zeros(B, dtype=torch.bool, device=device)
        flat, _, _ = model.spatial(spa, spa_mask, spa_lens, ids,
                                   model.spatial_token_id, has_spatial=hs)
        pos = torch.nonzero(ids == model.spatial_token_id, as_tuple=True)
        if flat.shape[0] != pos[0].shape[0]:
            raise RuntimeError("spatial injection mismatch")
        embeds = embeds.index_put(pos, flat.to(embeds.dtype))

    out = model.llm(inputs_embeds=embeds, attention_mask=attn, use_cache=True)
    past = out.past_key_values
    next_logits = out.logits[:, -1]

    im_end = tokenizer.convert_tokens_to_ids(IM_END)
    generated = [[] for _ in range(B)]
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    embed_layer = model.llm.get_input_embeddings()
    for _ in range(max_new):
        nxt = next_logits.argmax(-1)
        for i in range(B):
            if not finished[i]:
                if int(nxt[i]) == im_end:
                    finished[i] = True
                else:
                    generated[i].append(int(nxt[i]))
        if bool(finished.all()):
            break
        step = embed_layer(nxt.view(B, 1))
        attn = torch.cat([attn, torch.ones(B, 1, dtype=attn.dtype, device=device)], dim=1)
        out = model.llm(inputs_embeds=step, past_key_values=past,
                        attention_mask=attn, use_cache=True)
        past = out.past_key_values
        next_logits = out.logits[:, -1]
    return [tokenizer.decode(g).strip() for g in generated]


def main():
    args = parse_args()
    output_name = f"predictions_shard{args.shard}.jsonl"
    if os.path.exists(os.path.join(args.output_dir, output_name)):
        raise FileExistsError(f"Predictions already exist in {args.output_dir}; choose a new --output-dir.")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    state = (torch.load(args.checkpoint, map_location="cpu", weights_only=True)
             if args.checkpoint else None)
    has_null = state is not None and "spatial.spatial_null" in state
    model, tokenizer = build_af3(args.model_dir, args.beats_checkpoint,
                                 args.beats_repo, enable_replay=has_null)
    if state is not None:
        import argparse as _ap
        import train_so_af3 as T
        margs = _ap.Namespace(train_mode="beats_lora", lora_r=args.lora_r, lora_alpha=args.lora_alpha,
                              lora_dropout=args.lora_dropout)
        T.configure_trainable(model, margs)
        missing, unexpected = model.load_state_dict(state, strict=False)
        required = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        missing_required = sorted(required.intersection(missing))
        if unexpected or missing_required:
            raise ValueError(f"Incompatible checkpoint: unexpected={unexpected[:10]}, missing trainable keys={missing_required[:10]}")
        print(f"[eval] resume missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    model.to(device).eval()

    mel_extractor = WhisperFeatureExtractor(
        feature_size=128, sampling_rate=16000, hop_length=160,
        chunk_length=30, n_fft=400)

    records = [json.loads(l) for l in open(os.path.join(args.qa_root, f"{args.split}.jsonl"))]
    if args.max_samples:
        records = records[: args.max_samples]
    records = records[args.shard :: args.num_shards]
    print(f"[eval] {len(records)} records (shard {args.shard}/{args.num_shards})",
          flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"predictions_shard{args.shard}.jsonl")
    done = 0
    with open(out_path, "x", encoding="utf-8") as fout:
        for start in tqdm(range(0, len(records), args.batch_size), desc="Generate"):
            chunk = records[start : start + args.batch_size]
            preds = generate_batch(model, tokenizer, mel_extractor, chunk,
                                   args.audio_root, args.max_new_tokens, args.mono,
                                   no_spatial=args.no_spatial)
            for r, pred in zip(chunk, preds):
                row = dict(r)
                row["prediction"] = pred
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            done += len(chunk)
            if done % (args.batch_size * 10) < args.batch_size:
                print(f"[eval] {done}/{len(records)}", flush=True)
    print(f"[eval] done -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
