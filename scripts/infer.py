#!/usr/bin/env python3
"""Answer an audio question with SO-7B FOA or SO-7B-MIX FOA/mono input."""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--audio", required=True, help="16 kHz FOA audio (W in channel 0), or single-channel audio with --mono.")
    parser.add_argument("--question", required=True)
    parser.add_argument("--mono", action="store_true", help="Use MIX learned null tokens for single-channel 16 kHz audio.")
    parser.add_argument("--model-id", default=os.environ.get("SO_BASE_MODEL"))
    parser.add_argument("--beats-checkpoint", default=os.environ.get("SO_ENCODER_CKPT"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--dtype", default="bfloat16", choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--attn-impl", default="sdpa", choices=("sdpa", "eager", "flash_attention_2"))
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")
    import soundfile as sf
    import torch
    from transformers import set_seed
    from scripts.batch_bench_so_qa import (
        SpatialBeatsEvalCollator, get_model_device,
        instantiate_model_for_checkpoint, to_generation_inputs,
    )

    audio = Path(args.audio).expanduser().resolve()
    info = sf.info(str(audio))
    expected_channels = 1 if args.mono else 4
    if info.channels != expected_channels or info.samplerate != 16000 or info.frames == 0:
        raise ValueError(f"Expected nonempty {expected_channels}-channel 16 kHz audio; got {info.channels} channels, "
                         f"{info.samplerate} Hz, {info.frames} frames: {audio}")
    if args.mono:
        from spatial_omni.utils.release import load_release_settings
        settings = load_release_settings(args.checkpoint, model_id=args.model_id,
                                         encoder_checkpoint=args.beats_checkpoint)
        if not settings.get("mixed_spatial_replay", False):
            raise ValueError("--mono requires an SO-7B-MIX checkpoint with learned null tokens")
    logging.info("Audio: %s; %.2f s, %d Hz, %d channels", audio, info.duration,
                 info.samplerate, info.channels)
    if info.duration > 20:
        logging.warning("The released checkpoints use the first 20 seconds of the recording.")
    set_seed(args.seed)
    model, processor, _, _, result = instantiate_model_for_checkpoint(args, args.checkpoint)
    logging.info("Checkpoint loaded; only %d frozen base entries omitted.", len(result.missing_keys))
    record = {"audio_path": str(audio), "question": args.question,
              "prompt": args.question, "answer": ""}
    if args.mono:
        from train_so_qa import SpatialBeatsQACollator
        record["_replay_has_spatial"] = False
        batch = SpatialBeatsQACollator(processor=processor, enable_mono_replay=True,
                                      include_generation_inputs=True)([record])
    else:
        batch = SpatialBeatsEvalCollator(processor=processor)([record])
    inputs = to_generation_inputs(batch, get_model_device(model))
    with torch.inference_mode():
        generated = model.generate(**inputs, return_audio=False, do_sample=False,
                                   num_beams=1, max_new_tokens=args.max_new_tokens)
    prefix_length = inputs["input_ids"].shape[1]
    print(processor.tokenizer.decode(generated[0, prefix_length:].cpu(),
                                    skip_special_tokens=True).strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
