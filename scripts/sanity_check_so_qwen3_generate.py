#!/usr/bin/env python
"""End-to-end sanity check that generate() actually delivers spatial_audio
into Qwen3OmniMoeSpatialThinker.forward (SO-30B / Qwen3-Omni-MoE path).

Why: prior to the ``prepare_inputs_for_generation`` override, generate()
silently dropped spatial_audio at inference time even though training forward
saw it. This script:
  1. Builds the model + processor exactly like the bench script does.
  2. Loads ONE FOA test sample.
  3. Wraps the thinker's forward to log whether spatial_audio is None.
  4. Calls generate(...) for max_new_tokens=4.
  5. Asserts the FIRST forward call (prefill) received spatial_audio != None.

If the assertion fails, the override is broken. If it passes, the generate
path is delivering the spatial signal.

Paths are taken from environment variables / CLI flags so the script carries
no hard-coded absolute paths:

    QWEN3_TRANSFORMERS_FORK   transformers source tree containing qwen3_omni_moe
    SO_MODEL_ID               Qwen3-Omni-30B-A3B base model dir
    SO_ENCODER_CHECKPOINT     SO-Encoder (Spatial-BEATs) checkpoint .pt
    SO_ENCODER_REPO           BEATs repo (optional; vendored by default)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

# Repo root on sys.path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Qwen3 fork on sys.path BEFORE importing our model classes.
_FORK = os.environ.get("QWEN3_OMNI_FORK", os.environ.get("QWEN3_TRANSFORMERS_FORK", ""))
if _FORK and os.path.isdir(_FORK) and _FORK not in sys.path:
    sys.path.insert(0, _FORK)

# Mirror the train_so_qa_qwen3.py monkey-patch: torchaudio kaldi caches a
# module-level EPSILON tensor that hits a meta-tensor copy error under
# accelerate's low_cpu_mem_usage init. Reconstruct fresh each call.
import torchaudio.compliance.kaldi as _kaldi  # noqa: E402


def _safe_get_epsilon(device, dtype):
    return torch.tensor(torch.finfo(torch.float).eps, device=device, dtype=dtype)


_kaldi._get_epsilon = _safe_get_epsilon


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="best_trainable.pt to load")
    ap.add_argument(
        "--model-id",
        default=os.environ.get("SO_MODEL_ID", ""),
        help="Qwen3-Omni-30B-A3B base model dir (or set SO_MODEL_ID).",
    )
    ap.add_argument(
        "--beats-checkpoint",
        default=os.environ.get("SO_ENCODER_CHECKPOINT", ""),
        help="SO-Encoder checkpoint .pt (or set SO_ENCODER_CHECKPOINT).",
    )
    ap.add_argument(
        "--beats-repo",
        default=os.environ.get("SO_ENCODER_REPO", ""),
        help="BEATs repo path (vendored by default).",
    )
    ap.add_argument(
        "--qa-jsonl",
        default=os.environ.get("SO_QA_JSONL", ""),
        help="QA jsonl with at least one FOA test sample (or set SO_QA_JSONL).",
    )
    ap.add_argument("--max-new-tokens", type=int, default=4)
    args = ap.parse_args()

    if not args.model_id:
        print("[sanity] --model-id (or SO_MODEL_ID) is required", file=sys.stderr)
        return 2
    if not args.beats_checkpoint:
        print("[sanity] --beats-checkpoint (or SO_ENCODER_CHECKPOINT) is required", file=sys.stderr)
        return 2
    if not args.qa_jsonl:
        print("[sanity] --qa-jsonl (or SO_QA_JSONL) is required", file=sys.stderr)
        return 2

    from transformers import AutoTokenizer, AutoFeatureExtractor

    from spatial_omni.model.configuration_qwen3_omni import (
        Qwen3OmniMoeSpatialThinkerConfig,
    )
    from spatial_omni.model.modeling_so_thinker_qwen3 import (
        Qwen3OmniMoeSpatialForConditionalGeneration,
    )
    from spatial_omni.model.processing_so_qwen3 import (
        Qwen3OmniMoeSpatialProcessor,
    )

    # ---- build processor + model (mirrors train_so_qa_qwen3._build_*)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_id)
    processor = Qwen3OmniMoeSpatialProcessor(
        feature_extractor=feature_extractor, tokenizer=tokenizer
    )

    raw = json.load(open(os.path.join(args.model_id, "config.json")))
    thinker_kwargs = raw.get("thinker_config", raw)
    cfg = Qwen3OmniMoeSpatialThinkerConfig(**thinker_kwargs)
    cfg.spatial_encoder_type = "so_backbone"
    cfg.so_backbone_checkpoint_path = os.path.abspath(args.beats_checkpoint)
    if args.beats_repo:
        cfg.so_backbone_repo_path = os.path.abspath(args.beats_repo)
    cfg.so_encoder_dim = 768
    cfg.so_projector_hidden_dim = 768
    cfg.so_projector_type = "pixel_shuffle"
    cfg.so_projector_shuffle_factor = 4
    cfg.so_encoder_token_rate = 10.0
    cfg.so_backbone_target_token_rate = 2.5
    cfg.so_backbone_freeze_backbone = False
    cfg.so_backbone_max_audio_seconds = 20.0
    if hasattr(cfg.text_config, "router_aux_loss_coef"):
        cfg.text_config.router_aux_loss_coef = 0.0
        cfg.text_config.output_router_logits = False
    cfg.loss_type = "ForCausalLMLoss"
    cfg.text_config.loss_type = "ForCausalLMLoss"

    print("[sanity] from_pretrained ...")
    model = Qwen3OmniMoeSpatialForConditionalGeneration.from_pretrained(
        args.model_id,
        config=cfg,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
    )
    processor.sync_spatial_tokenizer_with_model(model)

    enc = model.so_encoder
    if enc is not None:
        enc._build_model()
    model.to("cuda:0")

    # Wrap with PEFT LoRA the SAME WAY the trainer does, otherwise the saved
    # ``trainable_state_dict`` keys (``base_model.model.model.layers.X.self_attn.q_proj.lora_A.weight``)
    # do not exist on the raw model and load_state_dict drops every tensor.
    from peft import LoraConfig, TaskType, get_peft_model

    lora_target_prefixes = ["model.layers"]
    lora_target_suffixes = {"q_proj", "k_proj", "v_proj", "o_proj"}
    target_modules = sorted(
        {
            name
            for name, _ in model.named_modules()
            if any(name.startswith(p) for p in lora_target_prefixes)
            and name.rsplit(".", 1)[-1] in lora_target_suffixes
        }
    )
    if not target_modules:
        print("[sanity] FAIL: no LoRA target modules resolved", file=sys.stderr)
        return 1
    print(f"[sanity] PEFT wrap: {len(target_modules)} target modules under {lora_target_prefixes}")
    lc = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
    model = get_peft_model(model, lc)

    # Load LoRA / projector / SO-Encoder weights from ckpt — keys live in
    # `trainable_state_dict` and use the PEFT-wrapped namespace.
    print(f"[sanity] loading trainable weights from {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ckpt.get("trainable_state_dict") or ckpt.get("state_dict") or ckpt.get("model_state_dict") or ckpt
    from spatial_omni.utils.ckpt_compat import remap_legacy_state_dict

    sd = remap_legacy_state_dict(sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(
        f"[sanity] sd has {len(sd)} keys; "
        f"after load -> unexpected={len(unexpected)}  missing_total={len(missing)}"
    )
    if unexpected:
        print(f"[sanity]   first unexpected: {unexpected[:3]}")
    # Verify the projector + a sample LoRA tensor actually carry data
    with torch.no_grad():
        sample_lora_key = next(
            (k for k in sd if "lora_B" in k and "q_proj" in k), None
        )
        sample_proj_key = next(
            (k for k in sd if "so_projector.fc1.weight" in k), None
        )
        if sample_lora_key is not None:
            print(f"[sanity]   sd['{sample_lora_key}'].abs().mean()={sd[sample_lora_key].float().abs().mean():.6f}")
        if sample_proj_key is not None:
            print(f"[sanity]   sd['{sample_proj_key}'].abs().mean()={sd[sample_proj_key].float().abs().mean():.6f}")

    model.eval()

    # ---- pick one FOA sample
    rec = None
    with open(args.qa_jsonl) as fh:
        for line in fh:
            r = json.loads(line)
            audio = r.get("audio_path") or r.get("scene_audio_path")
            if not audio or not os.path.isfile(audio):
                continue
            rec = r
            break
    if rec is None:
        print("[sanity] no readable FOA sample found in qa-jsonl", file=sys.stderr)
        return 1

    import soundfile as sf

    audio_path = rec.get("audio_path") or rec.get("scene_audio_path")
    waveform, sr = sf.read(audio_path)
    if waveform.ndim == 1:
        print(f"[sanity] sample is mono, but spatial requires FOA: {audio_path}")
        return 1
    if waveform.shape[0] != 4 and waveform.shape[1] == 4:
        waveform = waveform.T
    print(f"[sanity] audio: {audio_path}  shape={waveform.shape}  sr={sr}")
    if sr != 16000:
        import librosa  # noqa

        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)

    prompt = rec.get("prompt") or rec["question"]
    if "<|spatial|>" not in prompt:
        prompt = f"<|AUDIO|><|spatial|>\n{prompt}"

    pi = processor(
        text=prompt,
        audio=[waveform],
        return_tensors="pt",
    )
    pi = {k: (v.to("cuda:0") if isinstance(v, torch.Tensor) else v) for k, v in pi.items()}

    # ---- spy on forward
    # PEFT wraps the model: model is a PeftModel, model.base_model is
    # LoraModel, and the underlying spatial thinker is at
    # model.base_model.model. We must hook the THINKER's forward, because
    # generate() calls it directly via PeftModel.forward delegation, and our
    # spatial_audio injection lives there. Hooking at PEFT level would miss
    # the spatial_audio kwarg entirely (PEFT.forward strips peft-specific
    # args before calling the wrapped model).
    saw_spatial_in_forwards = []
    inner = model.base_model.model  # the actual Qwen3OmniMoeSpatialThinker
    print(f"[sanity] hooking forward on inner class: {type(inner).__name__}")
    orig_forward = inner.forward

    def spy_forward(*a, **kw):
        sa = kw.get("spatial_audio", None)
        st = kw.get("spatial_tokens", None)
        saw_spatial_in_forwards.append(
            {
                "iter": len(saw_spatial_in_forwards),
                "spatial_audio_is_none": sa is None,
                "spatial_audio_shape": tuple(sa.shape) if sa is not None else None,
                "spatial_tokens_is_none": st is None,
                "input_ids_seq_len": (
                    int(kw.get("input_ids").shape[1])
                    if kw.get("input_ids") is not None
                    else None
                ),
            }
        )
        return orig_forward(*a, **kw)

    inner.forward = spy_forward  # type: ignore[assignment]

    print("[sanity] calling generate(...) ...")
    # NOTE: Qwen3 has no talker — generate() does not accept return_audio.
    # The training script's monkey-patch (_qwen3_generate_with_eos_and_drop)
    # silently strips return_audio + injects eos_token_id; we replicate the
    # eos injection here directly so the generate path matches bench behavior.
    with torch.no_grad():
        out = model.generate(
            **pi,
            max_new_tokens=args.max_new_tokens,
            num_beams=1,
            do_sample=False,
            eos_token_id=[151645],  # <|im_end|>
            pad_token_id=151643,    # <|endoftext|>
        )
    text = processor.tokenizer.decode(out[0, pi["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"[sanity] generated text: {text!r}")

    print("[sanity] forward call audit:")
    for r in saw_spatial_in_forwards[:6]:
        print(f"   {r}")
    if len(saw_spatial_in_forwards) > 6:
        print(f"   ... ({len(saw_spatial_in_forwards)} forwards total)")

    if not saw_spatial_in_forwards:
        print("[sanity] FAIL: forward was never called", file=sys.stderr)
        return 1
    first = saw_spatial_in_forwards[0]
    if first["spatial_audio_is_none"]:
        print(
            "[sanity] FAIL: prefill forward got spatial_audio=None — fix did not take effect",
            file=sys.stderr,
        )
        return 1
    print(
        f"[sanity] PASS: prefill forward received spatial_audio shape="
        f"{first['spatial_audio_shape']}"
    )
    # Decode-step forwards should all have spatial_audio cleared:
    for r in saw_spatial_in_forwards[1:]:
        if not r["spatial_audio_is_none"]:
            print(
                f"[sanity] WARNING: decode iter {r['iter']} still carries spatial_audio "
                f"(shape {r['spatial_audio_shape']}); this would re-run the encoder every step."
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
