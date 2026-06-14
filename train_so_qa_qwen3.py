#!/usr/bin/env python
"""Qwen3-Omni-MoE Spatial-BEATs training entrypoint.

Thin wrapper around ``train_so_qa.py``: monkey-patches
``build_processor`` and ``build_model`` to use the Qwen3 spatial classes,
then defers to the original ``main`` for everything else (data loading,
LoRA wiring, distributed, checkpointing, validation).

Why a wrapper instead of forking the 1500-line script:
  - We want the Qwen2.5 path to remain bit-identical (same ckpts, same shell).
  - The only Qwen-version-specific bits in train_so_qa.py are the
    two ``build_*`` functions; everything else operates on the
    ``model.thinker.so_backbone_*`` API which we preserve via the
    Qwen3OmniMoeSpatialForConditionalGeneration wrapper.
"""

from __future__ import annotations

import json
import os
import sys
import time

# Inject the local transformers fork so qwen3_omni_moe is importable.
_FORK = os.environ.get(
    "QWEN3_OMNI_FORK",
    "${QWEN3_TRANSFORMERS_FORK}",
)
if os.path.isdir(_FORK) and _FORK not in sys.path:
    sys.path.insert(0, _FORK)

# Make sure the repo root is importable.
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import train_so_qa as _trainer  # noqa: E402

# ---------------------------------------------------------------------------
# Workaround for a torchaudio + accelerate interaction in torch 2.10:
# torchaudio.compliance.kaldi caches a module-level EPSILON tensor on CPU.
# When the model is loaded via ``from_pretrained(device_map="auto", low_cpu_mem_usage=True)``,
# accelerate's dispatch_model attaches hooks that interact poorly with the
# kaldi global tensor under ``torch.enable_grad()``: ``EPSILON.to(device=cuda)``
# crashes with ``Cannot copy out of meta tensor; no data!`` — even though the
# tensor itself is a real CPU tensor (verified by .device prints right before
# the call). Reconstructing the epsilon scalar from scratch each call avoids
# the bad cached path.
# ---------------------------------------------------------------------------
import torchaudio.compliance.kaldi as _kaldi  # noqa: E402

def _safe_get_epsilon(device, dtype):
    return torch.tensor(torch.finfo(torch.float).eps, device=device, dtype=dtype)

_kaldi._get_epsilon = _safe_get_epsilon

import torch  # noqa: E402
from transformers import AutoFeatureExtractor, AutoTokenizer  # noqa: E402

from spatial_omni.model.configuration_qwen3_omni import (  # noqa: E402
    Qwen3OmniMoeSpatialThinkerConfig,
)
from spatial_omni.model.modeling_so_thinker_qwen3 import (  # noqa: E402
    Qwen3OmniMoeSpatialForConditionalGeneration,
)
from spatial_omni.model.processing_so_qwen3 import (  # noqa: E402
    Qwen3OmniMoeSpatialProcessor,
)


# ---------------------------------------------------------------------------
# build_processor (Qwen3): load tokenizer + WhisperFeatureExtractor separately
# (top-level Qwen3OmniMoeProcessor.from_pretrained crashes on the talker config
# in our fork; we don't need the video processor for audio-only QA.)
# ---------------------------------------------------------------------------
def _build_processor_qwen3(model_id: str, sqr: str):
    if sqr not in sys.path:
        sys.path.insert(0, sqr)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
    return Qwen3OmniMoeSpatialProcessor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
    )


# ---------------------------------------------------------------------------
# build_model (Qwen3): instantiate the spatial thinker config from raw
# config.json (avoids the Qwen3OmniMoeConfig top-level talker bug), then
# Qwen3OmniMoeSpatialForConditionalGeneration.from_pretrained.
# ---------------------------------------------------------------------------
def _build_model_qwen3(args, processor):
    if args.so_repo not in sys.path:
        sys.path.insert(0, args.so_repo)

    cfg_path = os.path.join(args.model_id, "config.json")
    raw = json.load(open(cfg_path))
    thinker_kwargs = raw.get("thinker_config", raw)
    cfg = Qwen3OmniMoeSpatialThinkerConfig(**thinker_kwargs)

    # Spatial-BEATs configuration
    cfg.spatial_encoder_type = "so_backbone"
    cfg.so_backbone_checkpoint_path = os.path.abspath(args.beats_checkpoint)
    cfg.so_backbone_repo_path = os.path.abspath(args.beats_repo)
    cfg.so_encoder_dim = 768
    cfg.so_projector_hidden_dim = 768

    projector_type = getattr(args, "projector_type", "pixel_shuffle")
    shuffle_factor = int(getattr(args, "projector_shuffle_factor", 4))
    encoder_rate = float(getattr(args, "encoder_token_rate", _trainer.DEFAULT_ENCODER_TOKEN_RATE))
    if shuffle_factor < 1:
        raise ValueError("--projector-shuffle-factor must be >= 1")
    if projector_type != "pixel_shuffle":
        shuffle_factor = 1
    effective_rate = encoder_rate / float(shuffle_factor)
    if abs(effective_rate - _trainer.TARGET_TOKEN_RATE) > 1e-6:
        _trainer.rank0_print(
            f"[build_model_qwen3] WARNING: LLM-side spatial rate = "
            f"{encoder_rate}/{shuffle_factor} = {effective_rate} Hz "
            f"(conventional {_trainer.TARGET_TOKEN_RATE} Hz)"
        )
    cfg.so_encoder_token_rate = encoder_rate
    cfg.so_backbone_target_token_rate = effective_rate
    cfg.so_projector_type = projector_type
    cfg.so_projector_shuffle_factor = shuffle_factor

    # Stage1/2 freeze BEATs; stage3 unfreezes
    cfg.so_backbone_freeze_backbone = args.train_mode in {"projector_only", "encoder_lora"}
    cfg.so_backbone_max_audio_seconds = float(_trainer.MAX_AUDIO_SECONDS)

    # Disable router aux loss to keep training signal pure LM loss (MoE
    # load-balancing aux loss can interact badly with frozen experts when only
    # attention LoRA is trainable).
    if hasattr(cfg.text_config, "router_aux_loss_coef"):
        cfg.text_config.router_aux_loss_coef = 0.0
        cfg.text_config.output_router_logits = False

    cfg.loss_type = "ForCausalLMLoss"
    cfg.text_config.loss_type = "ForCausalLMLoss"

    # Resolve attn_impl
    attn_impl = getattr(args, "attn_impl", "auto")
    if attn_impl == "auto":
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            attn_impl = "sdpa"
        _trainer.rank0_print(f"[build_model_qwen3] attn_impl='auto' resolved to '{attn_impl}'")

    from_pretrained_kwargs = {
        "config": cfg,
        "torch_dtype": _trainer.dtype_from_name(args.dtype),
        "low_cpu_mem_usage": True,
    }
    if attn_impl and attn_impl != "auto":
        from_pretrained_kwargs["attn_implementation"] = attn_impl
    device_map = getattr(args, "device_map", None)
    if device_map is not None:
        from_pretrained_kwargs["device_map"] = device_map

    _trainer.rank0_print(
        f"[build_model_qwen3] from_pretrained: model_id={args.model_id} "
        f"dtype={args.dtype} attn={attn_impl} device_map={device_map}"
    )
    model = Qwen3OmniMoeSpatialForConditionalGeneration.from_pretrained(
        args.model_id, **from_pretrained_kwargs
    )
    _trainer.rank0_print(f"[build_model_qwen3] attn_implementation={attn_impl}")

    processor.sync_spatial_tokenizer_with_model(model)
    model.disable_talker()  # no-op on Qwen3 wrapper
    if args.gradient_checkpointing:
        _trainer.enable_gradient_checkpointing(model)
        model.config.use_cache = False

    # Build spatial-beats encoder lazily on CPU, then move to projector device
    enc = getattr(model, "so_encoder", None)
    proj = getattr(model, "so_projector", None)
    if enc is not None:
        _trainer.rank0_print(f"[{time.strftime('%H:%M:%S')}] Building SOBackbone on CPU ...")
        enc._build_model()
        _trainer.rank0_print(f"[{time.strftime('%H:%M:%S')}] SOBackbone built.")
        if device_map is not None:
            # When device_map='auto', accelerate has installed dispatch hooks
            # on every submodule. Our so_encoder & projector were
            # never registered with accelerate's device map (their weights are
            # NOT in the safetensors), so the hooks point them at the meta
            # device — calling forward then moves inputs to meta and crashes.
            # Strip the hooks and pin both modules to a real GPU manually.
            from accelerate.hooks import remove_hook_from_module
            target_dev = torch.device("cuda:0")
            if proj is not None:
                # Try to put projector on the LM head's GPU first (so the
                # masked_scatter into inputs_embeds stays on one device).
                try:
                    target_dev = next(model.lm_head.parameters()).device
                except Exception:
                    pass
            remove_hook_from_module(enc, recurse=True)
            enc.to(target_dev)
            if proj is not None:
                remove_hook_from_module(proj, recurse=True)
                proj.to(target_dev)
            _trainer.rank0_print(
                f"[build_model_qwen3] removed accelerate hooks and pinned "
                f"so_encoder + projector to {target_dev}"
            )

    return model


def main():
    # Patch the trainer module's two Qwen-specific factories.
    _trainer.build_processor = _build_processor_qwen3
    _trainer.build_model = _build_model_qwen3

    # The base trainer's argparse has no --device-map flag (DDP-only design).
    # For 30B Qwen3 on 8x 40GB we usually want HF accelerate sharding, which
    # is keyed off ``args.device_map``. Wrap parse_args() to add the flag.
    _orig_parse_args = _trainer.parse_args

    def _patched_parse_args():
        # Pre-process sys.argv to swallow --device-map before the inner parser
        # rejects it as unknown.
        import sys
        device_map = os.environ.get("DEVICE_MAP", None)
        argv = sys.argv
        out_argv = []
        i = 0
        while i < len(argv):
            tok = argv[i]
            if tok == "--device-map" and i + 1 < len(argv):
                device_map = argv[i + 1]
                i += 2
                continue
            if tok.startswith("--device-map="):
                device_map = tok.split("=", 1)[1]
                i += 1
                continue
            out_argv.append(tok)
            i += 1
        sys.argv = out_argv
        args = _orig_parse_args()
        args.device_map = device_map
        return args

    _trainer.parse_args = _patched_parse_args
    return _trainer.main()


if __name__ == "__main__":
    main()
