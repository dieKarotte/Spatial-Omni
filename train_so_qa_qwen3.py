#!/usr/bin/env python
"""SO-30B training entrypoint using the shared Spatial-Omni trainer.

The model and processor factories adapt the shared data, optimization, and
checkpoint code to the SO-30B Thinker.
"""

from __future__ import annotations

import json
import os
import sys
import time

# Optionally load Transformers from a caller-supplied source checkout.
_FORK = os.environ.get("QWEN3_OMNI_FORK", os.environ.get("QWEN3_TRANSFORMERS_FORK", ""))
if _FORK and os.path.isdir(_FORK) and _FORK not in sys.path:
    sys.path.insert(0, _FORK)

# Make sure the repo root is importable.
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import train_so_qa as _trainer  # noqa: E402

# ---------------------------------------------------------------------------
# Construct Kaldi epsilon on the requested device instead of reusing
# a cached tensor across model-dispatch contexts.
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
    register_qwen3_spatial_auto_classes,
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
    chat_template = getattr(tokenizer, "chat_template", None)
    chat_template_path = os.path.join(model_id, "chat_template.json")
    if not chat_template and os.path.isfile(chat_template_path):
        with open(chat_template_path, encoding="utf-8") as handle:
            chat_template = json.load(handle).get("chat_template")
    return Qwen3OmniMoeSpatialProcessor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        chat_template=chat_template,
    )


# ---------------------------------------------------------------------------
# build_model (Qwen3): instantiate the spatial thinker config from raw
# config.json (avoids the Qwen3OmniMoeConfig top-level talker bug), then
# Qwen3OmniMoeSpatialForConditionalGeneration.from_pretrained.
# ---------------------------------------------------------------------------
def _validate_qwen3_base_loading_info(loading_info) -> None:
    """Fail when any pretrained Thinker tensor was not loaded exactly."""

    allowed_missing_prefixes = ("so_encoder.", "so_projector.", "spatial_null")
    allowed_unexpected_prefixes = ("talker.", "code2wav.")
    missing = [
        key
        for key in loading_info.get("missing_keys", [])
        if not key.startswith(allowed_missing_prefixes)
    ]
    unexpected = [
        key
        for key in loading_info.get("unexpected_keys", [])
        if not key.startswith(allowed_unexpected_prefixes)
    ]
    # Transformers 5 returns sets for some loading-info fields, whereas older
    # releases returned lists. Normalize before slicing for diagnostics.
    mismatched = list(loading_info.get("mismatched_keys", []) or [])
    errors = list(loading_info.get("error_msgs", []) or [])
    if missing or unexpected or mismatched or errors:
        raise RuntimeError(
            "Qwen3 base checkpoint did not load cleanly. "
            f"missing={missing[:8]} unexpected={unexpected[:8]} "
            f"mismatched={mismatched[:8]} errors={errors[:3]}"
        )
    _trainer.rank0_print(
        "[build_model_qwen3] strict base-load check passed: "
        f"missing_spatial={len(loading_info.get('missing_keys', []))} "
        "base_missing=0 base_unexpected=0"
    )


def _build_model_qwen3(args, processor):
    if args.so_repo not in sys.path:
        sys.path.insert(0, args.so_repo)
    register_qwen3_spatial_auto_classes()

    cfg_path = os.path.join(args.model_id, "config.json")
    raw = json.load(open(cfg_path))
    thinker_kwargs = raw.get("thinker_config", raw)
    cfg = Qwen3OmniMoeSpatialThinkerConfig(**thinker_kwargs)

    # Spatial-BEATs configuration
    cfg.spatial_encoder_type = "so_backbone"
    cfg.so_backbone_checkpoint_path = os.path.abspath(args.beats_checkpoint)
    cfg.so_backbone_repo_path = os.path.abspath(args.beats_repo or args.so_repo)
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

    # Mono replay (mixed spatial+mono training): allocate the learned
    # spatial_null token bank sized to cover a max-length clip at the
    # LLM-side token rate (20s x 2.5Hz = 50 tokens with default rates).
    cfg.enable_spatial_replay = bool(getattr(args, "mixed_spatial_replay", False))
    if cfg.enable_spatial_replay:
        cfg.spatial_null_num_tokens = int(
            round(float(_trainer.MAX_AUDIO_SECONDS) * effective_rate)
        )
        cfg.spatial_null_alignment_weight = float(
            getattr(args, "null_alignment_weight", 0.05)
        )

    # Stage1/2 freeze BEATs; stage3 unfreezes
    cfg.so_backbone_freeze_backbone = args.train_mode in {"projector_only", "encoder_lora"}
    cfg.so_backbone_max_audio_seconds = float(_trainer.MAX_AUDIO_SECONDS)

    # Router adaptation is cheap (~12.6M parameters for the 30B thinker) and
    # useful after projector alignment. Keep it off unless explicitly enabled.
    if hasattr(cfg.text_config, "router_aux_loss_coef"):
        train_router = bool(getattr(args, "train_moe_router", False))
        cfg.text_config.router_aux_loss_coef = (
            float(getattr(args, "moe_router_aux_loss_coef", 1e-3))
            if train_router
            else 0.0
        )
        cfg.text_config.output_router_logits = train_router

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
        "output_loading_info": True,
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
    model, loading_info = Qwen3OmniMoeSpatialForConditionalGeneration.from_pretrained(
        args.model_id, **from_pretrained_kwargs
    )
    _validate_qwen3_base_loading_info(loading_info)
    _trainer.rank0_print(f"[build_model_qwen3] attn_implementation={attn_impl}")

    # `spatial_null` is introduced by this subclass and never appears in the
    # base checkpoint, so from_pretrained can leave it on meta / filled with
    # non-finite garbage. Re-init in that case (no-op for healthy resumes).
    if model.reinit_spatial_null_if_needed():
        sn = getattr(model, "spatial_null", None)
        if sn is not None:
            _trainer.rank0_print(
                f"[build_model_qwen3] Re-initialized spatial_null (was meta/NaN/Inf): "
                f"shape={tuple(sn.shape)} dtype={sn.dtype} "
                f"|max|={float(sn.abs().max()):.3e} std={float(sn.float().std()):.3e}"
            )

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

    # DDP mode: move entire model to local GPU. ``from_pretrained`` loads to
    # CPU when ``low_cpu_mem_usage=True`` is set without a device_map; DDP
    # later wraps with device_ids=[local_rank] which requires the params to
    # already live on that device. (The base trainer's build_model does this
    # via ``model.to(args.device)``; the Qwen3 monkey-patched replacement
    # must mirror that branch.)
    if device_map is None:
        model.to(args.device)
        _trainer.rank0_print(
            f"[build_model_qwen3] DDP mode: moved model to {args.device}"
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
        if getattr(args, "spatial_encoder_type", "so_backbone") != "so_backbone":
            raise ValueError("SO-30B currently supports only --spatial-encoder-type=so_backbone.")
        return args

    _trainer.parse_args = _patched_parse_args
    return _trainer.main()


if __name__ == "__main__":
    main()
