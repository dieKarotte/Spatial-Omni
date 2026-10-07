"""Portable release settings and validation of trained parameter coverage."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def load_release_settings(checkpoint_path, *, model_id=None, encoder_checkpoint=None) -> dict[str, Any]:
    """Read adjacent settings, or parent settings for the run/checkpoints layout."""
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    candidates = [checkpoint_path.parent / "train_args.json"]
    if checkpoint_path.parent.name == "checkpoints":
        candidates.append(checkpoint_path.parent.parent / "train_args.json")
    config_path = next((p for p in candidates if p.is_file()), None)
    if config_path is None:
        raise FileNotFoundError(f"train_args.json not found beside {checkpoint_path}")
    with config_path.open(encoding="utf-8") as handle:
        settings = json.load(handle)
    if not isinstance(settings, dict):
        raise ValueError(f"Expected an object in {config_path}")
    if model_id:
        settings["model_id"] = str(model_id)
    encoder = encoder_checkpoint or settings.get("beats_checkpoint")
    if not encoder:
        raise ValueError("Set --beats-checkpoint to the downloaded SO-Encoder.pt")
    encoder = Path(encoder).expanduser()
    if not encoder.is_absolute():
        encoder = (Path.cwd() if encoder_checkpoint else config_path.parent) / encoder
    encoder = encoder.resolve()
    if not encoder.is_file():
        raise FileNotFoundError(f"SO-Encoder checkpoint not found: {encoder}")
    settings["beats_checkpoint"] = str(encoder)
    settings["so_repo"] = str(Path(__file__).resolve().parents[2])
    settings["beats_repo"] = settings.get("beats_repo") or ""
    value = settings.get("beats_repo")
    if value:
        path = Path(value).expanduser()
        settings["beats_repo"] = str((config_path.parent / path).resolve() if not path.is_absolute() else path)
    return settings


def load_release_state(model, state_dict: Mapping[str, Any]):
    """Allow omitted frozen base weights; reject missing trained or MIX parameters."""
    parameters = dict(model.named_parameters())
    expected = {name for name, parameter in parameters.items()
                if parameter.requires_grad or "spatial_null" in name}
    available = set(model.state_dict())
    unexpected = sorted(set(state_dict) - available)
    missing = sorted(expected - set(state_dict))
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint does not match the selected model/configuration. "
            f"Missing trained parameters ({len(missing)}): {missing[:12]}; "
            f"unexpected parameters ({len(unexpected)}): {unexpected[:12]}"
        )
    return model.load_state_dict(state_dict, strict=False)


def apply_checkpoint_architecture(args, argv):
    """Use saved model structure for resume unless the caller overrides an option."""
    checkpoint = getattr(args, "resume_checkpoint_path", None)
    if not checkpoint and getattr(args, "resume_tag", None):
        checkpoint = Path(args.output_dir) / "checkpoints" / (args.resume_tag + "_trainable.pt")
    if not checkpoint:
        return
    checkpoint = Path(checkpoint).expanduser().resolve()
    candidates = [checkpoint.parent / "train_args.json"]
    if checkpoint.parent.name == "checkpoints":
        candidates.append(checkpoint.parent.parent / "train_args.json")
    config_path = next((p for p in candidates if p.is_file()), None)
    if config_path is None:
        return
    with config_path.open(encoding="utf-8") as handle:
        saved = json.load(handle)
    explicit = {str(option).split("=", 1)[0] for option in argv if str(option).startswith("--")}
    structural_fields = (
        "lora_r", "lora_alpha", "lora_dropout", "lora_target_modules", "lora_target_prefixes",
        "projector_type", "projector_shuffle_factor", "encoder_token_rate",
        "mixed_spatial_replay",
    )
    for key in structural_fields:
        option = "--" + key.replace("_", "-")
        if option not in explicit and key in saved and saved[key] is not None:
            setattr(args, key, saved[key])
    mode_flags = {"--projector-only", "--encoder-lora", "--beats-lora", "--train-all"}
    if not explicit.intersection(mode_flags) and saved.get("train_mode") is not None:
        if saved["train_mode"] not in {"projector_only", "encoder_lora", "beats_lora", "all"}:
            raise ValueError(f"Unsupported saved train_mode: {saved['train_mode']}")
        args.train_mode = saved["train_mode"]


def load_curriculum_state(model, state_dict):
    """Reject incomplete components while allowing whole newly enabled stage components."""
    parameters = dict(model.named_parameters())
    required = {name for name, parameter in parameters.items() if parameter.requires_grad}
    available = set(model.state_dict())
    newly_enabled = set()
    for marker in ("so_encoder.", "lora_", "spatial_null"):
        if not any(marker in name for name in state_dict):
            newly_enabled.update(name for name in required if marker in name)
    missing = sorted(required - set(state_dict) - newly_enabled)
    unexpected = sorted(set(state_dict) - available)
    if missing or unexpected:
        raise RuntimeError(f"Incompatible training checkpoint: missing={missing[:12]}, unexpected={unexpected[:12]}")
    return model.load_state_dict(state_dict, strict=False)
