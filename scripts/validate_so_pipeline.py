#!/usr/bin/env python3
"""Validate the SELD233 spatial branch with random variable-length FOA audio.

This script is intended for a machine that has `transformers` available. It
builds a `B=3` random FOA batch, runs:

1. spatial processor placeholder expansion
2. online `FOA -> 7ch` feature extraction
3. SELD233 backbone hidden-state extraction
4. `10 Hz -> 2.5 Hz` spatial-token adaptation

It also reports the final multimodal token tensor shape implied by the prompt.
If `--load-qwen-model` is passed, it additionally constructs the actual
`inputs_embeds` tensor after audio + spatial injection.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from spatial_omni.model.configuration import Qwen2_5OmniConfig
from spatial_omni.model.modeling_so_thinker import Qwen2_5OmniSpatialForConditionalGeneration
from spatial_omni.modules.seld_backbone import SeldBackbone
from spatial_omni.modules.seld_feature_bridge import SeldFeatureBridge
from spatial_omni.modules.seld_spatial_adapter import SeldSpatialAdapter
from spatial_omni.model.processing_qwen2_5_omni import Qwen2_5OmniProcessor
from spatial_omni.model.processing_so import Qwen2_5OmniSpatialProcessor


@dataclass
class SpatialComponents:
    feature_bridge: SeldFeatureBridge
    backbone: SeldBackbone
    adapter: SeldSpatialAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen2.5-Omni-7B",
        help="Base Qwen2.5-Omni model id or local path.",
    )
    parser.add_argument(
        "--baseline-repo-path",
        type=str,
        default="${DCASE_BASELINE_REPO}",
        help="Path to the DCASE baseline repo used by task 233.",
    )
    parser.add_argument(
        "--seld-checkpoint-path",
        type=str,
        default="${DCASE_BASELINE_REPO}/3_1_dev_split0_multiaccdoa_foa_model.h5",
        help="Checkpoint used by the SELD233 spatial backbone.",
    )
    parser.add_argument(
        "--seld-feature-stats-dir",
        type=str,
        default="${SELD_FEATURE_STATS_DIR}",
        help="Directory containing `foa_wts`.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Single-device target used when `--load-qwen-model` is enabled.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=("float32", "bfloat16", "float16"),
        help="Model dtype when `--load-qwen-model` is enabled.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed used to synthesize the FOA batch.",
    )
    parser.add_argument(
        "--load-qwen-model",
        action="store_true",
        help="If set, load the full Qwen spatial model and build actual multimodal embeddings.",
    )
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    return mapping[name]


def build_random_foa_batch(seed: int) -> tuple[list[np.ndarray], list[int]]:
    rng = np.random.default_rng(seed)
    length_seconds = rng.uniform(low=2.0, high=19.5, size=3)
    sample_lengths = [int(round(value * 16000)) for value in length_seconds]
    audio = [
        rng.standard_normal((4, sample_length)).astype(np.float32) * 0.05
        for sample_length in sample_lengths
    ]
    return audio, sample_lengths


def build_spatial_processor(model_id: str) -> Qwen2_5OmniSpatialProcessor:
    base_processor = Qwen2_5OmniProcessor.from_pretrained(model_id)
    return Qwen2_5OmniSpatialProcessor(
        image_processor=base_processor.image_processor,
        feature_extractor=base_processor.feature_extractor,
        tokenizer=base_processor.tokenizer,
        chat_template=base_processor.chat_template,
    )


def configure_spatial_omni(
    model_id: str,
    checkpoint_path: str,
    baseline_repo_path: str,
    feature_stats_dir: str,
) -> Qwen2_5OmniConfig:
    config = Qwen2_5OmniConfig.from_pretrained(model_id)
    thinker_config = config.thinker_config
    thinker_config.use_seld_spatial_modality = True
    thinker_config.seld_checkpoint_path = checkpoint_path
    thinker_config.seld_baseline_repo_path = baseline_repo_path
    thinker_config.seld_feature_stats_dir = feature_stats_dir
    thinker_config.seld_freeze_backbone = True
    thinker_config.seld_max_audio_seconds = 20.0
    return config


def build_spatial_components(config: Qwen2_5OmniConfig) -> SpatialComponents:
    thinker_config = config.thinker_config
    feature_bridge = SeldFeatureBridge(
        sample_rate=16000,
        max_audio_seconds=thinker_config.seld_max_audio_seconds,
        num_feature_channels=thinker_config.seld_num_feature_channels,
        num_mel_bins=thinker_config.seld_num_mel_bins,
        hop_length=320,
        baseline_repo_path=thinker_config.seld_baseline_repo_path,
        task_id=thinker_config.seld_task_id,
        feature_stats_dir=thinker_config.seld_feature_stats_dir,
    )
    backbone = SeldBackbone(
        baseline_repo_path=thinker_config.seld_baseline_repo_path,
        checkpoint_path=thinker_config.seld_checkpoint_path,
        task_id=thinker_config.seld_task_id,
        num_feature_channels=thinker_config.seld_num_feature_channels,
        num_mel_bins=thinker_config.seld_num_mel_bins,
        hidden_dim=thinker_config.seld_encoder_dim,
        feature_to_seld_ratio=5,
        freeze_backbone=thinker_config.seld_freeze_backbone,
    )
    adapter = SeldSpatialAdapter(
        feature_bridge=feature_bridge,
        backbone=backbone,
        hidden_dim=thinker_config.seld_encoder_dim,
        token_dim=thinker_config.seld_token_dim,
        downsample_factor=thinker_config.seld_downsample_factor,
    )
    return SpatialComponents(feature_bridge=feature_bridge, backbone=backbone, adapter=adapter)


def print_batch_summary(
    processor: Qwen2_5OmniSpatialProcessor,
    batch: dict,
    raw_lengths: Sequence[int],
    hidden_size: int,
) -> None:
    audio_token_id = int(processor.tokenizer.convert_tokens_to_ids(processor.audio_token))
    spatial_token_id = int(processor.tokenizer.convert_tokens_to_ids(processor.spatial_token))

    print("== Raw Batch Summary ==")
    print(f"input_ids shape: {tuple(batch['input_ids'].shape)}")
    print(f"attention_mask shape: {tuple(batch['attention_mask'].shape)}")
    print(f"input_features shape: {tuple(batch['input_features'].shape)}")
    print(f"feature_attention_mask shape: {tuple(batch['feature_attention_mask'].shape)}")
    print(f"spatial_audio shape: {tuple(batch['spatial_audio'].shape)}")
    print(f"spatial_audio_attention_mask shape: {tuple(batch['spatial_audio_attention_mask'].shape)}")
    print(f"spatial_audio_lengths: {batch['spatial_audio_lengths'].tolist()}")
    print(f"spatial_token_lengths: {batch['spatial_token_lengths'].tolist()}")
    print(f"expected multimodal token tensor shape: {(batch['input_ids'].shape[0], batch['input_ids'].shape[1], hidden_size)}")

    for index, original_length in enumerate(raw_lengths):
        audio_count = int((batch["input_ids"][index] == audio_token_id).sum().item())
        spatial_count = int((batch["input_ids"][index] == spatial_token_id).sum().item())
        print(
            f"[sample {index}] raw_samples={original_length} "
            f"raw_seconds={original_length / 16000.0:.2f} "
            f"kept_samples={int(batch['spatial_audio_lengths'][index])} "
            f"audio_tokens={audio_count} spatial_tokens={spatial_count}"
        )


def verify_lengths_and_placeholders(
    processor: Qwen2_5OmniSpatialProcessor,
    model,
    batch: dict,
    feature_output,
    hidden_output,
    spatial_output,
) -> None:
    audio_token_id = int(processor.tokenizer.convert_tokens_to_ids(processor.audio_token))
    spatial_token_id = int(processor.tokenizer.convert_tokens_to_ids(processor.spatial_token))

    spatial_audio_lengths = batch["spatial_audio_lengths"].to(torch.long)
    expected_feature_lengths = spatial_audio_lengths // 320
    expected_hidden_lengths = expected_feature_lengths // 5
    expected_spatial_lengths = (expected_hidden_lengths + 3) // 4

    print("\n== Length Consistency Checks ==")
    print(f"expected_feature_lengths from samples: {expected_feature_lengths.tolist()}")
    print(f"feature_lengths from bridge:         {feature_output.feature_lengths.tolist()}")
    print(f"expected_hidden_lengths from feat:  {expected_hidden_lengths.tolist()}")
    print(f"hidden_lengths from backbone:       {hidden_output.hidden_lengths.tolist()}")
    print(f"expected_spatial_lengths from hid:  {expected_spatial_lengths.tolist()}")
    print(f"spatial_token_lengths from adapter: {spatial_output.spatial_token_lengths.tolist()}")

    if not torch.equal(feature_output.feature_lengths.cpu(), expected_feature_lengths.cpu()):
        raise AssertionError("feature_lengths do not match floor(spatial_audio_lengths / 320)")
    if not torch.equal(hidden_output.hidden_lengths.cpu(), expected_hidden_lengths.cpu()):
        raise AssertionError("hidden_lengths do not match floor(feature_lengths / 5)")
    if not torch.equal(spatial_output.spatial_token_lengths.cpu(), expected_spatial_lengths.cpu()):
        raise AssertionError("spatial_token_lengths do not match ceil(hidden_lengths / 4)")

    audio_feature_lengths = batch["feature_attention_mask"].to(torch.long).sum(dim=1)
    _, audio_output_lengths = model.thinker.audio_tower._get_feat_extract_output_lengths(audio_feature_lengths)

    print("\n== Placeholder Count Checks ==")
    print(f"audio_feature_lengths from mask:     {audio_feature_lengths.tolist()}")
    print(f"audio_output_lengths from encoder:   {audio_output_lengths.tolist()}")

    for index in range(batch["input_ids"].shape[0]):
        audio_count = int((batch["input_ids"][index] == audio_token_id).sum().item())
        spatial_count = int((batch["input_ids"][index] == spatial_token_id).sum().item())
        expected_audio = int(audio_output_lengths[index].item())
        expected_spatial = int(spatial_output.spatial_token_lengths[index].item())
        print(
            f"[sample {index}] "
            f"audio_placeholder_count={audio_count} expected_audio_tokens={expected_audio} | "
            f"spatial_placeholder_count={spatial_count} expected_spatial_tokens={expected_spatial}"
        )
        if audio_count != expected_audio:
            raise AssertionError(
                f"Audio placeholder mismatch for sample {index}: {audio_count} vs {expected_audio}"
            )
        if spatial_count != expected_spatial:
            raise AssertionError(
                f"Spatial placeholder mismatch for sample {index}: {spatial_count} vs {expected_spatial}"
            )

    print("All length and placeholder checks passed.")


def maybe_load_qwen_model(
    args: argparse.Namespace,
    config: Qwen2_5OmniConfig,
    processor: Qwen2_5OmniSpatialProcessor,
):
    if not args.load_qwen_model:
        return None

    model = Qwen2_5OmniSpatialForConditionalGeneration.from_pretrained(
        args.model_id,
        config=config,
        torch_dtype=dtype_from_name(args.dtype),
    )
    model.disable_talker()
    model.eval()
    model.to(args.device)
    processor.sync_spatial_tokenizer_with_model(model)
    return model


def build_actual_multimodal_embeddings(model, batch: dict) -> torch.Tensor:
    thinker = model.thinker
    device = next(thinker.parameters()).device

    input_ids = batch["input_ids"].to(device)
    input_features = batch["input_features"].to(device)
    feature_attention_mask = batch["feature_attention_mask"].to(device)
    spatial_audio = batch["spatial_audio"].to(device)
    spatial_audio_attention_mask = batch["spatial_audio_attention_mask"].to(device)
    spatial_audio_lengths = batch["spatial_audio_lengths"].to(device)

    with torch.no_grad():
        inputs_embeds = thinker.get_input_embeddings()(input_ids)

        spatial_tokens, spatial_token_lengths = thinker._resolve_spatial_tokens(
            spatial_audio=spatial_audio,
            spatial_audio_attention_mask=spatial_audio_attention_mask,
            spatial_audio_lengths=spatial_audio_lengths,
            seld_features=None,
            seld_feature_attention_mask=None,
            seld_feature_lengths=None,
            spatial_tokens=None,
            spatial_token_lengths=batch["spatial_token_lengths"].to(device),
        )
        projected_spatial = thinker.seld_spatial_projector(spatial_tokens)
        packed_spatial = thinker._flatten_projected_spatial(projected_spatial, spatial_token_lengths)
        spatial_mask = thinker._build_spatial_mask(input_ids, inputs_embeds)
        thinker._validate_spatial_mask_count(spatial_mask, packed_spatial, spatial_token_lengths)
        inputs_embeds = inputs_embeds.masked_scatter(
            spatial_mask,
            packed_spatial.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype),
        )

        audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
        packed_audio_features = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)
        aftercnn_lengths, audio_output_lengths = thinker.audio_tower._get_feat_extract_output_lengths(audio_feature_lengths)
        audio_outputs = thinker.audio_tower(
            packed_audio_features,
            feature_lens=audio_feature_lengths,
            aftercnn_lens=aftercnn_lengths,
            spatial_features=None,
            spatial_audio=None,
            output_seld=False,
        )
        audio_features = audio_outputs.last_hidden_state
        if audio_features.shape[0] != int(audio_output_lengths.sum().item()):
            raise RuntimeError(
                "Audio feature count mismatch: "
                f"{audio_features.shape[0]} vs {int(audio_output_lengths.sum().item())}"
            )
        audio_mask = (
            (input_ids == thinker.config.audio_token_id)
            .unsqueeze(-1)
            .expand_as(inputs_embeds)
            .to(inputs_embeds.device)
        )
        inputs_embeds = inputs_embeds.masked_scatter(
            audio_mask,
            audio_features.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype),
        )
    return inputs_embeds


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    processor = build_spatial_processor(args.model_id)
    config = configure_spatial_omni(
        model_id=args.model_id,
        checkpoint_path=args.seld_checkpoint_path,
        baseline_repo_path=args.baseline_repo_path,
        feature_stats_dir=args.seld_feature_stats_dir,
    )

    model = maybe_load_qwen_model(args, config, processor)
    if model is not None:
        hidden_size = int(model.config.thinker_config.text_config.hidden_size)
        spatial_components = SpatialComponents(
            feature_bridge=model.thinker.seld_feature_bridge,
            backbone=model.thinker.seld_backbone,
            adapter=model.thinker.seld_spatial_adapter,
        )
    else:
        config.thinker_config.spatial_token_index = processor.spatial_token_id
        hidden_size = int(config.thinker_config.text_config.hidden_size)
        spatial_components = build_spatial_components(config)

    audio_batch, raw_lengths = build_random_foa_batch(args.seed)
    prompts = [
        "<|AUDIO|><|spatial|> Describe the sound events and spatial scene.",
        "<|AUDIO|><|spatial|> Describe the sound events and spatial scene.",
        "<|AUDIO|><|spatial|> Describe the sound events and spatial scene.",
    ]
    batch = processor(
        text=prompts,
        audio=audio_batch,
        padding=True,
        return_tensors="pt",
    )

    print_batch_summary(processor, batch, raw_lengths, hidden_size)

    with torch.no_grad():
        feature_output = spatial_components.feature_bridge(
            spatial_audio=batch["spatial_audio"],
            spatial_audio_attention_mask=batch["spatial_audio_attention_mask"],
            spatial_audio_lengths=batch["spatial_audio_lengths"],
        )
        hidden_output = spatial_components.backbone(
            seld_features=feature_output.features,
            seld_feature_attention_mask=feature_output.feature_attention_mask,
            seld_feature_lengths=feature_output.feature_lengths,
        )
        spatial_output = spatial_components.adapter(
            spatial_audio=batch["spatial_audio"],
            spatial_audio_attention_mask=batch["spatial_audio_attention_mask"],
            spatial_audio_lengths=batch["spatial_audio_lengths"],
        )

    print("\n== Spatial Branch Outputs ==")
    print(f"features shape: {tuple(feature_output.features.shape)}")
    print(f"feature_lengths: {feature_output.feature_lengths.tolist()}")
    print(f"hidden_states shape: {tuple(hidden_output.hidden_states.shape)}")
    print(f"hidden_lengths: {hidden_output.hidden_lengths.tolist()}")
    print(f"spatial_tokens shape: {tuple(spatial_output.spatial_tokens.shape)}")
    print(f"spatial_token_lengths: {spatial_output.spatial_token_lengths.tolist()}")

    if model is not None:
        verify_lengths_and_placeholders(
            processor=processor,
            model=model,
            batch=batch,
            feature_output=feature_output,
            hidden_output=hidden_output,
            spatial_output=spatial_output,
        )
        multimodal_embeds = build_actual_multimodal_embeddings(model, batch)
        print("\n== Full Qwen Injection ==")
        print(f"actual multimodal embedding shape: {tuple(multimodal_embeds.shape)}")
    else:
        print("\n== Full Qwen Injection ==")
        print("Skipped. Re-run with --load-qwen-model to build actual multimodal embeddings.")


if __name__ == "__main__":
    main()
