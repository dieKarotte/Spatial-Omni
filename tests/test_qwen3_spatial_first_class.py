#!/usr/bin/env python3
"""Lightweight contracts for the Qwen3 first-class spatial modality.

Run with the SO-30B environment; no model checkpoint or GPU is required:

    PYTHONPATH=. python tests/test_qwen3_spatial_first_class.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import sysconfig
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

try:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (
        AutoConfig,
        AutoModel,
        PreTrainedTokenizerFast,
        WhisperFeatureExtractor,
    )
    from transformers.models.auto.modeling_auto import (
        MODEL_FOR_CAUSAL_LM_MAPPING,
        MODEL_FOR_MULTIMODAL_LM_MAPPING,
        MODEL_MAPPING,
    )
    from transformers.models.auto.processing_auto import PROCESSOR_MAPPING

    from spatial_omni.model.configuration_qwen3_omni import (
        Qwen3OmniMoeSpatialThinkerConfig,
    )
    from spatial_omni.model import modeling_so_thinker_qwen3 as spatial_modeling
    from spatial_omni.model.modeling_so_thinker_qwen3 import (
        Qwen3OmniMoeSpatialForConditionalGeneration,
    )
    from spatial_omni.model.processing_so_qwen3 import (
        Qwen3OmniMoeSpatialProcessor,
    )

    QWEN3_AVAILABLE = True
    QWEN3_IMPORT_ERROR = None
except (ImportError, RuntimeError) as exc:  # pragma: no cover - 7B environments
    QWEN3_AVAILABLE = False
    QWEN3_IMPORT_ERROR = exc

try:
    from transformers.conversion_mapping import get_checkpoint_conversion_mapping

    CONVERSION_MAPPING_AVAILABLE = True
except ImportError:  # pragma: no cover - Transformers < 5
    CONVERSION_MAPPING_AVAILABLE = False


class _DummySOEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(()))

    def forward(self, **kwargs):  # pragma: no cover - tests inject projected tokens
        raise AssertionError("The lightweight tests must not run the real SO-Encoder.")


def _tiny_config():
    text_config = {
        "vocab_size": 64,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "moe_intermediate_size": 8,
        "num_experts_per_tok": 1,
        "num_experts": 2,
        "max_position_embeddings": 64,
        "rope_parameters": {"rope_type": "default", "mrope_section": [2, 1, 1]},
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": None,
    }
    audio_config = {
        "num_mel_bins": 8,
        "encoder_layers": 1,
        "encoder_attention_heads": 2,
        "encoder_ffn_dim": 16,
        "d_model": 8,
        "max_source_positions": 16,
        "output_dim": 16,
        "n_window": 4,
        "n_window_infer": 4,
        "conv_chunksize": 4,
        "downsample_hidden_size": 8,
    }
    vision_config = {
        "depth": 1,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_heads": 2,
        "in_channels": 3,
        "patch_size": 2,
        "spatial_merge_size": 1,
        "temporal_patch_size": 1,
        "out_hidden_size": 16,
        "num_position_embeddings": 16,
        "deepstack_visual_indexes": [],
    }
    return Qwen3OmniMoeSpatialThinkerConfig(
        text_config=text_config,
        audio_config=audio_config,
        vision_config=vision_config,
        audio_token_id=10,
        image_token_id=11,
        video_token_id=12,
        audio_start_token_id=13,
        audio_end_token_id=19,
        spatial_token_index=14,
        spatial_start_token_id=15,
        spatial_end_token_id=16,
        vision_start_token_id=17,
        so_backbone_checkpoint_path="unused-by-dummy-encoder",
        so_backbone_repo_path="unused-by-dummy-encoder",
        so_encoder_dim=8,
        so_projector_hidden_dim=8,
        so_projector_shuffle_factor=1,
        so_encoder_token_rate=2.5,
        so_backbone_target_token_rate=2.5,
    )


def _tiny_model(config=None):
    with mock.patch.object(spatial_modeling, "SOEncoder", _DummySOEncoder):
        model = Qwen3OmniMoeSpatialForConditionalGeneration(config or _tiny_config())
    model.generation_config.pad_token_id = 0
    model.generation_config.eos_token_id = None
    return model


def _valid_ids():
    # BOS, audio_start, audio x2, audio_end, spatial_start,
    # spatial x2, spatial_end, text
    return torch.tensor([[1, 13, 10, 10, 19, 15, 14, 14, 16, 2]])


def _build_test_processor():
    special_tokens = [
        "<unk>",
        "<pad>",
        "<|audio_start|>",
        "<|audio_pad|>",
        "<|audio_end|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|image_pad|>",
        "<|video_pad|>",
        "<|im_start|>",
        "<|im_end|>",
    ]
    vocabulary = {
        token: index
        for index, token in enumerate(
            special_tokens + ["user", "assistant", "where", "answer"]
        )
    }
    backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<|im_end|>",
        additional_special_tokens=special_tokens[2:],
    )
    tokenizer.audio_token = "<|audio_pad|>"
    tokenizer.audio_bos_token = "<|audio_start|>"
    tokenizer.audio_eos_token = "<|audio_end|>"
    tokenizer.image_token = "<|image_pad|>"
    tokenizer.video_token = "<|video_pad|>"
    tokenizer.vision_bos_token = "<|vision_start|>"
    tokenizer.vision_eos_token = "<|vision_end|>"
    feature_extractor = WhisperFeatureExtractor(
        feature_size=8,
        sampling_rate=16000,
        hop_length=160,
        chunk_length=2,
        n_fft=400,
        return_attention_mask=True,
    )
    return Qwen3OmniMoeSpatialProcessor(feature_extractor, tokenizer)


@unittest.skipUnless(QWEN3_AVAILABLE, f"Qwen3 unavailable: {QWEN3_IMPORT_ERROR}")
class SpatialModelContractTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = _tiny_model().eval()

    def test_placeholder_masks_expose_four_parallel_modalities(self):
        input_ids = torch.tensor([[11, 12, 10, 14, 3]])
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

        masks = self.model.get_placeholder_mask(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            return_spatial_mask=True,
        )

        self.assertEqual(self.model.input_modalities, ("image", "video", "audio", "spatial", "text"))
        self.assertEqual(len(masks), 4)
        for mask, expected_position in zip(masks, (0, 1, 2, 3)):
            self.assertEqual(int(mask[..., 0].sum()), 1)
            self.assertTrue(bool(mask[0, expected_position].all()))
        # Parent Qwen3 forward still receives its native three-tuple.
        self.assertEqual(
            len(self.model.get_placeholder_mask(input_ids, inputs_embeds)),
            3,
        )

    def test_spatial_rope_handles_left_padding(self):
        input_ids = torch.tensor(
            [
                [0, 0, 13, 10, 19, 15, 14, 14, 16, 2],
                [13, 10, 10, 19, 15, 14, 14, 16, 2, 3],
            ]
        )
        attention_mask = torch.tensor(
            [
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ]
        )

        position_ids, rope_deltas = self.model.get_rope_index(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        for batch_index in range(input_ids.shape[0]):
            valid = attention_mask[batch_index].bool()
            expected = torch.arange(
                int(valid.sum()), dtype=position_ids.dtype
            ).expand(3, -1)
            torch.testing.assert_close(position_ids[:, batch_index, valid], expected)
        self.assertTrue(bool((position_ids[:, 0, :2] == 1).all()))
        torch.testing.assert_close(
            rope_deltas,
            torch.zeros((2, 1), dtype=rope_deltas.dtype),
        )

    def test_spatial_rope_rejects_invalid_boundaries_and_order(self):
        invalid_sequences = {
            "missing spatial boundaries": [13, 10, 19, 14, 2],
            "noncontiguous spatial run": [13, 10, 19, 15, 14, 3, 14, 16],
            "audio after spatial": [15, 14, 16, 13, 10, 19],
        }
        for label, sequence in invalid_sequences.items():
            with self.subTest(label=label), self.assertRaises(ValueError):
                ids = torch.tensor([sequence])
                self.model.get_rope_index(ids, attention_mask=torch.ones_like(ids))

        with self.assertRaises(NotImplementedError):
            ids = _valid_ids()
            self.model.get_rope_index(
                ids,
                image_grid_thw=torch.tensor([[1, 1, 1]]),
                attention_mask=torch.ones_like(ids),
            )

    def test_projected_spatial_embeddings_are_injected_and_trainable(self):
        input_ids = _valid_ids()
        attention_mask = torch.ones_like(input_ids)
        projected = torch.randn(1, 2, 16, requires_grad=True)
        captured = {}

        def capture_inputs(_module, _args, kwargs):
            captured["inputs_embeds"] = kwargs["inputs_embeds"]

        handle = self.model.model.register_forward_pre_hook(capture_inputs, with_kwargs=True)
        try:
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                projected_spatial_tokens=projected,
                spatial_token_lengths=torch.tensor([2]),
            )
        finally:
            handle.remove()

        spatial_positions = input_ids == self.model.config.spatial_token_index
        injected = captured["inputs_embeds"][spatial_positions]
        torch.testing.assert_close(injected, projected[0], rtol=0, atol=0)
        output.logits.float().square().mean().backward()
        self.assertIsNotNone(projected.grad)
        self.assertGreater(float(projected.grad.abs().sum()), 0.0)

    def test_missing_or_misaligned_spatial_payload_fails_fast(self):
        input_ids = _valid_ids()
        with self.assertRaisesRegex(ValueError, "no spatial payload"):
            self.model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
        with self.assertRaisesRegex(ValueError, "exactly match"):
            self.model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                projected_spatial_tokens=torch.randn(1, 1, 16),
                spatial_token_lengths=torch.tensor([1]),
            )

    def test_spatial_payloads_are_strictly_mutually_exclusive(self):
        input_ids = _valid_ids()
        attention_mask = torch.ones_like(input_ids)
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                spatial_audio=torch.zeros(1, 160, 4),
                spatial_tokens=torch.zeros(1, 2, 8),
                spatial_token_lengths=torch.tensor([2]),
            )
        with self.assertRaisesRegex(ValueError, "require spatial_audio"):
            self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                projected_spatial_tokens=torch.zeros(1, 2, 16),
                spatial_audio_lengths=torch.tensor([160]),
                spatial_token_lengths=torch.tensor([2]),
            )

    def test_generate_keeps_spatial_only_for_prefill(self):
        from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe

        input_ids = _valid_ids()
        projected = torch.randn(1, 2, 16)
        payload_seen = []
        self.model.config.text_config.output_router_logits = True

        def capture_payload(_module, _args, kwargs):
            payload_seen.append(kwargs.get("projected_spatial_tokens") is not None)

        handle = self.model.register_forward_pre_hook(capture_payload, with_kwargs=True)
        try:
            with mock.patch.object(
                modeling_qwen3_omni_moe,
                "load_balancing_loss_func",
                wraps=modeling_qwen3_omni_moe.load_balancing_loss_func,
            ) as aux_loss:
                generated = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    projected_spatial_tokens=projected,
                    spatial_token_lengths=torch.tensor([2]),
                    max_new_tokens=2,
                    do_sample=False,
                    eos_token_id=None,
                    pad_token_id=0,
                    return_audio=False,
                )
        finally:
            handle.remove()

        self.assertEqual(tuple(generated.shape), (1, input_ids.shape[1] + 2))
        self.assertEqual(payload_seen, [True, False])
        aux_loss.assert_not_called()


@unittest.skipUnless(QWEN3_AVAILABLE, f"Qwen3 unavailable: {QWEN3_IMPORT_ERROR}")
class SpatialProcessorContractTests(unittest.TestCase):
    def setUp(self):
        self.processor = _build_test_processor()

    def test_audio_spatial_processor_sequence_and_rounding(self):
        prefix, answer = self.processor.build_qa_text_parts("where", "answer")
        ordered_tokens = (
            self.processor.audio_bos_token,
            self.processor.audio_token,
            self.processor.audio_eos_token,
            self.processor.spatial_start_token,
            self.processor.spatial_token,
            self.processor.spatial_end_token,
        )
        positions = [prefix.index(token) for token in ordered_tokens]
        self.assertEqual(positions, sorted(positions))
        self.assertTrue(answer.endswith("<|im_end|>"))

        # Model semantics: round 1.5s * 10Hz = 15 native tokens, then
        # pixel-shuffle floor(15 / 4) = 3 LLM-side spatial tokens.
        samples = torch.tensor([int(1.5 * 16000)])
        self.assertEqual(self.processor._samples_to_so_backbone_tokens(samples).tolist(), [3])

        foa = np.zeros((4, int(1.5 * 16000)), dtype=np.float32)
        batch = self.processor(
            text=prefix + answer,
            audio=[foa],
            return_tensors="pt",
            padding=True,
        )
        self.assertEqual(batch["spatial_token_lengths"].tolist(), [3])
        self.assertEqual(batch["spatial_audio_lengths"].tolist(), [foa.shape[1]])
        self.assertEqual(
            int((batch["input_ids"] == self.processor.spatial_token_id).sum()),
            3,
        )

    def test_projected_spatial_payload_is_preserved(self):
        prefix, answer = self.processor.build_qa_text_parts("where", "answer")
        projected = torch.randn(1, 2, 16)
        batch = self.processor(
            text=prefix + answer,
            audio=[np.zeros(16000, dtype=np.float32)],
            projected_spatial_tokens=projected,
            spatial_token_lengths=torch.tensor([2]),
            allow_mono_spatial_tokens=True,
            return_tensors="pt",
        )

        self.assertNotIn("spatial_audio", batch)
        torch.testing.assert_close(batch["projected_spatial_tokens"], projected)
        self.assertEqual(batch["spatial_token_lengths"].tolist(), [2])
        self.assertEqual(
            int((batch["input_ids"] == self.processor.spatial_token_id).sum()),
            2,
        )

        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            self.processor(
                text=prefix + answer,
                audio=[np.zeros((4, 16000), dtype=np.float32)],
                spatial_tokens=torch.zeros(1, 2, 8),
                projected_spatial_tokens=projected,
                spatial_token_lengths=torch.tensor([2]),
                return_tensors="pt",
            )
        with self.assertRaisesRegex(ValueError, r"within \[0, T_spat\]"):
            self.processor(
                text=prefix + answer,
                audio=[np.zeros((4, 16000), dtype=np.float32)],
                projected_spatial_tokens=projected,
                spatial_token_lengths=torch.tensor([3]),
                return_tensors="pt",
            )

    def test_model_input_names_are_audio_spatial_only(self):
        names = self.processor.model_input_names
        self.assertIn("input_features", names)
        self.assertIn("feature_attention_mask", names)
        self.assertIn("spatial_audio", names)
        self.assertIn("projected_spatial_tokens", names)
        self.assertNotIn("pixel_values", names)
        self.assertEqual(
            Qwen3OmniMoeSpatialProcessor.get_attributes(),
            ["feature_extractor", "tokenizer"],
        )
        for native_token in (
            "<|audio_start|>",
            "<|audio_pad|>",
            "<|audio_end|>",
            "<|im_start|>",
            "<|im_end|>",
        ):
            self.assertIn(native_token, self.processor.tokenizer.all_special_tokens)

    def test_auto_processor_save_load_roundtrip(self):
        from transformers import AutoProcessor

        self.processor.so_encoder_token_rate = 8.0
        self.processor.so_backbone_target_token_rate = 4.0
        self.processor.so_projector_shuffle_factor = 2
        self.processor.spatial_audio_max_seconds = 7.0
        with tempfile.TemporaryDirectory() as output_dir:
            self.processor.save_pretrained(output_dir)
            _tiny_config().save_pretrained(output_dir)
            restored = AutoProcessor.from_pretrained(
                output_dir,
                local_files_only=True,
            )

        self.assertIsInstance(restored, Qwen3OmniMoeSpatialProcessor)
        self.assertEqual(restored.audio_token, "<|audio_pad|>")
        self.assertEqual(restored.spatial_token, "<|spatial|>")
        self.assertEqual(
            restored.build_audio_spatial_prefix().count(restored.spatial_token),
            1,
        )
        self.assertEqual(restored.so_encoder_token_rate, 8.0)
        self.assertEqual(restored.so_backbone_target_token_rate, 4.0)
        self.assertEqual(restored.so_projector_shuffle_factor, 2)
        self.assertEqual(restored.spatial_audio_max_seconds, 7.0)
        prefix, answer = restored.build_qa_text_parts("where", "answer")
        processed = restored(
            text=prefix + answer,
            audio=[np.zeros((4, 16000), dtype=np.float32)],
            return_tensors="pt",
        )
        self.assertEqual(processed["spatial_token_lengths"].tolist(), [4])
        self.assertEqual(
            int((processed["input_ids"] == restored.spatial_token_id).sum()),
            4,
        )
        self.assertIn("spatial_audio", restored.model_input_names)

    def test_model_sync_preserves_reserved_vocab_rows(self):
        model = _tiny_model()
        model.config.spatial_start_token_id = None
        model.config.spatial_end_token_id = None
        original_vocab_size = model.get_input_embeddings().num_embeddings

        self.processor.sync_spatial_tokenizer_with_model(model)

        self.assertEqual(model.get_input_embeddings().num_embeddings, original_vocab_size)
        self.assertEqual(model.config.text_config.vocab_size, original_vocab_size)
        self.assertEqual(
            model.generation_config.eos_token_id,
            self.processor.tokenizer.eos_token_id,
        )
        self.assertEqual(
            model.generation_config.pad_token_id,
            self.processor.tokenizer.pad_token_id,
        )
        embeddings = model.get_input_embeddings().weight.detach()
        audio_start_id = self.processor.tokenizer.convert_tokens_to_ids(
            self.processor.audio_bos_token
        )
        audio_end_id = self.processor.tokenizer.convert_tokens_to_ids(
            self.processor.audio_eos_token
        )
        torch.testing.assert_close(
            embeddings[self.processor.spatial_start_token_id],
            embeddings[audio_start_id],
        )
        torch.testing.assert_close(
            embeddings[self.processor.spatial_end_token_id],
            embeddings[audio_end_id],
        )

    def test_bench_uses_the_training_generation_prefix(self):
        from scripts.batch_bench_so_qa import SpatialBeatsEvalCollator

        audio_time_major = np.zeros((24000, 4), dtype=np.float32)
        collator = SpatialBeatsEvalCollator(processor=self.processor)
        feature = {
            "audio_path": "unused.wav",
            "prompt": "where",
            "answer": "answer",
        }
        with mock.patch.object(
            collator,
            "_read_audio",
            return_value=(audio_time_major, 16000),
        ):
            batch = collator([feature])

        expected_prefix, _ = self.processor.build_qa_text_parts("where", "")
        expected = self.processor(
            text=expected_prefix,
            audio=[audio_time_major.T],
            padding=True,
            padding_side="right",
            return_tensors="pt",
        )
        torch.testing.assert_close(batch["input_ids"], expected["input_ids"])

    def test_bench_zero_spatial_uses_projected_payload_only(self):
        from scripts.batch_bench_so_qa import SpatialBeatsEvalCollator

        collator = SpatialBeatsEvalCollator(
            processor=self.processor,
            mono_audio_zero_spatial_tokens=True,
            zero_projected_spatial_dim=16,
        )
        feature = {
            "audio_path": "unused.wav",
            "prompt": "where",
            "answer": "answer",
        }
        mono_time_major = np.zeros((16000, 1), dtype=np.float32)
        with mock.patch.object(
            collator,
            "_read_audio",
            return_value=(mono_time_major, 16000),
        ):
            batch = collator([feature])

        self.assertIn("projected_spatial_tokens", batch)
        self.assertIn("gen_projected_spatial_tokens", batch)
        self.assertNotIn("spatial_audio", batch)
        self.assertTrue(bool((batch["projected_spatial_tokens"] == 0).all()))
        self.assertEqual(
            batch["projected_spatial_tokens"].shape[-1],
            16,
        )


@unittest.skipUnless(QWEN3_AVAILABLE, f"Qwen3 unavailable: {QWEN3_IMPORT_ERROR}")
class SpatialRegistrationAndTrainingTests(unittest.TestCase):
    def test_launcher_bootstraps_shared_conda_environment(self):
        conda_env = Path(sys.executable).resolve().parents[1]
        launcher = REPO_ROOT / "shell" / "launch_train_so_30b.sh"
        with tempfile.TemporaryDirectory() as temp_dir:
            asset_root = Path(temp_dir)
            model_dir = asset_root / "ckpts" / "Qwen3-Omni-30B-A3B-Instruct"
            model_dir.mkdir(parents=True)
            (model_dir / "config.json").write_text("{}\n", encoding="utf-8")
            (asset_root / "ckpts" / "SO-Encoder").mkdir()
            (asset_root / "ckpts" / "SO-Encoder" / "SO-Encoder.pt").touch()
            qa_root = asset_root / "SO-Dataset" / "qa"
            qa_root.mkdir(parents=True)
            for split in ("train", "valid", "test"):
                (qa_root / f"{split}.jsonl").touch()

            env = os.environ.copy()
            for name in (
                "PYTHON_BIN",
                "MODEL_ID",
                "SO_BASE_MODEL",
                "SO_DATASET_ROOT",
                "SO_ENCODER_CKPT",
                "BEATS_CKPT",
                "QA_ROOT",
                "AUDIO_ROOT",
            ):
                env.pop(name, None)
            env.update(
                {
                    "CONDA_ENV": str(conda_env),
                    "SO_ASSET_ROOT": str(asset_root),
                    "CHECK_ONLY": "1",
                    "GPUS": "0",
                    "NPROC": "1",
                }
            )
            result = subprocess.run(
                ["bash", str(launcher)],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                timeout=60,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(f"PYTHON_BIN={sys.executable}", result.stdout)
        self.assertIn(str(REPO_ROOT), result.stdout)
        self.assertIn(sysconfig.get_paths()["purelib"], result.stdout)
        self.assertIn("transformers=5.0.0", result.stdout)
        self.assertIn("[check-only]", result.stdout)

    def test_bench_rejects_incomplete_adaptation_state(self):
        from scripts.batch_bench_so_qa import validate_benchmark_checkpoint_load

        clean = SimpleNamespace(
            missing_keys=["model.frozen_weight"],
            unexpected_keys=[],
        )
        self.assertEqual(
            validate_benchmark_checkpoint_load(clean, {"so_projector.fc1.weight"}),
            [],
        )

        missing = SimpleNamespace(
            missing_keys=["so_projector.fc1.weight"],
            unexpected_keys=[],
        )
        with self.assertRaisesRegex(RuntimeError, "did not restore"):
            validate_benchmark_checkpoint_load(missing, {"so_projector.fc1.weight"})

        unexpected = SimpleNamespace(
            missing_keys=[],
            unexpected_keys=["base_model.model.unloaded_lora"],
        )
        with self.assertRaisesRegex(RuntimeError, "did not restore"):
            validate_benchmark_checkpoint_load(unexpected, set())

    def test_auto_classes_are_registered(self):
        config_kwargs = _tiny_config().to_dict()
        config_kwargs.pop("model_type", None)
        config = AutoConfig.for_model(
            Qwen3OmniMoeSpatialThinkerConfig.model_type,
            **config_kwargs,
        )
        self.assertIsInstance(config, Qwen3OmniMoeSpatialThinkerConfig)
        self.assertIs(
            MODEL_MAPPING[Qwen3OmniMoeSpatialThinkerConfig],
            Qwen3OmniMoeSpatialForConditionalGeneration,
        )
        self.assertIs(
            MODEL_FOR_CAUSAL_LM_MAPPING[Qwen3OmniMoeSpatialThinkerConfig],
            Qwen3OmniMoeSpatialForConditionalGeneration,
        )
        self.assertIs(
            MODEL_FOR_MULTIMODAL_LM_MAPPING[Qwen3OmniMoeSpatialThinkerConfig],
            Qwen3OmniMoeSpatialForConditionalGeneration,
        )
        self.assertIs(
            PROCESSOR_MAPPING[Qwen3OmniMoeSpatialThinkerConfig],
            Qwen3OmniMoeSpatialProcessor,
        )
        with mock.patch.object(spatial_modeling, "SOEncoder", _DummySOEncoder):
            auto_model = AutoModel.from_config(config)
        self.assertIsInstance(auto_model, Qwen3OmniMoeSpatialForConditionalGeneration)

    @unittest.skipUnless(
        CONVERSION_MAPPING_AVAILABLE,
        "Transformers checkpoint conversion registry is unavailable",
    )
    def test_checkpoint_converter_is_registered(self):
        spatial_mapping = get_checkpoint_conversion_mapping(
            Qwen3OmniMoeSpatialThinkerConfig.model_type
        )
        native_mapping = get_checkpoint_conversion_mapping("qwen3_omni_moe_thinker")
        self.assertIsNotNone(spatial_mapping)
        self.assertEqual(
            [converter.source_patterns for converter in spatial_mapping],
            [converter.source_patterns for converter in native_mapping],
        )

    def test_router_selection_and_optimizer_group(self):
        from train_so_qa import (
            build_qa_text_parts,
            build_optimizer,
            configure_encoder_lora_training,
            is_moe_router_parameter,
        )

        model = _tiny_model()
        args = SimpleNamespace(
            train_moe_router=True,
            lr=3e-5,
            projector_lr=1e-5,
            lora_lr=3e-5,
            beats_lr=1e-6,
            moe_router_lr=1e-6,
            spatial_null_lr=None,
            projector_weight_decay=None,
            weight_decay=0.01,
        )
        enabled = configure_encoder_lora_training(model, args)
        router_names = [name for name in enabled if is_moe_router_parameter(name)]
        self.assertEqual(router_names, ["model.layers.0.mlp.gate.weight"])
        self.assertTrue(
            all(
                not parameter.requires_grad
                for name, parameter in model.named_parameters()
                if ".mlp.experts." in name
            )
        )

        optimizer = build_optimizer(model, args)
        router_groups = [group for group in optimizer.param_groups if group["name"].startswith("router_")]
        self.assertEqual(len(router_groups), 1)
        self.assertEqual(router_groups[0]["lr"], 1e-6)
        self.assertEqual(router_groups[0]["weight_decay"], 0.0)

        legacy_processor = SimpleNamespace(
            audio_token="<audio>",
            spatial_token="<spatial>",
            tokenizer=SimpleNamespace(eos_token="</s>"),
        )
        self.assertEqual(
            build_qa_text_parts(legacy_processor, "where", "answer"),
            ("<audio><spatial>\nwhere\n", "answer</s>"),
        )

    def test_export_bundle_includes_spatial_and_router_state(self):
        from peft import get_peft_model_state_dict
        from train_so_qa import (
            apply_llm_lora,
            configure_encoder_lora_training,
            save_export_bundle,
        )

        args = SimpleNamespace(
            train_moe_router=True,
            lora_target_prefixes=["model.layers"],
            lora_target_modules=["q_proj"],
            lora_r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            gradient_checkpointing=False,
        )
        model, _ = apply_llm_lora(_tiny_model(), args)
        enabled = configure_encoder_lora_training(model, args)
        for name, parameter in model.named_parameters():
            if "so_encoder" in name:
                parameter.requires_grad_(True)
                enabled.append(name)
        self.assertTrue(any(".mlp.gate.modules_to_save." in name for name in enabled))
        self.assertTrue(any("so_encoder" in name for name in enabled))

        sentinel_parameters = {}
        predicates = {
            "projector": lambda name: "so_projector" in name,
            "encoder": lambda name: "so_encoder" in name,
            "router": lambda name: ".mlp.gate.modules_to_save." in name,
        }
        for value, (category, predicate) in enumerate(predicates.items(), start=1):
            name, parameter = next(
                (name, parameter)
                for name, parameter in model.named_parameters()
                if parameter.requires_grad and predicate(name)
            )
            with torch.no_grad():
                parameter.fill_(value + 0.25)
            sentinel_parameters[category] = (name, parameter.detach().cpu().clone())

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            adapter_state = get_peft_model_state_dict(model)
        self.assertTrue(any(key.endswith(".mlp.gate.weight") for key in adapter_state))

        with tempfile.TemporaryDirectory() as export_dir:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                save_export_bundle(
                    model,
                    _build_test_processor(),
                    export_dir,
                    SimpleNamespace(model_id="tiny-base"),
                )
            bundle = torch.load(
                os.path.join(export_dir, "spatial_trainable_state.pt"),
                map_location="cpu",
                weights_only=False,
            )
            exported_keys = set(bundle["trainable_state_dict"])
            self.assertTrue(any("so_projector" in key for key in exported_keys))
            self.assertTrue(any("so_encoder" in key for key in exported_keys))
            self.assertTrue(any(".mlp.gate.modules_to_save." in key for key in exported_keys))
            self.assertTrue(os.path.isfile(os.path.join(export_dir, "train_args.json")))
            self.assertTrue(os.path.isfile(os.path.join(export_dir, "spatial_export_manifest.json")))

            restored, _ = apply_llm_lora(_tiny_model(), args)
            configure_encoder_lora_training(restored, args)
            load_result = restored.load_state_dict(
                bundle["trainable_state_dict"],
                strict=False,
            )
            for category, (name, expected) in sentinel_parameters.items():
                with self.subTest(category=category):
                    self.assertNotIn(name, load_result.unexpected_keys)
                    torch.testing.assert_close(
                        dict(restored.named_parameters())[name].detach().cpu(),
                        expected,
                        rtol=0,
                        atol=0,
                    )

    def test_stage1_export_does_not_copy_frozen_base_model(self):
        from train_so_qa import freeze_all_but_projector, save_export_bundle

        model = _tiny_model()
        freeze_all_but_projector(model)
        with tempfile.TemporaryDirectory() as export_dir, mock.patch.object(
            model,
            "save_pretrained",
        ) as save_pretrained:
            save_export_bundle(
                model,
                _build_test_processor(),
                export_dir,
                SimpleNamespace(model_id="tiny-base"),
            )

            save_pretrained.assert_not_called()
            bundle = torch.load(
                os.path.join(export_dir, "spatial_trainable_state.pt"),
                map_location="cpu",
                weights_only=False,
            )
            with open(
                os.path.join(export_dir, "spatial_export_manifest.json"),
                encoding="utf-8",
            ) as handle:
                manifest = json.load(handle)

        self.assertEqual(bundle["format"], "spatial-omni-trainable-bundle-v1")
        self.assertTrue(
            all("so_projector" in name for name in bundle["trainable_state_dict"])
        )
        self.assertFalse(manifest["has_peft_adapter"])
        self.assertIsNone(manifest["adapter_directory"])
        self.assertFalse(manifest["merged_base_model"])

    def test_router_receives_gradient_from_spatial_sequence(self):
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
            load_balancing_loss_func,
        )

        config = _tiny_config()
        config.text_config.output_router_logits = True
        config.text_config.router_aux_loss_coef = 1e-3
        model = _tiny_model(config)
        model.train()
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        input_ids = _valid_ids()
        attention_mask = torch.ones_like(input_ids)
        captured_router = []

        def capture_router(_module, args, result):
            captured_router.append((args[0], result))

        gate = model.model.layers[0].mlp.gate
        handle = gate.register_forward_hook(capture_router)
        try:
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                projected_spatial_tokens=torch.randn(1, 2, 16),
                spatial_token_lengths=torch.tensor([2]),
                labels=input_ids,
                use_cache=False,
            )
        finally:
            handle.remove()

        gate_input, (raw_logits, routing_scores, selected_experts) = captured_router[0]
        expected_raw_logits = torch.nn.functional.linear(
            gate_input.reshape(-1, gate.hidden_dim),
            gate.weight,
        )
        torch.testing.assert_close(raw_logits, expected_raw_logits)
        probabilities = raw_logits.softmax(dim=-1, dtype=torch.float)
        expected_scores, expected_experts = torch.topk(
            probabilities,
            gate.top_k,
            dim=-1,
        )
        if gate.norm_topk_prob:
            expected_scores = expected_scores / expected_scores.sum(dim=-1, keepdim=True)
        torch.testing.assert_close(routing_scores, expected_scores)
        torch.testing.assert_close(selected_experts, expected_experts)
        expected_aux_loss = load_balancing_loss_func(
            (raw_logits,),
            model.num_experts,
            model.num_experts_per_tok,
            attention_mask,
        )
        self.assertIsNotNone(output.aux_loss)
        self.assertTrue(bool(torch.isfinite(output.aux_loss)))
        self.assertTrue(output.aux_loss.requires_grad)
        torch.testing.assert_close(output.aux_loss, expected_aux_loss)
        output.loss.backward()
        router = gate.weight
        self.assertIsNotNone(router.grad)
        self.assertGreater(float(router.grad.abs().sum()), 0.0)

    def test_peft_wrapped_router_keeps_aux_loss_and_gradient(self):
        from train_so_qa import (
            apply_llm_lora,
            configure_encoder_lora_training,
        )

        args = SimpleNamespace(
            train_moe_router=True,
            lora_target_prefixes=["model.layers"],
            lora_target_modules=["q_proj"],
            lora_r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            gradient_checkpointing=True,
        )
        model, _ = apply_llm_lora(_tiny_model(), args)
        configure_encoder_lora_training(model, args)
        model.train()
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        model.config.text_config.output_router_logits = True
        model.config.text_config.router_aux_loss_coef = 1e-3

        input_ids = _valid_ids()
        output = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            projected_spatial_tokens=torch.randn(1, 2, 16),
            spatial_token_lengths=torch.tensor([2]),
            labels=input_ids,
            use_cache=False,
        )

        self.assertIsNotNone(output.aux_loss)
        self.assertTrue(output.aux_loss.requires_grad)
        self.assertTrue(bool(torch.isfinite(output.aux_loss)))
        output.loss.backward()
        router_gradients = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if ".mlp.gate.modules_to_save." in name and parameter.requires_grad
        ]
        self.assertTrue(router_gradients)
        self.assertTrue(all(gradient is not None for gradient in router_gradients))
        self.assertGreater(
            sum(float(gradient.abs().sum()) for gradient in router_gradients),
            0.0,
        )

    def test_loading_info_accepts_transformers_five_sets(self):
        from train_so_qa_qwen3 import _validate_qwen3_base_loading_info

        clean_info = {
            "missing_keys": {"so_projector.fc1.weight"},
            "unexpected_keys": {
                "talker.model.embed_tokens.weight",
                "talker.model.layers.0.mlp.experts.down_proj",
            },
            "mismatched_keys": set(),
            "error_msgs": set(),
        }
        _validate_qwen3_base_loading_info(clean_info)

        broken_info = dict(clean_info, mismatched_keys={"model.layers.0.bad_weight"})
        with self.assertRaisesRegex(RuntimeError, "did not load cleanly"):
            _validate_qwen3_base_loading_info(broken_info)

        unknown_missing = dict(clean_info, missing_keys={"unknown_adapter.weight"})
        with self.assertRaisesRegex(RuntimeError, "did not load cleanly"):
            _validate_qwen3_base_loading_info(unknown_missing)

        unknown_unexpected = dict(clean_info, unexpected_keys={"unknown_head.weight"})
        with self.assertRaisesRegex(RuntimeError, "did not load cleanly"):
            _validate_qwen3_base_loading_info(unknown_unexpected)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    unittest.main(verbosity=2)
