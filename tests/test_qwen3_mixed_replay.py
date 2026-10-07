#!/usr/bin/env python3
"""Contracts for the Qwen3 mixed spatial+mono replay (MIX) path.

Covers the pieces ported from the 7B mono-replay implementation:
  - replay config fields on Qwen3OmniMoeSpatialThinkerConfig
  - spatial_null allocation / re-init guard
  - mixed-replay forward: null injection, W-only alignment loss, stats
  - trainer freeze policy + optimizer group for spatial_null
  - ChatML/OSS dataset normalization and the mixed collator batch layout

Run with the SO-30B environment; no model checkpoint or GPU is required:

    PYTHONPATH=. python tests/test_qwen3_mixed_replay.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

try:
    from spatial_omni.model.configuration_qwen3_omni import (
        Qwen3OmniMoeSpatialThinkerConfig,
    )
    from spatial_omni.model import modeling_so_thinker_qwen3 as spatial_modeling
    from spatial_omni.model.modeling_so_thinker_qwen3 import (
        Qwen3OmniMoeSpatialForConditionalGeneration,
    )

    QWEN3_AVAILABLE = True
    QWEN3_IMPORT_ERROR = None
except (ImportError, RuntimeError) as exc:  # pragma: no cover - 7B environments
    QWEN3_AVAILABLE = False
    QWEN3_IMPORT_ERROR = exc


class _DummySOEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(()))

    def forward(self, **kwargs):  # pragma: no cover - tests stub the encoder out
        raise AssertionError("The lightweight tests must not run the real SO-Encoder.")


def _tiny_config(**overrides):
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
    kwargs = dict(
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
    kwargs.update(overrides)
    return Qwen3OmniMoeSpatialThinkerConfig(**kwargs)


def _tiny_model(config=None):
    with mock.patch.object(spatial_modeling, "SOEncoder", _DummySOEncoder):
        model = Qwen3OmniMoeSpatialForConditionalGeneration(config or _tiny_config())
    model.generation_config.pad_token_id = 0
    model.generation_config.eos_token_id = None
    return model


def _mixed_input_ids():
    # Two rows, each with two <|spatial|> placeholders (index 14):
    #   row0 (spatial): BOS, audio_start, audio, audio_end, spatial_start,
    #                   spatial x2, spatial_end, text
    #   row1 (mono):    same layout with a different text tail
    return torch.tensor(
        [
            [1, 13, 10, 19, 15, 14, 14, 16, 2, 0],
            [1, 13, 10, 19, 15, 14, 14, 16, 3, 4],
        ]
    )


@unittest.skipUnless(QWEN3_AVAILABLE, f"Qwen3 unavailable: {QWEN3_IMPORT_ERROR}")
class MixedReplayConfigTests(unittest.TestCase):
    def test_replay_fields_default_off(self):
        config = _tiny_config()
        self.assertFalse(config.enable_spatial_replay)
        self.assertIsNone(config.spatial_null_num_tokens)
        self.assertAlmostEqual(config.spatial_null_alignment_weight, 0.05)

    def test_replay_fields_survive_config_roundtrip(self):
        config = _tiny_config(
            enable_spatial_replay=True,
            spatial_null_num_tokens=50,
            spatial_null_alignment_weight=0.1,
        )
        payload = config.to_dict()
        self.assertTrue(payload["enable_spatial_replay"])
        self.assertEqual(payload["spatial_null_num_tokens"], 50)
        restored = Qwen3OmniMoeSpatialThinkerConfig.from_dict(payload)
        self.assertTrue(restored.enable_spatial_replay)
        self.assertEqual(restored.spatial_null_num_tokens, 50)
        self.assertAlmostEqual(restored.spatial_null_alignment_weight, 0.1)


@unittest.skipUnless(QWEN3_AVAILABLE, f"Qwen3 unavailable: {QWEN3_IMPORT_ERROR}")
class SpatialNullParameterTests(unittest.TestCase):
    def test_spatial_null_allocated_only_when_enabled(self):
        disabled = _tiny_model(_tiny_config())
        self.assertIsNone(disabled.spatial_null)
        self.assertFalse(disabled.enable_spatial_replay)

        enabled = _tiny_model(
            _tiny_config(enable_spatial_replay=True, spatial_null_num_tokens=50)
        )
        self.assertIsInstance(enabled.spatial_null, nn.Parameter)
        self.assertEqual(tuple(enabled.spatial_null.shape), (50, 16))
        self.assertTrue(bool(torch.isfinite(enabled.spatial_null).all()))

    def test_spatial_null_default_size_uses_max_seconds_times_rate(self):
        config = _tiny_config(enable_spatial_replay=True)
        config.so_backbone_max_audio_seconds = 20.0
        model = _tiny_model(config)
        # 20s * 2.5Hz (so_backbone_target_token_rate) = 50 tokens.
        self.assertEqual(tuple(model.spatial_null.shape), (50, 16))

    def test_reinit_recovers_from_nan(self):
        model = _tiny_model(
            _tiny_config(enable_spatial_replay=True, spatial_null_num_tokens=4)
        )
        with torch.no_grad():
            model.spatial_null.fill_(float("nan"))
        self.assertTrue(model.reinit_spatial_null_if_needed())
        self.assertTrue(bool(torch.isfinite(model.spatial_null).all()))
        self.assertAlmostEqual(float(model.spatial_null.float().std()), 0.02, delta=0.02)
        # Healthy parameter -> no-op.
        self.assertFalse(model.reinit_spatial_null_if_needed())
        # Disabled replay -> no-op.
        self.assertFalse(_tiny_model().reinit_spatial_null_if_needed())

    def test_get_spatial_null_expands_and_pads(self):
        model = _tiny_model(
            _tiny_config(enable_spatial_replay=True, spatial_null_num_tokens=3)
        )
        tokens = model.get_spatial_null(2, token_lengths=torch.tensor([2, 3]))
        self.assertEqual(tuple(tokens.shape), (2, 3, 16))
        torch.testing.assert_close(tokens[0, :2], model.spatial_null[:2])
        # Request beyond the bank: last row is repeated.
        tokens = model.get_spatial_null(1, token_lengths=torch.tensor([5]))
        self.assertEqual(tuple(tokens.shape), (1, 5, 16))
        torch.testing.assert_close(tokens[0, 4], model.spatial_null[2])
        # Zero lengths -> empty.
        empty = model.get_spatial_null(2, token_lengths=torch.tensor([0, 0]))
        self.assertEqual(tuple(empty.shape), (2, 0, 16))

    def test_build_w_only_audio_layout(self):
        mono = torch.randn(2, 100)
        w_only, lengths = Qwen3OmniMoeSpatialForConditionalGeneration._build_w_only_audio(
            mono, None
        )
        self.assertEqual(tuple(w_only.shape), (2, 100, 4))
        torch.testing.assert_close(w_only[..., 0], mono)
        self.assertTrue(bool((w_only[..., 1:] == 0).all()))
        self.assertEqual(lengths.tolist(), [100, 100])
        with self.assertRaisesRegex(ValueError, r"\[B,T\], \[B,1,T\], or \[B,T,1\]"):
            Qwen3OmniMoeSpatialForConditionalGeneration._build_w_only_audio(
                torch.randn(2, 3, 4), None
            )


@unittest.skipUnless(QWEN3_AVAILABLE, f"Qwen3 unavailable: {QWEN3_IMPORT_ERROR}")
class MixedReplayForwardTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = _tiny_model(
            _tiny_config(
                enable_spatial_replay=True,
                spatial_null_num_tokens=4,
                spatial_null_alignment_weight=0.5,
            )
        ).eval()

    def _run_mixed_forward(self, model, labels=None):
        input_ids = _mixed_input_ids()
        attention_mask = torch.ones_like(input_ids)
        has_spatial = torch.tensor([True, False])
        B, T = input_ids.shape
        spatial_audio = torch.zeros(B, 160, 4)
        spatial_mask = torch.ones(B, 160, dtype=torch.bool)
        spatial_lengths = torch.tensor([160, 0])
        mono_audio = torch.zeros(B, 160)
        mono_lengths = torch.tensor([160, 160])

        def fake_project(spatial_audio, spatial_audio_attention_mask=None,
                         spatial_audio_lengths=None):
            n = spatial_audio.shape[0]
            projected = torch.full((n, 2, 16), 0.25, requires_grad=True)
            return projected, torch.full((n,), 2, dtype=torch.long)

        captured = {}
        handle = model.model.register_forward_pre_hook(
            lambda _m, _a, kw: captured.update(kw), with_kwargs=True
        )
        try:
            with mock.patch.object(model, "_project_spatial_audio", side_effect=fake_project):
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    spatial_audio=spatial_audio,
                    spatial_audio_attention_mask=spatial_mask,
                    spatial_audio_lengths=spatial_lengths,
                    has_spatial=has_spatial,
                    mono_audio=mono_audio,
                    mono_audio_lengths=mono_lengths,
                    labels=labels,
                )
        finally:
            handle.remove()
        return output, captured, input_ids

    def test_null_tokens_injected_for_mono_rows(self):
        output, captured, input_ids = self._run_mixed_forward(self.model)
        embeds = captured["inputs_embeds"]
        spatial_positions = input_ids == self.model.config.spatial_token_index
        # Mono row (index 1) placeholders == spatial_null bank rows.
        torch.testing.assert_close(
            embeds[1][spatial_positions[1]],
            self.model.spatial_null[:2].to(embeds.dtype),
            rtol=0,
            atol=0,
        )
        # Spatial row (index 0) placeholders == stubbed projector output.
        torch.testing.assert_close(
            embeds[0][spatial_positions[0]],
            torch.full((2, 16), 0.25, dtype=embeds.dtype),
            rtol=1e-5,
            atol=1e-6,
        )
        stats = self.model._last_spatial_replay_stats
        self.assertEqual(stats["spatial_samples"], 1.0)
        self.assertEqual(stats["replay_samples"], 1.0)
        self.assertIn("w_only_null_cosine", stats)

    def test_loss_composition_and_spatial_null_gradient(self):
        model = self.model
        model.train()
        labels = _mixed_input_ids().clone()
        output, _, _ = self._run_mixed_forward(model, labels=labels)

        self.assertIsNotNone(output.loss)
        self.assertTrue(bool(torch.isfinite(output.loss)))
        # loss == ce + weight * mse(w_only, spatial_null.detach())
        self.assertIsNotNone(getattr(output, "loss_ce", None))
        self.assertIsNotNone(getattr(output, "loss_null", None))
        expected = output.loss_ce + 0.5 * output.loss_null
        torch.testing.assert_close(output.loss, expected)

        output.loss.backward()
        self.assertIsNotNone(model.spatial_null.grad)
        # The injected null rows (mono placeholders) produce CE gradient.
        self.assertGreater(float(model.spatial_null.grad.abs().sum()), 0.0)
        stats = model._last_spatial_replay_stats
        for key in ("loss_null", "loss_ce", "loss_total"):
            self.assertIn(key, stats)
            self.assertGreaterEqual(stats[key], 0.0)

    def test_cached_mono_generation_uses_replay_payload_only_for_prefill(self):
        input_ids = _mixed_input_ids()[:1]
        payload_seen = []
        def capture(_module, _args, kwargs):
            payload_seen.append(tuple(
                kwargs.get(key) is not None
                for key in ("has_spatial", "mono_audio", "mono_audio_lengths")
            ))
        def fake_project(audio, spatial_audio_attention_mask=None, spatial_audio_lengths=None):
            return torch.zeros(audio.shape[0], 2, 16), torch.full((audio.shape[0],), 2, dtype=torch.long)
        handle = self.model.register_forward_pre_hook(capture, with_kwargs=True)
        try:
            with mock.patch.object(self.model, "_project_spatial_audio", side_effect=fake_project) as project:
                generated = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    spatial_audio=torch.zeros(1, 160, 4),
                    spatial_audio_lengths=torch.zeros(1, dtype=torch.long),
                    spatial_token_lengths=torch.tensor([2]),
                    has_spatial=torch.tensor([False]),
                    mono_audio=torch.zeros(1, 160),
                    mono_audio_lengths=torch.tensor([160]),
                    max_new_tokens=2, do_sample=False, use_cache=True,
                    eos_token_id=None, pad_token_id=0,
                )
        finally:
            handle.remove()
        self.assertEqual(generated.shape[1], input_ids.shape[1] + 2)
        self.assertEqual(payload_seen, [(True, True, True), (False, False, False)])
        self.assertEqual(project.call_count, 1)

    def test_replay_requires_enable_flag(self):
        model = _tiny_model().eval()
        input_ids = _mixed_input_ids()
        with self.assertRaisesRegex(ValueError, "enable_spatial_replay"):
            model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                spatial_audio=torch.zeros(2, 160, 4),
                has_spatial=torch.tensor([True, False]),
                mono_audio=torch.zeros(2, 160),
            )

    def test_default_forward_leaves_no_replay_stats(self):
        model = self.model
        input_ids = _mixed_input_ids()[:1]
        model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            projected_spatial_tokens=torch.randn(1, 2, 16),
            spatial_token_lengths=torch.tensor([2]),
        )
        self.assertEqual(model._last_spatial_replay_stats, {})


@unittest.skipUnless(QWEN3_AVAILABLE, f"Qwen3 unavailable: {QWEN3_IMPORT_ERROR}")
class MixedReplayTrainerTests(unittest.TestCase):
    def test_freeze_policy_and_optimizer_group(self):
        from train_so_qa import build_optimizer, configure_mixed_replay_training

        model = _tiny_model(
            _tiny_config(enable_spatial_replay=True, spatial_null_num_tokens=4)
        )
        args = SimpleNamespace(
            train_moe_router=False,
            lr=1e-5,
            projector_lr=None,
            lora_lr=None,
            beats_lr=None,
            moe_router_lr=1e-6,
            spatial_null_lr=2e-5,
            projector_weight_decay=None,
            weight_decay=0.01,
        )
        enabled = configure_mixed_replay_training(model, args)
        self.assertIn("spatial_null", enabled)
        self.assertTrue(model.spatial_null.requires_grad)
        self.assertTrue(
            any(
                "so_projector" in name and param.requires_grad
                for name, param in model.named_parameters()
            )
        )

        optimizer = build_optimizer(model, args)
        null_groups = [
            group for group in optimizer.param_groups if group["name"].startswith("null_")
        ]
        self.assertEqual(len(null_groups), 1)
        self.assertEqual(null_groups[0]["lr"], 2e-5)
        self.assertEqual(len(null_groups[0]["params"]), 1)

    def test_strict_base_load_allows_spatial_null_missing(self):
        from train_so_qa_qwen3 import _validate_qwen3_base_loading_info

        info = {
            "missing_keys": {"so_projector.fc1.weight", "spatial_null"},
            "unexpected_keys": {"talker.model.embed_tokens.weight"},
            "mismatched_keys": set(),
            "error_msgs": set(),
        }
        _validate_qwen3_base_loading_info(info)

        broken = dict(info, missing_keys={"model.layers.0.self_attn.q_proj.weight"})
        with self.assertRaisesRegex(RuntimeError, "did not load cleanly"):
            _validate_qwen3_base_loading_info(broken)

    def test_wrapper_no_longer_rejects_mixed_replay(self):
        import train_so_qa_qwen3

        source = Path(train_so_qa_qwen3.__file__).read_text(encoding="utf-8")
        self.assertNotIn("does not yet implement mono mixed-spatial replay", source)


@unittest.skipUnless(QWEN3_AVAILABLE, f"Qwen3 unavailable: {QWEN3_IMPORT_ERROR}")
class SubsetFullReplayRecipeTests(unittest.TestCase):
    def test_train_subset_ratio_is_deterministic_and_sized(self):
        from train_so_qa import sample_subset_indices

        first = sample_subset_indices(1000, 0.2, seed=1234, epoch=0)
        second = sample_subset_indices(1000, 0.2, seed=1234, epoch=0)
        other_seed = sample_subset_indices(1000, 0.2, seed=4321, epoch=0)

        self.assertEqual(len(first), 200)
        self.assertEqual(first, second)
        self.assertNotEqual(first, other_seed)
        self.assertEqual(first, sorted(first))
        self.assertEqual(len(sample_subset_indices(1000, 1.0, seed=1234, epoch=0)), 1000)

    def test_subset_plus_full_replay_concat_layout(self):
        from torch.utils.data import ConcatDataset, Subset

        from train_so_qa import TaggedDataset, sample_subset_indices

        spatial = [{"id": i} for i in range(100)]
        replay = [{"id": i} for i in range(7)]
        subset = Subset(spatial, sample_subset_indices(100, 0.2, seed=1234, epoch=0))
        mixed = ConcatDataset([
            TaggedDataset(subset, has_spatial=True),
            TaggedDataset(replay, has_spatial=False),
        ])

        self.assertEqual(len(mixed), 27)
        self.assertTrue(mixed[0]["_replay_has_spatial"])
        self.assertFalse(mixed[26]["_replay_has_spatial"])

    def test_parser_accepts_and_validates_new_recipe_flags(self):
        import train_so_qa

        with mock.patch.object(
            sys,
            "argv",
            [
                "prog",
                "--train-subset-ratio", "0.2",
                "--mix-full-replay",
                "--replay-qa-roots", "replay.jsonl",
            ],
        ):
            args = train_so_qa.parse_args()
        self.assertEqual(args.train_subset_ratio, 0.2)
        self.assertTrue(args.mix_full_replay)
        self.assertTrue(args.mixed_spatial_replay)

        for bad_argv in (
            ["prog", "--train-subset-ratio", "1.5"],
            ["prog", "--mix-full-replay"],
        ):
            with mock.patch.object(sys, "argv", bad_argv):
                with self.assertRaises(SystemExit):
                    train_so_qa.parse_args()


@unittest.skipUnless(QWEN3_AVAILABLE, f"Qwen3 unavailable: {QWEN3_IMPORT_ERROR}")
class MixedReplayDataTests(unittest.TestCase):
    def _write_manifest(self, handle, records):
        for record in records:
            handle.write(json.dumps(record) + "\n")
        handle.flush()

    def test_dataset_normalizes_chatml_and_skips_unsupported_uris(self):
        from train_so_qa import QAAudioDataset

        chatml_oss = {
            "messages": [
                {"role": "user", "content": [
                    {"audio": "oss://bucket/a.wav"},
                    {"text": "hello"},
                ]},
                {"role": "assistant", "content": [{"text": "world"}]},
            ]
        }
        chatml_http = {
            "messages": [
                {"role": "user", "content": [
                    {"audio": "https://example.com/b.wav"},
                    {"text": "skip me"},
                ]},
                {"role": "assistant", "content": [{"text": "skip"}]},
            ]
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = os.path.join(temp_dir, "train.jsonl")
            with open(manifest, "w", encoding="utf-8") as handle:
                self._write_manifest(handle, [chatml_oss, chatml_http])
            dataset = QAAudioDataset(manifest)

        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset.skipped_unsupported_audio_uris, 1)
        record = dataset[0]
        self.assertEqual(record["audio_path"], "oss://bucket/a.wav")
        self.assertIn("hello", record["prompt"])
        self.assertEqual(record["answer"], "world")

    def test_dataset_rejects_non_chatml_remote_uri(self):
        from train_so_qa import QAAudioDataset

        flat_remote = {
            "audio_path": "https://example.com/c.wav",
            "prompt": "q",
            "answer": "a",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = os.path.join(temp_dir, "train.jsonl")
            with open(manifest, "w", encoding="utf-8") as handle:
                self._write_manifest(handle, [flat_remote])
            with self.assertRaisesRegex(ValueError, "unsupported audio URI scheme"):
                QAAudioDataset(manifest)

    def test_dataset_skips_overlength_text(self):
        from train_so_qa import QAAudioDataset

        short = {"audio_path": "a.wav", "prompt": "q", "answer": "a"}
        long_rec = {"audio_path": "b.wav", "prompt": "x" * 5000, "answer": "y" * 4000}
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = os.path.join(temp_dir, "train.jsonl")
            with open(manifest, "w", encoding="utf-8") as handle:
                self._write_manifest(handle, [short, long_rec])
            dataset = QAAudioDataset(manifest, max_text_chars=8000)
            uncapped = QAAudioDataset(manifest)

        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset.skipped_overlength_text, 1)
        self.assertEqual(len(uncapped), 2)
        self.assertEqual(uncapped.skipped_overlength_text, 0)

    def test_mixed_collator_batch_layout(self):
        from tests.test_qwen3_spatial_first_class import _build_test_processor
        from train_so_qa import SpatialBeatsQACollator

        processor = _build_test_processor()
        collator = SpatialBeatsQACollator(
            processor=processor,
            enable_mono_replay=True,
        )
        foa = np.zeros((24000, 4), dtype=np.float32)
        mono = np.zeros((16000, 1), dtype=np.float32)
        features = [
            {"audio_path": "spatial.wav", "prompt": "where", "answer": "left",
             "_replay_has_spatial": True},
            {"audio_path": "oss://bucket/mono.wav", "prompt": "what", "answer": "speech",
             "_replay_has_spatial": False},
        ]
        reads = {"spatial.wav": (foa, 16000), "oss://bucket/mono.wav": (mono, 16000)}
        with mock.patch.object(
            collator, "_read_audio", side_effect=lambda path: reads[path]
        ):
            batch = collator(features)

        self.assertEqual(batch["has_spatial"].tolist(), [True, False])
        self.assertEqual(tuple(batch["mono_audio"].shape), (2, collator.max_audio_samples))
        self.assertEqual(tuple(batch["spatial_audio"].shape)[0], 2)
        self.assertEqual(batch["spatial_audio_lengths"].tolist(), [24000, 0])
        self.assertEqual(batch["mono_audio_lengths"].tolist(), [24000, 16000])
        # Placeholder count per row matches the declared spatial token lengths.
        counts = (batch["input_ids"] == processor.spatial_token_id).sum(dim=1)
        self.assertEqual(counts.tolist(), batch["spatial_token_lengths"].tolist())
        self.assertGreater(int(counts[0]), 0)
        self.assertGreater(int(counts[1]), 0)
        # Labels mask everything except the answer suffix.
        self.assertTrue(int((batch["labels"][0] != -100).sum()) > 0)
        self.assertEqual(
            [meta["has_spatial"] for meta in batch["meta"]], [True, False]
        )

    def test_collator_routes_oss_reads_through_reader(self):
        from tests.test_qwen3_spatial_first_class import _build_test_processor
        from train_so_qa import SpatialBeatsQACollator
        from spatial_omni.utils.replay_data import OssAudioReader

        collator = SpatialBeatsQACollator(
            processor=_build_test_processor(),
            oss_config="/nonexistent/oss.ini",
        )
        self.assertFalse(OssAudioReader.is_oss_path("/tmp/a.wav"))
        self.assertTrue(OssAudioReader.is_oss_path("oss://bucket/a.wav"))
        fake = np.zeros((8000, 1), dtype=np.float32)
        with mock.patch.object(
            OssAudioReader, "read_audio", return_value=(fake, 16000)
        ) as read_audio:
            wav, sr = collator._read_audio("oss://bucket/a.wav")
        self.assertEqual(sr, 16000)
        self.assertEqual(tuple(wav.shape), (8000, 1))
        read_audio.assert_called_once()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    unittest.main(verbosity=2)
