"""Regression tests for portable release configuration and parameter loading."""
import json
import tempfile
import unittest
from pathlib import Path

import torch

from spatial_omni.utils.release import load_release_settings, load_release_state


class ReleaseSettingsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.variant = self.root / "SO-7B" / "mix"
        self.variant.mkdir(parents=True)
        self.checkpoint = self.variant / "SO-7B-MIX.pt"
        self.checkpoint.touch()
        encoder = self.root / "SO-Encoder" / "SO-Encoder.pt"
        encoder.parent.mkdir()
        encoder.touch()
        self.encoder = encoder
        self.settings = {"model_id": "upstream/base", "beats_checkpoint": "../../SO-Encoder/SO-Encoder.pt",
                         "mixed_spatial_replay": True, "so_repo": "", "beats_repo": ""}
        (self.variant / "train_args.json").write_text(json.dumps(self.settings))

    def test_release_paths_are_relative_to_settings(self):
        result = load_release_settings(self.checkpoint)
        self.assertEqual(result["beats_checkpoint"], str(self.encoder.resolve()))
        self.assertTrue(result["mixed_spatial_replay"])
        self.assertEqual(result["model_id"], "upstream/base")

    def test_variant_settings_win_over_parent(self):
        (self.variant.parent / "train_args.json").write_text(json.dumps({"wrong": True}))
        self.assertTrue(load_release_settings(self.checkpoint)["mixed_spatial_replay"])

    def test_explicit_paths_override_recorded_paths(self):
        result = load_release_settings(self.checkpoint, model_id="another/base",
                                       encoder_checkpoint=self.encoder)
        self.assertEqual(result["model_id"], "another/base")
        self.assertEqual(result["beats_checkpoint"], str(self.encoder.resolve()))

    def test_missing_encoder_fails_before_model_download(self):
        (self.variant / "train_args.json").write_text(json.dumps(
            dict(self.settings, beats_checkpoint="missing.pt")))
        with self.assertRaisesRegex(FileNotFoundError, "SO-Encoder"):
            load_release_settings(self.checkpoint)

    def test_training_run_layout_remains_supported(self):
        run = self.root / "run"
        (run / "checkpoints").mkdir(parents=True)
        checkpoint = run / "checkpoints" / "best_trainable.pt"
        checkpoint.touch()
        (run / "train_args.json").write_text(json.dumps(
            dict(self.settings, beats_checkpoint=str(self.encoder))))
        self.assertTrue(load_release_settings(checkpoint)["mixed_spatial_replay"])


class ReleaseStateTests(unittest.TestCase):
    def model(self):
        model = torch.nn.Module()
        model.base = torch.nn.Linear(2, 2)
        model.base.requires_grad_(False)
        model.so_projector = torch.nn.Linear(2, 2)
        model.spatial_null = torch.nn.Parameter(torch.zeros(2, 2), requires_grad=False)
        return model

    def state(self, model):
        return {k: torch.ones_like(v) for k, v in model.state_dict().items()
                if not k.startswith("base.")}

    def test_frozen_base_can_be_absent(self):
        model = self.model()
        load_release_state(model, self.state(model))
        self.assertTrue(torch.equal(model.so_projector.weight, torch.ones(2, 2)))

    def test_missing_projector_is_rejected(self):
        model = self.model()
        state = self.state(model)
        del state["so_projector.weight"]
        with self.assertRaisesRegex(RuntimeError, "so_projector.weight"):
            load_release_state(model, state)

    def test_missing_mix_null_is_rejected_even_when_frozen(self):
        model = self.model()
        state = self.state(model)
        del state["spatial_null"]
        with self.assertRaisesRegex(RuntimeError, "spatial_null"):
            load_release_state(model, state)

    def test_unexpected_key_is_rejected(self):
        model = self.model()
        state = self.state(model)
        state["wrong_adapter.weight"] = torch.zeros(2, 2)
        with self.assertRaisesRegex(RuntimeError, "wrong_adapter"):
            load_release_state(model, state)


if __name__ == "__main__":
    unittest.main()
