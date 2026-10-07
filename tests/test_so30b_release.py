"""Release entrypoint and checkpoint-loading contracts; no model assets required."""
import argparse
import importlib.util
from pathlib import Path
import sys
import os
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import batch_bench_so_qa as bench


class ReleaseCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.runtime = argparse.Namespace(device="cpu", dtype="float32")
        self.model = torch.nn.Module()
        self.model.so_projector = torch.nn.Linear(2, 2)
        self.model.spatial_null = torch.nn.Parameter(torch.ones(2, 2))
        self.settings = {"model_id": "unused", "so_repo": str(ROOT),
                         "beats_checkpoint": "unused", "beats_repo": str(ROOT),
                         "train_mode": "projector_only"}
        self.processor = SimpleNamespace(tokenizer=SimpleNamespace(padding_side="right"))

    def load(self, state, **runtime_values):
        for key, value in runtime_values.items():
            setattr(self.runtime, key, value)
        with mock.patch("spatial_omni.utils.release.load_release_settings", return_value=self.settings), \
             mock.patch.object(bench, "build_processor", return_value=self.processor), \
             mock.patch.object(bench, "build_model", return_value=self.model), \
             mock.patch.object(bench.torch, "load", return_value={"trainable_state_dict": state}) as load:
            result = bench.instantiate_model_for_checkpoint(self.runtime, "unused.pt")
        return result, load.call_args.kwargs

    def test_loader_returns_six_values_and_uses_safe_pickle_default(self):
        result, kwargs = self.load(self.model.state_dict())
        self.assertEqual(len(result), 6)
        self.assertEqual(result[5]["missing_trainable"], 0)
        self.assertTrue(kwargs["weights_only"])

    def test_frozen_mix_null_parameter_must_still_be_present(self):
        state = dict(self.model.state_dict())
        state.pop("spatial_null")
        with self.assertRaisesRegex(RuntimeError, "spatial_null"):
            self.load(state)

    def test_legacy_pickle_requires_explicit_flag(self):
        _, kwargs = self.load(self.model.state_dict(), trust_checkpoint=True)
        self.assertFalse(kwargs["weights_only"])

    def test_wrapper_accepts_public_trainer_namespace(self):
        path = next(ROOT.glob("train_so_qa_*3.py"))
        spec = importlib.util.spec_from_file_location("so30b_release_wrapper", path)
        wrapper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(wrapper)
        trainer = wrapper._trainer
        with mock.patch.object(trainer, "parse_args", return_value=SimpleNamespace()), \
             mock.patch.object(trainer, "main", side_effect=lambda: trainer.parse_args()), \
             mock.patch.object(trainer, "build_processor"), \
             mock.patch.object(trainer, "build_model"), \
             mock.patch.object(sys, "argv", [str(path)]), \
             mock.patch.dict("os.environ", {}, clear=False):
            args = wrapper.main()
        self.assertTrue(hasattr(args, "device_map"))


class ReplayLauncherTests(unittest.TestCase):
    def test_check_mode_validates_inputs_without_creating_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "model").mkdir()
            (root / "encoder.pt").touch()
            (root / "adaptation.pt").touch()
            (root / "data" / "qa").mkdir(parents=True)
            (root / "data" / "qa" / "train.jsonl").write_text("{}\n")
            (root / "data" / "qa" / "valid.jsonl").write_text("{}\n")
            (root / "replay.jsonl").write_text("{}\n")
            (root / "audio").mkdir()
            env = os.environ.copy()
            env.update(PYTHON_BIN=sys.executable, SO_BASE_MODEL=str(root / "model"),
                       SO_ENCODER_CKPT=str(root / "encoder.pt"), SO_DATASET_ROOT=str(root / "data"),
                       REPLAY_QA_ROOT=str(root / "replay.jsonl"), REPLAY_AUDIO_ROOT=str(root / "audio"),
                       RESUME_CKPT=str(root / "adaptation.pt"), OUTPUT_DIR=str(root / "output"))
            result = subprocess.run(["bash", str(ROOT / "shell" / "launch_train_so_30b_mix.sh"), "check"],
                                    env=env, capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("[check-only]", result.stdout)
            self.assertFalse((root / "output").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
