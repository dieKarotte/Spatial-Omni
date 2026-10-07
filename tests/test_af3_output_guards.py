"""Check existing outputs are rejected before any model or GPU initialization."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

MODEL_ARG = '--model-dir'
TRAINER = 'train_so_af3.py'
EVALUATOR = 'so_audiollm/eval_af3_sobench.py'
PREDICTIONS = 'predictions_shard0.jsonl'

class OutputGuardTest(unittest.TestCase):
    def assert_rejected(self, entrypoint, output, extra):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
        command = [sys.executable, "-B", str(ROOT / entrypoint), MODEL_ARG, "unused-base",
                   "--qa-root", "unused-qa", "--audio-root", "unused-audio",
                   "--output-dir", str(output)] + extra
        result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("FileExistsError", result.stderr)
        self.assertNotIn("CUDA error", result.stderr)

    def test_existing_training_run_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "train_args.json"
            path.write_text('{"keep": true}')
            self.assert_rejected(TRAINER, folder, ["--beats-checkpoint", "unused-encoder"])
            self.assertEqual(path.read_text(), '{"keep": true}')

    def test_existing_predictions_are_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / PREDICTIONS
            path.write_text('{"prediction": "keep"}\n')
            self.assert_rejected(EVALUATOR, folder, ["--checkpoint", "unused-checkpoint"])
            self.assertEqual(path.read_text(), '{"prediction": "keep"}\n')

if __name__ == "__main__":
    unittest.main()
