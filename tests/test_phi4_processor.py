"""Offline processor regression checks; model weights and GPUs are not used."""
import os
import unittest
from unittest.mock import patch

import numpy as np

from so_phi4.processing import SoPhi4Processor
from so_phi4.collator import SoPhi4QACollator

class Phi4ProcessorTest(unittest.TestCase):
    def test_token_rounding(self):
        self.assertEqual([SoPhi4Processor.spatial_token_count(n) for n in
                          [8000, 16000, 24000, 319999, 320000]], [1, 2, 4, 50, 50])

    @unittest.skipUnless(os.environ.get("SO_PHI4_MODEL_DIR"), "Set SO_PHI4_MODEL_DIR to a local base model")
    def test_spatial_and_replay_batch(self):
        from transformers import AutoProcessor
        base = AutoProcessor.from_pretrained(os.environ["SO_PHI4_MODEL_DIR"],
                                            trust_remote_code=True, local_files_only=True)
        processor = SoPhi4Processor(base)
        features = [{"audio_path": "foa", "prompt": "Where is the sound?", "answer": "On the left."},
                    {"audio_path": "mono", "prompt": "What is heard?", "answer": "A bell.", "has_spatial": False}]
        with patch("so_phi4.collator._load_foa", return_value=np.zeros((16000, 4), np.float32)), \
             patch("so_phi4.collator._load_mono", return_value=np.zeros(8000, np.float32)):
            batch = SoPhi4QACollator(processor)(features)
        self.assertEqual(batch["has_spatial"].tolist(), [True, False])
        self.assertEqual((batch["input_ids"] == processor.spatial_token_id).sum(1).tolist(), [2, 1])
        self.assertEqual(int(batch["spatial_audio"][1, :, 1:].count_nonzero()), 0)
        self.assertTrue(bool((batch["labels"][batch["attention_mask"] == 0] == -100).all()))
        self.assertTrue(bool((batch["labels"] != -100).any(1).all()))

if __name__ == "__main__":
    unittest.main()
