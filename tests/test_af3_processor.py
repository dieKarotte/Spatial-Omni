"""Offline processor regression checks; model weights and GPUs are not used."""
import os
import unittest
from unittest.mock import patch

import numpy as np

import so_audiollm.af3 as af3
from so_audiollm.common import spatial_token_count, pack_spatial_batch

class AF3ProcessorTest(unittest.TestCase):
    def test_token_rounding_and_padding(self):
        self.assertEqual([spatial_token_count(n) for n in [8000, 16000, 24000, 319999, 320000]],
                         [1, 2, 4, 50, 50])
        audio, mask, lengths, tokens = pack_spatial_batch([np.zeros((16000, 4), np.float32),
                                                         np.zeros((8000, 4), np.float32)])
        self.assertEqual(lengths.tolist(), [16000, 8000])
        self.assertEqual(tokens.tolist(), [2, 1])
        self.assertEqual(mask.sum(1).tolist(), [16000, 8000])
        with self.assertRaises(ValueError):
            pack_spatial_batch([np.zeros((16000, 2), np.float32)])

    @unittest.skipUnless(os.environ.get("SO_AF3_MODEL_DIR"), "Set SO_AF3_MODEL_DIR to a local base model")
    def test_spatial_and_replay_batch(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(os.environ["SO_AF3_MODEL_DIR"], "llm"),
                                                  local_files_only=True)
        tokenizer.add_special_tokens({"additional_special_tokens": [af3.SPATIAL_TOKEN]})
        features = [{"audio_path": "foa", "prompt": "Where is the sound?", "answer": "On the left."},
                    {"audio_path": "mono", "prompt": "What is heard?", "answer": "A bell.", "has_spatial": False}]
        with patch.object(af3, "load_foa", return_value=np.zeros((16000, 4), np.float32)), \
             patch.object(af3, "load_mono", return_value=np.zeros(8000, np.float32)):
            batch = af3.SoAF3Collator(tokenizer)(features)
        self.assertEqual(batch["has_spatial"].tolist(), [True, False])
        self.assertEqual(batch["sound_lengths"].tolist(), [50, 20])
        self.assertEqual((batch["input_ids"] == tokenizer.convert_tokens_to_ids(af3.SPATIAL_TOKEN)).sum(1).tolist(), [2, 1])
        self.assertEqual(tuple(batch["sound_mel"].shape), (2, 128, 3000))
        self.assertEqual(int(batch["spatial_audio"][1, :, 1:].count_nonzero()), 0)
        self.assertTrue(bool((batch["labels"][batch["attention_mask"] == 0] == -100).all()))
        self.assertTrue(bool((batch["labels"] != -100).any(1).all()))

if __name__ == "__main__":
    unittest.main()
