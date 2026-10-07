"""Regression tests for left-padded validation generation."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_so_qa as trainer


class LeftPaddedGenerationTests(unittest.TestCase):
    def test_padding_preserves_prefixes_and_masks(self):
        ids = torch.tensor([[11, 12, 90, 0], [21, 22, 23, 91]])
        attention = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
        prefix_lengths = torch.tensor([2, 3])
        padded, mask = trainer.build_left_padded_batch(ids, attention, prefix_lengths, 0)
        self.assertEqual(padded.tolist(), [[0, 11, 12], [21, 22, 23]])
        self.assertEqual(mask.tolist(), [[0, 1, 1], [1, 1, 1]])

    def test_validation_decodes_only_new_tokens_for_every_prefix_length(self):
        class Model(torch.nn.Module):
            def generate(self, input_ids, **kwargs):
                answers = torch.tensor([[31, 32], [41, 42]])
                return torch.cat([input_ids, answers], dim=1)

        tokenizer = SimpleNamespace(decode=lambda tokens, **kwargs: " ".join(map(str, tokens.tolist())))
        batch = {
            "gen_input_ids": torch.tensor([[0, 11, 12], [21, 22, 23]]),
            "gen_attention_mask": torch.tensor([[0, 1, 1], [1, 1, 1]]),
            "meta": [{"pair_id": "a", "answer": "31 32"}, {"pair_id": "b", "answer": "41 42"}],
        }
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.object(trainer, "tqdm", side_effect=lambda values, **kwargs: values):
            trainer.run_validation_generation(
                Model(), SimpleNamespace(tokenizer=tokenizer), [batch], "cpu", 1,
                directory, max_new_tokens=2, num_beams=1, do_sample=False,
            )
            path = Path(directory) / "valid_predictions" / "epoch_001.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual([row["prediction"] for row in rows], ["31 32", "41 42"])
        self.assertEqual([row["exact_match"] for row in rows], [1, 1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
