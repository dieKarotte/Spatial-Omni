"""Checkpoint completeness across staged training and replay initialization."""
import unittest
import torch
from train_so_phi4 import load_trainable_weights


class StageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.so_projector = torch.nn.Linear(2, 2)
        self.so_encoder = torch.nn.Linear(2, 2)
        self.lora_A = torch.nn.Linear(2, 2)
        self.spatial_null = torch.nn.Parameter(torch.zeros(2, 2))


class CheckpointLoadingTest(unittest.TestCase):
    def test_complete_checkpoint_loads(self):
        model = StageModel()
        self.assertEqual(load_trainable_weights(model, model.state_dict()), ([], []))

    def test_projector_only_stage_can_initialize_new_components(self):
        model = StageModel()
        state = {k: v for k, v in model.state_dict().items() if k.startswith("so_projector.")}
        missing, unexpected = load_trainable_weights(model, state)
        self.assertIn("spatial_null", missing)
        self.assertEqual(unexpected, [])

    def test_partial_trained_component_is_rejected(self):
        for name in ["so_projector.bias", "so_encoder.bias", "lora_A.bias"]:
            with self.subTest(name=name):
                model = StageModel()
                state = dict(model.state_dict())
                del state[name]
                with self.assertRaisesRegex(ValueError, "missing trained keys"):
                    load_trainable_weights(model, state)

    def test_unexpected_checkpoint_key_is_rejected(self):
        model = StageModel()
        state = dict(model.state_dict())
        state["wrong_backend.weight"] = torch.zeros(1)
        with self.assertRaisesRegex(ValueError, "unexpected"):
            load_trainable_weights(model, state)


if __name__ == "__main__":
    unittest.main()
