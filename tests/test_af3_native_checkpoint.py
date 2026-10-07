"""Native AF3 tower weights must be complete; no real weights are loaded."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
import so_audiollm.af3 as af3

class NativeTowerCheckpointTest(unittest.TestCase):
    def build_with_state(self, state):
        llm = torch.nn.Module()
        llm.config = SimpleNamespace(hidden_size=2)
        tower = torch.nn.Linear(2, 2)
        spatial = torch.nn.Module()
        spatial.build = Mock()
        spatial.reset_projector = Mock()
        projector = {
            "layers.0.weight": torch.zeros(2, 2), "layers.0.bias": torch.zeros(2),
            "layers.2.weight": torch.zeros(2, 2), "layers.2.bias": torch.zeros(2),
        }
        with patch.object(af3.Qwen2ForCausalLM, "from_pretrained", return_value=llm), \
             patch.object(af3.WhisperConfig, "from_pretrained", return_value=SimpleNamespace(d_model=2)), \
             patch.object(af3, "WhisperEncoder", return_value=tower), \
             patch.object(af3, "load_file", side_effect=[state, projector]), \
             patch.object(af3, "SpatialBranch", return_value=spatial):
            return af3.SoAF3Model("unused-base", "unused-encoder", None)

    def test_missing_native_weight_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Missing key"):
            self.build_with_state({"bias": torch.zeros(2)})

    def test_complete_native_weights_load(self):
        model = self.build_with_state({"weight": torch.ones(2, 2), "bias": torch.zeros(2)})
        self.assertTrue(torch.equal(model.sound_tower.weight.float(), torch.ones(2, 2)))

if __name__ == "__main__":
    unittest.main()
