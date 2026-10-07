"""Curriculum changes may add components but must not silently drop partial weights."""
import unittest
import torch
from spatial_omni.utils.release import load_curriculum_state

class CurriculumCheckpointTests(unittest.TestCase):
    def model(self):
        model=torch.nn.Module()
        model.so_encoder=torch.nn.Linear(2,2)
        model.so_projector=torch.nn.Linear(2,2)
        model.lora_adapter=torch.nn.Linear(2,2)
        model.spatial_null=torch.nn.Parameter(torch.zeros(2,2))
        return model
    def test_projector_warmup_can_initialize_later_stage(self):
        model=self.model()
        state={k:v.clone() for k,v in model.state_dict().items() if "so_projector." in k}
        load_curriculum_state(model,state)
    def test_partial_encoder_is_rejected(self):
        model=self.model();state=model.state_dict();del state["so_encoder.weight"]
        with self.assertRaisesRegex(RuntimeError,"so_encoder.weight"):load_curriculum_state(model,state)
    def test_missing_projector_is_rejected(self):
        model=self.model();state=model.state_dict();del state["so_projector.weight"]
        with self.assertRaisesRegex(RuntimeError,"so_projector.weight"):load_curriculum_state(model,state)
    def test_extra_lora_name_is_rejected(self):
        model=self.model();state=model.state_dict();state["wrong.lora_A.weight"]=torch.zeros(2,2)
        with self.assertRaisesRegex(RuntimeError,"wrong.lora_A"):load_curriculum_state(model,state)
if __name__=="__main__":unittest.main()
