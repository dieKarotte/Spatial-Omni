"""Regression coverage for complete trainable components at curriculum boundaries."""
import unittest
import torch
from spatial_omni.utils.release import load_curriculum_state

class CurriculumResumeTests(unittest.TestCase):
    def model(self):
        m = torch.nn.Module()
        m.so_projector = torch.nn.Linear(2, 2)
        m.so_encoder = torch.nn.Linear(2, 2)
        m.lora_A = torch.nn.Linear(2, 2, bias=False)
        m.spatial_null = torch.nn.Parameter(torch.zeros(2, 2))
        m.layers = torch.nn.ModuleList([torch.nn.Module(), torch.nn.Module()])
        for layer in m.layers:
            layer.mlp = torch.nn.Module()
            layer.mlp.gate = torch.nn.Linear(2, 2, bias=False)
        return m

    def test_stage_one_can_initialize_new_encoder_lora_router_and_null(self):
        model = self.model()
        state = {k:v.clone() for k,v in model.state_dict().items() if k.startswith("so_projector.")}
        load_curriculum_state(model,state)

    def test_partial_encoder_and_router_are_rejected(self):
        for missing in ("so_encoder.weight", "layers.1.mlp.gate.weight"):
            with self.subTest(missing=missing):
                model=self.model()
                state=dict(model.state_dict())
                del state[missing]
                with self.assertRaisesRegex(RuntimeError,"Incompatible training checkpoint"):
                    load_curriculum_state(model,state)

    def test_missing_projector_and_unexpected_parameters_are_rejected(self):
        for invalid in ("missing", "unexpected"):
            with self.subTest(invalid=invalid):
                model=self.model()
                state=dict(model.state_dict())
                if invalid=="missing": del state["so_projector.weight"]
                else: state["wrong.weight"]=torch.zeros(2,2)
                with self.assertRaisesRegex(RuntimeError,"Incompatible training checkpoint"):
                    load_curriculum_state(model,state)

    def test_complete_adapter_is_restored(self):
        model=self.model()
        state={k:torch.ones_like(v) for k,v in model.state_dict().items()}
        load_curriculum_state(model,state)
        self.assertTrue(all(torch.equal(v,torch.ones_like(v)) for v in model.state_dict().values()))

if __name__=="__main__":
    unittest.main()
