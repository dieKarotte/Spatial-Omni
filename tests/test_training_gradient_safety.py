"""Finite-gradient checks for the real training loop."""
import sys
from pathlib import Path
import unittest
from unittest import mock
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import train_so_qa as trainer

class GradientSafetyTests(unittest.TestCase):
    def run_case(self, nonfinite):
        model=torch.nn.Linear(1,1,bias=False)
        with torch.no_grad():model.weight.fill_(0.0 if nonfinite else 1.0)
        opt=torch.optim.SGD(model.parameters(),lr=0.1)
        writer=mock.Mock()
        def loss(*args):
            value=model.weight.sqrt().sum() if nonfinite else model.weight.square().sum()
            return value,{"loss":float(value.detach()),"supervised_tokens":1}
        with mock.patch.object(trainer,"compute_batch_loss",side_effect=loss):
            result=trainer.train_one_epoch(model,[{}],opt,None,"cpu",1,1.0,1,1,True,writer=writer)
        return model,writer,result
    def test_nonfinite_gradient_fails_before_update(self):
        with self.assertRaisesRegex(RuntimeError,"non-finite"):
            self.run_case(True)
    def test_finite_gradient_updates_parameter_and_logs_norm(self):
        model,writer,_=self.run_case(False)
        self.assertAlmostEqual(model.weight.item(),0.9,places=5)
        events=[c for c in writer.add_scalar.call_args_list if c.args[0]=="train/grad_norm"]
        self.assertEqual(len(events),1)
        self.assertAlmostEqual(events[0].args[1],2.0)
if __name__=="__main__":unittest.main()
