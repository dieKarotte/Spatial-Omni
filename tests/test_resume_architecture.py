"""Resume must rebuild the architecture recorded beside a checkpoint."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from spatial_omni.utils.release import apply_checkpoint_architecture

class ResumeArchitectureTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.ckpt=self.root/"model.pt"
        self.ckpt.touch()
        self.saved={"lora_r":16,"lora_alpha":32,"lora_target_modules":["q_proj","k_proj","v_proj","o_proj"],
                    "lora_target_prefixes":["thinker.model"],"train_mode":"beats_lora",
                    "mixed_spatial_replay":True,"projector_type":"pixel_shuffle",
                    "projector_shuffle_factor":4,"encoder_token_rate":10.0}
        (self.root/"train_args.json").write_text(json.dumps(self.saved))
    def args(self):
        return SimpleNamespace(resume_checkpoint_path=str(self.ckpt),resume_tag=None,
                               output_dir=str(self.root),lora_r=8,lora_alpha=16,
                               lora_target_modules=["q_proj","down_proj"],train_mode="projector_only")
    def test_released_attention_lora_replaces_default_mlp_targets(self):
        args=self.args()
        apply_checkpoint_architecture(args,[])
        self.assertEqual(args.lora_target_modules,self.saved["lora_target_modules"])
        self.assertEqual(args.train_mode,"beats_lora")
        self.assertTrue(args.mixed_spatial_replay)
    def test_explicit_architecture_and_stage_transition_win(self):
        args=self.args()
        apply_checkpoint_architecture(args,["--lora-r=8","--lora-target-modules","q_proj","down_proj","--projector-only"])
        self.assertEqual(args.lora_r,8)
        self.assertEqual(args.lora_target_modules,["q_proj","down_proj"])
        self.assertEqual(args.train_mode,"projector_only")
    def test_training_run_layout(self):
        directory=self.root/"checkpoints";directory.mkdir()
        args=self.args();args.resume_checkpoint_path=str(directory/"last_trainable.pt")
        apply_checkpoint_architecture(args,[])
        self.assertEqual(args.lora_target_modules,self.saved["lora_target_modules"])
    def test_from_scratch_defaults_are_unchanged(self):
        args=self.args();args.resume_checkpoint_path=None
        apply_checkpoint_architecture(args,[])
        self.assertEqual(args.lora_target_modules,["q_proj","down_proj"])
        self.assertEqual(args.train_mode,"projector_only")
if __name__=="__main__":unittest.main()
