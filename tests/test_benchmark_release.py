"""Benchmark routing and output preservation without loading model weights."""
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

from scripts import run_bench
from scripts import bench_test_generate


class BenchmarkReleaseTests(unittest.TestCase):
    def test_main_dispatcher_only_advertises_existing_entrypoints(self):
        root = Path(__file__).resolve().parents[1]
        self.assertNotIn("af3", run_bench.BASELINE_TO_MODULE)
        self.assertNotIn("so-30b", run_bench.BASELINE_TO_MODULE)
        for module, _ in run_bench.BASELINE_TO_MODULE.values():
            self.assertTrue((root / (module.replace(".", "/") + ".py")).is_file())

    def test_release_paths_and_audio_root_are_forwarded(self):
        args = SimpleNamespace(baseline="so-7b", model_id="base", beats_checkpoint="encoder.pt",
                               audio_root="audio", attn_impl="sdpa", device_map=None)
        argv = run_bench.build_sub_argv(args, "scripts.bench_test_generate", [])
        self.assertEqual(argv, ["--model-id", "base", "--beats-checkpoint", "encoder.pt",
                                "--audio-root", "audio", "--attn-impl", "sdpa"])

    def test_generated_qa_id_aligns_with_scorer_despite_local_audio_path(self):
        import torch
        from scripts.score_sobench import prepare_records

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))
            def generate(self, input_ids, **kwargs):
                return torch.cat([input_ids, torch.tensor([[9]])], dim=1)

        batch = {"gen_input_ids": torch.tensor([[1, 2]]),
                 "gen_attention_mask": torch.ones(1, 2, dtype=torch.long),
                 "meta": [{"qa_id": "qa-stable", "pair_id": "legacy",
                           "audio_path": "/local/audio.wav", "task_name": "count",
                           "question": "How many?", "answer": "one"}]}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            qa = {"qa_id": "qa-stable", "audio_path": "audio.wav",
                  "task_name": "count", "question": "How many?", "answer": "one"}
            (root / "test.jsonl").write_text(json.dumps(qa) + "\n")
            output = root / "predictions.jsonl"
            with mock.patch.object(bench_test_generate, "tqdm", side_effect=lambda x, **kwargs: x):
                bench_test_generate.run_generation_bench_with_ablation(
                    Model(), SimpleNamespace(tokenizer=SimpleNamespace(decode=lambda *a, **k: "one")),
                    [batch], str(output), 1, 1, False, "test")
            records = prepare_records(SimpleNamespace(qa_root=str(root), split="test",
                                                       predictions_jsonl=str(output), limit=None))
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["qa_id"], "qa-stable")

    def test_final_merge_cannot_replace_another_completed_job(self):
        from scripts.batch_bench_so_qa import finalize_distributed_prediction_file
        with tempfile.TemporaryDirectory() as temp:
            final = Path(temp) / "predictions.jsonl"
            shard = Path(str(final) + ".rank0.jsonl")
            final.write_text("completed job\n")
            shard.write_text('{"prediction":"new job"}\n')
            with self.assertRaises(FileExistsError):
                finalize_distributed_prediction_file(str(final))
            self.assertEqual(final.read_text(), "completed job\n")
            self.assertTrue(shard.exists())

    def test_existing_predictions_are_preserved_before_model_load(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            qa = root / "qa"
            qa.mkdir()
            (qa / "test.jsonl").write_text(json.dumps(
                {"audio_path": "unused.wav", "question": "What?", "answer": "a"}) + "\n")
            ckpt = root / "model.pt"
            ckpt.touch()
            output = root / "out"
            prediction = output / "model" / "predictions.jsonl"
            prediction.parent.mkdir(parents=True)
            prediction.write_text("precious output\n")
            argv = ["bench", "--qa-root", str(qa), "--checkpoint-paths", str(ckpt),
                    "--output-dir", str(output), "--device", "cpu"]
            with mock.patch.object(sys, "argv", argv), \
                 mock.patch.object(bench_test_generate, "instantiate_model_for_checkpoint") as load:
                with self.assertRaises(FileExistsError):
                    bench_test_generate.main()
                load.assert_not_called()
            self.assertEqual(prediction.read_text(), "precious output\n")


if __name__ == "__main__":
    unittest.main()
