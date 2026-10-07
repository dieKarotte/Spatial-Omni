"""SO-Bench metric and CLI regression checks; no model or judge service required."""
import argparse
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("score_sobench", ROOT / "scripts" / "score_sobench.py")
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)


def metric_args(**overrides):
    values = dict(angle_threshold_deg=20.0, elevation_threshold_deg=10.0,
                  distance_threshold_m=1.0, time_threshold_s=0.2,
                  detect_source_time_policy="event_only",
                  spatial_temporal_time_policy="semantic_times_iou",
                  accept_broader_source=False)
    values.update(overrides)
    return argparse.Namespace(**values)


def record(task, meta=None, **values):
    result = dict(pair_id="sample", task_name=task, answer="reference",
                  answer_meta=meta or {}, prediction="prediction")
    result.update(values)
    return result


class MetricTests(unittest.TestCase):
    def test_distance_is_absolute_one_meter(self):
        near = scorer.evaluate_record(record("estimate_distance", {"distance_m": 1}),
                                      {"prediction": {"distance_m": 1.9}}, metric_args())
        far = scorer.evaluate_record(record("estimate_distance", {"distance_m": 10}),
                                     {"prediction": {"distance_m": 11.1}}, metric_args())
        self.assertEqual(near["correct"], 1.0)
        self.assertEqual(far["correct"], 0.0)

    def test_angle_wrap_and_elevation_threshold(self):
        result = scorer.evaluate_record(record("estimate_azimuth", {"azimuth_deg": 350}),
                                        {"prediction": {"angle_deg": 10}}, metric_args())
        self.assertEqual(result["doaerr_deg"], 20.0)
        self.assertEqual(result["correct"], 1.0)
        result = scorer.evaluate_record(record("estimate_elevation", {"elevation_deg": 0}),
                                        {"prediction": {"elevation_deg": 11}}, metric_args())
        self.assertEqual(result["correct"], 0.0)

    def test_onset_threshold_is_point_two_seconds(self):
        for prediction, expected in [(1.2, 1.0), (1.201, 0.0)]:
            with self.subTest(prediction=prediction):
                result = scorer.evaluate_record(record("onset_from_location", {"onset_time": 1}),
                                                {"prediction": {"onset_time_s": prediction}}, metric_args())
                self.assertEqual(result["correct"], expected)

    def test_time_scores_are_fractional_iou(self):
        result = scorer.evaluate_record(record("detect_time", {"time_span": [0, 2]}),
                                        {"prediction": {"time_span": [1, 3]}}, metric_args())
        self.assertAlmostEqual(result["correct"], 1 / 3)

    def test_source_and_spatial_temporal_policies(self):
        judgement = {"prediction": {"time_span": [1, 3]},
                     "semantic": {"source_match": True, "source_match_level": "exact_or_synonym", "direction_match": True}}
        event = scorer.evaluate_record(record("detect_source", {"time_span": [0, 2]}), judgement, metric_args())
        joint = scorer.evaluate_record(record("spatial_temporal", {"time_span": [0, 2]}), judgement, metric_args())
        self.assertEqual(event["correct"], 1.0)
        self.assertAlmostEqual(joint["correct"], 1 / 3)
        judgement["semantic"]["source_match_level"] = "compatible_but_broader"
        self.assertFalse(scorer.source_correct(judgement, accept_broader=False))
        self.assertTrue(scorer.source_correct(judgement, accept_broader=True))

    def test_speech_and_aggregate_are_not_plain_accuracy(self):
        speech = scorer.evaluate_record(record("speech_content", answer="a dog barks", prediction="a cat barks"), {}, metric_args())
        temporal = scorer.evaluate_record(record("detect_time", {"time_span": [0, 2]}),
                                          {"prediction": {"time_span": [1, 3]}}, metric_args())
        self.assertAlmostEqual(speech["wer"], 1 / 3)
        self.assertEqual(speech["correct"], 1.0)
        summary = scorer.summarize([speech, temporal])
        self.assertAlmostEqual(summary["task_aware_accuracy"], 2 / 3)
        self.assertAlmostEqual(summary["per_task"]["speech_content"]["wer_mean"], 1 / 3)

    def test_fallback_cache_is_not_treated_as_judge_output(self):
        self.assertTrue(scorer.cache_row_has_api_error({"judgement": {"notes": "local regex fallback only"}}))
        self.assertTrue(scorer.cache_row_has_api_error({"judgement": {"notes": "api_error=test"}}))
        self.assertFalse(scorer.cache_row_has_api_error({"judgement": {"notes": "valid judgement"}}))


class CliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="so-bench-test-")
        self.root = Path(self.temp.name)
        self.qa = self.root / "qa"
        self.qa.mkdir()
        self.rows = [record("estimate_distance", {"distance_m": 1}, pair_id="d", question="Distance?", audio_path="d.wav"),
                     record("speech_content", pair_id="s", question="Transcript?", audio_path="s.wav", answer="a dog barks", prediction="a dog barks")]
        self.predictions = self.root / "predictions.jsonl"
        for path in [self.qa / "test.jsonl", self.predictions]:
            path.write_text("".join(json.dumps(row) + "\n" for row in self.rows))
        self.arguments = ["score_sobench.py", "--predictions-jsonl", str(self.predictions), "--qa-root", str(self.qa),
                          "--output-json", str(self.root / "result.json"), "--judged-jsonl", str(self.root / "judged.jsonl"),
                          "--cache-jsonl", str(self.root / "cache.jsonl"), "--expected-examples", "2", "--max-rpm", "0"]

    def tearDown(self):
        self.temp.cleanup()

    def run_cli(self, extra=()):
        with patch.object(sys, "argv", self.arguments + list(extra)), contextlib.redirect_stdout(io.StringIO()):
            scorer.main()

    def test_dry_run_never_calls_api_or_writes(self):
        with patch.object(scorer, "call_chat_completion", side_effect=AssertionError("network forbidden")):
            self.run_cli(["--dry-run"])
        self.assertFalse((self.root / "result.json").exists())
        self.assertFalse((self.root / "cache.jsonl").exists())

    def test_mocked_judge_and_offline_resume_match(self):
        judgement = {"prediction": {"distance_m": 1.9}, "semantic": {}}
        with patch.object(scorer, "call_chat_completion", return_value=judgement) as call:
            self.run_cli(["--require-api", "--no-skip-sensitive-api-errors"])
        self.assertEqual(call.call_count, 1)  # Speech is scored locally.
        first = json.loads((self.root / "result.json").read_text())
        with patch.object(scorer, "call_chat_completion", side_effect=AssertionError("network forbidden")):
            self.run_cli(["--offline", "--overwrite"])
        second = json.loads((self.root / "result.json").read_text())
        self.assertEqual(first["task_aware_accuracy"], second["task_aware_accuracy"])
        self.assertEqual(second["examples"], 2)
        self.assertEqual(second["api_fallback_used"], 0)
        self.assertEqual(second["cache_hits"], 2)

    def test_offline_missing_cache_fails_without_request(self):
        with patch.object(scorer, "call_chat_completion", side_effect=AssertionError("network forbidden")):
            with self.assertRaisesRegex(RuntimeError, "No usable cached judgement"):
                self.run_cli(["--offline"])

    def test_wrong_denominator_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Expected 3 examples"):
            self.run_cli(["--dry-run", "--expected-examples", "3"])

    def test_unmatched_prediction_is_rejected(self):
        row = dict(self.rows[0], pair_id="unknown", audio_path="other.wav", question="Unmatched?")
        self.predictions.write_text(json.dumps(row) + "\n")
        with self.assertRaisesRegex(ValueError, "did not match a QA record"):
            self.run_cli(["--dry-run"])

    def test_existing_output_and_input_collision_are_rejected(self):
        (self.root / "result.json").write_text("preserve")
        with self.assertRaises(FileExistsError):
            self.run_cli(["--dry-run"])
        self.assertEqual((self.root / "result.json").read_text(), "preserve")
        with self.assertRaisesRegex(ValueError, "must be distinct"):
            self.run_cli(["--output-json", str(self.predictions), "--overwrite", "--dry-run"])

    def test_api_key_comes_only_from_environment(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-only-value"}), patch.object(sys, "argv", self.arguments):
            args = scorer.parse_args()
        self.assertEqual(args.api_key, "test-only-value")


if __name__ == "__main__":
    unittest.main()
