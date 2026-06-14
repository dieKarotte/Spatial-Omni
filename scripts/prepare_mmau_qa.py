"""Convert MMAU/MMAU-Pro data into the QA-root schema consumed by run_bench.py."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", required=True,
                        help="Path to mmau-test-mini.json, MMAU-Pro jsonl, or a compatible file.")
    parser.add_argument("--output-root", required=True,
                        help="Directory where <split>.jsonl will be written.")
    parser.add_argument("--split", default="test",
                        help="Output split name. Default: test.")
    parser.add_argument("--audio-root", default=None,
                        help="Root used to resolve relative audio_id values. "
                             "Defaults to the input JSON parent directory.")
    parser.add_argument("--verify-audio", action="store_true",
                        help="Stat every resolved audio_path and fail if any are missing.")
    parser.add_argument("--min-choices", type=int, default=1,
                        help="Skip records with fewer choices. Use 2 for closed-choice MMAU-Pro accuracy.")
    parser.add_argument("--multi-audio-policy", choices=("skip", "first"), default="skip",
                        help="How to handle records whose audio_path/audio_id is a list with multiple files. "
                             "The current bench runners accept one audio file per QA record.")
    parser.add_argument("--answer-format", choices=("text", "letter"), default="text",
                        help="Instruction appended to MCQ prompts. Scoring accepts both letters and text.")
    parser.add_argument("--include-categories", nargs="+", default=None,
                        help="Optional category allowlist, e.g. sound music speech.")
    parser.add_argument("--exclude-categories", nargs="+", default=None,
                        help="Optional category denylist, e.g. open 'instruction following'.")
    return parser.parse_args()


def load_records(path: Path) -> List[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected MMAU JSON to be a list, got {type(payload).__name__}")
    return payload


def build_prompt(sample: Dict[str, Any], answer_format: str = "text") -> str:
    question = str(sample.get("question", "")).strip()
    choices = sample.get("choices") or []
    choice_lines = [
        f"{chr(ord('A') + index)}. {choice}"
        for index, choice in enumerate(choices)
    ]
    if choice_lines:
        instruction = (
            "Answer with the option letter only."
            if answer_format == "letter"
            else "Answer with the option text only."
        )
        return (
            f"{question}\n"
            "Choices:\n"
            + "\n".join(choice_lines)
            + f"\n{instruction}"
        )
    return f"{question}\nAnswer with the option text only."


def first_audio_path(sample: Dict[str, Any], multi_audio_policy: str) -> str:
    value = sample.get("audio_path", sample.get("audio_id"))
    if isinstance(value, list):
        if not value:
            raise ValueError(f"Sample {sample.get('id')} has an empty audio path list")
        if len(value) > 1 and multi_audio_policy == "skip":
            return ""
        return str(value[0])
    if value is None:
        raise ValueError(f"Sample {sample.get('id')} missing audio_path/audio_id")
    return str(value)


def convert_one(sample: Dict[str, Any], audio_root: Path, answer_format: str = "text") -> Dict[str, Any]:
    sample_id = sample.get("id")
    if not sample_id:
        raise ValueError(f"MMAU sample missing id: {sample}")
    audio_id = first_audio_path(sample, multi_audio_policy="first")
    answer = sample.get("answer")
    if answer is None:
        raise ValueError(f"MMAU sample {sample_id} missing answer")

    domain = str(sample.get("task") or sample.get("category") or "").strip() or "unknown"
    audio_path = Path(audio_id)
    if not audio_path.is_absolute():
        audio_path = audio_root / str(audio_id).lstrip("./")
    audio_path = Path(os.path.abspath(os.fspath(audio_path)))
    return {
        "pair_id": sample_id,
        "task_name": f"MMAU-Pro-{domain}" if "category" in sample and "task" not in sample else f"MMAU-{domain}",
        "question_class": domain,
        "question": sample.get("question"),
        "prompt": build_prompt(sample, answer_format=answer_format),
        "answer": answer,
        "canonical_answer": answer,
        "audio_path": str(audio_path),
        "scene_id": sample_id,
        "answer_meta": {
            "dataset": sample.get("dataset"),
            "task": sample.get("task") or sample.get("category"),
            "split": sample.get("split"),
            "category": sample.get("category"),
            "sub-category": sample.get("sub-category"),
            "sub-cat": sample.get("sub-cat"),
            "difficulty": sample.get("difficulty"),
            "length_type": sample.get("length_type"),
            "perceptual_skills": sample.get("perceptual_skills"),
            "reasoning_skills": sample.get("reasoning_skills"),
            "choices": sample.get("choices"),
        },
    }


def main() -> int:
    args = parse_args()
    input_json = Path(args.input_json).resolve()
    audio_root = Path(args.audio_root).resolve() if args.audio_root else input_json.parent
    output_root = Path(args.output_root).resolve()
    output_path = output_root / f"{args.split}.jsonl"

    records = []
    skipped = {
        "missing_answer": 0,
        "category_filter": 0,
        "too_few_choices": 0,
        "multi_audio": 0,
    }
    for sample in load_records(input_json):
        if sample.get("answer") in (None, ""):
            skipped["missing_answer"] += 1
            continue
        category = str(sample.get("category") or sample.get("task") or "")
        if args.include_categories is not None and category not in set(args.include_categories):
            skipped["category_filter"] += 1
            continue
        if args.exclude_categories is not None and category in set(args.exclude_categories):
            skipped["category_filter"] += 1
            continue
        choices = sample.get("choices") or []
        if len(choices) < int(args.min_choices):
            skipped["too_few_choices"] += 1
            continue
        audio_value = sample.get("audio_path", sample.get("audio_id"))
        if isinstance(audio_value, list) and len(audio_value) > 1 and args.multi_audio_policy == "skip":
            skipped["multi_audio"] += 1
            continue
        if isinstance(audio_value, list):
            sample = dict(sample)
            sample["audio_path"] = [audio_value[0]]
        records.append(convert_one(sample, audio_root, answer_format=args.answer_format))
    if args.verify_audio:
        missing = [record["audio_path"] for record in records if not os.path.exists(record["audio_path"])]
        if missing:
            preview = "\n".join(missing[:10])
            raise FileNotFoundError(
                f"{len(missing)} audio files are missing. First missing paths:\n{preview}"
            )

    output_root.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    counts: Dict[str, int] = {}
    for record in records:
        domain = str(record.get("question_class") or "unknown")
        counts[domain] = counts.get(domain, 0) + 1
    print(f"Wrote {len(records)} records -> {output_path}")
    print("Skipped:")
    for key, count in skipped.items():
        print(f"  {key}: {count}")
    print("Domain counts:")
    for domain, count in sorted(counts.items()):
        print(f"  {domain}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
