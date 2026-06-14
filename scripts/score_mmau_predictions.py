"""Score MMAU predictions emitted by scripts/run_bench.py."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List


DEFAULT_DOMAINS = ("sound", "music", "speech")
OPTION_RE = re.compile(r"^\s*(?:answer\s*(?:is|:)\s*)?[\(\[]?\s*([A-Z])\s*[\)\].:\-]?", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-jsonl", required=True,
                        help="Path to predictions.jsonl from a bench run.")
    parser.add_argument("--qa-root", default=None,
                        help="Optional QA root containing <split>.jsonl/json. "
                             "Used to recover choices/answer_meta when baseline "
                             "predictions did not copy those fields.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-json", default=None,
                        help="Aggregate score JSON. Defaults to mmau_score.json next to predictions.")
    parser.add_argument("--per-record-jsonl", default=None,
                        help="Per-record score JSONL. Defaults to mmau_score_by_record.jsonl next to predictions.")
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def resolve_qa_split_path(qa_root: Path, split: str) -> Path:
    for suffix in (".jsonl", ".json"):
        path = qa_root / f"{split}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing {split}.jsonl or {split}.json under {qa_root}")


def load_qa_records(qa_root: str, split: str) -> List[Dict[str, Any]]:
    path = resolve_qa_split_path(Path(qa_root).resolve(), split)
    if path.suffix == ".jsonl":
        return load_jsonl(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        return payload["records"]
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return payload["data"]
    raise ValueError(f"Unsupported QA file structure: {path}")


def enrich_predictions_from_qa(
    predictions: List[Dict[str, Any]],
    qa_records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    by_pair_id = {
        str(record.get("pair_id")): record
        for record in qa_records
        if record.get("pair_id") is not None
    }
    enriched: List[Dict[str, Any]] = []
    for pred in predictions:
        merged = dict(pred)
        qa = by_pair_id.get(str(pred.get("pair_id")))
        if qa is not None:
            for key in ("answer_meta", "canonical_answer", "question_class", "task_name", "answer"):
                if merged.get(key) is None and qa.get(key) is not None:
                    merged[key] = qa.get(key)
        enriched.append(merged)
    return enriched


def normalize_answer(text: Any) -> str:
    text = str(text or "").lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def phrase_match(gold: Any, pred: Any) -> bool:
    gold_norm = normalize_answer(gold)
    pred_norm = normalize_answer(pred)
    if not gold_norm or not pred_norm:
        return False
    if gold_norm == pred_norm:
        return True
    return re.search(r"(?:^|\s)" + re.escape(gold_norm) + r"(?:\s|$)", pred_norm) is not None


def get_choices(record: Dict[str, Any]) -> List[Any]:
    answer_meta = record.get("answer_meta")
    if isinstance(answer_meta, dict) and isinstance(answer_meta.get("choices"), list):
        return answer_meta["choices"]
    choices = record.get("choices")
    if isinstance(choices, list):
        return choices
    return []


def parse_option_letter(prediction: Any, num_choices: int) -> tuple[str | None, int | None]:
    text = str(prediction or "").strip()
    if not text:
        return None, None
    match = OPTION_RE.match(text)
    if not match:
        return None, None
    letter = match.group(1).upper()
    index = ord(letter) - ord("A")
    if index < 0 or index >= num_choices:
        return letter, None
    return letter, index


def score_prediction(record: Dict[str, Any]) -> Dict[str, Any]:
    gold = record.get("canonical_answer") or record.get("answer")
    prediction = record.get("prediction_cleaned") or record.get("prediction")
    choices = get_choices(record)
    letter, index = parse_option_letter(prediction, len(choices))
    mapped_prediction = None
    method = "text"
    if index is not None:
        mapped_prediction = choices[index]
        is_correct = phrase_match(gold, mapped_prediction)
        method = "option_letter"
    else:
        is_correct = phrase_match(gold, prediction)
    return {
        "gold": gold,
        "prediction": prediction,
        "prediction_option_letter": letter,
        "prediction_option_index": index,
        "prediction_option_text": mapped_prediction,
        "match_method": method,
        "correct": bool(is_correct),
    }


def infer_domain(record: Dict[str, Any]) -> str:
    question_class = record.get("question_class")
    if question_class:
        return str(question_class).strip().lower()
    answer_meta = record.get("answer_meta")
    if isinstance(answer_meta, dict) and answer_meta.get("task"):
        return str(answer_meta["task"]).strip().lower()
    task_name = str(record.get("task_name") or "").strip().lower()
    if task_name.startswith("mmau-pro-"):
        return task_name[len("mmau-pro-"):]
    if task_name.startswith("mmau-"):
        return task_name.split("-", 1)[1]
    return task_name or "unknown"


def accuracy(correct: int, total: int) -> float:
    return float(correct) / float(total) if total else 0.0


def score(records: Iterable[Dict[str, Any]]) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    per_record: List[Dict[str, Any]] = []
    totals: Dict[str, int] = {}
    corrects: Dict[str, int] = {}
    total = 0
    correct = 0

    for record in records:
        domain = infer_domain(record)
        scored = score_prediction(record)
        is_correct = bool(scored["correct"])
        total += 1
        correct += int(is_correct)
        totals[domain] = totals.get(domain, 0) + 1
        corrects[domain] = corrects.get(domain, 0) + int(is_correct)
        per_record.append(
            {
                "pair_id": record.get("pair_id"),
                "domain": domain,
                "answer": scored["gold"],
                "prediction": record.get("prediction"),
                "prediction_cleaned": record.get("prediction_cleaned"),
                "prediction_option_letter": scored["prediction_option_letter"],
                "prediction_option_index": scored["prediction_option_index"],
                "prediction_option_text": scored["prediction_option_text"],
                "match_method": scored["match_method"],
                "correct": bool(is_correct),
            }
        )

    domain_metrics = {
        domain: {
            "accuracy": accuracy(corrects[domain], totals[domain]),
            "correct": corrects[domain],
            "total": totals[domain],
        }
        for domain in sorted(totals)
    }
    present_domain_accs = [
        stats["accuracy"]
        for stats in domain_metrics.values()
        if stats["total"] > 0
    ]
    default_domain_accs = [
        domain_metrics[domain]["accuracy"]
        for domain in DEFAULT_DOMAINS
        if domain in domain_metrics
        if domain_metrics[domain]["total"] > 0
    ]
    summary: Dict[str, Any] = {
        "scoring": "MMAU multiple-choice aware: leading option letters are mapped through answer_meta.choices before matching.",
        "overall_accuracy": accuracy(correct, total),
        "overall_correct": correct,
        "overall_total": total,
        "sound_accuracy": domain_metrics.get("sound", {}).get("accuracy", 0.0),
        "music_accuracy": domain_metrics.get("music", {}).get("accuracy", 0.0),
        "speech_accuracy": domain_metrics.get("speech", {}).get("accuracy", 0.0),
        "domain_macro_avg": (
            sum(default_domain_accs) / len(default_domain_accs)
            if default_domain_accs else (
                sum(present_domain_accs) / len(present_domain_accs)
                if present_domain_accs else 0.0
            )
        ),
        "category_macro_avg": (
            sum(present_domain_accs) / len(present_domain_accs)
            if present_domain_accs else 0.0
        ),
        "domains": domain_metrics,
    }
    return summary, per_record


def main() -> int:
    args = parse_args()
    predictions_path = Path(args.predictions_jsonl).resolve()
    output_json = Path(args.output_json).resolve() if args.output_json else predictions_path.parent / "mmau_score.json"
    per_record_jsonl = (
        Path(args.per_record_jsonl).resolve()
        if args.per_record_jsonl
        else predictions_path.parent / "mmau_score_by_record.jsonl"
    )

    predictions = load_jsonl(predictions_path)
    if args.qa_root:
        predictions = enrich_predictions_from_qa(
            predictions,
            load_qa_records(args.qa_root, args.split),
        )
    summary, per_record = score(predictions)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
    with per_record_jsonl.open("w", encoding="utf-8") as handle:
        for record in per_record:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[mmau-score] wrote {output_json}")
    print(f"[mmau-score] wrote {per_record_jsonl}")
    print(
        "[mmau-score] "
        f"overall={summary['overall_accuracy']:.4f} "
        f"sound={summary['sound_accuracy']:.4f} "
        f"music={summary['music_accuracy']:.4f} "
        f"speech={summary['speech_accuracy']:.4f} "
        f"avg={summary['domain_macro_avg']:.4f} "
        f"category_avg={summary['category_macro_avg']:.4f}"
    )
    for domain, stats in summary["domains"].items():
        print(
            "[mmau-score] "
            f"{domain}={stats['accuracy']:.4f} "
            f"({stats['correct']}/{stats['total']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
