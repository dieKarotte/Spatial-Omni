#!/usr/bin/env python3
"""LLM-assisted task-aware scoring for open-form spatial QA predictions.

This scorer is intended for baseline outputs whose wording is highly variable.
It asks an OpenAI-compatible chat-completions API to normalize each
question/answer/prediction triple into a small JSON object, then applies local
numeric and task-specific metrics to that JSON. The API is used only for
semantic normalization/judging; thresholds for angle, distance, time, and IoU
are applied locally.

Note: the ``speech_content`` task does NOT call the LLM judge. It is scored
locally using word error rate against the gold transcript (see the
``speech_wer`` branch in ``evaluate_record``). All other tasks go through the
LLM judge.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tqdm.auto import tqdm


DEFAULT_MODEL = os.environ.get("LLM_JUDGE_MODEL", "gpt-4o-mini")
DEFAULT_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

FLOAT_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")

SEMANTIC_SOURCE_TASKS = {
    "detect_source",
    "identify_source_by_doa",
    "identify_source_by_location",
    "multi_hop",
}
RELATION_TASKS = {
    "compare_distance",
    "compare_elevation",
    "relative_left_right",
}


SYSTEM_PROMPT = """You are a strict evaluator for spatial audio question answering.

You compare only the provided QUESTION, REFERENCE_ANSWER, and MODEL_PREDICTION.
Do not use outside audio knowledge. Do not give credit for information that is
not present in the model prediction.

Normalize wording into structured JSON:
- Treat sound-event synonyms as matches when they refer to the same audible class:
  "speech", "voice", "speaker", "man/woman talking" match; "guitar" and
  "guitar playing" match; "vehicle" and "car" may match if the reference is
  broad enough, but "musical instrument" is too generic for "guitar".
- Mark source_match_level as one of:
  "exact_or_synonym", "compatible_but_broader", "related_but_wrong", "wrong", "unknown".
- For left/right/front/back/above/below/near/far relations, extract the relation
  explicitly stated by the prediction. Do not infer missing relations.
- For transcript questions, ignore punctuation/case and allow minor ASR word
  errors, but do not accept unrelated or hallucinated transcript content.
- For numeric values, extract numbers with units when present. Use null when
  no value is stated.

Return ONLY valid JSON with this schema:
{
  "prediction": {
    "sources": ["normalized source labels"],
    "primary_source": string|null,
    "angle_deg": number|null,
    "elevation_deg": number|null,
    "distance_m": number|null,
    "time_span": [number, number]|null,
    "onset_time_s": number|null,
    "count": integer|null,
    "motion_label": "stationary"|"moving_toward"|"moving_away"|"moving_left"|"moving_right"|"moving"|null,
    "yes_no": "yes"|"no"|null,
    "left_right": "left"|"right"|null,
    "height_relation": "higher"|"lower"|"same"|null,
    "distance_relation": "closer"|"farther"|"same"|null,
    "direction_phrase": string|null,
    "transcript": string|null
  },
  "semantic": {
    "source_match_level": "exact_or_synonym"|"compatible_but_broader"|"related_but_wrong"|"wrong"|"unknown",
    "source_match": boolean,
    "relation_match": boolean|null,
    "direction_match": boolean|null,
    "motion_match": boolean|null,
    "transcript_match": boolean|null
  },
  "notes": string
}
"""


def task_prompt(task_name: str) -> str:
    common = (
        f"TASK_NAME: {task_name}\n"
        "Extract the model prediction into the JSON schema. Also compare it to "
        "the reference answer for semantic source/relation fields."
    )
    if task_name in {"estimate_azimuth", "estimate_elevation"}:
        return common + "\nFocus on extracting the final numeric angle in degrees from MODEL_PREDICTION."
    if task_name == "estimate_distance":
        return common + "\nFocus on extracting the final numeric distance in meters, not only the near/far category."
    if task_name == "detect_time":
        return common + "\nExtract the time interval for the target sound event as [start_s, end_s]."
    if task_name == "detect_source":
        return common + "\nExtract the sound event label(s) and any time interval. Judge event-name semantic match."
    if task_name in {"identify_source_by_doa", "identify_source_by_location", "multi_hop"}:
        return common + "\nExtract the answer source/event label and judge whether it semantically matches the reference source."
    if task_name in RELATION_TASKS:
        return common + "\nExtract the comparative spatial relation and judge if the relation matches the reference."
    if task_name == "spatial_temporal":
        return common + "\nExtract source, direction phrase, and time interval. Judge source and direction semantic match."
    if task_name == "speech_content":
        return common + "\nExtract the spoken transcript and judge transcript_match against the quoted reference sentence."
    if task_name == "classify_motion":
        return common + "\nNormalize the motion label and judge motion_match."
    return common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM-assisted scorer for spatial QA predictions.")
    parser.add_argument("--predictions-jsonl", required=True)
    parser.add_argument("--qa-root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--judged-jsonl", default=None)
    parser.add_argument("--cache-jsonl", default=None)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL,
                        help="OpenAI-compatible chat-completions base URL. Falls back to $OPENAI_BASE_URL.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--concurrency", "--num-workers", dest="concurrency", type=int, default=128)
    parser.add_argument("--max-rpm", type=float, default=180.0,
                        help="Maximum requests per minute across all workers; 0 disables the rate limit.")
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument(
        "--require-api",
        action="store_true",
        help="Fail immediately if the LLM judge API cannot be called instead of using local regex fallback.",
    )
    parser.add_argument(
        "--skip-sensitive-api-errors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When the provider rejects a request for sensitive words, mark that record wrong and continue.",
    )
    parser.add_argument(
        "--angle-threshold-deg",
        type=float,
        default=20.0,
        help="Azimuth threshold in degrees. Kept for compatibility with older commands.",
    )
    parser.add_argument("--elevation-threshold-deg", type=float, default=10.0)
    parser.add_argument("--distance-threshold-m", type=float, default=1.0)
    parser.add_argument("--time-threshold-s", type=float, default=0.2)
    parser.add_argument(
        "--detect-source-time-policy",
        choices=("event_only", "event_times_iou"),
        default="event_only",
        help="For detect_source, score semantic event only or multiply by time IoU when both spans exist.",
    )
    parser.add_argument(
        "--spatial-temporal-time-policy",
        choices=("semantic_only", "semantic_times_iou"),
        default="semantic_times_iou",
    )
    parser.add_argument(
        "--accept-broader-source",
        action="store_true",
        help="Accept compatible_but_broader source matches as correct. Default only accepts exact/synonym.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and report the scoring plan without API calls or writes.")
    parser.add_argument("--offline", action="store_true", help="Use cached judgements only; fail if any non-speech record is uncached.")
    parser.add_argument("--expected-examples", type=int, default=None, help="Require this many aligned records before scoring.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing result and judged-record files; the cache is append-only.")
    parser.set_defaults(api_key=os.environ.get("OPENAI_API_KEY", ""))
    return parser.parse_args()


def resolve_split_path(qa_root: str, split: str) -> Path:
    root = Path(qa_root)
    for suffix in (".json", ".jsonl"):
        candidate = root / f"{split}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing {split}.json/.jsonl under {qa_root}")


def load_json_records(path: Path) -> List[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        return payload["records"]
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return payload["data"]
    raise ValueError(f"Unsupported JSON structure: {path}")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_pair_id(record: Dict[str, Any], fallback_index: int) -> str:
    qa_id = record.get("qa_id")
    if qa_id is not None:
        return str(qa_id)
    pair_id = record.get("pair_id")
    if pair_id is not None:
        return str(pair_id)
    eval_index = record.get("eval_index")
    if eval_index is not None:
        return str(eval_index)
    return str(fallback_index)


def match_signature(record: Dict[str, Any]) -> Tuple[str, str, str, str]:
    return (
        str(record.get("audio_path") or ""),
        str(record.get("task_name") or ""),
        normalize_text(record.get("answer")),
        normalize_text(record.get("canonical_answer")),
    )


def normalize_text(text: Any) -> str:
    return " ".join(str(text or "").strip().lower().split())


def canonicalize_transcript(text: Any) -> str:
    value = str(text or "").lower().replace("_", " ")
    value = re.sub(r"[^\w\s']", " ", value)
    return " ".join(value.split())


def extract_transcript_candidates(text: Any) -> List[str]:
    value = str(text or "").strip()
    if not value:
        return [""]
    candidates = [value]

    quoted = re.findall(r'"([^"]+)"|“([^”]+)”|\'([^\']+)\'', value)
    for groups in quoted:
        candidate = next((part for part in groups if part), "").strip()
        if candidate:
            candidates.append(candidate)

    marker_patterns = [
        r"(?i)(?:transcript|spoken words|speech content|the speaker says|speaker says|says|said|answer)\s*[:：]\s*(.+)$",
        r"(?i)(?:says|said)\s*,?\s*[\"“']?(.+?)[\"”']?$",
    ]
    for pattern in marker_patterns:
        match = re.search(pattern, value, flags=re.DOTALL)
        if match and match.group(1).strip():
            candidates.append(match.group(1).strip())

    cleaned: List[str] = []
    seen = set()
    for candidate in candidates:
        candidate = candidate.strip(" \t\r\n\"'“”")
        key = canonicalize_transcript(candidate)
        if key and key not in seen:
            seen.add(key)
            cleaned.append(candidate)
    return cleaned or [value]


def transcript_gold_text(record: Dict[str, Any]) -> str:
    canonical = str(record.get("canonical_answer") or "").strip()
    if canonical:
        return canonical
    answer = str(record.get("answer") or "").strip()
    candidates = extract_transcript_candidates(answer)
    if len(candidates) > 1:
        return candidates[1]
    return answer


def word_error_rate(reference: Any, hypothesis: Any) -> Tuple[float, int, int]:
    ref_words = canonicalize_transcript(reference).split()
    hyp_words = canonicalize_transcript(hypothesis).split()
    n = len(ref_words)
    if n == 0:
        return (0.0 if not hyp_words else 1.0), len(hyp_words), n
    prev = list(range(len(hyp_words) + 1))
    for i, ref_word in enumerate(ref_words, start=1):
        cur = [i] + [0] * len(hyp_words)
        for j, hyp_word in enumerate(hyp_words, start=1):
            sub_cost = 0 if ref_word == hyp_word else 1
            cur[j] = min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + sub_cost,
            )
        prev = cur
    errors = prev[-1]
    return errors / n, errors, n


def first_float(text: Any) -> Optional[float]:
    match = FLOAT_RE.search(str(text or ""))
    return float(match.group(0)) if match else None


def all_floats(text: Any) -> List[float]:
    return [float(match.group(0)) for match in FLOAT_RE.finditer(str(text or ""))]


def local_time_span(text: Any) -> Optional[List[float]]:
    values = all_floats(text)
    if len(values) < 2:
        return None
    start, end = values[0], values[1]
    if end < start:
        start, end = end, start
    return [start, end]


def local_category(text: Any, choices: Iterable[str]) -> Optional[str]:
    norm = normalize_text(text)
    for choice in choices:
        if choice in norm:
            return choice
    return None


def local_yes_no(text: Any) -> Optional[str]:
    norm = normalize_text(text)
    if re.search(r"\bno\b", norm):
        return "no"
    if re.search(r"\byes\b", norm):
        return "yes"
    return None


def angle_error_deg(prediction_deg: float, target_deg: float) -> float:
    delta = prediction_deg - target_deg
    while delta > 180.0:
        delta -= 360.0
    while delta <= -180.0:
        delta += 360.0
    return abs(delta)


def coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def interval_iou(pred_span: Optional[List[float]], target_span: Optional[List[float]]) -> Optional[float]:
    if pred_span is None or target_span is None or len(pred_span) != 2 or len(target_span) != 2:
        return None
    pred_start = coerce_float(pred_span[0])
    pred_end = coerce_float(pred_span[1])
    gt_start = coerce_float(target_span[0])
    gt_end = coerce_float(target_span[1])
    if pred_start is None or pred_end is None or gt_start is None or gt_end is None:
        return None
    if pred_end < pred_start:
        pred_start, pred_end = pred_end, pred_start
    if gt_end < gt_start:
        gt_start, gt_end = gt_end, gt_start
    intersection = max(0.0, min(pred_end, gt_end) - max(pred_start, gt_start))
    union = max(pred_end, gt_end) - min(pred_start, gt_start)
    return intersection / union if union > 0 else 0.0


def cache_key(record: Dict[str, Any]) -> str:
    payload = {
        "task_name": record.get("task_name"),
        "question": record.get("question"),
        "answer": record.get("answer"),
        "prediction": record.get("prediction"),
        "prediction_raw": record.get("prediction_raw"),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def extract_json_object(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            return json.loads(stripped[start : end + 1])
        raise


class RateLimiter:
    """Limit request starts independently of the number of concurrent workers."""

    def __init__(self, max_rpm: Optional[float]) -> None:
        self._interval = 60.0 / max_rpm if max_rpm and max_rpm > 0 else 0.0
        self._lock = threading.Lock()
        self._next_slot = 0.0

    def acquire(self) -> None:
        if self._interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next_slot)
            self._next_slot = slot + self._interval
        delay = slot - now
        if delay > 0:
            time.sleep(delay)


_RATE_LIMITER = RateLimiter(None)


def call_chat_completion(
    args: argparse.Namespace,
    record: Dict[str, Any],
) -> Dict[str, Any]:
    if not args.api_key:
        raise RuntimeError("Missing API key. Set OPENAI_API_KEY.")
    prediction_text = record.get("prediction_cleaned") or record.get("prediction") or record.get("prediction_raw") or ""
    user_prompt = (
        f"{task_prompt(str(record.get('task_name') or ''))}\n\n"
        f"QUESTION:\n{record.get('question')}\n\n"
        f"REFERENCE_ANSWER:\n{record.get('answer')}\n\n"
        f"ANSWER_META_JSON:\n{json.dumps(record.get('answer_meta') or {}, ensure_ascii=False)}\n\n"
        f"MODEL_PREDICTION:\n{prediction_text}\n"
    )
    body = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    url = args.base_url.rstrip("/") + "/chat/completions"
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {args.api_key}",
        },
        method="POST",
    )
    last_error: Optional[BaseException] = None
    for attempt in range(args.max_retries):
        try:
            _RATE_LIMITER.acquire()
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = json.loads(response.read().decode("utf-8"))
            content = payload["choices"][0]["message"]["content"]
            return extract_json_object(content)
        except Exception as error:  # API/network/model JSON errors are retried.
            if isinstance(error, urllib.error.HTTPError):
                try:
                    body = error.read().decode("utf-8", errors="replace")
                except Exception:
                    body = ""
                last_error = RuntimeError(f"HTTP Error {error.code}: {body[:1000].replace(args.api_key, '[redacted]')}")
            else:
                last_error = RuntimeError(str(error).replace(args.api_key, "[redacted]"))
            sleep_s = args.retry_sleep * (2 ** attempt)
            time.sleep(sleep_s)
    raise RuntimeError(f"LLM judge API failed after {args.max_retries} retries: {last_error}")


def fallback_judgement(record: Dict[str, Any]) -> Dict[str, Any]:
    prediction_text = record.get("prediction_cleaned") or record.get("prediction") or record.get("prediction_raw") or ""
    return {
        "prediction": {
            "sources": [],
            "primary_source": None,
            "angle_deg": first_float(prediction_text),
            "elevation_deg": first_float(prediction_text),
            "distance_m": first_float(prediction_text),
            "time_span": local_time_span(prediction_text),
            "onset_time_s": first_float(prediction_text),
            "count": int(first_float(prediction_text)) if first_float(prediction_text) is not None else None,
            "motion_label": None,
            "yes_no": local_yes_no(prediction_text),
            "left_right": local_category(prediction_text, ("left", "right")),
            "height_relation": local_category(prediction_text, ("higher", "lower", "same")),
            "distance_relation": local_category(prediction_text, ("closer", "farther", "same")),
            "direction_phrase": None,
            "transcript": None,
        },
        "semantic": {
            "source_match_level": "unknown",
            "source_match": False,
            "relation_match": None,
            "direction_match": None,
            "motion_match": None,
            "transcript_match": None,
        },
        "notes": "local regex fallback only",
    }


def error_scored_record(record: Dict[str, Any], error: BaseException) -> Dict[str, Any]:
    return {
        "pair_id": record.get("pair_id"),
        "eval_index": record.get("eval_index"),
        "_record_order": record.get("_record_order"),
        "task_name": record.get("task_name"),
        "answer": record.get("answer"),
        "prediction": record.get("prediction_cleaned") or record.get("prediction") or record.get("prediction_raw") or "",
        "llm_judgement": {"error": str(error)},
        "correct": 0.0,
        "metric_type": "scoring_error",
        "parseable": 0,
        "scoring_error": str(error),
    }


def api_skipped_record(record: Dict[str, Any], error: BaseException, reason: str) -> Dict[str, Any]:
    return {
        "pair_id": record.get("pair_id"),
        "eval_index": record.get("eval_index"),
        "_record_order": record.get("_record_order"),
        "task_name": record.get("task_name"),
        "answer": record.get("answer"),
        "prediction": record.get("prediction_cleaned") or record.get("prediction") or record.get("prediction_raw") or "",
        "llm_judgement": {"error": str(error), "skip_reason": reason},
        "correct": 0.0,
        "metric_type": f"api_skipped_{reason}",
        "parseable": 0,
        "api_skipped": True,
        "api_skip_reason": reason,
        "api_error": str(error),
    }


def load_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    cache: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = row.get("cache_key")
            if key:
                cache[key] = row
    return cache


def append_cache(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def cache_row_has_api_error(row: Dict[str, Any]) -> bool:
    judgement = row.get("judgement") or {}
    notes = str(judgement.get("notes") or "")
    return "api_error=" in notes or notes.startswith("local regex fallback only")


def is_sensitive_api_error(error: BaseException) -> bool:
    text = str(error).lower()
    return "sensitive_words" in text or "sensitive words" in text


def record_context(record: Dict[str, Any]) -> str:
    prediction = record.get("prediction_cleaned") or record.get("prediction") or record.get("prediction_raw") or ""
    return json.dumps(
        {
            "record_order": record.get("_record_order"),
            "pair_id": record.get("pair_id"),
            "eval_index": record.get("eval_index"),
            "task_name": record.get("task_name"),
            "question": str(record.get("question") or "")[:300],
            "answer": str(record.get("answer") or "")[:300],
            "prediction": str(prediction)[:300],
        },
        ensure_ascii=False,
    )


def source_correct(judgement: Dict[str, Any], accept_broader: bool) -> bool:
    semantic = judgement.get("semantic") or {}
    level = semantic.get("source_match_level")
    if semantic.get("source_match") is True:
        if level == "compatible_but_broader" and not accept_broader:
            return False
        return True
    return level == "exact_or_synonym" or (accept_broader and level == "compatible_but_broader")


def get_pred(judgement: Dict[str, Any], key: str) -> Any:
    pred = judgement.get("prediction")
    if isinstance(pred, dict):
        return pred.get(key)
    return None


def evaluate_record(record: Dict[str, Any], judgement: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    task = str(record.get("task_name") or "")
    answer_meta = record.get("answer_meta") or {}
    answer_text = record.get("answer")
    semantic = judgement.get("semantic") or {}
    result: Dict[str, Any] = {
        "pair_id": record.get("pair_id"),
        "eval_index": record.get("eval_index"),
        "_record_order": record.get("_record_order"),
        "task_name": task,
        "answer": answer_text,
        "prediction": record.get("prediction_cleaned") or record.get("prediction") or record.get("prediction_raw") or "",
        "llm_judgement": judgement,
        "correct": 0.0,
        "metric_type": "llm_semantic",
        "parseable": 1,
    }

    if task == "estimate_azimuth":
        target = answer_meta.get("azimuth_deg")
        pred = get_pred(judgement, "angle_deg")
        result["metric_type"] = "er20_azimuth_llm_extract"
        if target is None or pred is None:
            result["parseable"] = 0
            return result
        err = angle_error_deg(float(pred), float(target))
        result["doaerr_deg"] = err
        result["correct"] = float(err <= args.angle_threshold_deg)
        return result

    if task == "estimate_elevation":
        target = answer_meta.get("elevation_deg")
        pred = get_pred(judgement, "elevation_deg")
        if pred is None:
            pred = get_pred(judgement, "angle_deg")
        result["metric_type"] = "er10_elevation_llm_extract"
        if target is None or pred is None:
            result["parseable"] = 0
            return result
        err = abs(float(pred) - float(target))
        result["doaerr_deg"] = err
        result["correct"] = float(err <= args.elevation_threshold_deg)
        return result

    if task == "estimate_distance":
        target = answer_meta.get("distance_m")
        pred = get_pred(judgement, "distance_m")
        result["metric_type"] = "distance_threshold_llm_extract"
        if target is None or pred is None:
            result["parseable"] = 0
            return result
        err = abs(float(pred) - float(target))
        result["distance_error_m"] = err
        result["correct"] = float(err <= args.distance_threshold_m)
        return result

    if task == "detect_time":
        target = answer_meta.get("time_span")
        pred_span = get_pred(judgement, "time_span")
        iou = interval_iou(pred_span, target)
        result["metric_type"] = "time_span_iou_llm_extract"
        if iou is None:
            result["parseable"] = 0
            return result
        result["interval_iou"] = iou
        result["correct"] = float(iou)
        return result

    if task == "onset_from_location":
        target = answer_meta.get("onset_time")
        pred = get_pred(judgement, "onset_time_s")
        result["metric_type"] = "onset_threshold_llm_extract"
        if target is None or pred is None:
            result["parseable"] = 0
            return result
        err = abs(float(pred) - float(target))
        result["time_error_s"] = err
        result["correct"] = float(err <= args.time_threshold_s)
        return result

    if task == "count_sources":
        target = answer_meta.get("active_count")
        pred = get_pred(judgement, "count")
        result["metric_type"] = "integer_exact_llm_extract"
        if target is None or pred is None:
            result["parseable"] = 0
            return result
        result["correct"] = float(int(pred) == int(target))
        return result

    if task == "classify_motion":
        result["metric_type"] = "motion_semantic_llm"
        result["correct"] = float(semantic.get("motion_match") is True)
        return result

    if task in SEMANTIC_SOURCE_TASKS:
        result["metric_type"] = "source_semantic_llm"
        result["source_match_level"] = (judgement.get("semantic") or {}).get("source_match_level")
        result["correct"] = float(source_correct(judgement, args.accept_broader_source))
        if task == "detect_source" and args.detect_source_time_policy == "event_times_iou":
            iou = interval_iou(get_pred(judgement, "time_span"), answer_meta.get("time_span"))
            if iou is not None:
                result["interval_iou"] = iou
                result["correct"] *= float(iou)
        return result

    if task in RELATION_TASKS:
        result["metric_type"] = "relation_semantic_llm"
        result["correct"] = float(semantic.get("relation_match") is True)
        return result

    if task == "spatial_temporal":
        result["metric_type"] = "source_direction_time_llm"
        semantic_ok = source_correct(judgement, args.accept_broader_source) and semantic.get("direction_match") is True
        if args.spatial_temporal_time_policy == "semantic_only":
            result["correct"] = float(semantic_ok)
            return result
        iou = interval_iou(get_pred(judgement, "time_span"), answer_meta.get("time_span"))
        result["interval_iou"] = iou
        result["correct"] = float(semantic_ok) * float(iou if iou is not None else 0.0)
        if iou is None:
            result["parseable"] = 0
        return result

    if task == "speech_content":
        gold = transcript_gold_text(record)
        prediction = result["prediction"]
        raw_wer, raw_errors, raw_ref_words = word_error_rate(gold, prediction)
        best_wer = raw_wer
        best_errors = raw_errors
        best_candidate = prediction
        best_candidate_index = -1
        for index, candidate in enumerate(extract_transcript_candidates(prediction)):
            candidate_wer, candidate_errors, _ = word_error_rate(gold, candidate)
            if candidate_wer < best_wer:
                best_wer = candidate_wer
                best_errors = candidate_errors
                best_candidate = candidate
                best_candidate_index = index
        result["metric_type"] = "speech_wer"
        result["gold_transcript"] = gold
        result["best_transcript_candidate"] = best_candidate
        result["best_transcript_candidate_index"] = best_candidate_index
        result["wer"] = float(best_wer)
        result["wer_raw"] = float(raw_wer)
        result["wer_errors"] = int(best_errors)
        result["wer_ref_words"] = int(raw_ref_words)
        result["correct"] = float(best_wer <= 0.5)
        return result

    result["metric_type"] = "normalized_exact_match"
    result["correct"] = float(normalize_text(result["prediction"]) == normalize_text(answer_text))
    return result


def mean(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def median(values: List[float]) -> Optional[float]:
    return float(statistics.median(values)) if values else None


def summarize(scored: List[Dict[str, Any]]) -> Dict[str, Any]:
    per_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in scored:
        per_task[str(row.get("task_name"))].append(row)
    summary: Dict[str, Any] = {
        "examples": len(scored),
        "task_aware_accuracy": mean([float(row.get("correct", 0.0)) for row in scored]) or 0.0,
        "per_task": {},
    }
    for task, rows in sorted(per_task.items()):
        task_summary: Dict[str, Any] = {
            "examples": len(rows),
            "task_aware_accuracy": mean([float(row.get("correct", 0.0)) for row in rows]) or 0.0,
            "parse_rate": mean([float(row.get("parseable", 0)) for row in rows]) or 0.0,
            "metric_type": rows[0].get("metric_type"),
        }
        if any("doaerr_deg" in row for row in rows):
            errors = [float(row["doaerr_deg"]) for row in rows if row.get("doaerr_deg") is not None and math.isfinite(float(row["doaerr_deg"]))]
            task_summary["doaerr_mean_deg"] = mean(errors)
            task_summary["doaerr_median_deg"] = median(errors)
        if any("distance_error_m" in row for row in rows):
            errors = [float(row["distance_error_m"]) for row in rows if row.get("distance_error_m") is not None]
            task_summary["distance_error_mean_m"] = mean(errors)
            task_summary["distance_error_median_m"] = median(errors)
        if any("interval_iou" in row for row in rows):
            ious = [float(row["interval_iou"]) for row in rows if row.get("interval_iou") is not None]
            task_summary["interval_iou_mean"] = mean(ious)
            task_summary["interval_iou_median"] = median(ious)
        if any("time_error_s" in row for row in rows):
            errors = [float(row["time_error_s"]) for row in rows if row.get("time_error_s") is not None]
            task_summary["time_error_mean_s"] = mean(errors)
            task_summary["time_error_median_s"] = median(errors)
        if task == "speech_content":
            wers = [float(row["wer"]) for row in rows if row.get("wer") is not None]
            wers_raw = [float(row["wer_raw"]) for row in rows if row.get("wer_raw") is not None]
            task_summary["wer_mean"] = mean(wers)
            task_summary["wer_median"] = median(wers)
            task_summary["wer_clipped_mean"] = mean([min(w, 1.0) for w in wers])
            task_summary["wer_clipped_median"] = median([min(w, 1.0) for w in wers])
            task_summary["wer_raw_mean"] = mean(wers_raw)
            task_summary["wer_raw_median"] = median(wers_raw)
            task_summary["acc_wer_le_0.3"] = mean([1.0 if w <= 0.3 else 0.0 for w in wers])
            task_summary["acc_wer_le_0.5"] = mean([1.0 if w <= 0.5 else 0.0 for w in wers])
            task_summary["acc_wer_le_1.0"] = mean([1.0 if w <= 1.0 else 0.0 for w in wers])
        summary["per_task"][task] = task_summary
    return summary


def print_summary(summary: Dict[str, Any]) -> None:
    print(f"examples={summary['examples']} task_aware_accuracy={summary['task_aware_accuracy']:.4f}")
    for task, row in summary["per_task"].items():
        line = f"{task}: n={row['examples']} acc={row['task_aware_accuracy']:.4f} parse_rate={row['parse_rate']:.4f} metric={row['metric_type']}"
        for key in (
            "doaerr_mean_deg",
            "doaerr_median_deg",
            "distance_error_mean_m",
            "distance_error_median_m",
            "interval_iou_mean",
            "interval_iou_median",
            "time_error_mean_s",
            "time_error_median_s",
            "wer_mean",
            "wer_median",
            "wer_clipped_mean",
            "acc_wer_le_0.5",
        ):
            if key in row:
                line += f" {key}={row[key]}"
        print(line)


def prepare_records(args: argparse.Namespace) -> List[Dict[str, Any]]:
    qa_records = load_json_records(resolve_split_path(args.qa_root, args.split))
    pred_records = load_jsonl(Path(args.predictions_jsonl))
    qa_by_pair_id = {normalize_pair_id(row, idx): (idx, row) for idx, row in enumerate(qa_records)}
    qa_by_signature: Dict[Tuple[str, str, str, str], List[Tuple[int, Dict[str, Any]]]] = defaultdict(list)
    for idx, row in enumerate(qa_records):
        signature = match_signature(row)
        qa_by_signature[signature].append((idx, row))
    records: List[Dict[str, Any]] = []
    used_qa_indices = set()
    for idx, pred in enumerate(pred_records):
        pair_id = normalize_pair_id(pred, idx)
        qa_index: Optional[int] = None
        qa: Optional[Dict[str, Any]] = None
        pair_match = qa_by_pair_id.get(pair_id)
        if pair_match is not None and pair_match[0] not in used_qa_indices:
            qa_index, qa = pair_match
        if qa is None:
            for candidate_index, candidate in qa_by_signature.get(match_signature(pred), []):
                if candidate_index not in used_qa_indices:
                    qa_index, qa = candidate_index, candidate
                    break
        if qa is None and idx < len(qa_records) and idx not in used_qa_indices:
            candidate = qa_records[idx]
            if (
                str(candidate.get("audio_path") or "") == str(pred.get("audio_path") or "")
                and str(candidate.get("task_name") or "") == str(pred.get("task_name") or "")
            ):
                qa_index = idx
                qa = candidate
        if qa is None:
            raise ValueError(f"Prediction {idx} (pair_id={pair_id}) did not match a QA record; refusing to reduce the evaluation denominator.")
        if qa_index is not None:
            used_qa_indices.add(qa_index)
        merged = dict(pred)
        merged["_record_order"] = idx
        merged["pair_id"] = pair_id
        merged["question"] = qa.get("question", pred.get("question"))
        merged["answer"] = qa.get("answer", pred.get("answer"))
        merged["canonical_answer"] = qa.get("canonical_answer", pred.get("canonical_answer"))
        merged["answer_format"] = qa.get("answer_format", pred.get("answer_format"))
        merged["source_refs"] = qa.get("source_refs", pred.get("source_refs"))
        merged["task_name"] = qa.get("task_name", pred.get("task_name"))
        merged["question_class"] = qa.get("question_class", pred.get("question_class"))
        merged["answer_meta"] = qa.get("answer_meta") or {}
        records.append(merged)
    if args.limit is not None:
        records = records[: args.limit]
    return records


def main() -> None:
    args = parse_args()
    global _RATE_LIMITER
    _RATE_LIMITER = RateLimiter(args.max_rpm)
    predictions_path = Path(args.predictions_jsonl).resolve()
    output_json = Path(args.output_json).resolve() if args.output_json else predictions_path.parent / "llm_result.json"
    judged_jsonl = Path(args.judged_jsonl).resolve() if args.judged_jsonl else predictions_path.parent / "llm_judged_records.jsonl"
    cache_jsonl = Path(args.cache_jsonl).resolve() if args.cache_jsonl else predictions_path.parent / "llm_judge_cache.jsonl"

    input_paths = {predictions_path, resolve_split_path(args.qa_root, args.split).resolve()}
    output_paths = (output_json, judged_jsonl, cache_jsonl)
    if len(set(output_paths)) != len(output_paths) or any(path in input_paths for path in output_paths):
        raise ValueError("Result, judged-record and cache paths must be distinct from one another and from inputs.")
    for path in (output_json, judged_jsonl):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Output already exists: {path}. Choose a new path or pass --overwrite.")
    records = prepare_records(args)
    if not records:
        raise ValueError("No matched predictions to score.")
    if args.expected_examples is not None and len(records) != args.expected_examples:
        raise ValueError(f"Expected {args.expected_examples} examples, found {len(records)} aligned predictions.")
    if args.dry_run:
        print(json.dumps({"examples": len(records), "model": args.model, "offline": args.offline,
                          "output_json": str(output_json), "judged_jsonl": str(judged_jsonl),
                          "cache_jsonl": str(cache_jsonl)}, indent=2))
        return
    cache = {} if args.force_refresh else load_cache(cache_jsonl)

    scored: List[Dict[str, Any]] = []
    new_cache_rows: List[Dict[str, Any]] = []
    cache_lock = threading.Lock()

    def flush_cache_rows() -> None:
        with cache_lock:
            rows = list(new_cache_rows)
            new_cache_rows.clear()
        if rows:
            append_cache(cache_jsonl, rows)

    def judge_one(record: Dict[str, Any]) -> Dict[str, Any]:
        key = cache_key(record)
        cached = cache.get(key)
        api_fallback_used = False
        api_error = ""
        api_skipped = False
        api_skip_reason = ""
        invalid_cache_ignored = False
        if str(record.get("task_name") or "") == "speech_content":
            judgement = fallback_judgement(record)
            cache_hit = True
        elif cached is not None and "judgement" in cached and not cache_row_has_api_error(cached):
            judgement = cached["judgement"]
            cache_hit = True
        elif args.offline:
            raise RuntimeError(f"No usable cached judgement for pair_id={record.get('pair_id')}; offline mode cannot call the judge.")
        else:
            invalid_cache_ignored = cached is not None and "judgement" in cached
            try:
                judgement = call_chat_completion(args, record)
                cache_hit = False
            except Exception as error:
                if args.skip_sensitive_api_errors and is_sensitive_api_error(error):
                    scored_record = api_skipped_record(record, error, "sensitive_words")
                    scored_record["cache_key"] = key
                    scored_record["cache_hit"] = False
                    scored_record["api_fallback_used"] = False
                    scored_record["invalid_cache_ignored"] = invalid_cache_ignored
                    return scored_record
                if args.require_api:
                    raise RuntimeError(f"{error}; record={record_context(record)}") from error
                judgement = fallback_judgement(record)
                api_fallback_used = True
                api_error = str(error)
                api_skipped = False
                api_skip_reason = ""
                judgement["notes"] = f"{judgement.get('notes', '')}; api_error={api_error}"
                cache_hit = False
        try:
            scored_record = evaluate_record(record, judgement, args)
        except Exception as error:
            scored_record = error_scored_record(record, error)
        scored_record["cache_key"] = key
        scored_record["cache_hit"] = cache_hit
        scored_record["api_fallback_used"] = api_fallback_used
        scored_record["api_error"] = api_error
        scored_record["api_skipped"] = api_skipped
        scored_record["api_skip_reason"] = api_skip_reason
        scored_record["invalid_cache_ignored"] = invalid_cache_ignored
        if not cache_hit and not api_fallback_used:
            with cache_lock:
                new_cache_rows.append({"cache_key": key, "judgement": judgement})
        return scored_record

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(args.concurrency, 1)) as executor:
        futures = [executor.submit(judge_one, record) for record in records]
        for index, future in enumerate(tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Scoring"), 1):
            try:
                scored.append(future.result())
            except Exception:
                flush_cache_rows()
                raise
            if index % 100 == 0:
                flush_cache_rows()

    scored.sort(key=lambda row: int(row.get("_record_order", 0)))
    flush_cache_rows()
    judged_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with judged_jsonl.open("w", encoding="utf-8") as handle:
        for row in scored:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = summarize(scored)
    summary.update(
        {
            "predictions_jsonl": str(predictions_path),
            "qa_root": str(Path(args.qa_root).resolve()),
            "split": args.split,
            "model": args.model,
            "concurrency": args.concurrency,
            "max_rpm": args.max_rpm,
            "judged_jsonl": str(judged_jsonl),
            "cache_jsonl": str(cache_jsonl),
            "cache_hits": sum(1 for row in scored if row.get("cache_hit") is True),
            "api_fallback_used": sum(1 for row in scored if row.get("api_fallback_used") is True),
            "api_skipped": sum(1 for row in scored if row.get("api_skipped") is True),
            "invalid_cache_ignored": sum(1 for row in scored if row.get("invalid_cache_ignored") is True),
            "scoring_errors": sum(1 for row in scored if row.get("scoring_error")),
            "angle_threshold_deg": args.angle_threshold_deg,
            "elevation_threshold_deg": args.elevation_threshold_deg,
            "distance_threshold_m": args.distance_threshold_m,
            "time_threshold_s": args.time_threshold_s,
            "detect_source_time_policy": args.detect_source_time_policy,
            "spatial_temporal_time_policy": args.spatial_temporal_time_policy,
            "accept_broader_source": args.accept_broader_source,
        }
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, ensure_ascii=False)
    print_summary(summary)
    print(f"Saved judged records to {judged_jsonl}")
    print(f"Saved summary to {output_json}")


if __name__ == "__main__":
    main()
