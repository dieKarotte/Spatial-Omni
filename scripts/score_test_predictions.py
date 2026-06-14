"""Score test-split predictions from a Spatial-Omni bench run.

This script is **evaluation-only** — it does not load the model. It consumes a
`predictions.jsonl` (produced by `scripts/bench_test_generate.py` or the older
`batch_bench_so_qa.py`) plus the original QA split, joins on
`pair_id`, and computes per-task metrics.

Supported task_names:
    estimate_azimuth / estimate_elevation  - numeric angle
    identify_source_by_doa                  - single source label
    identify_source_by_location             - single source label
    detect_time                             - one time span per event
    detect_source                           - list of (event, start, end)

Key features:
    * Robust regex-based extractors with per-task error categories.
    * Optional LLM judge (OpenAI-compatible, uses `gpt4o_api.py` endpoint style)
      for semantic equivalence on source-identification tasks where
      surface-form exact match fails but the prediction might still be correct
      (e.g. "bell" vs "church_bell"). LLM is ALSO used as a last-resort
      answer-extractor for verbose generations before scoring.
    * Detailed parse-fail tracking: every record records a
      `parse_status` ∈ {"ok", "fail_regex", "fail_llm_extract", "fail_empty",
      "fail_no_answer_meta"}.
    * Aggregate report distinguishes task-correctness, parse rate, and
      LLM-assist rate.

Usage (no LLM):
    python scripts/score_test_predictions.py \\
        --predictions-jsonl runs/.../bench/test/<ckpt>/predictions.jsonl \\
        --qa-root /path/to/SO-Dataset/qa \\
        --output-json runs/.../bench/test/<ckpt>/result.json

Usage (LLM judge on ambiguous source identification):
    python scripts/score_test_predictions.py \\
        --predictions-jsonl ... --qa-root ... \\
        --llm-judge --llm-model gpt-4o \\
        --llm-concurrency 8

The LLM judge is only invoked when:
    1. exact match fails,
    2. the task is `identify_source_by_doa` / `identify_source_by_location`
       (where synonyms are common) or the user passes `--llm-judge-all-tasks`.
    3. both prediction and answer are non-empty after cleaning.

Fully deterministic regex path still runs regardless of --llm-judge, so the
non-LLM `correct` column is always populated as a fallback.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from statistics import median
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Regex + small parsers
# ---------------------------------------------------------------------------

FLOAT_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
# "0.1s to 5.8s" / "from 0.1 s to 5.8 s" / "0.1 - 5.8 seconds"
TIME_SPAN_RE = re.compile(
    r"(?P<start>[-+]?\d+(?:\.\d+)?)\s*(?:s|sec|seconds)?\s*(?:to|-|–|—|until)\s*(?P<end>[-+]?\d+(?:\.\d+)?)\s*(?:s|sec|seconds)?",
    re.IGNORECASE,
)
# Event spans embedded in a longer sentence, e.g.
# "The audio contains a camera from 0.0s to 9.6s, glass from 0.0s to 0.3s, ..."
EVENT_SPAN_RE = re.compile(
    # Captures "<label> from X to Y" OR "<label> active from X to Y"
    r"(?P<label>[A-Za-z][A-Za-z\s_'-]{0,40})\s+"
    r"(?:active\s+)?(?:from\s+)?(?P<start>[-+]?\d+(?:\.\d+)?)\s*(?:s|sec|seconds)?\s*"
    r"(?:to|-|–|—|until)\s*(?P<end>[-+]?\d+(?:\.\d+)?)\s*(?:s|sec|seconds)?",
    re.IGNORECASE,
)
STOPWORDS = {
    "the", "a", "an", "sound", "source", "is", "at", "from", "to", "and",
    "of", "that", "which", "are", "can", "be", "heard", "coming", "audio",
    "clip", "contains", "features", "active", "during", "this", "in", "with",
}


def normalize_text(text: Any) -> str:
    return " ".join(str(text).strip().lower().split())


def strip_stopwords(text: str) -> str:
    return " ".join(w for w in normalize_text(text).split() if w not in STOPWORDS)


def canonicalize_label(text: str) -> str:
    """Canonicalize a source/event label for comparison.

    Lowercase + underscore -> space + strip trailing punctuation + singularize
    common plurals. This is intentionally simple; LLM judge handles harder
    synonym cases.
    """
    text = normalize_text(text).replace("_", " ")
    text = text.strip(".,;:!?\"'()[]")
    # Drop articles.
    for prefix in ("a ", "an ", "the "):
        if text.startswith(prefix):
            text = text[len(prefix):]
    # Very naive plural -> singular. Guard against -es / -ches / -ies.
    if text.endswith("ies") and len(text) > 4:
        text = text[:-3] + "y"
    elif text.endswith(("sses", "shes", "ches", "xes")):
        text = text[:-2]
    elif text.endswith("s") and len(text) > 2 and not text.endswith(("ss", "us", "is", "os")):
        text = text[:-1]
    return text.strip()


_LABEL_LEADING_JUNK = re.compile(
    r"^(?:the\s+audio\s+(?:contains|features|includes)|audio\s+(?:contains|features|includes)|"
    r"the\s+clip\s+(?:contains|features)|this\s+(?:audio\s+)?(?:clip\s+)?(?:contains|features)|"
    r"and\s+|followed\s+by\s+|then\s+|next\s+)\s*",
    re.IGNORECASE,
)
_LABEL_TRAILING_JUNK = re.compile(
    r"\s*(?:active|audible|heard|happening|occurring|present|from)\s*$",
    re.IGNORECASE,
)


def _clean_event_label(raw: str) -> str:
    """Trim connector phrases / filler from a raw label span."""
    s = raw.strip()
    # Remove a few common leading phrases.
    while True:
        new = _LABEL_LEADING_JUNK.sub("", s)
        if new == s:
            break
        s = new
    s = _LABEL_TRAILING_JUNK.sub("", s)
    return canonicalize_label(s)


def parse_first_float(text: Any) -> Optional[float]:
    match = FLOAT_RE.search(str(text))
    if match is None:
        return None
    return float(match.group(0))


def parse_time_span_first(text: Any) -> Optional[Tuple[float, float]]:
    m = TIME_SPAN_RE.search(str(text))
    if m is None:
        # Fallback: take the first two floats.
        floats = [float(x) for x in FLOAT_RE.findall(str(text))]
        if len(floats) < 2:
            return None
        s, e = floats[0], floats[1]
        if e < s:
            s, e = e, s
        return s, e
    s = float(m.group("start"))
    e = float(m.group("end"))
    if e < s:
        s, e = e, s
    return s, e


def parse_all_events(text: Any) -> List[Tuple[str, float, float]]:
    """Extract (label, start, end) tuples from a detect_source-style answer."""
    events: List[Tuple[str, float, float]] = []
    for m in EVENT_SPAN_RE.finditer(str(text)):
        label_raw = m.group("label")
        label = _clean_event_label(label_raw)
        if not label or label in STOPWORDS:
            continue
        try:
            s = float(m.group("start"))
            e = float(m.group("end"))
        except ValueError:
            continue
        if e < s:
            s, e = e, s
        events.append((label, s, e))
    return events


def angle_err_deg(pred: float, target: float) -> float:
    d = pred - target
    while d > 180.0:
        d -= 360.0
    while d <= -180.0:
        d += 360.0
    return abs(d)


def interval_iou(ps: float, pe: float, gs: float, ge: float) -> float:
    inter = max(0.0, min(pe, ge) - max(ps, gs))
    union = max(pe, ge) - min(ps, gs)
    if union <= 0:
        return 0.0
    return inter / union


def mean_or_none(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def median_or_none(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    return float(median(xs))


# ---------------------------------------------------------------------------
# LLM judge / extractor (optional, threaded)
# ---------------------------------------------------------------------------


@dataclass
class LLMConfig:
    enabled: bool = False
    model: str = "gpt-4o"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = os.environ.get(
        "SO_LLM_API_KEY",
        "",
    )
    concurrency: int = 16
    max_retries: int = 3
    timeout_s: float = 60.0
    judge_all_tasks: bool = False
    # Safety: don't let the judge rewrite history; we ONLY use it for
    # borderline exact-match misses on single-label tasks.
    judge_max_calls: int = 5000
    # Optional path. When set, every LLM call (prompt + raw response +
    # latency + caller-supplied metadata) is appended to this file as
    # JSON-Lines so you can audit / debug the judge's behavior after the
    # run. Set to None to disable logging entirely.
    log_path: Optional[str] = None


class LLMJudge:
    """Thin OpenAI-compatible wrapper. Lazily imports openai to keep the
    scorer runnable with no API deps when --llm-judge is off.
    """

    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        self._client = None
        self._lock_calls = 0
        # Lock for serialized writes when concurrent threads call _chat.
        import threading
        self._log_lock = threading.Lock()
        # Separate lock for incrementing the call counter / lazy-init of the
        # OpenAI client. Without this, 16 concurrent threads can all read
        # `_lock_calls < cap` simultaneously and overshoot the cap, and they
        # can also each instantiate a separate OpenAI() in _client_or_none().
        self._call_lock = threading.Lock()
        # Truncate (or create) the log file once so a re-run does not
        # silently append to stale records from a previous invocation.
        if cfg.log_path:
            os.makedirs(os.path.dirname(os.path.abspath(cfg.log_path)) or ".",
                          exist_ok=True)
            with open(cfg.log_path, "w", encoding="utf-8") as fh:
                # Header line for self-describing audit.
                fh.write(json.dumps({
                    "_meta": {
                        "model": cfg.model,
                        "base_url": cfg.base_url,
                        "max_retries": cfg.max_retries,
                        "timeout_s": cfg.timeout_s,
                        "judge_max_calls": cfg.judge_max_calls,
                        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    },
                }, ensure_ascii=False) + "\n")

    def _client_or_none(self):
        if not self.cfg.enabled:
            return None
        if self._client is not None:
            return self._client
        with self._call_lock:
            if self._client is not None:
                return self._client
            try:
                from openai import OpenAI  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "--llm-judge requires `pip install openai`"
                ) from exc
            self._client = OpenAI(base_url=self.cfg.base_url, api_key=self.cfg.api_key)
            return self._client

    def _append_log(self, entry: Dict[str, Any]) -> None:
        if not self.cfg.log_path:
            return
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        with self._log_lock:
            with open(self.cfg.log_path, "a", encoding="utf-8") as fh:
                fh.write(line)

    def _chat(self, prompt: str,
              meta: Optional[Dict[str, Any]] = None) -> Optional[str]:
        # Atomically check + increment the call counter so 16 concurrent
        # threads can't all slip past the cap simultaneously.
        with self._call_lock:
            if self._lock_calls >= self.cfg.judge_max_calls:
                self._append_log({
                    "status": "rate_limited",
                    "prompt": prompt,
                    "response": None,
                    "meta": meta or {},
                    "ts": time.time(),
                })
                return None
            self._lock_calls += 1
        client = self._client_or_none()
        if client is None:
            self._append_log({
                "status": "no_client",
                "prompt": prompt,
                "response": None,
                "meta": meta or {},
                "ts": time.time(),
            })
            return None
        last_exc: Optional[Exception] = None
        t_start = time.time()
        for attempt in range(self.cfg.max_retries):
            try:
                completion = client.chat.completions.create(
                    model=self.cfg.model,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=self.cfg.timeout_s,
                )
                resp = (completion.choices[0].message.content or "").strip()
                self._append_log({
                    "status": "ok",
                    "prompt": prompt,
                    "response": resp,
                    "attempts": attempt + 1,
                    "latency_s": time.time() - t_start,
                    "meta": meta or {},
                    "ts": time.time(),
                })
                return resp
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                time.sleep(min(2.0 * (attempt + 1), 10.0))
        sys.stderr.write(f"[llm-judge] call failed after {self.cfg.max_retries}: {last_exc}\n")
        self._append_log({
            "status": "failed",
            "prompt": prompt,
            "response": None,
            "error": repr(last_exc),
            "attempts": self.cfg.max_retries,
            "latency_s": time.time() - t_start,
            "meta": meta or {},
            "ts": time.time(),
        })
        return None

    # ---- task-specific prompts ----

    def extract_label(self, prediction: str, candidates: Optional[List[str]] = None) -> Optional[str]:
        """Ask the LLM to boil a verbose prediction down to a single short
        label (1-3 words).
        """
        hint = ""
        if candidates:
            hint = (
                "The sound source should be one of these canonical names "
                f"if possible: {', '.join(sorted(set(candidates))[:80])}. "
                "If the text mentions something not in the list, return the "
                "closest English name, 1-3 words, lowercase, no punctuation.\n"
            )
        prompt = (
            "Extract the single sound-source name mentioned as the ANSWER "
            "from the following text. Return ONLY the label in 1-3 English "
            "words, lowercase, no punctuation, no explanation. If nothing "
            "identifiable is mentioned, return the literal token UNKNOWN.\n"
            f"{hint}"
            f"TEXT:\n{prediction}\n"
            "ANSWER:"
        )
        out = self._chat(prompt, meta={
            "kind": "extract_label",
            "prediction": prediction,
            "candidates_used": bool(candidates),
        })
        if out is None:
            return None
        out = out.strip().splitlines()[0].strip(" \t\"'.,`").lower()
        if not out or out == "unknown":
            return None
        return out

    def judge_equivalent(self, prediction_label: str, gold_label: str,
                          task_name: str) -> Optional[bool]:
        """Ask: is `prediction_label` a semantically valid rewording of
        `gold_label`? Returns True/False or None on API failure.
        """
        prompt = (
            "You are evaluating a spatial-audio QA system. Decide whether "
            "the MODEL ANSWER and the GOLD ANSWER refer to the SAME sound "
            "source / event category. Surface wording may differ "
            "(e.g. 'footstep' vs 'footsteps', 'bell' vs 'church_bell' vs "
            "'bell ringing'). Output exactly one token: YES or NO.\n\n"
            f"TASK: {task_name}\n"
            f"MODEL ANSWER: {prediction_label}\n"
            f"GOLD ANSWER:  {gold_label}\n\n"
            "VERDICT:"
        )
        out = self._chat(prompt, meta={
            "kind": "judge_equivalent",
            "task": task_name,
            "prediction_label": prediction_label,
            "gold_label": gold_label,
        })
        if out is None:
            return None
        head = out.strip().splitlines()[0].upper()
        if head.startswith("YES"):
            return True
        if head.startswith("NO"):
            return False
        return None

    def judge_compare(self, question: str, prediction: str, gold_answer: str,
                       task_name: str) -> Optional[bool]:
        """Compare-tasks judge (compare_azimuth / compare_distance /
        compare_elevation): the gold answer is short (e.g. "guitar" /
        "the keyboard instrument") but the model may reply with a sentence
        that includes direction words ("the guitar is at a higher
        elevation than the camera"). The judge needs to know the QUESTION
        to decide if the prediction picks the same source AND assigns it
        the same role (closer / farther / higher / lower / left / right).
        """
        prompt = (
            "You are evaluating a spatial-audio QA system on a comparison "
            "task. Given the QUESTION, decide whether the MODEL ANSWER "
            "agrees with the GOLD ANSWER. Two answers agree if:\n"
            "  1. They identify the SAME sound source (synonyms allowed: "
            "'speech' == 'spoken voice', 'telephone alarm' == 'phone "
            "ringing', 'keyboard instrument' == 'piano').\n"
            "  2. They assign that source the SAME comparative role with "
            "respect to the question (e.g. 'closer to the listener', "
            "'higher elevation', 'greater leftward azimuth').\n"
            "If the model answer picks a different source, or picks the "
            "right source but with the opposite comparative role, answer NO.\n"
            "If the gold answer is just a single source name, treat the "
            "model answer as correct as long as it picks that same source "
            "(the comparative role is assumed implicit in the gold).\n\n"
            f"TASK: {task_name}\n"
            f"QUESTION: {question}\n"
            f"MODEL ANSWER: {prediction}\n"
            f"GOLD ANSWER:  {gold_answer}\n\n"
            "Output exactly one token: YES or NO.\n"
            "VERDICT:"
        )
        out = self._chat(prompt, meta={
            "kind": "judge_compare",
            "task": task_name,
            "question": question,
            "prediction": prediction,
            "gold_answer": gold_answer,
        })
        if out is None:
            return None
        head = out.strip().splitlines()[0].upper()
        if head.startswith("YES"):
            return True
        if head.startswith("NO"):
            return False
        return None


# ---------------------------------------------------------------------------
# Per-task scorers
# ---------------------------------------------------------------------------


@dataclass
class TaskScore:
    """Result of scoring a single (prediction, qa) pair."""
    pair_id: Any
    task_name: str
    prediction: str
    answer: str
    canonical_answer: Optional[str]
    correct: float = 0.0                # 0.0 / 1.0 (or IoU for spans)
    parse_status: str = "ok"            # ok, fail_regex, fail_llm_extract, fail_empty, fail_no_answer_meta
    metric_type: str = "exact_match"
    details: Dict[str, Any] = field(default_factory=dict)
    llm_used: bool = False


def score_estimate_angle(record: Dict[str, Any], pred_text: str,
                           is_azimuth: bool, angle_threshold_deg: float,
                           ) -> TaskScore:
    task = str(record["task_name"])
    meta = record.get("answer_meta") or {}
    key = "azimuth_deg" if is_azimuth else "elevation_deg"
    target = meta.get(key)
    # Fallback: parse from answer text.
    if target is None:
        target = parse_first_float(record.get("answer", ""))

    ts = TaskScore(
        pair_id=record.get("pair_id"),
        task_name=task,
        prediction=pred_text,
        answer=str(record.get("answer", "")),
        canonical_answer=record.get("canonical_answer"),
        metric_type=("er%d_angle" % int(angle_threshold_deg)
                     if is_azimuth
                     else "abs%d_angle" % int(angle_threshold_deg)),
    )

    if target is None:
        ts.parse_status = "fail_no_answer_meta"
        return ts
    if not str(pred_text).strip():
        ts.parse_status = "fail_empty"
        return ts

    pred = parse_first_float(pred_text)
    if pred is None:
        ts.parse_status = "fail_regex"
        return ts

    err = angle_err_deg(float(pred), float(target)) if is_azimuth \
        else abs(float(pred) - float(target))
    ts.details = {
        "predicted_deg": float(pred),
        "target_deg": float(target),
        "error_deg": float(err),
        "threshold_deg": angle_threshold_deg,
    }
    ts.correct = float(err <= angle_threshold_deg)
    return ts


def score_identify_source(record: Dict[str, Any], pred_text: str,
                            llm: LLMJudge, llm_allowed: bool,
                            candidate_labels: Optional[List[str]] = None,
                            ) -> TaskScore:
    task = str(record["task_name"])
    gold = record.get("canonical_answer") or record.get("answer", "")
    gold_norm = canonicalize_label(str(gold))

    ts = TaskScore(
        pair_id=record.get("pair_id"),
        task_name=task,
        prediction=pred_text,
        answer=str(record.get("answer", "")),
        canonical_answer=record.get("canonical_answer"),
        metric_type="source_label_match",
    )

    if not gold_norm:
        ts.parse_status = "fail_no_answer_meta"
        return ts
    if not str(pred_text).strip():
        ts.parse_status = "fail_empty"
        return ts

    pred_norm = canonicalize_label(pred_text)
    # Stage 1: direct canonical label match.
    if pred_norm == gold_norm:
        ts.correct = 1.0
        ts.details = {"pred_label": pred_norm, "gold_label": gold_norm, "match_stage": "canonical"}
        return ts

    # Stage 2: substring match (either direction).
    if gold_norm and gold_norm in pred_norm:
        ts.correct = 1.0
        ts.details = {"pred_label": pred_norm, "gold_label": gold_norm, "match_stage": "substring"}
        return ts

    # Stage 3: stopword-stripped match.
    pred_s = strip_stopwords(pred_text)
    gold_s = strip_stopwords(str(gold))
    if pred_s and gold_s and (pred_s == gold_s or gold_s in pred_s):
        ts.correct = 1.0
        ts.details = {"pred_label": pred_s, "gold_label": gold_s, "match_stage": "stopword_stripped"}
        return ts

    # Stage 4: LLM extractor — boil verbose prediction to a single label,
    # then compare normalized.
    details: Dict[str, Any] = {"pred_label": pred_norm, "gold_label": gold_norm, "match_stage": "none"}
    if llm_allowed and llm.cfg.enabled:
        extracted = llm.extract_label(pred_text, candidates=candidate_labels)
        if extracted is not None:
            ts.llm_used = True
            extracted_norm = canonicalize_label(extracted)
            details["llm_extracted"] = extracted_norm
            if extracted_norm == gold_norm:
                ts.correct = 1.0
                details["match_stage"] = "llm_extract"
                ts.details = details
                return ts
            # Stage 5: LLM judge semantic equivalence.
            verdict = llm.judge_equivalent(extracted_norm or pred_norm, gold_norm, task)
            if verdict is True:
                ts.correct = 1.0
                details["match_stage"] = "llm_judge"
                ts.details = details
                return ts
            if verdict is None and extracted is None:
                ts.parse_status = "fail_llm_extract"
    # Fell through all stages.
    ts.details = details
    return ts


def score_detect_time(record: Dict[str, Any], pred_text: str,
                        iou_threshold: float) -> TaskScore:
    task = str(record["task_name"])
    meta = record.get("answer_meta") or {}
    refs = record.get("source_refs") or []

    # Pick the gold span: prefer answer_meta.time_span / start_time+end_time,
    # else source_refs[0], else parse the answer text.
    gold_span: Optional[Tuple[float, float]] = None
    span_meta = meta.get("time_span")
    if isinstance(span_meta, (list, tuple)) and len(span_meta) == 2:
        gold_span = (float(span_meta[0]), float(span_meta[1]))
    elif meta.get("start_time") is not None and meta.get("end_time") is not None:
        gold_span = (float(meta["start_time"]), float(meta["end_time"]))
    elif refs and isinstance(refs[0], dict):
        r0 = refs[0]
        if r0.get("start_time") is not None and r0.get("end_time") is not None:
            gold_span = (float(r0["start_time"]), float(r0["end_time"]))
    if gold_span is None:
        gold_span = parse_time_span_first(record.get("answer", ""))

    ts = TaskScore(
        pair_id=record.get("pair_id"),
        task_name=task,
        prediction=pred_text,
        answer=str(record.get("answer", "")),
        canonical_answer=record.get("canonical_answer"),
        metric_type="time_span_iou",
    )

    if gold_span is None:
        ts.parse_status = "fail_no_answer_meta"
        return ts
    if not str(pred_text).strip():
        ts.parse_status = "fail_empty"
        return ts
    pred_span = parse_time_span_first(pred_text)
    if pred_span is None:
        ts.parse_status = "fail_regex"
        return ts
    ps, pe = pred_span
    gs, ge = gold_span
    iou = interval_iou(ps, pe, gs, ge)
    ts.details = {
        "predicted_span": [ps, pe],
        "target_span": [gs, ge],
        "iou": iou,
        "start_error_s": abs(ps - gs),
        "end_error_s": abs(pe - ge),
        "iou_threshold": iou_threshold,
    }
    # Use IoU directly as soft-correct (0..1), plus a binary @threshold.
    ts.correct = float(iou)
    ts.details["correct_binary"] = int(iou >= iou_threshold)
    return ts


def score_detect_source(record: Dict[str, Any], pred_text: str,
                          llm: LLMJudge, llm_allowed: bool,
                          iou_threshold: float) -> TaskScore:
    """Detect-source: list of (label, start, end). Score with event-level F1
    under (label_match AND iou>=thr). Label match uses canonical normalization;
    optional LLM synonym matching on a label-by-label basis (disabled by default
    because of API cost on long lists — enable via --llm-judge-all-tasks).
    """
    task = str(record["task_name"])
    refs = record.get("source_refs") or []
    gold_events: List[Tuple[str, float, float]] = []
    for r in refs:
        if not isinstance(r, dict):
            continue
        label = canonicalize_label(str(r.get("class_name") or ""))
        if label and r.get("start_time") is not None and r.get("end_time") is not None:
            gold_events.append((label, float(r["start_time"]), float(r["end_time"])))
    if not gold_events:
        # Fallback: parse the answer text.
        gold_events = parse_all_events(record.get("answer", ""))

    ts = TaskScore(
        pair_id=record.get("pair_id"),
        task_name=task,
        prediction=pred_text,
        answer=str(record.get("answer", "")),
        canonical_answer=record.get("canonical_answer"),
        metric_type="detect_source_f1",
    )

    if not gold_events:
        ts.parse_status = "fail_no_answer_meta"
        return ts
    if not str(pred_text).strip():
        ts.parse_status = "fail_empty"
        return ts
    pred_events = parse_all_events(pred_text)
    if not pred_events:
        ts.parse_status = "fail_regex"
        return ts

    # Greedy matching: for each gold event, find the highest-IoU pred event
    # whose label matches.
    matched_pred = [False] * len(pred_events)
    tp = 0
    ious: List[float] = []
    for (gl, gs, ge) in gold_events:
        best_idx = -1
        best_iou = 0.0
        for i, (pl, ps, pe) in enumerate(pred_events):
            if matched_pred[i]:
                continue
            label_ok = (pl == gl)
            if not label_ok and llm_allowed and llm.cfg.enabled:
                verdict = llm.judge_equivalent(pl, gl, task)
                if verdict is True:
                    label_ok = True
                    ts.llm_used = True
            if not label_ok:
                continue
            iou = interval_iou(ps, pe, gs, ge)
            if iou > best_iou:
                best_iou = iou
                best_idx = i
        if best_idx >= 0 and best_iou >= iou_threshold:
            matched_pred[best_idx] = True
            tp += 1
            ious.append(best_iou)
    n_gold = len(gold_events)
    n_pred = len(pred_events)
    precision = tp / max(n_pred, 1)
    recall = tp / max(n_gold, 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    ts.details = {
        "n_gold_events": n_gold,
        "n_pred_events": n_pred,
        "tp": tp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matched_iou_mean": mean_or_none(ious),
        "iou_threshold": iou_threshold,
    }
    ts.correct = float(f1)
    return ts


# ---------------------------------------------------------------------------
# Medium-split per-task scorers (count_sources, classify_motion,
# estimate_distance, onset_from_location)
# ---------------------------------------------------------------------------


def _parse_first_int(text: Any) -> Optional[int]:
    """Return the first integer found in text. Used by count_sources where
    the GT is a small integer ('1', '2', '3', ...) and the prediction may
    be embedded in a sentence ('There are 2 sources active.')."""
    m = re.search(r"-?\d+", str(text))
    if m is None:
        return None
    try:
        return int(m.group(0))
    except ValueError:
        return None


def score_count_sources(record: Dict[str, Any], pred_text: str) -> "TaskScore":
    """count_sources: GT is an integer (active_count). Just extract the
    first integer from prediction and compare. No LLM needed.
    """
    task = str(record["task_name"])
    meta = record.get("answer_meta") or {}
    target = meta.get("active_count")
    if target is None:
        target = _parse_first_int(record.get("answer", ""))

    ts = TaskScore(
        pair_id=record.get("pair_id"),
        task_name=task,
        prediction=pred_text,
        answer=str(record.get("answer", "")),
        canonical_answer=record.get("canonical_answer"),
        metric_type="count_exact_match",
    )

    if target is None:
        ts.parse_status = "fail_no_answer_meta"
        return ts
    if not str(pred_text).strip():
        ts.parse_status = "fail_empty"
        return ts
    pred = _parse_first_int(pred_text)
    if pred is None:
        ts.parse_status = "fail_regex"
        return ts
    err = abs(int(pred) - int(target))
    ts.details = {
        "predicted_count": int(pred),
        "target_count": int(target),
        "abs_error": int(err),
    }
    ts.correct = float(int(pred) == int(target))
    return ts


def score_estimate_distance(record: Dict[str, Any], pred_text: str,
                              rel_threshold: float = 0.3) -> "TaskScore":
    """estimate_distance: GT is a float (meters). The "correct" signal is
    the relative error: rel_err = |pred - gt| / max(|gt|, eps), with a
    threshold (default 0.3, i.e. within 30% of the true distance).

    Why relative instead of absolute: a 1m error is huge for a near-source
    (gt=1.5m -> 67% off) but acceptable for a far source (gt=8m -> 12.5%
    off). A relative threshold gives a fair signal across the [near, far]
    range.

    Aggregates report:
        - mean / median absolute error (m)
        - mean / median relative error
        - acc within `rel_threshold` (the binary "correct" signal)
        - extra acc points at relaxed/tight thresholds (rel<0.2 / rel<0.5)
    """
    task = str(record["task_name"])
    meta = record.get("answer_meta") or {}
    target = meta.get("distance_m")
    if target is None:
        target = parse_first_float(record.get("answer", ""))

    ts = TaskScore(
        pair_id=record.get("pair_id"),
        task_name=task,
        prediction=pred_text,
        answer=str(record.get("answer", "")),
        canonical_answer=record.get("canonical_answer"),
        metric_type=f"rel{rel_threshold:.2f}_distance",
    )

    if target is None:
        ts.parse_status = "fail_no_answer_meta"
        return ts
    if not str(pred_text).strip():
        ts.parse_status = "fail_empty"
        return ts
    pred = parse_first_float(pred_text)
    if pred is None:
        ts.parse_status = "fail_regex"
        return ts
    err = abs(float(pred) - float(target))
    rel_err = err / max(abs(float(target)), 1e-3)
    ts.details = {
        "predicted_m": float(pred),
        "target_m": float(target),
        "abs_error_m": float(err),
        "rel_error": float(rel_err),
        "rel_threshold": float(rel_threshold),
    }
    ts.correct = float(rel_err <= rel_threshold)
    return ts


def score_onset_from_location(record: Dict[str, Any], pred_text: str,
                                 within_s: float = 0.4) -> "TaskScore":
    """onset_from_location: GT is the onset time (seconds). Score reports
    the time error and acc within `within_s` (default 0.4s)."""
    task = str(record["task_name"])
    meta = record.get("answer_meta") or {}
    target = meta.get("onset_time")
    # Fallback: try to pull a "X.Xs" pattern from canonical/answer text.
    if target is None:
        target = parse_first_float(record.get("canonical_answer") or
                                    record.get("answer", ""))

    ts = TaskScore(
        pair_id=record.get("pair_id"),
        task_name=task,
        prediction=pred_text,
        answer=str(record.get("answer", "")),
        canonical_answer=record.get("canonical_answer"),
        metric_type=f"abs{within_s}s_onset",
    )

    if target is None:
        ts.parse_status = "fail_no_answer_meta"
        return ts
    if not str(pred_text).strip():
        ts.parse_status = "fail_empty"
        return ts
    pred = parse_first_float(pred_text)
    if pred is None:
        ts.parse_status = "fail_regex"
        return ts
    err = abs(float(pred) - float(target))
    ts.details = {
        "predicted_s": float(pred),
        "target_s": float(target),
        "abs_error_s": float(err),
        "within_s_threshold": float(within_s),
    }
    ts.correct = float(err <= within_s)
    return ts


# Default canonical motion labels. The QA generator uses 3 categories:
# stationary / moving / approaching|receding (some variants may say
# 'approaching' or 'moving towards'). Substring + LLM judge handle synonyms.
_MOTION_CANONICAL_LABELS = (
    "stationary", "moving", "approaching", "receding",
    "moving towards", "moving away",
)


def score_classify_motion(record: Dict[str, Any], pred_text: str,
                            llm: "LLMJudge") -> "TaskScore":
    """classify_motion: GT is a short label ('stationary' / 'moving' / ...).
    Strategy: substring match first (covers 90% of cases like
    'The laughter remains stationary throughout its duration.'),
    then fall back to LLM judge for ambiguous synonyms.
    """
    task = str(record["task_name"])
    meta = record.get("answer_meta") or {}
    gold = meta.get("motion_label") or record.get("canonical_answer") or \
        record.get("answer", "")
    gold_norm = canonicalize_label(str(gold))

    ts = TaskScore(
        pair_id=record.get("pair_id"),
        task_name=task,
        prediction=pred_text,
        answer=str(record.get("answer", "")),
        canonical_answer=record.get("canonical_answer"),
        metric_type="motion_label_match",
    )

    if not gold_norm:
        ts.parse_status = "fail_no_answer_meta"
        return ts
    if not str(pred_text).strip():
        ts.parse_status = "fail_empty"
        return ts

    pred_norm = canonicalize_label(pred_text)
    details: Dict[str, Any] = {
        "predicted_norm": pred_norm,
        "gold_norm": gold_norm,
    }

    # Stage 1: exact / substring match on canonicalized labels.
    if pred_norm == gold_norm:
        details["match_stage"] = "exact"
        ts.correct = 1.0
        ts.details = details
        return ts
    if gold_norm in pred_norm.split():
        details["match_stage"] = "word"
        ts.correct = 1.0
        ts.details = details
        return ts
    if gold_norm in pred_norm:
        details["match_stage"] = "substring"
        ts.correct = 1.0
        ts.details = details
        return ts

    # Stage 2: LLM judge. classify_motion has many phrasings ("staying still"
    # vs "stationary", "approaching" vs "moving towards the listener", etc.)
    # so we always allow the LLM to break ties when substring fails.
    if llm.cfg.enabled:
        verdict = llm.judge_equivalent(pred_norm, gold_norm, task)
        ts.llm_used = True
        if verdict is True:
            details["match_stage"] = "llm_judge"
            ts.correct = 1.0
        elif verdict is False:
            details["match_stage"] = "none"
            ts.correct = 0.0
        else:
            ts.parse_status = "fail_llm_extract"
            details["match_stage"] = "llm_failed"
        ts.details = details
        return ts

    # No LLM available -> mark as miss but keep parse_status=ok.
    details["match_stage"] = "none"
    ts.correct = 0.0
    ts.details = details
    return ts


# ---------------------------------------------------------------------------
# Hard-split per-task scorers
# ---------------------------------------------------------------------------


# Canonical direction tokens. Each entry: (canonical, alternates_regex).
# Alternates are matched as whole-word phrases (word-boundary anchored) and
# normalized to the canonical form. This catches both:
#   - separator variants: "front-left" / "front left" / "front_left"
#   - synonym variants:   "upper" -> "above", "behind" -> "back", etc.
# Order matters: compound directions first (so "front-left" beats "front").
_DIRECTION_ALTERNATES: List[Tuple[str, "re.Pattern[str]"]] = [
    # Compound 2-axis first.
    ("front-left",  re.compile(r'\b(front[\s_\-]+left|left[\s_\-]+front|upper[\s_\-]+left[\s_\-]+front|forward[\s_\-]+left)\b', re.I)),
    ("front-right", re.compile(r'\b(front[\s_\-]+right|right[\s_\-]+front|forward[\s_\-]+right)\b', re.I)),
    ("back-left",   re.compile(r'\b(back[\s_\-]+left|left[\s_\-]+back|rear[\s_\-]+left|behind[\s_\-]+(?:and[\s_\-]+)?left|left[\s_\-]+behind)\b', re.I)),
    ("back-right",  re.compile(r'\b(back[\s_\-]+right|right[\s_\-]+back|rear[\s_\-]+right|behind[\s_\-]+(?:and[\s_\-]+)?right|right[\s_\-]+behind)\b', re.I)),
    # Single axis.
    ("front", re.compile(r'\b(front|forward|ahead|in[\s_\-]+front)\b', re.I)),
    ("back",  re.compile(r'\b(back|behind|rear|backward|backwards)\b', re.I)),
    ("left",  re.compile(r'\b(left|leftward|leftwards|port|leftside|left[\s_\-]+side)\b', re.I)),
    ("right", re.compile(r'\b(right|rightward|rightwards|starboard|rightside|right[\s_\-]+side)\b', re.I)),
    # Vertical.
    ("above", re.compile(r'\b(above|over|overhead|up|upward|upwards|upper|top|high|elevated)\b', re.I)),
    ("below", re.compile(r'\b(below|under|underneath|down|downward|downwards|lower|bottom|beneath)\b', re.I)),
]

# Flat list of canonical tokens (for diagnostic purposes).
_DIRECTION_TOKENS = tuple(c for c, _ in _DIRECTION_ALTERNATES)


def _extract_directions(text: str) -> List[str]:
    """Return canonical direction tokens that appear in `text`, in order of
    first occurrence. Compound directions (front-left etc.) are matched
    before their single-axis components, and synonym variants ("upper",
    "behind", "overhead") map to the canonical form. Separator variants
    ("front left", "front_left", "front-left") are all accepted.
    """
    s = (text or "").lower()
    if not s:
        return []
    # Find all (start_pos, canonical) by iterating regexes in priority order.
    # When a span matches a compound, mask it out so single-axis regexes
    # don't double-count "front" inside a "front-left" span.
    masked = list(s)
    hits: List[Tuple[int, str]] = []
    for canonical, pat in _DIRECTION_ALTERNATES:
        for m in pat.finditer("".join(masked)):
            hits.append((m.start(), canonical))
            # Mask the matched span so later regexes don't see it.
            for i in range(m.start(), m.end()):
                masked[i] = " "
    # Dedupe by canonical, preserving earliest position.
    seen: Dict[str, int] = {}
    for pos, canon in hits:
        if canon not in seen or pos < seen[canon]:
            seen[canon] = pos
    return [c for c, _ in sorted(seen.items(), key=lambda kv: kv[1])]


def score_compare(record: Dict[str, Any], pred_text: str,
                    llm: "LLMJudge",
                    strict: bool = False) -> "TaskScore":
    """compare_azimuth / compare_distance / compare_elevation.

    Default mode: substring match the gold source label in the prediction;
    fall back to LLM judge when substring fails. This gives high recall
    but cannot detect "correct source, wrong comparative role".

    Strict mode (`strict=True`): always invoke the LLM judge, even when the
    substring matches. Use this when you need to penalize answers that pick
    the right source but assign it the OPPOSITE comparative role
    (e.g. "X is farther" when gold says "X is closer"). Costs N LLM calls
    instead of ~10% of N. Requires --llm-judge.
    """
    task = str(record["task_name"])
    gold = (record.get("canonical_answer") or record.get("answer") or "")
    gold_norm = canonicalize_label(str(gold))

    ts = TaskScore(
        pair_id=record.get("pair_id"),
        task_name=task,
        prediction=pred_text,
        answer=str(record.get("answer", "")),
        canonical_answer=record.get("canonical_answer"),
        metric_type="compare_semantic_match",
    )
    if not gold_norm:
        ts.parse_status = "fail_no_answer_meta"
        return ts
    if not str(pred_text).strip():
        ts.parse_status = "fail_empty"
        return ts

    pred_norm = canonicalize_label(pred_text)
    details: Dict[str, Any] = {"predicted_norm": pred_norm, "gold_norm": gold_norm}

    # Strict mode: always go to the LLM (when enabled).
    if strict and llm.cfg.enabled:
        verdict = llm.judge_compare(
            question=str(record.get("question", "")),
            prediction=pred_text,
            gold_answer=str(record.get("answer", "")),
            task_name=task,
        )
        ts.llm_used = True
        if verdict is True:
            details["match_stage"] = "llm_judge"
            ts.correct = 1.0
        elif verdict is False:
            details["match_stage"] = "llm_judge_no"
            ts.correct = 0.0
        else:
            ts.parse_status = "fail_llm_extract"
            details["match_stage"] = "llm_failed"
        ts.details = details
        return ts

    if gold_norm == pred_norm:
        details["match_stage"] = "exact"
        ts.correct = 1.0; ts.details = details; return ts
    if gold_norm in pred_norm:
        details["match_stage"] = "substring"
        ts.correct = 1.0; ts.details = details; return ts

    if llm.cfg.enabled:
        verdict = llm.judge_compare(
            question=str(record.get("question", "")),
            prediction=pred_text,
            gold_answer=str(record.get("answer", "")),
            task_name=task,
        )
        ts.llm_used = True
        if verdict is True:
            details["match_stage"] = "llm_judge"
            ts.correct = 1.0
        elif verdict is False:
            details["match_stage"] = "none"
            ts.correct = 0.0
        else:
            ts.parse_status = "fail_llm_extract"
            details["match_stage"] = "llm_failed"
        ts.details = details
        return ts

    details["match_stage"] = "none"
    ts.correct = 0.0; ts.details = details; return ts


def score_relative(record: Dict[str, Any], pred_text: str) -> "TaskScore":
    """relative_left_right / relative_position.

    Canonical answer is a full sentence like 'crackle is to the right of
    vehicle.' Use direction-token overlap as the metric: a prediction is
    correct if all directional tokens that appear in the canonical answer
    also appear in the prediction (and the prediction does not contradict
    by including the OPPOSITE direction).
    """
    task = str(record["task_name"])
    gold = (record.get("canonical_answer") or record.get("answer") or "")

    ts = TaskScore(
        pair_id=record.get("pair_id"),
        task_name=task,
        prediction=pred_text,
        answer=str(record.get("answer", "")),
        canonical_answer=record.get("canonical_answer"),
        metric_type="relative_direction_match",
    )
    if not gold:
        ts.parse_status = "fail_no_answer_meta"; return ts
    if not str(pred_text).strip():
        ts.parse_status = "fail_empty"; return ts

    gold_dirs = _extract_directions(gold)
    pred_dirs = _extract_directions(pred_text)
    pred_set = set(pred_dirs)

    OPPOSITES = {"left": "right", "right": "left",
                 "front": "back", "back": "front",
                 "above": "below", "below": "above",
                 "up": "down", "down": "up",
                 "front-left": "back-right", "back-right": "front-left",
                 "front-right": "back-left", "back-left": "front-right"}

    matched = [d for d in gold_dirs if d in pred_set]
    contradicted = [d for d in gold_dirs if OPPOSITES.get(d) in pred_set]

    details = {
        "gold_directions": gold_dirs,
        "pred_directions": pred_dirs,
        "matched_directions": matched,
        "contradicted_directions": contradicted,
    }
    # Correct if all gold directions matched and no opposite is asserted.
    if gold_dirs and len(matched) == len(gold_dirs) and not contradicted:
        ts.correct = 1.0
        details["match_stage"] = "all_directions"
    elif matched and not contradicted:
        # Partial direction match (e.g. only the left/right axis but not
        # the up/down axis). Score as fractional credit but the binary
        # "correct" flag is 0 (matches the convention of the other tasks).
        ts.correct = 0.0
        details["match_stage"] = "partial"
        details["partial_credit"] = len(matched) / max(len(gold_dirs), 1)
    else:
        ts.correct = 0.0
        details["match_stage"] = "none"
    ts.details = details
    return ts


def score_spatial_temporal(record: Dict[str, Any], pred_text: str) -> "TaskScore":
    """spatial_temporal: canonical answer is '<source> from <direction>.'

    Scoring (continuous, not hard AND):
      - src_match: gold source token(s) appear in prediction (substring on
        canonicalized text). Partial credit: fraction of gold-source words
        present in pred.
      - dir_recall: |gold_dirs ∩ pred_dirs| / |gold_dirs|. If gold has no
        directions, treat as 1.0.
      - dir_precision: |gold_dirs ∩ pred_dirs| / |pred_dirs| when pred_dirs
        non-empty (diagnostic only; not in correct).
      - correct = src_match * dir_recall  (in [0,1])
      - hard_correct = float(src_match==1.0 AND dir_recall==1.0)
        (kept for backward-compat with old summaries)
    Time IoU is reported as auxiliary detail (gold span comes from
    answer_meta['time_span'] when available).
    """
    task = str(record["task_name"])
    gold_can = (record.get("canonical_answer") or "").strip()
    if not gold_can:
        gold_can = record.get("answer", "")
    gold_can_norm = canonicalize_label(gold_can)
    pred_norm = canonicalize_label(pred_text)

    ts = TaskScore(
        pair_id=record.get("pair_id"),
        task_name=task,
        prediction=pred_text,
        answer=str(record.get("answer", "")),
        canonical_answer=record.get("canonical_answer"),
        metric_type="spatial_temporal_match",
    )
    if not gold_can_norm:
        ts.parse_status = "fail_no_answer_meta"; return ts
    if not pred_norm:
        ts.parse_status = "fail_empty"; return ts

    # Extract source: text before " from " in canonical.
    parts = gold_can_norm.split(" from ", 1)
    if len(parts) == 2:
        gold_source = parts[0].strip()
        gold_dir_text = parts[1].strip()
    else:
        gold_source = gold_can_norm
        gold_dir_text = ""
    gold_dirs = _extract_directions(gold_dir_text)
    pred_dirs = _extract_directions(pred_text)
    pred_dirs_set = set(pred_dirs)
    gold_dirs_set = set(gold_dirs)

    # Source match: try whole-phrase substring first; fall back to
    # word-overlap fraction so multi-word sources still get credit if the
    # model paraphrases ("door bell" vs "doorbell").
    if gold_source and gold_source in pred_norm:
        src_match = 1.0
        src_stage = "phrase"
    elif gold_source:
        gold_words = [w for w in gold_source.split() if w]
        pred_words = set(pred_norm.split())
        if gold_words:
            hits = sum(1 for w in gold_words if w in pred_words)
            # Also accept space-stripped concatenation ("doorbell" matching "door bell").
            joined = "".join(gold_words)
            if joined and joined in pred_norm.replace(" ", ""):
                src_match = 1.0
                src_stage = "concat"
            else:
                src_match = hits / len(gold_words)
                src_stage = "word_overlap" if src_match > 0 else "miss"
        else:
            src_match = 0.0
            src_stage = "miss"
    else:
        src_match = 1.0  # no gold source to match
        src_stage = "no_gold_source"

    if gold_dirs_set:
        dir_inter = gold_dirs_set & pred_dirs_set
        dir_recall = len(dir_inter) / len(gold_dirs_set)
        dir_precision = (
            len(dir_inter) / len(pred_dirs_set) if pred_dirs_set else 0.0
        )
    else:
        dir_recall = 1.0
        dir_precision = 1.0

    soft_correct = float(src_match * dir_recall)
    hard_correct = float(src_match >= 1.0 and dir_recall >= 1.0)

    details: Dict[str, Any] = {
        "gold_source": gold_source,
        "gold_directions": gold_dirs,
        "pred_directions": pred_dirs,
        "src_match": float(src_match),
        "src_stage": src_stage,
        "dir_recall": float(dir_recall),
        "dir_precision": float(dir_precision),
        "hard_correct": hard_correct,
    }

    # Aux: time IoU when gold time_span and a parseable [a, b] pattern in
    # prediction both exist.
    meta = record.get("answer_meta") or {}
    gt_span = meta.get("time_span")
    if gt_span and len(gt_span) == 2:
        pred_span = parse_time_span_first(pred_text)
        if pred_span is not None:
            iou = interval_iou(pred_span[0], pred_span[1],
                                 float(gt_span[0]), float(gt_span[1]))
            details["pred_time_span"] = list(pred_span)
            details["gold_time_span"] = [float(gt_span[0]), float(gt_span[1])]
            details["time_iou"] = float(iou)

    ts.correct = soft_correct
    ts.details = details
    if hard_correct:
        ts.details["match_stage"] = "ok"
    elif src_match >= 1.0:
        ts.details["match_stage"] = "src_only"
    elif dir_recall >= 1.0:
        ts.details["match_stage"] = "dir_only"
    else:
        ts.details["match_stage"] = "partial" if soft_correct > 0 else "none"
    return ts


def score_multi_hop(record: Dict[str, Any], pred_text: str,
                       llm: "LLMJudge") -> "TaskScore":
    """multi_hop: canonical is a single source label (e.g. 'telephone alarm').
    Score with substring + LLM-judge fallback (same recipe as
    identify_source). Also report event-F1 (binary, since gold is single
    event) and time-IoU when source_refs and a parseable time span are
    available in the prediction.
    """
    task = str(record["task_name"])
    gold = (record.get("canonical_answer") or "").strip()
    gold_norm = canonicalize_label(gold)
    pred_norm = canonicalize_label(pred_text)

    ts = TaskScore(
        pair_id=record.get("pair_id"),
        task_name=task,
        prediction=pred_text,
        answer=str(record.get("answer", "")),
        canonical_answer=record.get("canonical_answer"),
        metric_type="multi_hop_match",
    )
    if not gold_norm:
        ts.parse_status = "fail_no_answer_meta"; return ts
    if not pred_norm:
        ts.parse_status = "fail_empty"; return ts

    details: Dict[str, Any] = {"predicted_norm": pred_norm, "gold_norm": gold_norm}

    if gold_norm == pred_norm:
        details["match_stage"] = "exact"
        ts.correct = 1.0
    elif gold_norm in pred_norm:
        details["match_stage"] = "substring"
        ts.correct = 1.0
    elif llm.cfg.enabled:
        verdict = llm.judge_equivalent(pred_norm, gold_norm, task)
        ts.llm_used = True
        if verdict is True:
            details["match_stage"] = "llm_judge"; ts.correct = 1.0
        elif verdict is False:
            details["match_stage"] = "none"; ts.correct = 0.0
        else:
            ts.parse_status = "fail_llm_extract"
            details["match_stage"] = "llm_failed"
    else:
        details["match_stage"] = "none"
        ts.correct = 0.0

    # Event F1 with single gold event (binary by definition):
    details["event_f1"] = float(ts.correct)
    details["event_precision"] = float(ts.correct)
    details["event_recall"] = float(ts.correct)

    # Time IoU when both gold (from source_refs of the matching class) and
    # a parseable time span in prediction are available.
    refs = record.get("source_refs") or []
    if refs:
        # Pick the gold event whose class_name matches gold_norm best.
        gold_event = None
        for r in refs:
            cn = canonicalize_label(str(r.get("class_name", "")))
            if cn == gold_norm or gold_norm in cn or cn in gold_norm:
                gold_event = r; break
        if gold_event is None:
            gold_event = refs[0]
        gs = float(gold_event.get("start_time", 0.0))
        ge = float(gold_event.get("end_time", 0.0))
        pred_span = parse_time_span_first(pred_text)
        if pred_span is not None and ge > gs:
            iou = interval_iou(pred_span[0], pred_span[1], gs, ge)
            details["pred_time_span"] = list(pred_span)
            details["gold_time_span"] = [gs, ge]
            details["time_iou"] = float(iou)
    ts.details = details
    return ts


# Patterns for stripping wrapper prose around the actual transcript.
# Models tend to emit things like:
#   The speaker said, "The cat was unable to stroll."
#   The audio says: turn left at the next intersection.
#   They mention that they are going home.
# We try several candidate extractions and take the one with the lowest WER
# against the gold (lowest = closest to gold = best wrapper-stripping).
_SPEECH_QUOTE_RE = re.compile(
    r'["“”‘’\'`]([^"“”‘’\'`]{1,500}?)'
    r'["“”‘’\'`]',
    re.S,
)
_SPEECH_PREFIX_RE = re.compile(
    r'^\s*(?:the\s+)?'
    r'(?:speaker|person|man|woman|voice|audio|recording|narrator|speech)\s+'
    r'(?:said|says|is\s+saying|stated|states|mentioned|mentions|'
    r'told?\s+\w+|told|tells?\s+\w+|asked?|asks?|exclaim(?:s|ed)?|'
    r'utter(?:s|ed)?|spoke|spoken)'
    r'\s*(?:that\s+|,\s*|:\s*|-\s*)?',
    re.I,
)
_SPEECH_COLON_RE = re.compile(r'^[^:\n]{0,80}:\s*(.+)$', re.S)


def _speech_candidates(text: str) -> List[str]:
    """Extract candidate transcript spans from a (possibly verbose) prediction.
    Order is best-effort; the caller picks the candidate with the lowest WER
    against the gold transcript. Original text is always included as a fallback.
    """
    raw = (text or "").strip()
    cands: List[str] = []

    # 1. Quoted spans. Take ALL of them (concatenated) and the longest one
    # individually -- transcripts can be split across multiple quoted clauses.
    quoted = _SPEECH_QUOTE_RE.findall(raw)
    if quoted:
        cands.append(max(quoted, key=len))
        if len(quoted) > 1:
            cands.append(" ".join(quoted))

    # 2. After-colon tail (e.g. "the audio says: <transcript>").
    m = _SPEECH_COLON_RE.match(raw)
    if m:
        cands.append(m.group(1))

    # 3. Strip "the speaker said / the audio says / ..." prefix.
    stripped = _SPEECH_PREFIX_RE.sub("", raw)
    if stripped and stripped != raw:
        cands.append(stripped)
        # Combine: prefix-stripped + after-colon (handles "the speaker said: X")
        m2 = _SPEECH_COLON_RE.match(stripped)
        if m2:
            cands.append(m2.group(1))

    # 4. Original text as final fallback.
    cands.append(raw)

    # Trim quotes / punctuation, dedupe while preserving order.
    cleaned: List[str] = []
    seen = set()
    for c in cands:
        c = (c or "").strip().strip('.,;:!?"“”‘’\'` ')
        if c and c not in seen:
            seen.add(c)
            cleaned.append(c)
    return cleaned or [raw]


def _wer(reference: str, hypothesis: str) -> Tuple[float, int, int]:
    """Standard word error rate: (S+D+I)/N over reference. Returns
    (wer, num_errors, num_ref_words). Lowercased word-level Levenshtein.
    """
    ref_words = canonicalize_label(reference).split()
    hyp_words = canonicalize_label(hypothesis).split()
    n = len(ref_words); m = len(hyp_words)
    if n == 0:
        return (0.0 if m == 0 else 1.0), m, 0
    # Levenshtein distance on word lists.
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            cur[j] = min(prev[j] + 1,        # deletion
                          cur[j - 1] + 1,    # insertion
                          prev[j - 1] + cost)  # substitution
        prev = cur
    errs = prev[m]
    return errs / n, errs, n


# Yes/No detection patterns. Models often answer with hedging or full
# sentences (e.g. "Yes, both sources are at roughly the same azimuth"),
# so exact-match against the literal "yes"/"no" gold is wrong. We:
#   1. detect explicit yes/no tokens at the start of the sentence
#   2. if both appear, take whichever comes first (the leading verdict)
#   3. fall back to negation cues for "no" (not / aren't / isn't / different)
#      and affirmation cues for "yes" (same / matching / equal)
#
# Returns "yes" / "no" / None (couldn't decide).
_YES_TOKENS = re.compile(
    r'\b(yes|yeah|yep|yup|correct|right|true|affirmative|indeed)\b',
    re.I,
)
_NO_TOKENS = re.compile(
    r'\b(no|nope|nah|incorrect|false|negative|not\s+really)\b',
    re.I,
)
_NEG_CUES = re.compile(
    r'\b(not|aren\'t|are\s+not|isn\'t|is\s+not|don\'t|do\s+not|'
    r'doesn\'t|does\s+not|different|differ|differs|differing|distinct)\b',
    re.I,
)
_POS_CUES = re.compile(
    r'\b(same|identical|matching|matches|match|equal|equivalent|'
    r'identically)\b',
    re.I,
)


def _classify_yes_no(text: str) -> Optional[str]:
    """Boil a free-form English answer down to 'yes' / 'no' / None."""
    t = (text or "").strip().lower()
    if not t:
        return None
    # 1. explicit yes/no token; whichever appears first wins.
    yes_m = _YES_TOKENS.search(t)
    no_m = _NO_TOKENS.search(t)
    if yes_m and no_m:
        return "yes" if yes_m.start() < no_m.start() else "no"
    if yes_m:
        return "yes"
    if no_m:
        return "no"
    # 2. cue-based fallback: "not the same" -> no, "they are different" -> no,
    # "the same azimuth" -> yes.
    has_neg = bool(_NEG_CUES.search(t))
    has_pos = bool(_POS_CUES.search(t))
    if has_neg and not has_pos:
        return "no"
    if has_pos and not has_neg:
        return "yes"
    if has_pos and has_neg:
        # "not the same" -> no (negation wins over affirmation)
        return "no"
    return None


def score_yesno(record: Dict[str, Any], pred_text: str) -> "TaskScore":
    """Generic yes/no scorer used for binary tasks like same_azimuth.
    Gold is normalized to 'yes' / 'no'; prediction is mapped via
    `_classify_yes_no` (lexical + cue-based). Anything that can't be
    classified falls back to normalized exact match.
    """
    task = str(record["task_name"])
    gold_raw = (record.get("canonical_answer") or record.get("answer") or "")
    gold = _classify_yes_no(str(gold_raw))

    ts = TaskScore(
        pair_id=record.get("pair_id"),
        task_name=task,
        prediction=pred_text,
        answer=str(record.get("answer", "")),
        canonical_answer=record.get("canonical_answer"),
        metric_type="yes_no",
    )
    if not gold:
        # Gold itself is unparseable; don't penalize the model.
        ts.parse_status = "fail_no_answer_meta"
        return ts
    if not str(pred_text).strip():
        ts.parse_status = "fail_empty"
        return ts

    pred_label = _classify_yes_no(pred_text)
    ts.details = {
        "gold_label": gold,
        "pred_label": pred_label,
        "pred_text": pred_text,
    }
    if pred_label is None:
        # Last-ditch: normalized exact match against gold token (rare).
        if normalize_text(pred_text) == gold:
            ts.correct = 1.0
            ts.details["match_stage"] = "normalized_exact"
        else:
            ts.correct = 0.0
            ts.details["match_stage"] = "unparsed"
    else:
        ts.correct = float(pred_label == gold)
        ts.details["match_stage"] = "classified"
    return ts


# ---------------------------------------------------------------------------
# Categorical scorers: distance_category, elevation_category.
#
# Gold is one of a small finite set of labels (e.g. "near"/"medium"/"far",
# "low"/"middle"/"high"). The model's natural response wraps that label
# inside a sentence ("the source is far away from the listener"), so
# normalized exact match catches almost nothing. We do:
#   1. Map both gold and pred to a canonical bucket via keyword matching.
#   2. If multiple buckets match in the prediction, pick the bucket with the
#      highest-priority (most specific) keyword.
#   3. If nothing matches, fall back to substring containment of the gold
#      label inside the (lowercased) prediction.
# ---------------------------------------------------------------------------

# Synonym groups ordered (longer / more-specific phrases first so they win
# the search). Each entry: (canonical_label, regex pattern).
_DISTANCE_BUCKETS: List[Tuple[str, "re.Pattern[str]"]] = [
    ("near",   re.compile(r'\b(very\s+close|right\s+next\s+to|close\s+by|nearby|near|close|proximate|adjacent)\b', re.I)),
    ("far",    re.compile(r'\b(very\s+far|far\s+away|far\s+off|far|distant|remote)\b', re.I)),
    ("medium", re.compile(r'\b(medium|middle|moderate|mid(?:[-\s]?range)?|intermediate|average\s+distance)\b', re.I)),
]

_ELEVATION_BUCKETS: List[Tuple[str, "re.Pattern[str]"]] = [
    ("high",   re.compile(r'\b(very\s+high|up\s+high|overhead|above|high|elevated|upper|top)\b', re.I)),
    ("low",    re.compile(r'\b(very\s+low|down\s+low|below|low(?:er)?|underneath|bottom|ground[-\s]?level)\b', re.I)),
    ("middle", re.compile(r'\b(middle|mid(?:[-\s]?level)?|eye[-\s]?level|same\s+(?:level|height)|level|center|center\s+height)\b', re.I)),
]


def _categorize(text: str, buckets: List[Tuple[str, "re.Pattern[str]"]]) -> Optional[str]:
    """Return the first bucket whose pattern matches (earliest position wins
    when multiple match, except buckets are searched in declaration order so
    longer/more-specific phrases get a chance first)."""
    if not text:
        return None
    t = text.lower()
    best: Optional[Tuple[int, str]] = None
    for label, pat in buckets:
        m = pat.search(t)
        if m is None:
            continue
        if best is None or m.start() < best[0]:
            best = (m.start(), label)
    return best[1] if best else None


def _score_categorical(record: Dict[str, Any], pred_text: str,
                        buckets: List[Tuple[str, "re.Pattern[str]"]],
                        metric_label: str) -> "TaskScore":
    task = str(record["task_name"])
    gold_raw = (record.get("canonical_answer") or record.get("answer") or "")
    gold = _categorize(str(gold_raw), buckets)
    # If the gold's first word IS a canonical label literally, take that
    # short-circuit (covers cases where the gold is just "far" / "high").
    if gold is None:
        first = canonicalize_label(str(gold_raw)).split()
        if first:
            for label, _ in buckets:
                if first[0] == label:
                    gold = label
                    break

    ts = TaskScore(
        pair_id=record.get("pair_id"),
        task_name=task,
        prediction=pred_text,
        answer=str(record.get("answer", "")),
        canonical_answer=record.get("canonical_answer"),
        metric_type=metric_label,
    )
    if not gold:
        ts.parse_status = "fail_no_answer_meta"
        return ts
    if not str(pred_text).strip():
        ts.parse_status = "fail_empty"
        return ts

    pred_label = _categorize(pred_text, buckets)
    ts.details = {
        "gold_label": gold,
        "pred_label": pred_label,
        "pred_text": pred_text,
    }
    if pred_label is None:
        # Last-ditch: gold token literally appears in normalized pred.
        if gold in canonicalize_label(pred_text).split():
            ts.correct = 1.0
            ts.details["match_stage"] = "substring"
        else:
            ts.correct = 0.0
            ts.details["match_stage"] = "unparsed"
    else:
        ts.correct = float(pred_label == gold)
        ts.details["match_stage"] = "classified"
    return ts


def score_distance_category(record: Dict[str, Any], pred_text: str) -> "TaskScore":
    return _score_categorical(record, pred_text, _DISTANCE_BUCKETS,
                                metric_label="distance_category_match")


def score_elevation_category(record: Dict[str, Any], pred_text: str) -> "TaskScore":
    return _score_categorical(record, pred_text, _ELEVATION_BUCKETS,
                                metric_label="elevation_category_match")


def score_speech_content(record: Dict[str, Any], pred_text: str) -> "TaskScore":
    """speech_content: free-form ASR-like task. Compare prediction to the
    canonical transcript using word error rate (WER). Lower is better.
    "correct" = WER <= 0.5 (a loose threshold; tighter ones reported too).
    """
    task = str(record["task_name"])
    # Prefer the canonical_answer (raw transcript). Fall back to answer.
    gold = (record.get("canonical_answer") or "").strip()
    if not gold:
        gold = (record.get("answer") or "").strip()

    ts = TaskScore(
        pair_id=record.get("pair_id"),
        task_name=task,
        prediction=pred_text,
        answer=str(record.get("answer", "")),
        canonical_answer=record.get("canonical_answer"),
        metric_type="speech_wer",
    )
    if not gold:
        ts.parse_status = "fail_no_answer_meta"; return ts
    if not str(pred_text).strip():
        ts.parse_status = "fail_empty"; return ts

    wer, errs, ref_n = _wer(gold, pred_text)
    # Try wrapper-stripping candidates: quoted spans, after-colon tail,
    # prefix-stripped versions. Pick the one with lowest WER.
    candidates = _speech_candidates(pred_text)
    best_wer, best_errs, best_n, best_cand, best_idx = wer, errs, ref_n, pred_text, -1
    for idx, cand in enumerate(candidates):
        if cand == pred_text:
            continue
        w, e, n = _wer(gold, cand)
        if w < best_wer:
            best_wer, best_errs, best_n, best_cand, best_idx = w, e, n, cand, idx

    pred_word_count = len(canonicalize_label(pred_text).split())
    extracted_word_count = len(canonicalize_label(best_cand).split())
    ts.details = {
        "wer": float(best_wer),
        "wer_raw": float(wer),                    # WER on un-extracted prediction
        "edit_distance": int(best_errs),
        "ref_word_count": int(ref_n),
        "pred_word_count": int(pred_word_count),
        "pred_extracted": best_cand,
        "pred_extracted_word_count": int(extracted_word_count),
        "pred_n_candidates": len(candidates),
        "extraction_used": (best_idx >= 0),       # whether wrapper stripping helped
    }
    # "correct" = WER <= 0.5 by default (most ASR papers use 0.5 as
    # "intelligible vs not"). We also report tighter thresholds in agg.
    ts.correct = float(best_wer <= 0.5)
    return ts


# ---------------------------------------------------------------------------
# Dispatch + aggregate
# ---------------------------------------------------------------------------


def score_record(qa: Dict[str, Any], pred_text: str, llm: LLMJudge,
                   thresholds: Dict[str, float],
                   candidate_labels: Optional[List[str]],
                   ) -> TaskScore:
    task = str(qa.get("task_name") or "")
    if task == "estimate_azimuth":
        return score_estimate_angle(qa, pred_text, True,
                                      thresholds.get("azimuth_deg", thresholds["angle_deg"]))
    if task == "estimate_elevation":
        return score_estimate_angle(qa, pred_text, False,
                                      thresholds.get("elevation_deg", thresholds["angle_deg"]))
    if task in ("identify_source_by_doa", "identify_source_by_location"):
        return score_identify_source(qa, pred_text, llm,
                                       llm_allowed=True,
                                       candidate_labels=candidate_labels)
    if task == "detect_time":
        return score_detect_time(qa, pred_text, thresholds["iou"])
    if task == "detect_source":
        return score_detect_source(qa, pred_text, llm,
                                     llm_allowed=llm.cfg.judge_all_tasks,
                                     iou_threshold=thresholds["iou"])
    # Medium-split tasks.
    if task == "count_sources":
        return score_count_sources(qa, pred_text)
    if task == "estimate_distance":
        return score_estimate_distance(qa, pred_text,
                                          rel_threshold=thresholds.get("distance_rel", 0.3))
    if task == "onset_from_location":
        return score_onset_from_location(qa, pred_text,
                                            within_s=thresholds.get("onset_s", 0.4))
    if task == "classify_motion":
        # classify_motion always tries LLM-judge fallback for synonym
        # robustness, regardless of --llm-judge-all-tasks. The LLM call only
        # actually fires if --llm-judge is enabled (LLMJudge.cfg.enabled).
        return score_classify_motion(qa, pred_text, llm)
    # Yes/No medium task: same_azimuth. Don't use exact-match -- model
    # typically wraps the answer ("Yes, they are at the same azimuth").
    if task == "same_azimuth":
        return score_yesno(qa, pred_text)
    # Categorical medium tasks: distance_category, elevation_category.
    # Gold is a short bucket label ("near"/"far", "high"/"low") but model
    # responses are full sentences -- exact-match would fail almost always.
    if task == "distance_category":
        return score_distance_category(qa, pred_text)
    if task == "elevation_category":
        return score_elevation_category(qa, pred_text)
    # Hard-split tasks.
    if task in ("compare_azimuth", "compare_distance", "compare_elevation"):
        return score_compare(qa, pred_text, llm,
                                 strict=bool(thresholds.get("strict_compare", False)))
    if task in ("relative_left_right", "relative_position"):
        return score_relative(qa, pred_text)
    if task == "spatial_temporal":
        return score_spatial_temporal(qa, pred_text)
    if task == "multi_hop":
        return score_multi_hop(qa, pred_text, llm)
    if task == "speech_content":
        return score_speech_content(qa, pred_text)
    # Unknown task -> fall back to normalized exact match.
    ts = TaskScore(
        pair_id=qa.get("pair_id"),
        task_name=task or "unknown",
        prediction=pred_text,
        answer=str(qa.get("answer", "")),
        canonical_answer=qa.get("canonical_answer"),
        metric_type="normalized_exact_match",
    )
    if not pred_text.strip():
        ts.parse_status = "fail_empty"
        return ts
    ts.correct = float(normalize_text(pred_text) == normalize_text(qa.get("answer", "")))
    return ts


def summarize(scored: List[TaskScore], thresholds: Dict[str, float],
                parse_status_order: List[str]) -> Dict[str, Any]:
    by_task: Dict[str, List[TaskScore]] = defaultdict(list)
    for s in scored:
        by_task[s.task_name].append(s)

    parse_ok = [s for s in scored if s.parse_status == "ok"]
    correct_all = mean_or_none([float(s.correct) for s in scored])
    correct_parseable = mean_or_none([float(s.correct) for s in parse_ok])

    summary: Dict[str, Any] = {
        "examples": len(scored),
        "overall_correct": correct_all,
        "overall_correct_parseable": correct_parseable,
        "parse_rate": mean_or_none([1.0 if s.parse_status == "ok" else 0.0 for s in scored]),
        "llm_used_rate": mean_or_none([1.0 if s.llm_used else 0.0 for s in scored]),
        "thresholds": thresholds,
        "parse_status_counts": {
            status: sum(1 for s in scored if s.parse_status == status)
            for status in parse_status_order
        },
        "per_task": {},
    }

    for task_name, records in sorted(by_task.items()):
        n = len(records)
        parse_ok_records = [r for r in records if r.parse_status == "ok"]
        per_task: Dict[str, Any] = {
            "examples": n,
            "metric_type": records[0].metric_type,
            "correct_all": mean_or_none([float(r.correct) for r in records]),
            "correct_parseable": mean_or_none([float(r.correct) for r in parse_ok_records]),
            "parse_rate": mean_or_none([1.0 if r.parse_status == "ok" else 0.0 for r in records]),
            "parse_status_counts": {
                status: sum(1 for r in records if r.parse_status == status)
                for status in parse_status_order
            },
            "llm_used_rate": mean_or_none([1.0 if r.llm_used else 0.0 for r in records]),
        }
        # Task-specific aggregates.
        if task_name in ("estimate_azimuth", "estimate_elevation"):
            errs = [float(r.details.get("error_deg"))
                    for r in parse_ok_records if "error_deg" in r.details]
            per_task["error_deg_mean"] = mean_or_none(errs)
            per_task["error_deg_median"] = median_or_none(errs)
            per_task["correct_is_within_threshold"] = per_task["correct_all"]
            # ------- collapse / template-answer diagnostics -------
            # Purpose: distinguish a model that genuinely localizes from one
            # that emits a constant / few-value template and rides the GT
            # distribution bias (elevation is centered near 0° -- a constant
            # 0° predictor already hits 33% acc@10° without reading the audio).
            preds = [float(r.details.get("predicted_deg"))
                     for r in parse_ok_records if "predicted_deg" in r.details]
            tgts = [float(r.details.get("target_deg"))
                    for r in parse_ok_records if "target_deg" in r.details]
            n_pred = len(preds)
            if n_pred > 0:
                # Output diversity: unique predictions / n. <0.1 means the
                # model is emitting a very small vocabulary of angles.
                uniq = len(set(round(p, 1) for p in preds))
                per_task["unique_prediction_ratio"] = uniq / n_pred
                per_task["unique_prediction_count"] = uniq
                # Top-1 prediction frequency: if one value dominates (>30%
                # for elevation, >15% for azimuth), the model is likely
                # template-answering.
                from collections import Counter
                pc = Counter(round(p, 1) for p in preds)
                top1_val, top1_cnt = pc.most_common(1)[0]
                per_task["top1_prediction"] = float(top1_val)
                per_task["top1_prediction_share"] = top1_cnt / n_pred
                # Constant-baseline comparison: how does the best single
                # constant prediction (median of GT) compare? This is the
                # "free floor" that any trivial model can hit. If the model's
                # acc / median_err is close to this floor, it hasn't learned
                # spatial localization.
                if tgts:
                    import statistics
                    gt_med = statistics.median(tgts)
                    if task_name == "estimate_azimuth":
                        # Wrap-around aware
                        def _err(p, g):
                            return abs(((p - g + 180.0) % 360.0) - 180.0)
                    else:
                        def _err(p, g):
                            return abs(p - g)
                    const_errs = [_err(gt_med, g) for g in tgts]
                    thr = (thresholds.get("azimuth_deg") if task_name == "estimate_azimuth"
                           else thresholds.get("elevation_deg")) or thresholds.get("angle_deg") or 20.0
                    per_task["const_median_baseline_value_deg"] = float(gt_med)
                    per_task["const_median_baseline_median_err"] = median_or_none(const_errs)
                    per_task["const_median_baseline_mean_err"] = mean_or_none(const_errs)
                    per_task["const_median_baseline_acc"] = mean_or_none(
                        [1.0 if e <= thr else 0.0 for e in const_errs])
                    # Gain vs constant baseline: positive = model actually
                    # learned; near-zero or negative = template collapse.
                    per_task["acc_gain_over_constant"] = (
                        (per_task["correct_all"] or 0.0)
                        - (per_task["const_median_baseline_acc"] or 0.0))
                    per_task["median_err_reduction_vs_constant"] = (
                        (per_task["const_median_baseline_median_err"] or 0.0)
                        - (per_task["error_deg_median"] or 0.0))
                # Stratified accuracy: bin predictions by GT angle and compute
                # acc@threshold within each bin, then macro-average. A model
                # that only predicts near-zero (common for elevation) will
                # pass easy bins but fail hard bins, and macro-avg penalizes
                # that bias.
                if task_name == "estimate_elevation":
                    bins = [(-90, -30), (-30, -10), (-10, 10), (10, 30), (30, 90)]
                else:  # estimate_azimuth
                    bins = [(-180, -90), (-90, -30), (-30, 30), (30, 90), (90, 180)]
                thr = (thresholds.get("azimuth_deg") if task_name == "estimate_azimuth"
                       else thresholds.get("elevation_deg")) or thresholds.get("angle_deg") or 20.0
                bin_accs: List[float] = []
                bin_counts: List[int] = []
                for lo, hi in bins:
                    mask_errs = []
                    for p, g in zip(preds, tgts):
                        if lo <= g < hi:
                            if task_name == "estimate_azimuth":
                                e = abs(((p - g + 180.0) % 360.0) - 180.0)
                            else:
                                e = abs(p - g)
                            mask_errs.append(1.0 if e <= thr else 0.0)
                    bin_counts.append(len(mask_errs))
                    if mask_errs:
                        bin_accs.append(sum(mask_errs) / len(mask_errs))
                if bin_accs:
                    per_task["acc_macro_by_gt_bin"] = sum(bin_accs) / len(bin_accs)
                    per_task["acc_per_gt_bin"] = {
                        f"[{lo},{hi})": {
                            "n": bin_counts[i],
                            "acc": (bin_accs[i] if i < len(bin_accs) else None),
                        }
                        for i, (lo, hi) in enumerate(bins)
                    }
        elif task_name == "detect_time":
            ious = [float(r.details.get("iou"))
                     for r in parse_ok_records if "iou" in r.details]
            starts = [float(r.details.get("start_error_s"))
                     for r in parse_ok_records if "start_error_s" in r.details]
            ends = [float(r.details.get("end_error_s"))
                     for r in parse_ok_records if "end_error_s" in r.details]
            per_task["iou_mean"] = mean_or_none(ious)
            per_task["iou_median"] = median_or_none(ious)
            per_task["start_error_mean_s"] = mean_or_none(starts)
            per_task["end_error_mean_s"] = mean_or_none(ends)
            per_task["iou_at_threshold"] = mean_or_none(
                [float(r.details.get("correct_binary") or 0)
                 for r in parse_ok_records])
        elif task_name == "detect_source":
            f1s = [float(r.details.get("f1"))
                    for r in parse_ok_records if "f1" in r.details]
            precs = [float(r.details.get("precision"))
                    for r in parse_ok_records if "precision" in r.details]
            recs = [float(r.details.get("recall"))
                    for r in parse_ok_records if "recall" in r.details]
            per_task["f1_mean"] = mean_or_none(f1s)
            per_task["precision_mean"] = mean_or_none(precs)
            per_task["recall_mean"] = mean_or_none(recs)
        elif task_name in ("identify_source_by_doa", "identify_source_by_location"):
            stages: Dict[str, int] = defaultdict(int)
            for r in records:
                stages[str(r.details.get("match_stage", "none"))] += 1
            per_task["match_stage_counts"] = dict(stages)
        elif task_name == "count_sources":
            errs = [float(r.details.get("abs_error"))
                    for r in parse_ok_records if "abs_error" in r.details]
            per_task["abs_error_mean"] = mean_or_none(errs)
            per_task["abs_error_median"] = median_or_none(errs)
            # acc within tolerance N for sanity
            for tol in (0, 1):
                per_task[f"acc_within_{tol}"] = mean_or_none(
                    [1.0 if e <= tol else 0.0 for e in errs])
        elif task_name == "estimate_distance":
            errs = [float(r.details.get("abs_error_m"))
                    for r in parse_ok_records if "abs_error_m" in r.details]
            rels = [float(r.details.get("rel_error"))
                    for r in parse_ok_records if "rel_error" in r.details]
            per_task["abs_error_m_mean"] = mean_or_none(errs)
            per_task["abs_error_m_median"] = median_or_none(errs)
            per_task["rel_error_mean"] = mean_or_none(rels)
            per_task["rel_error_median"] = median_or_none(rels)
            # The "correct" signal is rel_err <= 0.3 (the binary metric).
            per_task["acc_rel_within_0.3"] = per_task["correct_all"]
            # Extra context: tighter (0.2) and looser (0.5) thresholds.
            per_task["acc_rel_within_0.2"] = mean_or_none(
                [1.0 if r <= 0.2 else 0.0 for r in rels])
            per_task["acc_rel_within_0.5"] = mean_or_none(
                [1.0 if r <= 0.5 else 0.0 for r in rels])
            # Also keep the old absolute-distance acc as a sanity reference.
            per_task["acc_abs_within_1m"] = mean_or_none(
                [1.0 if e <= 1.0 else 0.0 for e in errs])
        elif task_name == "onset_from_location":
            errs = [float(r.details.get("abs_error_s"))
                    for r in parse_ok_records if "abs_error_s" in r.details]
            per_task["abs_error_s_mean"] = mean_or_none(errs)
            per_task["abs_error_s_median"] = median_or_none(errs)
            per_task["acc_within_0.4s"] = per_task["correct_all"]
            per_task["acc_within_0.2s"] = mean_or_none(
                [1.0 if e <= 0.2 else 0.0 for e in errs])
            per_task["acc_within_1.0s"] = mean_or_none(
                [1.0 if e <= 1.0 else 0.0 for e in errs])
        elif task_name == "classify_motion":
            stages: Dict[str, int] = defaultdict(int)
            for r in records:
                stages[str(r.details.get("match_stage", "none"))] += 1
            per_task["match_stage_counts"] = dict(stages)
        elif task_name in ("compare_azimuth", "compare_distance", "compare_elevation"):
            stages: Dict[str, int] = defaultdict(int)
            for r in records:
                stages[str(r.details.get("match_stage", "none"))] += 1
            per_task["match_stage_counts"] = dict(stages)
        elif task_name in ("relative_left_right", "relative_position"):
            stages: Dict[str, int] = defaultdict(int)
            partials = []
            for r in records:
                stages[str(r.details.get("match_stage", "none"))] += 1
                pc = r.details.get("partial_credit")
                if pc is not None:
                    partials.append(float(pc))
            per_task["match_stage_counts"] = dict(stages)
            # Direction-level partial credit: average over examples that had
            # a non-zero match. Useful when the LLM gets one axis right but
            # not the other (e.g. "left" right but "above/below" wrong).
            partial_all = []
            for r in records:
                if r.details.get("match_stage") == "all_directions":
                    partial_all.append(1.0)
                elif r.details.get("partial_credit") is not None:
                    partial_all.append(float(r.details["partial_credit"]))
                else:
                    partial_all.append(0.0)
            per_task["direction_partial_credit_mean"] = mean_or_none(partial_all)
        elif task_name == "spatial_temporal":
            stages: Dict[str, int] = defaultdict(int)
            time_ious = []
            for r in records:
                stages[str(r.details.get("match_stage", "none"))] += 1
                if "time_iou" in r.details:
                    time_ious.append(float(r.details["time_iou"]))
            per_task["match_stage_counts"] = dict(stages)
            # src_match / dir_recall are now floats in [0,1] (continuous
            # partial credit). Report mean of each component as well as
            # the strict-AND `hard_correct` for back-compat.
            per_task["src_match_mean"] = mean_or_none(
                [float(r.details.get("src_match", 0.0)) for r in records])
            per_task["dir_recall_mean"] = mean_or_none(
                [float(r.details.get("dir_recall", 0.0)) for r in records])
            per_task["dir_precision_mean"] = mean_or_none(
                [float(r.details.get("dir_precision", 0.0)) for r in records])
            per_task["hard_correct_rate"] = mean_or_none(
                [float(r.details.get("hard_correct", 0.0)) for r in records])
            # Back-compat keys (now derived from the soft scores).
            per_task["src_match_rate"] = mean_or_none(
                [1.0 if float(r.details.get("src_match", 0.0)) >= 1.0 else 0.0
                 for r in records])
            per_task["dirs_match_rate"] = mean_or_none(
                [1.0 if float(r.details.get("dir_recall", 0.0)) >= 1.0 else 0.0
                 for r in records])
            if time_ious:
                per_task["time_iou_mean"] = mean_or_none(time_ious)
                per_task["time_iou_median"] = median_or_none(time_ious)
                per_task["time_iou_n_with_pred_span"] = len(time_ious)
        elif task_name == "multi_hop":
            stages: Dict[str, int] = defaultdict(int)
            f1s, time_ious = [], []
            for r in records:
                stages[str(r.details.get("match_stage", "none"))] += 1
                if "event_f1" in r.details:
                    f1s.append(float(r.details["event_f1"]))
                if "time_iou" in r.details:
                    time_ious.append(float(r.details["time_iou"]))
            per_task["match_stage_counts"] = dict(stages)
            per_task["event_f1_mean"] = mean_or_none(f1s)
            if time_ious:
                per_task["time_iou_mean"] = mean_or_none(time_ious)
                per_task["time_iou_median"] = median_or_none(time_ious)
                per_task["time_iou_n_with_pred_span"] = len(time_ious)
        elif task_name == "speech_content":
            wers = [float(r.details.get("wer"))
                    for r in parse_ok_records if "wer" in r.details]
            wers_raw = [float(r.details.get("wer_raw"))
                        for r in parse_ok_records if "wer_raw" in r.details]
            per_task["wer_mean"] = mean_or_none(wers)
            per_task["wer_median"] = median_or_none(wers)
            # Clipped WER: cap each example at 1.0 before averaging. Bounded
            # in [0, 1], so cross-task comparison is meaningful and
            # hallucinated-prefix outliers don't dominate the mean.
            per_task["wer_clipped_mean"] = mean_or_none(
                [min(w, 1.0) for w in wers])
            per_task["wer_clipped_median"] = median_or_none(
                [min(w, 1.0) for w in wers])
            if wers_raw:
                per_task["wer_raw_mean"] = mean_or_none(wers_raw)
                per_task["wer_raw_median"] = median_or_none(wers_raw)
            per_task["acc_wer_le_0.3"] = mean_or_none(
                [1.0 if w <= 0.3 else 0.0 for w in wers])
            per_task["acc_wer_le_0.5"] = mean_or_none(
                [1.0 if w <= 0.5 else 0.0 for w in wers])
            per_task["acc_wer_le_1.0"] = mean_or_none(
                [1.0 if w <= 1.0 else 0.0 for w in wers])
            # Extraction diagnostics: how often did wrapper-stripping help,
            # and how does pred length compare to ref length.
            extracted_used = [
                1.0 if r.details.get("extraction_used") else 0.0
                for r in parse_ok_records if "extraction_used" in r.details
            ]
            if extracted_used:
                per_task["extraction_used_rate"] = mean_or_none(extracted_used)
            pred_wc = [int(r.details.get("pred_word_count"))
                       for r in parse_ok_records
                       if "pred_word_count" in r.details]
            ref_wc = [int(r.details.get("ref_word_count"))
                      for r in parse_ok_records
                      if "ref_word_count" in r.details]
            ext_wc = [int(r.details.get("pred_extracted_word_count"))
                      for r in parse_ok_records
                      if "pred_extracted_word_count" in r.details]
            if pred_wc:
                per_task["mean_pred_word_count"] = mean_or_none(
                    [float(x) for x in pred_wc])
            if ref_wc:
                per_task["mean_ref_word_count"] = mean_or_none(
                    [float(x) for x in ref_wc])
            if ext_wc:
                per_task["mean_extracted_word_count"] = mean_or_none(
                    [float(x) for x in ext_wc])
        summary["per_task"][task_name] = per_task

    return summary


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resolve_qa_split(qa_root: str, split: str) -> str:
    for ext in (".jsonl", ".json"):
        p = os.path.join(qa_root, f"{split}{ext}")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Missing {split}.jsonl or {split}.json under {qa_root}")


def load_qa_split(qa_root: str, split: str) -> List[Dict[str, Any]]:
    path = resolve_qa_split(qa_root, split)
    if path.endswith(".jsonl"):
        return load_jsonl(path)
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return payload
    return payload.get("records") or payload.get("data") or []


def build_candidate_labels(qa_records: List[Dict[str, Any]]) -> List[str]:
    """Collect all canonical source labels seen in the split — used as
    hints for the LLM extractor."""
    labels: set = set()
    for r in qa_records:
        c = r.get("canonical_answer")
        if c:
            labels.add(canonicalize_label(str(c)))
        for ref in (r.get("source_refs") or []):
            if isinstance(ref, dict) and ref.get("class_name"):
                labels.add(canonicalize_label(str(ref["class_name"])))
    labels.discard("")
    return sorted(labels)


def clean_generated(text: str) -> str:
    """Trim the typical decoder tail so scorers see just the answer."""
    s = str(text).replace("\r\n", "\n").strip()
    for marker in ("Human:", "Question:", "\nHuman:", "\nQuestion:"):
        if marker in s:
            s = s.split(marker, 1)[0].strip()
    # Keep multi-line for detect_source / detect_time (they can have multiple
    # events). Only trim if the first non-empty line looks like a short
    # label-style answer.
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if not lines:
        return ""
    # Heuristic: if first line is short and later lines start with "Answer:" /
    # "Explanation:", take only the first line. Otherwise keep all.
    if (len(lines) > 1 and len(lines[0]) <= 120
            and any(ln.lower().startswith(("explanation:", "reason:", "note:", "because")) for ln in lines[1:])):
        return lines[0]
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions-jsonl", required=True,
                    help="Path to predictions.jsonl produced by the bench script.")
    p.add_argument("--qa-root", default=None,
                    help="QA root containing <split>.jsonl with the gold fields. "
                         "Optional: if omitted, scorer uses the `answer` / "
                         "`canonical_answer` / `answer_meta` / `source_refs` "
                         "fields embedded in predictions.jsonl directly "
                         "(self-scoring mode).")
    p.add_argument("--split", default="test")
    p.add_argument("--output-json", default=None,
                    help="Summary output path. Defaults to <predictions-dir>/score_result.json.")
    p.add_argument("--per-record-jsonl", default=None,
                    help="Optional path to dump per-record scoring detail.")
    p.add_argument("--angle-threshold-deg", type=float, default=20.0,
                    help="Default angle tolerance. Used only if the per-task "
                         "--azimuth/--elevation flags are left unset.")
    p.add_argument("--azimuth-threshold-deg", type=float, default=20.0,
                    help="estimate_azimuth tolerance (deg). Default: 20.")
    p.add_argument("--elevation-threshold-deg", type=float, default=10.0,
                    help="estimate_elevation tolerance (deg). Default: 10. "
                         "Elevation uses a tighter threshold because GT is "
                         "concentrated near 0° (a constant-0 predictor "
                         "already hits ~34%% at 10°; at 20° it would hit ~55%% "
                         "which obscures template-answer baselines).")
    p.add_argument("--iou-threshold", type=float, default=0.5,
                    help="IoU threshold used for detect_time/detect_source binary metrics.")
    p.add_argument("--distance-rel-threshold", type=float, default=0.3,
                    help="estimate_distance tolerance as a relative error "
                         "(|pred-gt|/|gt|). Default: 0.3 (i.e. correct if "
                         "predicted distance is within 30%% of the truth). "
                         "Relative error is fairer than absolute meters when "
                         "GT spans both near (<2m) and far (>5m) sources.")
    p.add_argument("--onset-threshold-s", type=float, default=0.4,
                    help="onset_from_location tolerance (seconds). Default: 0.4.")
    p.add_argument("--strict-compare-judge", action="store_true",
                    help="For compare_{azimuth,distance,elevation}: always "
                         "invoke the LLM judge instead of substring-matching "
                         "the gold source name. Substring match cannot detect "
                         "'right source, wrong comparative role' (e.g. model "
                         "says 'X is farther' but gold says 'X is closer'). "
                         "Strict mode catches that but costs 1 LLM call per "
                         "compare example. Requires --llm-judge.")
    p.add_argument("--llm-judge", action="store_true",
                    help="Enable OpenAI-compatible LLM judge / extractor for "
                         "identify_source_* when regex match fails.")
    p.add_argument("--llm-judge-all-tasks", action="store_true",
                    help="Also use LLM for detect_source label synonym matching. Slow.")
    p.add_argument("--llm-model", default="gpt-4o")
    p.add_argument("--llm-base-url", default="https://api.openai.com/v1")
    p.add_argument("--llm-concurrency", type=int, default=16,
                    help="Number of parallel LLM calls. Default 16; LLM calls "
                         "are network-bound so 16 typically gives ~4x speedup "
                         "over the old default of 4. Set to 1 for ordered debugging.")
    p.add_argument("--llm-max-calls", type=int, default=5000,
                    help="Hard cap on LLM calls (safety valve for cost).")
    p.add_argument("--llm-log-path", type=str, default=None,
                    help="If set, log every LLM call (prompt + response + meta) "
                         "to this JSONL file. The file is truncated at startup; "
                         "first line is a `_meta` header with the run config; "
                         "subsequent lines are one record per call with fields "
                         "{status: ok|failed|rate_limited|no_client, prompt, "
                         "response, attempts, latency_s, meta:{kind,task,...}, ts}. "
                         "Useful for auditing judge decisions.")
    p.add_argument("--keep-duplicate-pair-ids", action="store_true")
    p.add_argument("--model-name", type=str, default="spatial_omni25",
                    help="Model identifier shown in the markdown summary table.")
    p.add_argument("--version", type=str, default="-",
                    help="Difficulty / split tag shown in the markdown summary "
                         "table (e.g. easy, medium, hard).")
    p.add_argument("--prediction-tag", type=str, default="-",
                    help="Free-form prediction tag for the markdown table "
                         "(e.g. ckpt step / beam config). Default '-'.")
    p.add_argument("--md-output", type=str, default=None,
                    help="If set, also write the markdown summary table to "
                         "this path. Otherwise it is only printed to stdout.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    predictions = load_jsonl(args.predictions_jsonl)
    if args.qa_root:
        qa_records = load_qa_split(args.qa_root, args.split)
    else:
        # Self-scoring mode: synthesize qa_records from predictions themselves.
        # The `answer` field in preds IS the ground truth; the `canonical_answer`
        # / `answer_meta` / `source_refs` fields are absent only for legacy runs.
        print("[score] self-scoring mode (no --qa-root): using answer fields "
              "from predictions.jsonl as ground truth.")
        qa_records = [
            {
                "pair_id": r.get("pair_id"),
                "task_name": r.get("task_name"),
                "question": r.get("question"),
                "answer": r.get("answer"),
                "audio_path": r.get("audio_path"),
                "scene_id": r.get("scene_id"),
                "segment_stem": r.get("segment_stem"),
                "canonical_answer": r.get("canonical_answer"),
                "answer_meta": r.get("answer_meta"),
                "source_refs": r.get("source_refs"),
            }
            for r in predictions
        ]

    # Synthesize a stable pair_id for any record (pred or QA) whose source
    # split has pair_id=None. Must match the formula used by the bench
    # collator (scripts/batch_bench_so_qa.py) so predictions and
    # QA align exactly.
    def _ensure_pair_id(r: Dict[str, Any]) -> None:
        pid = r.get("pair_id")
        if pid is None or pid == "":
            import hashlib
            key = "|".join(
                str(r.get(k, ""))
                for k in ("scene_id", "segment_stem", "task_name", "question", "audio_path")
            )
            r["pair_id"] = "auto_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]

    for r in qa_records:
        _ensure_pair_id(r)
    for r in predictions:
        _ensure_pair_id(r)

    # Build a join key. Prefer pair_id, but many generated QA splits have
    # pair_id = None for every record (the SO-Dataset release is one of
    # them). In that case fall back to (question + task_name) which is
    # unique in practice (and further (question + task_name + audio_path)
    # if even that collides).
    def _primary_key(r):
        pid = r.get("pair_id")
        # Only trust pair_id if it's a "real" (non-synthetic) id. Synthetic
        # ids start with "auto_" and are only stable when both sides saw the
        # same audio_path/scene_id, which is not true for older bench runs
        # that dropped audio_path. Skip to richer fallback in that case.
        if pid is not None and not str(pid).startswith("auto_"):
            return ("id", str(pid))
        # Best: (task, question, audio_path) — uniquely identifies the sample.
        ap = r.get("audio_path")
        if ap:
            return ("tqa", str(r.get("task_name", "")), str(r.get("question", "")), str(ap))
        # Fallback for legacy predictions that lack audio_path: use the
        # generated answer string, which together with (task,question)
        # uniquely picks the right QA record in ~91% of test records.
        return (
            "tqans",
            str(r.get("task_name", "")),
            str(r.get("question", "")),
            str(r.get("answer", "")),
        )

    def _fallback_key(r):
        return (
            "qta",
            str(r.get("question", "")),
            str(r.get("task_name", "")),
            str(r.get("audio_path", "")),
        )

    qa_index: Dict[Any, Dict[str, Any]] = {}
    qa_collisions = 0
    for r in qa_records:
        # Register the record under EVERY possible key that a prediction
        # might produce, so lookups succeed regardless of which fields the
        # prediction file carries (legacy runs dropped audio_path).
        keys = []
        pid = r.get("pair_id")
        if pid is not None and not str(pid).startswith("auto_"):
            keys.append(("id", str(pid)))
        ap = r.get("audio_path")
        if ap:
            keys.append(("tqa", str(r.get("task_name", "")),
                         str(r.get("question", "")), str(ap)))
        # Legacy key: (task, question, answer). Useful when preds lack audio_path.
        keys.append(("tqans", str(r.get("task_name", "")),
                     str(r.get("question", "")), str(r.get("answer", ""))))
        for k in keys:
            if k in qa_index:
                qa_collisions += 1
            else:
                qa_index[k] = r

    if qa_collisions:
        print(f"[score] WARN: {qa_collisions} QA records collided on "
              f"(question,task_name); falling back to (question,task_name,audio_path) "
              f"for those.")

    keyed_mode = "pair_id"
    if all(r.get("pair_id") is None for r in qa_records[:200]):
        keyed_mode = "(question, task_name)"
    print(f"[score] join key: {keyed_mode}; "
          f"qa_records={len(qa_records)} predictions={len(predictions)}")

    # Deduplicate predictions by join key (DistributedSampler padding emits dups).
    if not args.keep_duplicate_pair_ids:
        seen = set()
        deduped = []
        for rec in predictions:
            k = _primary_key(rec)
            if k in seen:
                continue
            seen.add(k)
            deduped.append(rec)
        predictions = deduped
        print(f"[score] after dedup: {len(predictions)} predictions")

    thresholds = {
        "angle_deg": args.angle_threshold_deg,
        "azimuth_deg": args.azimuth_threshold_deg,
        "elevation_deg": args.elevation_threshold_deg,
        "iou": args.iou_threshold,
        "distance_rel": args.distance_rel_threshold,
        "onset_s": args.onset_threshold_s,
        "strict_compare": bool(args.strict_compare_judge),
    }

    llm_cfg = LLMConfig(
        enabled=bool(args.llm_judge or args.llm_judge_all_tasks),
        model=args.llm_model,
        base_url=args.llm_base_url,
        concurrency=max(1, args.llm_concurrency),
        judge_all_tasks=bool(args.llm_judge_all_tasks),
        judge_max_calls=args.llm_max_calls,
        log_path=args.llm_log_path,
    )
    llm = LLMJudge(llm_cfg)
    candidate_labels = build_candidate_labels(qa_records) if llm_cfg.enabled else None

    # Score in parallel when LLM is on (LLM calls dominate cost); otherwise
    # score serially to keep logs readable.
    parse_status_order = ["ok", "fail_regex", "fail_llm_extract",
                           "fail_empty", "fail_no_answer_meta"]

    scored: List[TaskScore] = []
    unmatched = [0]  # counter, mutable via closure

    def _do_one(rec: Dict[str, Any]) -> Optional[TaskScore]:
        k = _primary_key(rec)
        qa = qa_index.get(k)
        if qa is None:
            # Try fallback key for collisions resolved above.
            qa = qa_index.get(_fallback_key(rec))
        if qa is None:
            # If prediction record itself already carries the answer, build a
            # lightweight qa dict from it — that way we still get a score even
            # on splits where no pair_id / question match is possible.
            if rec.get("answer") is not None and rec.get("task_name"):
                qa = {
                    "pair_id": rec.get("pair_id"),
                    "task_name": rec.get("task_name"),
                    "question": rec.get("question"),
                    "answer": rec.get("answer"),
                    "canonical_answer": rec.get("canonical_answer"),
                    "answer_meta": rec.get("answer_meta"),
                    "source_refs": rec.get("source_refs"),
                }
            else:
                unmatched[0] += 1
                return None
        raw_pred = rec.get("prediction_cleaned") or rec.get("prediction") or ""
        pred_text = clean_generated(raw_pred)
        ts = score_record(qa, pred_text, llm, thresholds, candidate_labels)
        return ts

    if llm_cfg.enabled and llm_cfg.concurrency > 1:
        with ThreadPoolExecutor(max_workers=llm_cfg.concurrency) as pool:
            futures = [pool.submit(_do_one, rec) for rec in predictions]
            for i, fut in enumerate(as_completed(futures)):
                ts = fut.result()
                if ts is not None:
                    scored.append(ts)
                if (i + 1) % 500 == 0:
                    print(f"  scored {i+1}/{len(predictions)}", flush=True)
    else:
        for i, rec in enumerate(predictions):
            ts = _do_one(rec)
            if ts is not None:
                scored.append(ts)
            if (i + 1) % 2000 == 0:
                print(f"  scored {i+1}/{len(predictions)}", flush=True)

    summary = summarize(scored, thresholds, parse_status_order)

    out_json = args.output_json or os.path.join(
        os.path.dirname(os.path.abspath(args.predictions_jsonl)), "score_result.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, sort_keys=True)
    print(f"\n[score] wrote {out_json}")

    if args.per_record_jsonl:
        with open(args.per_record_jsonl, "w", encoding="utf-8") as handle:
            for s in scored:
                handle.write(json.dumps({
                    "pair_id": s.pair_id,
                    "task_name": s.task_name,
                    "correct": s.correct,
                    "parse_status": s.parse_status,
                    "metric_type": s.metric_type,
                    "llm_used": s.llm_used,
                    "prediction": s.prediction,
                    "answer": s.answer,
                    "canonical_answer": s.canonical_answer,
                    "details": s.details,
                }, ensure_ascii=False) + "\n")
        print(f"[score] wrote per-record scoring to {args.per_record_jsonl}")

    # Print a compact tabular summary for humans.
    def _fmt(v, spec=".4f"):
        if v is None:
            return "n/a"
        try:
            return format(v, spec)
        except Exception:
            return str(v)

    print(f"\n=== overall ({summary['examples']} records) ===")
    if unmatched[0] > 0:
        print(f"  unmatched predictions: {unmatched[0]} (no QA join key)")
    print(f"  parse_rate               = {_fmt(summary['parse_rate'])}")
    print(f"  correct (all records)    = {_fmt(summary['overall_correct'])}")
    print(f"  correct (parseable only) = {_fmt(summary['overall_correct_parseable'])}")
    print(f"  llm_used_rate            = {_fmt(summary['llm_used_rate'])}")
    print("  parse_status_counts      = " + json.dumps(summary["parse_status_counts"]))

    # ------------------------------------------------------------------
    # Markdown summary table.
    #
    # Schema (one row per scored run):
    #   model | version | prediction
    #   easy oa | detect_source_f1 | detect_time_iou_mean
    #   estimate_azimuth | error_deg_median
    #   estimate_elevation | error_deg_median
    #   identify_source_by_doa | identify_source_by_location
    #   medium oa | classify_motion | count_sources | distance_category |
    #     elevation_category | estimate_azimuth | estimate_distance |
    #     abs_error_m_median | estimate_elevation | onset_from_location |
    #     same_azimuth
    #   hard oa | compare_azimuth | compare_distance | compare_elevation |
    #     multi_hop | relative_left_right | relative_position |
    #     spatial_temporal | speech_content
    #
    # Cells are filled with `correct_all` (the headline accuracy/F1 used by
    # this scorer's per-task agg) unless the column is one of the
    # specifically named medians, in which case we emit the median value.
    # Missing tasks become "-".
    # ------------------------------------------------------------------
    per_task = summary.get("per_task", {})

    def _t(name: str) -> Dict[str, Any]:
        return per_task.get(name, {}) or {}

    def _correct(name: str) -> str:
        v = _t(name).get("correct_all")
        return "-" if v is None else f"{v:.4f}"

    def _val(name: str, key: str) -> str:
        v = _t(name).get(key)
        if v is None:
            return "-"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    def _oa_for(task_list: List[str], gate_tasks: List[str]) -> str:
        """Macro-average correct_all over `task_list` (those that ran).
        Returns '-' unless at least one task in `gate_tasks` actually ran;
        this prevents an easy-only run from inflating the medium oa column
        via overlap tasks like estimate_azimuth/elevation that exist in
        both splits.
        """
        if not any(g in per_task for g in gate_tasks):
            return "-"
        vals = [_t(n).get("correct_all") for n in task_list]
        vals = [v for v in vals if v is not None]
        if not vals:
            return "-"
        return f"{sum(vals) / len(vals):.4f}"

    EASY_TASKS = [
        "detect_source", "detect_time",
        "estimate_azimuth", "estimate_elevation",
        "identify_source_by_doa", "identify_source_by_location",
    ]
    # Tasks that ONLY appear in easy split. Used to gate the easy-oa column.
    EASY_ONLY_GATE = [
        "detect_source", "detect_time",
        "identify_source_by_doa", "identify_source_by_location",
    ]
    MEDIUM_TASKS = [
        "classify_motion", "count_sources",
        "distance_category", "elevation_category",
        "estimate_azimuth", "estimate_distance",
        "estimate_elevation", "onset_from_location", "same_azimuth",
    ]
    # Tasks that ONLY appear in medium split (not in easy/hard). Used to gate.
    MEDIUM_ONLY_GATE = [
        "classify_motion", "count_sources",
        "distance_category", "elevation_category",
        "estimate_distance", "onset_from_location", "same_azimuth",
    ]
    HARD_TASKS = [
        "compare_azimuth", "compare_distance", "compare_elevation",
        "multi_hop", "relative_left_right", "relative_position",
        "spatial_temporal", "speech_content",
    ]
    # speech_content is reported as WER (lower-is-better), not acc, so it is
    # excluded from the hard-oa macro average to avoid mixing units.
    HARD_OA_TASKS = [t for t in HARD_TASKS if t != "speech_content"]
    HARD_ONLY_GATE = HARD_TASKS  # all hard tasks are hard-exclusive

    # Column order matches the user-supplied header exactly.
    headers = [
        "model", "version", "prediction",
        # easy
        "oa", "detect_source_f1", "detect_time_iou_mean",
        "estimate_azimuth", "error_deg_median",
        "estimate_elevation", "error_deg_median",
        "identify_source_by_doa", "identify_source_by_location",
        # medium
        "oa", "classify_motion", "count_sources",
        "distance_category", "elevation_category",
        "estimate_azimuth", "estimate_distance", "abs_error_m_median",
        "estimate_elevation", "onset_from_location", "same_azimuth",
        # hard
        "oa", "compare_azimuth", "compare_distance", "compare_elevation",
        "multi_hop", "relative_left_right", "relative_position",
        "spatial_temporal", "speech_content",
    ]

    row = [
        args.model_name, args.version, args.prediction_tag,
        # easy
        _oa_for(EASY_TASKS, EASY_ONLY_GATE),
        _val("detect_source", "f1_mean"),
        _val("detect_time", "iou_mean"),
        _correct("estimate_azimuth"),
        _val("estimate_azimuth", "error_deg_median"),
        _correct("estimate_elevation"),
        _val("estimate_elevation", "error_deg_median"),
        _correct("identify_source_by_doa"),
        _correct("identify_source_by_location"),
        # medium
        _oa_for(MEDIUM_TASKS, MEDIUM_ONLY_GATE),
        _correct("classify_motion"),
        _correct("count_sources"),
        _correct("distance_category"),
        _correct("elevation_category"),
        _correct("estimate_azimuth"),
        _correct("estimate_distance"),
        _val("estimate_distance", "abs_error_m_median"),
        _correct("estimate_elevation"),
        _correct("onset_from_location"),
        _correct("same_azimuth"),
        # hard
        _oa_for(HARD_OA_TASKS, HARD_ONLY_GATE),
        _correct("compare_azimuth"),
        _correct("compare_distance"),
        _correct("compare_elevation"),
        _correct("multi_hop"),
        _correct("relative_left_right"),
        _correct("relative_position"),
        _correct("spatial_temporal"),
        # speech_content cell shows WER (lower-is-better) instead of acc.
        # Use clipped WER so it stays in [0,1] and isn't dominated by
        # hallucinated-prefix outliers; see _speech_candidates.
        _val("speech_content", "wer_clipped_mean"),
    ]

    md_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        "| " + " | ".join(row) + " |",
    ]
    md_text = "\n".join(md_lines)
    print("\n" + md_text)

    if args.md_output:
        os.makedirs(os.path.dirname(os.path.abspath(args.md_output)) or ".",
                     exist_ok=True)
        with open(args.md_output, "w", encoding="utf-8") as fh:
            fh.write(md_text + "\n")
        print(f"[score] wrote {args.md_output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
