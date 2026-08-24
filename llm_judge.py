"""
llm_judge.py — LLM-as-a-Judge evaluator.

Scores a generated response against the same rubric a human evaluator uses, so
that the two are directly comparable. The judge is not the reference standard; it
is a scalable approximation of one, and its agreement with human raters is itself
measured (see alignment.py).

Three design decisions worth naming:

1. NO CHAIN-OF-THOUGHT IS REQUESTED. The judge returns scores and a short
   justification of the rating. Asking a model to expose private reasoning is both
   unnecessary here and a poor practice to demonstrate.

2. THE JUDGE'S ARITHMETIC IS NOT TRUSTED. It reports an overall_score and a
   pass/fail, and we recompute both from its own dimension scores using the same
   written rule humans apply. Where its self-report disagrees with the rule, that
   is recorded as a judge self-consistency defect rather than silently overwritten.

3. THE JUDGE MODEL SHOULD DIFFER FROM THE GENERATOR MODEL. A model evaluating its
   own output exhibits self-preference bias. The default judge model is a
   different, larger model than the generator; when the fallback chain lands on the
   same model as the generator, the run artifact records that fact so the bias
   caveat can be surfaced in the UI rather than forgotten.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Optional

from failure_taxonomy import FAILURE_MODES, is_critical, normalize_failure_mode
from human_evals import (
    CRITICAL_RULE,
    DIMENSIONS,
    PASS_RULE,
    RUBRIC,
    SCALE,
    applies_pass_rule,
    overall_score,
)

logger = logging.getLogger(__name__)

JUDGE_VERSION = "judge-v1"

# Tried in order. The first model the API accepts is used, and the run artifact
# records which one actually ran — never the one we hoped for.
#
# The chain advances only on a PERMANENT error (model not found, auth). A rate limit is
# transient and must not cause a permanent downgrade: an earlier run advanced the chain
# on every 429, walked past both working models, and failed 22 judgements against a
# model that does not exist on this account.
JUDGE_MODEL_CHAIN = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
]

JUDGE_TEMPERATURE = 0.0   # deterministic scoring; a judge that drifts run to run is not a measurement

# The gpt-oss models are reasoning models: they spend output tokens thinking before
# emitting anything, and that reasoning counts against max_tokens. At 700 tokens with
# default reasoning, a full run truncated 8 of 60 judgements mid-field and four returned
# nothing at all. Truncation is indistinguishable from a genuine judge failure in the
# stored artifact, so it is engineered out rather than reported as a finding.
#
# reasoning_effort="low" is the fix that matters. The judge is applying a rubric that is
# already spelled out in the prompt, not solving a problem, so extended deliberation buys
# little — and it roughly triples token usage, which on a rate-limited tier turns a slow
# run into a failing one.
JUDGE_MAX_TOKENS = 1200
JUDGE_REASONING_EFFORT = "low"

# Groq's client retries 429s internally; when it finally gives up, we back off and retry
# the SAME model rather than downgrading.
RATE_LIMIT_BACKOFF_SECONDS = 25
RATE_LIMIT_MAX_ATTEMPTS = 4


# ── Prompt ────────────────────────────────────────────────────────────────────


def _rubric_block() -> str:
    """The same anchors the human evaluator sees, rendered compactly for the prompt."""
    lines = []
    for dim in DIMENSIONS:
        spec = RUBRIC[dim]
        lines.append(f"{dim} — {spec['question']}")
        lines.append(f"  Score on: {spec['scored_on']}")
        lines.append(f"  Do NOT score on: {spec['ignore']}")
        for score in (5, 3, 1):
            lines.append(f"  {score} = {spec['anchors'][score]}")
        lines.append("")
    return "\n".join(lines)


def _taxonomy_block() -> str:
    lines = []
    for key, entry in FAILURE_MODES.items():
        if key == "none":
            continue
        lines.append(f"  {key} — {entry['description']}")
    lines.append("  none — the response met the expected behaviour")
    return "\n".join(lines)


JUDGE_SYSTEM_PROMPT = f"""You are an evaluation judge for a governed analytics assistant. You score a
generated answer against a rubric. You are not answering the question yourself.

ABSOLUTE RULES

1. The GOVERNED CONTEXT is the only source of truth. Judge every factual claim
   against it. Your own knowledge of business analytics is irrelevant and must not
   be used to excuse a claim the context does not support.
2. Penalise unsupported claims. Any figure, benchmark, causal explanation or trend
   that is not in the governed context is unsupported, however plausible it sounds.
3. Distinguish inference from evidence. A conclusion drawn from context figures is
   acceptable when it is presented as a conclusion. The same conclusion presented
   as a fact from the data is an unsupported claim. Judge how it is presented, not
   only whether it is reasonable.
4. Penalise overconfidence. When the governed context is incomplete or ambiguous
   and the answer is delivered with full confidence and no caveat, that is a
   defect, not a style choice.
5. Do not reward length, fluency or a confident tone. A short correct answer beats
   a long plausible one.
6. Do not reward an answer for restating the governed context back. That is
   grounded but not helpful.
7. An answer that refuses or hedges when the context DOES contain what was asked is
   a failure (unnecessary_refusal), not caution.

SCALE
{chr(10).join(f"  {k} = {v}" for k, v in SCALE.items())}

RUBRIC
{_rubric_block()}
PASS RULE (apply exactly)
  {PASS_RULE}

CRITICAL FAILURE (apply exactly)
  {CRITICAL_RULE}

FAILURE MODE — choose exactly one, the one a reviewer would fix first.
Use these keys verbatim. Do not invent categories.
{_taxonomy_block()}

CONFIDENCE
  Report your own confidence in this evaluation from 0.0 to 1.0. Lower it when the
  rubric was genuinely hard to apply to this case. Do not report high confidence by
  default.

OUTPUT
Return a single JSON object and nothing else. No markdown fences, no commentary
before or after. Do not include any reasoning beyond the reasoning_summary field.

{{
  "relevance": <1-5>,
  "groundedness": <1-5>,
  "correctness": <1-5>,
  "instruction_following": <1-5>,
  "helpfulness": <1-5>,
  "clarity": <1-5>,
  "overall_score": <mean of the six, one or two decimals>,
  "pass": <true|false>,
  "critical_failure": <true|false>,
  "failure_mode": "<one key from the list above, or none>",
  "confidence": <0.0-1.0>,
  "reasoning_summary": "<one or two sentences justifying the scores. State the
                        specific claim or figure that drove the lowest score.>"
}}"""


def build_judge_prompt(case: dict, response_text: str, governed_context: str) -> str:
    """
    Assemble the user message.

    The expected behaviour and required/forbidden facts from the golden case are
    supplied to the judge, which makes this a reference-based evaluation rather
    than a reference-free one. That is the stronger setup, and it is also the
    honest one to disclose: the judge is not discovering the standard, it is
    applying a standard the dataset author wrote.
    """
    required = case.get("required_facts") or []
    required_display = [
        " | ".join(str(x) for x in f) if isinstance(f, list) else str(f) for f in required
    ]
    forbidden = case.get("forbidden_claims") or []

    parts = [
        "GOVERNED CONTEXT (the only permitted source of truth)",
        "─" * 60,
        governed_context.strip() or "(no governed context was supplied for this case)",
        "",
        "─" * 60,
        f"USER QUESTION\n{case.get('question', '')}",
        "",
        f"EXPECTED BEHAVIOUR (what a correct response does)\n{case.get('expected_behavior', '')}",
        "",
        f"REFERENCE ANSWER SUMMARY\n{case.get('expected_answer_summary', '')}",
    ]

    if required_display:
        parts += ["", "REQUIRED FACTS (a correct answer contains each of these; "
                      "alternatives separated by |)"]
        parts += [f"  - {r}" for r in required_display]

    if forbidden:
        parts += ["", "FORBIDDEN CLAIMS (a correct answer contains none of these, "
                      "in these or equivalent words)"]
        parts += [f"  - {f}" for f in forbidden]

    if case.get("governed_context", {}).get("available") is False:
        parts += [
            "",
            "NOTE: this case is designed so the governed context does NOT contain what the "
            "question asks for. The correct behaviour is to say so. Supplying a figure anyway "
            "is a serious failure, not helpfulness.",
        ]

    parts += [
        "",
        "─" * 60,
        f"RESPONSE TO EVALUATE\n{response_text.strip()}",
        "",
        "─" * 60,
        "Return only the JSON object.",
    ]
    return "\n".join(parts)


# ── Parsing ───────────────────────────────────────────────────────────────────


class JudgeParseError(Exception):
    """Raised when a judge response cannot be recovered into valid structured output."""


def _extract_json_object(raw: str) -> str:
    """
    Recover the JSON object from a model response.

    Handles the three things models actually do wrong: wrapping in markdown fences,
    prefixing with prose, and appending a trailing explanation. Uses brace matching
    rather than a greedy regex so nested objects survive.
    """
    text = raw.strip()

    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    start = text.find("{")
    if start == -1:
        raise JudgeParseError("no JSON object found in judge response")

    depth, in_string, escaped = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    raise JudgeParseError("unbalanced braces in judge response")


def _coerce_score(value, field: str) -> Optional[int]:
    """Coerce a dimension score to an int in 1-5. Out-of-range values are clamped, not dropped."""
    if value is None:
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        raise JudgeParseError(f"{field} is not numeric: {value!r}")
    clamped = max(1, min(5, int(round(num))))
    return clamped


def _coerce_bool(value) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        low = value.strip().lower()
        if low in ("true", "yes", "y", "pass", "1"):
            return True
        if low in ("false", "no", "n", "fail", "0"):
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return None


def parse_judge_response(raw: str, eval_id: Optional[str] = None) -> dict:
    """
    Parse and validate a judge response into a normalised record.

    The judge's own overall_score and pass are recorded as *_reported, and the
    authoritative values are recomputed from its dimension scores using the same
    written rules a human applies. Where the two differ, the discrepancy flags are
    set. This is what lets the app report judge self-consistency instead of
    assuming it.

    Raises JudgeParseError when the response cannot be recovered at all.
    """
    obj = json.loads(_extract_json_object(raw))
    if not isinstance(obj, dict):
        raise JudgeParseError(f"judge returned {type(obj).__name__}, expected object")

    scores = {}
    missing = []
    for dim in DIMENSIONS:
        if dim not in obj or obj[dim] is None:
            missing.append(dim)
            scores[dim] = None
        else:
            scores[dim] = _coerce_score(obj[dim], dim)

    if len(missing) == len(DIMENSIONS):
        raise JudgeParseError("judge returned no dimension scores")

    mode_raw = obj.get("failure_mode")
    mode = normalize_failure_mode(mode_raw)

    computed_overall = overall_score(scores)
    reported_overall = obj.get("overall_score")
    try:
        reported_overall = float(reported_overall) if reported_overall is not None else None
    except (TypeError, ValueError):
        reported_overall = None

    reported_pass = _coerce_bool(obj.get("pass"))
    reported_critical = _coerce_bool(obj.get("critical_failure"))

    computed_critical = is_critical(mode)
    computed_pass = applies_pass_rule(scores, mode)

    confidence = obj.get("confidence")
    try:
        confidence = max(0.0, min(1.0, float(confidence))) if confidence is not None else None
    except (TypeError, ValueError):
        confidence = None

    return {
        "eval_id": eval_id,
        "scores": scores,
        "missing_dimensions": missing,

        # Authoritative values, recomputed under the same rules humans apply.
        "overall_score": computed_overall,
        "pass": computed_pass,
        "critical_failure": computed_critical,
        "failure_mode": mode,

        # What the judge claimed, kept for self-consistency measurement.
        "overall_score_reported": reported_overall,
        "pass_reported": reported_pass,
        "critical_failure_reported": reported_critical,
        "failure_mode_raw": mode_raw,

        # Self-consistency flags.
        "arithmetic_error": (
            reported_overall is not None
            and computed_overall is not None
            and abs(reported_overall - computed_overall) > 0.15
        ),
        "pass_rule_error": reported_pass is not None and reported_pass != computed_pass,
        "off_taxonomy_label": mode == "unclassified",

        "confidence": confidence,
        "reasoning_summary": str(obj.get("reasoning_summary", "")).strip(),
        "parse_ok": True,
        "parse_error": None,
    }


def _failed_judge_record(eval_id: Optional[str], error: str, raw: str = "") -> dict:
    """
    A judge failure is data, not an absence of data.

    Returned with parse_ok=False and no scores, so downstream aggregation excludes
    it from quality metrics while still counting it in judge reliability.
    """
    return {
        "eval_id": eval_id,
        "scores": {d: None for d in DIMENSIONS},
        "missing_dimensions": list(DIMENSIONS),
        "overall_score": None,
        "pass": None,
        "critical_failure": None,
        "failure_mode": None,
        "overall_score_reported": None,
        "pass_reported": None,
        "critical_failure_reported": None,
        "failure_mode_raw": None,
        "arithmetic_error": False,
        "pass_rule_error": False,
        "off_taxonomy_label": False,
        "confidence": None,
        "reasoning_summary": "",
        "parse_ok": False,
        "parse_error": error,
        "raw_excerpt": raw[:400],
    }


# ── Model call ────────────────────────────────────────────────────────────────


def get_api_key(explicit: Optional[str] = None) -> str:
    """
    Resolve the Groq key from, in order: an explicit argument, the environment, then
    Streamlit secrets. Never from a file in the repository.
    """
    if explicit:
        return explicit
    env = os.environ.get("GROQ_API_KEY", "")
    if env:
        return env
    try:
        import streamlit as st
        return st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        return ""


def judge_response(
    case: dict,
    response_text: str,
    governed_context: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    max_retries: int = 1,
) -> dict:
    """
    Run the judge on one response.

    On malformed JSON the judge is retried once with an explicit repair instruction.
    A second failure is recorded as a judge failure rather than being silently
    dropped — judge reliability is one of the things this system measures.
    """
    key = get_api_key(api_key)
    if not key:
        return _failed_judge_record(case.get("eval_id"), "no GROQ_API_KEY available")

    try:
        from groq import Groq
    except ImportError:
        return _failed_judge_record(case.get("eval_id"), "groq package not installed")

    client = Groq(api_key=key)
    user_message = build_judge_prompt(case, response_text, governed_context)
    models = [model] if model else JUDGE_MODEL_CHAIN
    eval_id = case.get("eval_id")

    last_error = "no model attempted"
    for model_name in models:
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        json_retries = 0
        rate_limit_attempts = 0
        permanent_failure = False

        while json_retries <= max_retries and rate_limit_attempts < RATE_LIMIT_MAX_ATTEMPTS:
            try:
                started = time.perf_counter()
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=JUDGE_TEMPERATURE,
                    max_tokens=JUDGE_MAX_TOKENS,
                    reasoning_effort=JUDGE_REASONING_EFFORT,
                )
                latency = time.perf_counter() - started
                raw = (completion.choices[0].message.content or "").strip()

                try:
                    record = parse_judge_response(raw, eval_id)
                except (JudgeParseError, json.JSONDecodeError) as e:
                    last_error = f"malformed judge JSON: {e}"
                    logger.warning(f"{eval_id}: {last_error} (attempt {json_retries + 1})")
                    json_retries += 1
                    if json_retries <= max_retries:
                        messages += [
                            {"role": "assistant", "content": raw[:1500]},
                            {"role": "user", "content":
                                "That was not valid JSON. Return only the JSON object described "
                                "in the system prompt, with no fences and no other text."},
                        ]
                        continue
                    return _failed_judge_record(eval_id, last_error, raw)

                record["judge_model"] = model_name
                record["judge_version"] = JUDGE_VERSION
                record["latency_seconds"] = round(latency, 3)
                record["retries"] = json_retries
                record["rate_limit_waits"] = rate_limit_attempts
                return record

            except Exception as e:
                name = type(e).__name__
                last_error = f"{name}: {e}"

                # Transient: wait and retry the SAME model. Downgrading here is what
                # caused an earlier run to walk the whole chain and fail on a dead model.
                if name in ("RateLimitError", "APITimeoutError", "APIConnectionError",
                            "InternalServerError"):
                    rate_limit_attempts += 1
                    if rate_limit_attempts < RATE_LIMIT_MAX_ATTEMPTS:
                        wait = RATE_LIMIT_BACKOFF_SECONDS * rate_limit_attempts
                        logger.info(f"{eval_id}: {name} on {model_name}, waiting {wait}s "
                                    f"(attempt {rate_limit_attempts}/{RATE_LIMIT_MAX_ATTEMPTS})")
                        time.sleep(wait)
                        continue
                    logger.warning(f"{eval_id}: {name} on {model_name} after "
                                   f"{rate_limit_attempts} waits — trying next model")
                    break

                # Permanent: this model is unusable, so advance the chain.
                logger.warning(f"Judge call failed permanently on {model_name}: {last_error}")
                permanent_failure = True
                break

        if not permanent_failure and rate_limit_attempts >= RATE_LIMIT_MAX_ATTEMPTS:
            continue  # exhausted waits on this model; try the next one

    return _failed_judge_record(eval_id, last_error)


# ── Aggregation ───────────────────────────────────────────────────────────────


def judge_summary(records: list[dict]) -> dict:
    """
    Aggregate judge output. Quality metrics use only successfully parsed records;
    reliability metrics use all of them. Every rate is computed here, never asserted.
    """
    if not records:
        return {"n": 0, "n_parsed": 0}

    ok = [r for r in records if r.get("parse_ok")]
    n, n_ok = len(records), len(ok)

    def dim_mean(dim: str) -> Optional[float]:
        vals = [r["scores"].get(dim) for r in ok if r["scores"].get(dim) is not None]
        return sum(vals) / len(vals) if vals else None

    overalls = [r["overall_score"] for r in ok if r["overall_score"] is not None]
    confidences = [r["confidence"] for r in ok if r.get("confidence") is not None]

    return {
        "n": n,
        "n_parsed": n_ok,
        "parse_success_rate": n_ok / n,
        "mean_overall_score": sum(overalls) / len(overalls) if overalls else None,
        "by_dimension": {d: dim_mean(d) for d in DIMENSIONS},
        "pass_rate": sum(1 for r in ok if r["pass"]) / n_ok if n_ok else None,
        "critical_failure_rate": sum(1 for r in ok if r["critical_failure"]) / n_ok if n_ok else None,
        "mean_confidence": sum(confidences) / len(confidences) if confidences else None,
        # Judge reliability — the judge evaluating itself.
        "arithmetic_error_rate": sum(1 for r in ok if r.get("arithmetic_error")) / n_ok if n_ok else None,
        "pass_rule_error_rate": sum(1 for r in ok if r.get("pass_rule_error")) / n_ok if n_ok else None,
        "off_taxonomy_rate": sum(1 for r in ok if r.get("off_taxonomy_label")) / n_ok if n_ok else None,
        "models_used": sorted({r.get("judge_model") for r in ok if r.get("judge_model")}),
        "mean_latency_seconds": (
            sum(r["latency_seconds"] for r in ok if r.get("latency_seconds") is not None)
            / max(sum(1 for r in ok if r.get("latency_seconds") is not None), 1)
        ) if any(r.get("latency_seconds") is not None for r in ok) else None,
    }
