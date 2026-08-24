"""
demo_raters.py — Scripted rubric profiles. NOT human annotators.

WHAT THIS IS
Two deterministic annotation profiles that read the actual generated response and
apply the human rubric through explicit anchor rules. They exist so the calibration
workflow — inter-rater agreement, disagreement analysis, rubric refinement — has two
differing raters to reason about without claiming that people produced the ratings.

WHAT THIS IS NOT
These are not human annotations, not crowdworker output, and not a substitute for
human judgement. Every record they produce carries rater_type="demo_profile", and
every statistic computed over them is labelled accordingly in the UI. Human
annotations authored in the Human Evaluation page carry rater_type="human" and are
reported separately everywhere.

WHY TWO PROFILES
They differ in a documented, principled way rather than randomly:

  demo_strict   Applies anchors literally. Where a response sits between two anchors,
                takes the lower one. Treats any unmarked interpretive claim as a
                groundedness problem. Models a rater optimising for precision.

  demo_lenient  Gives benefit of the doubt between adjacent anchors. Scores
                helpfulness and clarity on whether a business reader could use the
                answer, not on whether it is tightly written. Models a rater
                optimising for usefulness.

That difference is the point: it produces systematic, explainable disagreement of
the kind real annotation programmes have to detect and calibrate away, rather than
noise.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from failure_taxonomy import normalize_failure_mode
from human_evals import DIMENSIONS, make_annotation

logger = logging.getLogger(__name__)

DEMO_RATERS = {
    "demo_strict": {
        "evaluator_id": "demo_strict",
        "label": "Demo profile — strict",
        "policy": (
            "Applies rubric anchors literally. Between two anchors, takes the lower. "
            "Treats unmarked interpretation as a groundedness defect. Optimises for precision."
        ),
        "leniency": -1,
    },
    "demo_lenient": {
        "evaluator_id": "demo_lenient",
        "label": "Demo profile — lenient",
        "policy": (
            "Gives benefit of the doubt between adjacent anchors. Scores helpfulness and "
            "clarity on usability rather than tightness. Optimises for recall of good answers."
        ),
        "leniency": +1,
    },
}

DISCLOSURE = (
    "Demo profiles are scripted rubric applications, not human annotators. "
    "They read the real generated response and apply explicit anchor rules."
)

_HEDGE = re.compile(
    r"i (?:cannot|can't|don't|do not) (?:determine|know|have)|"
    r"insufficient (?:data|information)|would need access to|unable to answer",
    re.I,
)
_INFERENCE_MARKED = re.compile(
    r"\b(hypothesis|hypothesise|hypothesize|inference|inferred|suggests that|"
    r"cannot be confirmed|not (?:stated|available|present|defined) in the|"
    r"the governed (?:layer|context|metrics) (?:does not|do not)|no .{0,30}data (?:exists|is available))\b",
    re.I,
)
_TEMPLATE = re.compile(r"\*\*(insight|why it matters|recommended action)\*\*", re.I)


def _clamp(value: int) -> int:
    return max(1, min(5, int(value)))


def _score_dimensions(case: dict, answer: str, det: dict, leniency: int) -> tuple[dict, dict]:
    """
    Apply the rubric anchors to one response.

    Returns (scores, rationales). Every score carries the rule that produced it, so a
    reader can check the profile rather than trust it.
    """
    checks = det["checks"]
    facts = checks["required_facts"]
    forbidden = checks["forbidden_claims"]
    grounding = checks["numeric_grounding"]["status"]
    relevance_h = checks["relevance_heuristic"]["status"]
    unsupported = checks["unsupported_claim_heuristic"]["status"]
    routed_ok = checks["expected_domain"]["status"]

    test_type = case.get("test_type")
    ctx_available = case.get("governed_context", {}).get("available", True)
    marked_inference = bool(_INFERENCE_MARKED.search(answer))
    hedged = bool(_HEDGE.search(answer))
    has_template = bool(_TEMPLATE.search(answer))
    length = len(answer.strip())

    scores: dict[str, Optional[int]] = {}
    why: dict[str, str] = {}

    # ── groundedness ──────────────────────────────────────────────────────────
    # The correctly-declined case is checked first. When the governed context does
    # not hold the answer, containing no figures is the correct outcome, not a
    # grounding warning — the generic heuristic would otherwise penalise the one
    # behaviour these cases exist to reward.
    if ctx_available is False and marked_inference and forbidden["status"] != "FAIL":
        scores["groundedness"] = 5
        why["groundedness"] = (
            "Anchor 5: the governed context does not contain what was asked, and the answer "
            "states that explicitly rather than supplying a figure."
        )
    elif forbidden["status"] == "FAIL":
        scores["groundedness"] = 1
        why["groundedness"] = f"Anchor 1: contains a forbidden claim ({forbidden['violations'][0]!r})."
    elif grounding == "FAIL":
        scores["groundedness"] = 1
        why["groundedness"] = "Anchor 1: figures in the answer are absent from the governed context."
    elif ctx_available is False and not marked_inference and length > 120:
        scores["groundedness"] = 2
        why["groundedness"] = (
            "Anchor 2: the governed context does not contain what was asked, and the answer "
            "does not say so."
        )
    elif unsupported == "FOUND":
        scores["groundedness"] = 2
        why["groundedness"] = "Anchor 2: contains a generalisation the governed context does not support."
    elif grounding == "WARN":
        scores["groundedness"] = 3 if leniency < 0 else 4
        why["groundedness"] = "Anchor 3-4: some figures matched the context, others could not be confirmed."
    elif marked_inference:
        scores["groundedness"] = 5
        why["groundedness"] = "Anchor 5: figures grounded and inference explicitly marked as such."
    else:
        scores["groundedness"] = 4 if leniency < 0 else 5
        why["groundedness"] = (
            "Anchor 4-5: all figures grounded. Strict withholds 5 because interpretation is unmarked."
        )

    # ── correctness ───────────────────────────────────────────────────────────
    if facts["status"] == "FAIL":
        scores["correctness"] = 2
        why["correctness"] = f"Anchor 2: required facts missing ({', '.join(facts['missing'][:3])})."
    elif forbidden["status"] == "FAIL":
        scores["correctness"] = 2
        why["correctness"] = "Anchor 2: asserts something the case explicitly forbids."
    elif facts["status"] == "WARN":
        scores["correctness"] = 3
        why["correctness"] = f"Anchor 3: partial fact coverage ({facts['coverage']:.0%})."
    elif routed_ok == "FAIL":
        scores["correctness"] = 2
        why["correctness"] = "Anchor 2: grounded in the wrong domain's governed context."
    else:
        scores["correctness"] = 4 if leniency < 0 else 5
        why["correctness"] = "Anchor 4-5: figures map to the right metrics and conclusions hold."

    # ── relevance ─────────────────────────────────────────────────────────────
    if relevance_h == "FAIL":
        scores["relevance"] = 2
        why["relevance"] = "Anchor 2: hedging language suggests the question was not addressed."
    elif hedged and ctx_available:
        scores["relevance"] = 2
        why["relevance"] = "Anchor 2: declined to answer although the context contains the figures."
    elif relevance_h == "WARN":
        scores["relevance"] = 3 if leniency < 0 else 4
        why["relevance"] = "Anchor 3-4: addresses the topic but term overlap with the question is low."
    else:
        scores["relevance"] = 5
        why["relevance"] = "Anchor 5: directly addresses what was asked."

    # ── instruction following ─────────────────────────────────────────────────
    if test_type == "instruction_following":
        if forbidden["status"] == "FAIL":
            scores["instruction_following"] = 2
            why["instruction_following"] = (
                f"Anchor 2: the question gave an explicit format instruction and the answer "
                f"violates it ({forbidden['violations'][0]!r})."
            )
        elif has_template:
            scores["instruction_following"] = 2
            why["instruction_following"] = (
                "Anchor 2: the default three-section template was used despite an explicit "
                "instruction specifying a different format."
            )
        else:
            scores["instruction_following"] = 5
            why["instruction_following"] = "Anchor 5: the explicit format instruction was followed."
    else:
        scores["instruction_following"] = 5
        why["instruction_following"] = "Anchor 5: no explicit format instruction was given."

    # ── helpfulness ───────────────────────────────────────────────────────────
    if length < 120:
        scores["helpfulness"] = 2 if leniency < 0 else 3
        why["helpfulness"] = "Anchor 2-3: too brief to convey a decision-relevant point."
    elif facts["status"] == "FAIL" or relevance_h == "FAIL":
        scores["helpfulness"] = 2
        why["helpfulness"] = "Anchor 2: does not give the reader what they needed."
    elif ctx_available is False and marked_inference:
        # Correctly saying "this is not available" is genuinely useful.
        scores["helpfulness"] = 4 if leniency < 0 else 5
        why["helpfulness"] = (
            "Anchor 4-5: correctly identifies that the governed layer cannot answer this, "
            "which saves the reader a false conclusion."
        )
    elif length > 1400:
        scores["helpfulness"] = 3 if leniency < 0 else 4
        why["helpfulness"] = (
            "Anchor 3-4: substantive but padded. Strict scores restatement lower; lenient "
            "scores on whether the reader can act on it."
        )
    else:
        scores["helpfulness"] = 4 if leniency < 0 else 5
        why["helpfulness"] = "Anchor 4-5: surfaces the comparison the reader needed."

    # ── clarity ───────────────────────────────────────────────────────────────
    if length > 1800:
        scores["clarity"] = 2 if leniency < 0 else 3
        why["clarity"] = "Anchor 2-3: long enough to require rereading."
    elif has_template or "\n" in answer.strip():
        scores["clarity"] = 4 if leniency < 0 else 5
        why["clarity"] = "Anchor 4-5: structured and readable."
    elif length < 120:
        scores["clarity"] = 4
        why["clarity"] = "Anchor 4: short and unambiguous."
    else:
        scores["clarity"] = 3 if leniency < 0 else 4
        why["clarity"] = "Anchor 3-4: readable but unstructured."

    return {k: _clamp(v) for k, v in scores.items()}, why


def _infer_failure_mode(case: dict, answer: str, det: dict, scores: dict) -> str:
    """
    Choose the single failure mode a reviewer would fix first.

    Ordered by severity so a hallucinated figure is never reported as a clarity
    problem. Returns "none" when nothing crossed a failure threshold.
    """
    checks = det["checks"]
    ctx_available = case.get("governed_context", {}).get("available", True)
    marked = bool(_INFERENCE_MARKED.search(answer))

    if checks["expected_domain"]["status"] == "FAIL":
        return "wrong_domain"
    if checks["numeric_grounding"]["status"] == "FAIL":
        return "hallucinated_number"
    if checks["forbidden_claims"]["status"] == "FAIL":
        # A forbidden phrase in an instruction case is a format violation, not a fabrication.
        if case.get("test_type") == "instruction_following":
            return "instruction_violation"
        # Prefer the mode the case was designed to elicit, but a forbidden-claim
        # violation is never "no failure" — fall through to the generic category.
        designed = normalize_failure_mode(case.get("expected_failure_mode"))
        return designed if designed not in ("none", "unclassified") else "unsupported_claim"
    if ctx_available is False and not marked:
        return "missing_context_failure"
    if case.get("test_type") == "instruction_following" and scores["instruction_following"] <= 2:
        return "instruction_violation"
    if _HEDGE.search(answer) and ctx_available:
        return "unnecessary_refusal"
    if checks["unsupported_claim_heuristic"]["status"] == "FOUND":
        return "unsupported_claim"
    if checks["required_facts"]["status"] == "FAIL":
        return "incomplete_answer"
    if scores["relevance"] <= 2:
        return "irrelevant_answer"
    if checks["required_facts"]["status"] == "WARN":
        return "incomplete_answer"
    return "none"


def annotate(rater_id: str, case: dict, answer: str, deterministic: dict, timestamp: Optional[str] = None) -> dict:
    """Produce one demo-profile annotation for one response."""
    profile = DEMO_RATERS[rater_id]
    scores, rationales = _score_dimensions(case, answer, deterministic, profile["leniency"])
    mode = _infer_failure_mode(case, answer, deterministic, scores)

    # Confidence drops where the rubric was genuinely hard to apply, which is what
    # makes low-confidence cases worth reviewing as a batch.
    confidence = 0.85
    if case.get("test_type") in ("ambiguous", "cross_domain"):
        confidence = 0.55
    elif case.get("governed_context", {}).get("available") is False:
        confidence = 0.7
    if any(s == 3 for s in scores.values()):
        confidence -= 0.1

    notes = " ".join(f"[{d}] {rationales[d]}" for d in DIMENSIONS if d in rationales)

    ann = make_annotation(
        eval_id=case["eval_id"],
        evaluator_id=rater_id,
        scores=scores,
        rater_type="demo_profile",
        failure_mode=mode,
        evaluator_confidence=round(max(0.3, confidence), 2),
        notes=notes,
        timestamp=timestamp,
    )
    ann["rationales"] = rationales
    ann["profile_policy"] = profile["policy"]
    ann["disclosure"] = DISCLOSURE
    return ann


def annotate_all(
    cases_by_id: dict,
    responses: list[dict],
    deterministic_by_id: dict,
    timestamp: Optional[str] = None,
    rater_ids: Optional[list[str]] = None,
) -> list[dict]:
    """Run every demo profile over every response that has a deterministic record."""
    rater_ids = rater_ids or list(DEMO_RATERS)
    out = []
    for resp in responses:
        eval_id = resp.get("eval_id")
        case = cases_by_id.get(eval_id)
        det = deterministic_by_id.get(eval_id)
        if not case or not det or resp.get("error"):
            continue
        for rid in rater_ids:
            out.append(annotate(rid, case, resp.get("answer", ""), det, timestamp))
    logger.info(f"Demo profiles produced {len(out)} annotations across {len(rater_ids)} profiles")
    return out
