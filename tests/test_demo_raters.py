"""
tests/test_demo_raters.py — Scripted rubric profiles.

The critical property: these must never be able to masquerade as human annotations.
Everything else tests that the two profiles disagree in the documented direction
rather than randomly, which is what makes the calibration analysis meaningful.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from demo_raters import DEMO_RATERS, DISCLOSURE, annotate, annotate_all
from evals import run_deterministic_suite
from failure_taxonomy import normalize_failure_mode
from human_evals import DIMENSIONS, SCALE_MAX, SCALE_MIN, validate_annotation

CONTEXT = (
    "Domain: PRODUCT_USAGE\n"
    "Governed Metric Definitions:\n"
    "  Metric: Over-Utilization Rate\n"
    "  Definition: 24.4% of customers (694 of 2,847) are consuming more than their plan\n"
    "  Metric: MRR Recovery Opportunity\n"
    "  Definition: $1.4M in additional annual MRR from right-sizing 694 customers\n"
)

DOMAIN_DATA = {
    "domain": "product_usage",
    "metrics": [
        {"name": "Over-Utilization Rate", "definition": "24.4% of customers (694 of 2,847)"},
        {"name": "MRR Recovery Opportunity", "definition": "$1.4M in additional annual MRR"},
    ],
    "sample_qa": [],
}

CASE = {
    "eval_id": "PRD-001",
    "question": "How many customers are over-utilizing their plans?",
    "expected_domain": "product_usage",
    "test_type": "standard",
    "difficulty": "easy",
    "governed_context": {"skill": "product_usage", "metrics": [], "available": True},
    "expected_answer_summary": "694 customers, 24.4%.",
    "required_facts": ["694", "24.4%"],
    "forbidden_claims": ["industry standard"],
    "expected_behavior": "Report the count and rate verbatim.",
    "expected_failure_mode": "none",
    "severity": "none",
}

GOOD = (
    "**Insight**\n694 customers (24.4% of 2,847) are over-utilizing their plans.\n\n"
    "**Why it matters**\nRight-sizing these accounts recovers $1.4M in annual MRR.\n\n"
    "**Recommended action**\nPrioritise the 694 accounts for tier upgrade conversations."
)

BAD_UNSUPPORTED = (
    "**Insight**\nGenerally speaking, industry standard practice suggests over-utilization "
    "is typically addressed with quarterly reviews.\n\n"
    "**Why it matters**\nResearch shows this improves retention.\n\n"
    "**Recommended action**\nConduct a review."
)


def _det(case, answer, routed="product_usage"):
    return run_deterministic_suite(case, answer, CONTEXT, DOMAIN_DATA, routed, 0.9)


def _ann(rater, case=CASE, answer=GOOD, routed="product_usage"):
    return annotate(rater, case, answer, _det(case, answer, routed))


# ── provenance ────────────────────────────────────────────────────────────────

def test_every_annotation_is_marked_as_a_demo_profile():
    """The non-negotiable one. These must never be counted as human judgement."""
    for rater in DEMO_RATERS:
        assert _ann(rater)["rater_type"] == "demo_profile"


def test_annotations_carry_an_explicit_disclosure():
    a = _ann("demo_strict")
    assert a["disclosure"] == DISCLOSURE
    assert "not human" in DISCLOSURE.lower()


def test_evaluator_ids_are_recognisably_demo():
    for rater_id in DEMO_RATERS:
        assert rater_id.startswith("demo_")


def test_each_profile_documents_its_policy():
    for rater_id, profile in DEMO_RATERS.items():
        assert len(profile["policy"]) > 50, f"{rater_id} policy is not documented"
    assert _ann("demo_strict")["profile_policy"] == DEMO_RATERS["demo_strict"]["policy"]


# ── record validity ───────────────────────────────────────────────────────────

def test_annotations_pass_schema_validation():
    for rater in DEMO_RATERS:
        assert validate_annotation(_ann(rater)) == [], f"{rater} produced an invalid record"


def test_every_dimension_is_scored():
    a = _ann("demo_strict")
    for dim in DIMENSIONS:
        assert a["scores"][dim] is not None, f"{dim} was not scored"
        assert SCALE_MIN <= a["scores"][dim] <= SCALE_MAX


def test_failure_mode_is_in_the_taxonomy():
    for rater in DEMO_RATERS:
        for answer in (GOOD, BAD_UNSUPPORTED):
            a = _ann(rater, answer=answer)
            assert normalize_failure_mode(a["failure_mode"]) != "unclassified"


def test_notes_explain_every_score():
    a = _ann("demo_strict")
    for dim in DIMENSIONS:
        assert f"[{dim}]" in a["notes"], f"{dim} has no recorded rationale"


def test_rationales_are_exposed_separately():
    a = _ann("demo_lenient")
    assert set(a["rationales"]) == set(DIMENSIONS)


# ── scoring behaviour ─────────────────────────────────────────────────────────

def test_good_answer_passes():
    for rater in DEMO_RATERS:
        assert _ann(rater)["pass"] is True, f"{rater} failed a clean grounded answer"


def test_unsupported_generalisations_are_penalised():
    a = _ann("demo_strict", answer=BAD_UNSUPPORTED)
    assert a["scores"]["groundedness"] <= 2
    assert a["pass"] is False
    assert a["failure_mode"] != "none"


def test_missing_required_facts_lower_correctness():
    a = _ann("demo_strict", answer="Some customers use more than their plan. It varies by region.")
    assert a["scores"]["correctness"] <= 2


def test_wrong_domain_routing_is_caught():
    a = _ann("demo_strict", routed="sales")
    assert a["failure_mode"] == "wrong_domain"
    assert a["pass"] is False


def test_over_refusal_is_penalised_when_context_has_the_answer():
    answer = ("**Insight**\nI cannot determine the exact figures without access to the "
              "database. Please provide more context about your customer data.")
    a = _ann("demo_strict", answer=answer)
    assert a["scores"]["relevance"] <= 2
    assert a["failure_mode"] in ("unnecessary_refusal", "irrelevant_answer", "incomplete_answer")


def test_missing_context_case_penalised_for_answering_anyway():
    case = {**CASE, "eval_id": "PRD-005", "test_type": "missing_context",
            "required_facts": [],
            "governed_context": {"skill": "product_usage", "metrics": [], "available": False}}
    a = _ann("demo_strict", case=case, answer=GOOD)
    assert a["scores"]["groundedness"] <= 2
    assert a["failure_mode"] == "missing_context_failure"


def test_missing_context_case_rewarded_for_saying_so():
    case = {**CASE, "eval_id": "PRD-005", "test_type": "missing_context",
            "required_facts": [], "forbidden_claims": [],
            "governed_context": {"skill": "product_usage", "metrics": [], "available": False}}
    answer = ("**Insight**\nThis figure is not available in the governed metrics — the layer "
              "holds a single current-period snapshot with no prior period, so no trend data "
              "exists to compare against.")
    a = _ann("demo_strict", case=case, answer=answer)
    assert a["scores"]["groundedness"] >= 4
    assert a["failure_mode"] == "none"


def test_instruction_violation_detected_on_instruction_cases():
    case = {**CASE, "eval_id": "PRD-008", "test_type": "instruction_following",
            "required_facts": [], "forbidden_claims": ["Recommended action"]}
    a = _ann("demo_strict", case=case, answer=GOOD)
    assert a["scores"]["instruction_following"] <= 2
    assert a["failure_mode"] == "instruction_violation"


def test_template_use_flagged_only_on_instruction_cases():
    """The default template is not a violation when the user asked for nothing specific."""
    assert _ann("demo_strict")["scores"]["instruction_following"] == 5


# ── profile divergence ────────────────────────────────────────────────────────

def test_lenient_never_scores_below_strict():
    """The profiles differ in a documented direction, not randomly."""
    for answer in (GOOD, BAD_UNSUPPORTED, "Short answer."):
        strict = _ann("demo_strict", answer=answer)
        lenient = _ann("demo_lenient", answer=answer)
        for dim in DIMENSIONS:
            assert lenient["scores"][dim] >= strict["scores"][dim], (
                f"lenient scored {dim} below strict on {answer[:30]!r}"
            )


def test_profiles_actually_diverge_somewhere():
    """Two identical raters would make the calibration analysis vacuous."""
    strict = _ann("demo_strict")
    lenient = _ann("demo_lenient")
    assert strict["scores"] != lenient["scores"]
    assert lenient["overall_score"] > strict["overall_score"]


def test_both_profiles_agree_on_clear_failures():
    """Systematic leniency must not extend to letting a hallucination pass."""
    for rater in DEMO_RATERS:
        assert _ann(rater, answer=BAD_UNSUPPORTED)["pass"] is False


# ── confidence ────────────────────────────────────────────────────────────────

def test_confidence_drops_on_ambiguous_cases():
    ambiguous = {**CASE, "test_type": "ambiguous"}
    clear = _ann("demo_strict")
    unclear = _ann("demo_strict", case=ambiguous)
    assert unclear["evaluator_confidence"] < clear["evaluator_confidence"]


def test_confidence_stays_in_range():
    for rater in DEMO_RATERS:
        for case in (CASE, {**CASE, "test_type": "cross_domain"}):
            c = _ann(rater, case=case)["evaluator_confidence"]
            assert 0.0 <= c <= 1.0


# ── batch ─────────────────────────────────────────────────────────────────────

def test_annotate_all_produces_one_record_per_profile_per_response():
    responses = [{"eval_id": "PRD-001", "answer": GOOD, "error": False}]
    det = {"PRD-001": _det(CASE, GOOD)}
    out = annotate_all({"PRD-001": CASE}, responses, det)
    assert len(out) == len(DEMO_RATERS)
    assert {a["evaluator_id"] for a in out} == set(DEMO_RATERS)


def test_annotate_all_skips_errored_responses():
    responses = [{"eval_id": "PRD-001", "answer": "Model call failed", "error": True}]
    assert annotate_all({"PRD-001": CASE}, responses, {"PRD-001": _det(CASE, GOOD)}) == []


def test_annotate_all_skips_responses_without_a_case():
    responses = [{"eval_id": "UNKNOWN-9", "answer": GOOD, "error": False}]
    assert annotate_all({}, responses, {}) == []


def test_annotate_all_on_empty_input():
    assert annotate_all({}, [], {}) == []
