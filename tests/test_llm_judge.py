"""
tests/test_llm_judge.py — Judge output parsing, validation and aggregation.

No network calls. Everything here tests the boundary between an untrusted model
response and the structured record the dashboards consume — which is where an LLM
judge actually breaks in practice.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from human_evals import DIMENSIONS
from llm_judge import (
    JUDGE_SYSTEM_PROMPT,
    JUDGE_VERSION,
    JudgeParseError,
    _extract_json_object,
    build_judge_prompt,
    judge_summary,
    parse_judge_response,
)

VALID = {
    "relevance": 4, "groundedness": 5, "correctness": 4,
    "instruction_following": 5, "helpfulness": 4, "clarity": 4,
    "overall_score": 4.33, "pass": True, "critical_failure": False,
    "failure_mode": "none", "confidence": 0.91,
    "reasoning_summary": "All figures are supported by the supplied context.",
}

CASE = {
    "eval_id": "PRD-001",
    "question": "How many customers are over-utilizing their plans?",
    "expected_behavior": "Report the count and rate verbatim.",
    "expected_answer_summary": "694 customers, 24.4%.",
    "required_facts": ["694", ["24.4%", "24.4 percent"]],
    "forbidden_claims": ["industry standard"],
    "governed_context": {"skill": "product_usage", "metrics": [], "available": True},
    "test_type": "standard",
}


def _raw(**overrides) -> str:
    return json.dumps({**VALID, **overrides})


# ── prompt construction ───────────────────────────────────────────────────────

def test_system_prompt_states_the_hard_rules():
    p = JUDGE_SYSTEM_PROMPT.lower()
    assert "only source of truth" in p
    assert "unsupported" in p
    assert "overconfid" in p
    assert "inference" in p and "evidence" in p


def test_system_prompt_does_not_request_chain_of_thought():
    """Asking a model to expose private reasoning is not something to demonstrate."""
    p = JUDGE_SYSTEM_PROMPT.lower()
    for phrase in ["chain of thought", "chain-of-thought", "think step by step",
                   "show your reasoning", "step-by-step reasoning"]:
        assert phrase not in p, f"judge prompt requests {phrase!r}"


def test_system_prompt_lists_every_rubric_dimension():
    for dim in DIMENSIONS:
        assert dim in JUDGE_SYSTEM_PROMPT, f"{dim} missing from judge prompt"


def test_prompt_includes_context_question_and_response():
    prompt = build_judge_prompt(CASE, "694 customers (24.4%).", "GOVERNED: 694 of 2,847")
    assert "GOVERNED: 694 of 2,847" in prompt
    assert CASE["question"] in prompt
    assert "694 customers (24.4%)." in prompt


def test_prompt_surfaces_required_and_forbidden():
    prompt = build_judge_prompt(CASE, "answer", "context")
    assert "694" in prompt
    assert "24.4% | 24.4 percent" in prompt, "alternative surface forms should be shown to the judge"
    assert "industry standard" in prompt


def test_prompt_flags_deliberately_absent_context():
    case = {**CASE, "governed_context": {"skill": "hr", "metrics": [], "available": False}}
    prompt = build_judge_prompt(case, "answer", "context")
    assert "does NOT contain" in prompt


def test_prompt_handles_empty_context_without_crashing():
    prompt = build_judge_prompt(CASE, "answer", "")
    assert "no governed context" in prompt


# ── JSON extraction ───────────────────────────────────────────────────────────

def test_extract_plain_json():
    assert json.loads(_extract_json_object('{"a": 1}'))["a"] == 1


def test_extract_from_markdown_fence():
    wrapped = '```json\n{"a": 1}\n```'
    assert json.loads(_extract_json_object(wrapped))["a"] == 1


def test_extract_from_unlabelled_fence():
    assert json.loads(_extract_json_object('```\n{"a": 1}\n```'))["a"] == 1


def test_extract_ignores_prose_before_and_after():
    messy = 'Here is my evaluation:\n{"a": 1}\nLet me know if you need more.'
    assert json.loads(_extract_json_object(messy))["a"] == 1


def test_extract_handles_nested_objects():
    nested = '{"outer": {"inner": {"deep": 1}}, "b": 2}'
    assert json.loads(_extract_json_object(nested))["outer"]["inner"]["deep"] == 1


def test_extract_handles_braces_inside_strings():
    """Brace matching must not be fooled by a brace in the reasoning text."""
    tricky = '{"reasoning_summary": "the answer used {placeholder} syntax", "a": 1}'
    assert json.loads(_extract_json_object(tricky))["a"] == 1


def test_extract_handles_escaped_quotes_in_strings():
    tricky = '{"reasoning_summary": "it said \\"694\\" correctly", "a": 1}'
    assert json.loads(_extract_json_object(tricky))["a"] == 1


def test_extract_raises_when_no_object():
    for bad in ["", "no json here at all", "[1, 2, 3]"]:
        try:
            _extract_json_object(bad)
            assert False, f"should raise on {bad!r}"
        except JudgeParseError:
            pass


def test_extract_raises_on_unbalanced_braces():
    try:
        _extract_json_object('{"a": 1')
        assert False, "should raise on truncated JSON"
    except JudgeParseError:
        pass


# ── parsing ───────────────────────────────────────────────────────────────────

def test_parse_valid_response():
    r = parse_judge_response(_raw(), "PRD-001")
    assert r["parse_ok"] is True
    assert r["eval_id"] == "PRD-001"
    assert r["scores"]["groundedness"] == 5
    assert r["failure_mode"] == "none"
    assert r["confidence"] == 0.91


def test_parse_recomputes_overall_rather_than_trusting_it():
    """The judge's arithmetic is not the source of truth."""
    r = parse_judge_response(_raw(overall_score=1.0))
    assert abs(r["overall_score"] - (4 + 5 + 4 + 5 + 4 + 4) / 6) < 1e-9
    assert r["overall_score_reported"] == 1.0
    assert r["arithmetic_error"] is True


def test_correct_arithmetic_is_not_flagged():
    assert parse_judge_response(_raw())["arithmetic_error"] is False


def test_parse_recomputes_pass_under_the_written_rule():
    """Judge says pass; groundedness of 2 fails the gate, so the record says fail."""
    r = parse_judge_response(_raw(groundedness=2, **{"pass": True}))
    assert r["pass"] is False
    assert r["pass_reported"] is True
    assert r["pass_rule_error"] is True


def test_critical_failure_derived_from_the_taxonomy():
    r = parse_judge_response(_raw(failure_mode="hallucinated_number", critical_failure=False))
    assert r["critical_failure"] is True, "taxonomy severity overrides the judge's own flag"
    assert r["critical_failure_reported"] is False


def test_failure_mode_alias_is_normalised():
    r = parse_judge_response(_raw(failure_mode="Hallucination"))
    assert r["failure_mode"] == "hallucinated_number"
    assert r["failure_mode_raw"] == "Hallucination"


def test_off_taxonomy_label_is_flagged_not_coerced():
    r = parse_judge_response(_raw(failure_mode="the_answer_was_boring"))
    assert r["failure_mode"] == "unclassified"
    assert r["off_taxonomy_label"] is True


def test_scores_are_clamped_to_range():
    r = parse_judge_response(_raw(relevance=9, clarity=-3))
    assert r["scores"]["relevance"] == 5
    assert r["scores"]["clarity"] == 1


def test_float_scores_are_rounded():
    assert parse_judge_response(_raw(relevance=4.4))["scores"]["relevance"] == 4
    assert parse_judge_response(_raw(relevance=4.6))["scores"]["relevance"] == 5


def test_string_booleans_are_coerced():
    r = parse_judge_response(_raw(**{"pass": "false"}))
    assert r["pass_reported"] is False


def test_confidence_is_clamped():
    assert parse_judge_response(_raw(confidence=1.8))["confidence"] == 1.0
    assert parse_judge_response(_raw(confidence=-0.5))["confidence"] == 0.0


def test_non_numeric_confidence_becomes_none_not_zero():
    """A missing confidence must not read as 'the judge was certain it was uncertain'."""
    assert parse_judge_response(_raw(confidence="high"))["confidence"] is None


def test_missing_dimension_is_recorded_not_invented():
    payload = {k: v for k, v in VALID.items() if k != "helpfulness"}
    r = parse_judge_response(json.dumps(payload))
    assert r["scores"]["helpfulness"] is None
    assert "helpfulness" in r["missing_dimensions"]
    assert r["overall_score"] == (4 + 5 + 4 + 5 + 4) / 5, "mean must exclude the missing dimension"


def test_all_dimensions_missing_raises():
    try:
        parse_judge_response(json.dumps({"pass": True, "reasoning_summary": "x"}))
        assert False, "should raise when no dimension was scored"
    except JudgeParseError:
        pass


def test_non_numeric_score_raises():
    try:
        parse_judge_response(_raw(relevance="very good"))
        assert False, "should raise on a non-numeric score"
    except JudgeParseError:
        pass


def test_array_payload_raises():
    try:
        parse_judge_response("[1, 2, 3]")
        assert False, "should raise on a non-object payload"
    except JudgeParseError:
        pass


def test_malformed_json_raises_recoverably():
    for bad in ['{"relevance": 4,,}', '{"relevance": }', 'null']:
        try:
            parse_judge_response(bad)
            assert False, f"should raise on {bad!r}"
        except (JudgeParseError, json.JSONDecodeError):
            pass


def test_parse_survives_fenced_and_prefixed_output():
    messy = 'Sure! Here is the evaluation.\n```json\n' + _raw() + '\n```\nHope that helps.'
    assert parse_judge_response(messy)["parse_ok"] is True


# ── aggregation ───────────────────────────────────────────────────────────────

def _rec(**over):
    r = parse_judge_response(_raw(**over))
    r["judge_model"] = "test-model"
    r["latency_seconds"] = 1.0
    return r


def _failed():
    from llm_judge import _failed_judge_record
    return _failed_judge_record("X", "malformed judge JSON")


def test_summary_of_empty_input():
    s = judge_summary([])
    assert s["n"] == 0 and s["n_parsed"] == 0


def test_summary_computes_rates_from_records():
    records = [_rec(), _rec(groundedness=1, failure_mode="hallucinated_number")]
    s = judge_summary(records)
    assert s["n"] == 2 and s["n_parsed"] == 2
    assert s["parse_success_rate"] == 1.0
    assert s["pass_rate"] == 0.5
    assert s["critical_failure_rate"] == 0.5


def test_summary_excludes_failed_parses_from_quality_but_not_reliability():
    s = judge_summary([_rec(), _failed()])
    assert s["n"] == 2
    assert s["n_parsed"] == 1
    assert s["parse_success_rate"] == 0.5
    assert s["pass_rate"] == 1.0, "quality metrics use parsed records only"


def test_summary_reports_judge_self_consistency():
    s = judge_summary([_rec(overall_score=1.0), _rec()])
    assert s["arithmetic_error_rate"] == 0.5


def test_summary_reports_models_actually_used():
    """The artifact records the model that ran, not the one that was requested."""
    assert judge_summary([_rec()])["models_used"] == ["test-model"]


def test_summary_dimension_means():
    s = judge_summary([_rec(relevance=2), _rec(relevance=4)])
    assert s["by_dimension"]["relevance"] == 3.0


def test_summary_all_failed_gives_none_not_zero():
    s = judge_summary([_failed(), _failed()])
    assert s["n_parsed"] == 0
    assert s["pass_rate"] is None
    assert s["mean_overall_score"] is None


def test_judge_version_is_pinned():
    assert JUDGE_VERSION.startswith("judge-v")
