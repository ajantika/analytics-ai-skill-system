"""
tests/test_evals.py — Unit tests for evaluation framework
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evals import evaluate_response, EvalResult


MOCK_DOMAIN_DATA = {
    "domain": "product_usage",
    "description": "Product usage analytics",
    "metrics": [
        {"name": "Over-Utilization Rate", "definition": "24.4% of customers (694 of 2,847)"},
        {"name": "MRR Recovery Opportunity", "definition": "$1.4M in additional annual MRR"},
        {"name": "Plan Fit Score", "definition": "Below 0.5 = churn risk"},
    ],
    "sample_qa": [],
}

MOCK_CONTEXT = """
Domain: PRODUCT_USAGE
Governed Metric Definitions:
  Metric: Over-Utilization Rate
  Definition: 24.4% of customers (694 of 2,847) are consuming more than their current plan
  Metric: MRR Recovery Opportunity
  Definition: $1.4M in additional annual MRR identified by right-sizing
  Metric: Plan Fit Score
  Definition: Below 0.5 = churn risk, 0.5-1.0 = healthy, above 1.0 = over-utilized
"""

GOOD_ANSWER = """
**Insight**
24.4% of customers (694 of 2,847) are over-utilizing their plans, with LATAM leading at 41% over-utilization rate.

**Why it matters**
Right-sizing these accounts would recover $1.4M in annual MRR without acquiring new customers.

**Recommended action**
Prioritise LATAM and NA accounts for tier upgrade conversations given their highest over-utilization rates.
"""

BAD_ANSWER_GENERIC = """
Generally speaking, over-utilization typically indicates that customers need to be upgraded to higher tiers.
Industry standard practice suggests conducting utilization reviews quarterly.
"""

NON_ANSWER = """
I cannot determine the exact figures without access to the database.
Please provide more context about your specific customer data.
"""


def test_good_answer_passes():
    result = evaluate_response(
        question="Which customers are over-utilizing their plans?",
        answer=GOOD_ANSWER,
        context=MOCK_CONTEXT,
        domain_data=MOCK_DOMAIN_DATA,
        routing_confidence=0.9
    )
    assert isinstance(result, EvalResult)
    assert result.groundedness in ("PASS", "WARN"), f"Expected PASS/WARN groundedness, got {result.groundedness}"
    assert result.quality_label in ("HIGH", "MEDIUM")
    assert result.overall_quality > 0.5


def test_generic_answer_flagged():
    result = evaluate_response(
        question="Which customers are over-utilizing?",
        answer=BAD_ANSWER_GENERIC,
        context=MOCK_CONTEXT,
        domain_data=MOCK_DOMAIN_DATA,
        routing_confidence=0.6
    )
    # Should flag unsupported generalisation phrases
    assert result.unsupported_claims == "FOUND", "Generic phrases should be flagged as unsupported claims"


def test_non_answer_fails_relevance():
    result = evaluate_response(
        question="What is the MRR recovery opportunity?",
        answer=NON_ANSWER,
        context=MOCK_CONTEXT,
        domain_data=MOCK_DOMAIN_DATA,
        routing_confidence=0.8
    )
    assert result.relevance == "FAIL", "Non-answer should fail relevance check"


def test_result_has_all_fields():
    result = evaluate_response(
        question="Test question",
        answer="Test answer with $1.4M figure",
        context=MOCK_CONTEXT,
        domain_data=MOCK_DOMAIN_DATA,
        routing_confidence=0.7
    )
    assert hasattr(result, "groundedness")
    assert hasattr(result, "metric_validity")
    assert hasattr(result, "relevance")
    assert hasattr(result, "unsupported_claims")
    assert hasattr(result, "overall_quality")
    assert hasattr(result, "quality_label")
    assert hasattr(result, "details")
    assert hasattr(result, "methods")


def test_quality_score_bounded():
    result = evaluate_response(
        question="Test",
        answer="Test",
        context=MOCK_CONTEXT,
        domain_data=MOCK_DOMAIN_DATA,
        routing_confidence=1.0
    )
    assert 0.0 <= result.overall_quality <= 1.0


def test_methods_are_labelled():
    result = evaluate_response(
        question="Test",
        answer="Test",
        context=MOCK_CONTEXT,
        domain_data=MOCK_DOMAIN_DATA,
        routing_confidence=0.5
    )
    # All evaluation methods must be explicitly labelled
    for method_key, method_label in result.methods.items():
        assert any(word in method_label.upper() for word in ["DETERMINISTIC", "HEURISTIC", "LLM"]), \
            f"Method '{method_key}' must be labelled as DETERMINISTIC, HEURISTIC, or LLM-BASED"


def test_low_routing_confidence_lowers_quality():
    result_high = evaluate_response(
        question="Which customers are over-utilizing?",
        answer=GOOD_ANSWER,
        context=MOCK_CONTEXT,
        domain_data=MOCK_DOMAIN_DATA,
        routing_confidence=0.95
    )
    result_low = evaluate_response(
        question="Which customers are over-utilizing?",
        answer=GOOD_ANSWER,
        context=MOCK_CONTEXT,
        domain_data=MOCK_DOMAIN_DATA,
        routing_confidence=0.20
    )
    assert result_high.overall_quality >= result_low.overall_quality, \
        "Higher routing confidence should produce equal or higher quality score"


if __name__ == "__main__":
    tests = [
        test_good_answer_passes,
        test_generic_answer_flagged,
        test_non_answer_fails_relevance,
        test_result_has_all_fields,
        test_quality_score_bounded,
        test_methods_are_labelled,
        test_low_routing_confidence_lowers_quality,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ✓ {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  ✗ {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ {t.__name__} (error): {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
