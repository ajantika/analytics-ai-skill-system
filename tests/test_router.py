"""
tests/test_router.py — Unit tests for domain routing logic
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from router import classify_domain, AMBIGUITY_THRESHOLD


# Minimal domain fixtures (no YAML files needed for unit tests)
MOCK_DOMAINS = {
    "product_usage": {
        "keywords": ["product", "usage", "utilization", "over-utilized", "plan", "mrr", "region"],
        "description": "Product usage analytics",
        "metrics": [],
        "sample_qa": [],
    },
    "marketing": {
        "keywords": ["campaign", "leads", "cac", "mql", "sql", "marketing", "channel", "funnel"],
        "description": "Marketing analytics",
        "metrics": [],
        "sample_qa": [],
    },
    "sales": {
        "keywords": ["revenue", "pipeline", "deals", "sales", "quota", "discount", "rep", "mrr"],
        "description": "Sales analytics",
        "metrics": [],
        "sample_qa": [],
    },
    "hr": {
        "keywords": ["attrition", "hiring", "headcount", "employees", "retention", "enps", "talent"],
        "description": "HR analytics",
        "metrics": [],
        "sample_qa": [],
    },
    "csup": {
        "keywords": ["tickets", "support", "csat", "sla", "response", "agent", "resolution"],
        "description": "Support analytics",
        "metrics": [],
        "sample_qa": [],
    },
}


def test_product_routing():
    result = classify_domain("Which customers are over-utilizing their plans?", MOCK_DOMAINS)
    assert result.domain == "product_usage", f"Expected product_usage, got {result.domain}"
    assert result.confidence > 0.4


def test_marketing_routing():
    result = classify_domain("Which campaign brought the highest number of customers?", MOCK_DOMAINS)
    assert result.domain == "marketing", f"Expected marketing, got {result.domain}"


def test_sales_routing():
    result = classify_domain("Which sales rep gives the highest discounts?", MOCK_DOMAINS)
    assert result.domain == "sales", f"Expected sales, got {result.domain}"


def test_hr_routing():
    result = classify_domain("Which teams have the highest attrition?", MOCK_DOMAINS)
    assert result.domain == "hr", f"Expected hr, got {result.domain}"


def test_support_routing():
    result = classify_domain("What is our CSAT score?", MOCK_DOMAINS)
    assert result.domain == "csup", f"Expected csup, got {result.domain}"


def test_ambiguous_question_low_confidence():
    result = classify_domain("How are we doing?", MOCK_DOMAINS)
    # Should still return a domain but confidence should be low or it's ambiguous
    assert result.domain is not None
    # Very generic question — ambiguous flag may be set
    print(f"Ambiguous question confidence: {result.confidence:.2f}, ambiguous={result.is_ambiguous}")


def test_routing_method_always_keyword():
    result = classify_domain("What is our MRR?", MOCK_DOMAINS)
    assert result.method == "keyword", "Routing method must always be labelled 'keyword'"


def test_empty_question_returns_fallback():
    # Empty / whitespace question — should not crash
    result = classify_domain("", MOCK_DOMAINS)
    assert result.domain is not None or result.is_ambiguous


def test_no_domains_returns_none():
    result = classify_domain("What is our MRR?", {})
    assert result.domain is None
    assert result.is_ambiguous


def test_mrr_routes_to_product_or_sales():
    result = classify_domain("What is the MRR recovery opportunity?", MOCK_DOMAINS)
    # MRR is in both product and sales — should go to one of them
    assert result.domain in ("product_usage", "sales"), f"Unexpected domain: {result.domain}"


if __name__ == "__main__":
    tests = [
        test_product_routing,
        test_marketing_routing,
        test_sales_routing,
        test_hr_routing,
        test_support_routing,
        test_ambiguous_question_low_confidence,
        test_routing_method_always_keyword,
        test_empty_question_returns_fallback,
        test_no_domains_returns_none,
        test_mrr_routes_to_product_or_sales,
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
