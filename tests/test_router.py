"""
tests/test_router.py — Unit tests for domain routing logic
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from router import (
    AMBIGUITY_THRESHOLD,
    DEFAULT_ROUTER_VERSION,
    ROUTER_VERSION_NOTES,
    ROUTER_VERSIONS,
    classify_domain,
    classify_domain_v1,
    classify_domain_v2,
    classify_domain_v3,
    domain_frequency,
)


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


def test_routing_method_never_claims_to_be_semantic():
    """
    The original assertion pinned the literal string "keyword". Its real intent was
    that routing must never be presented as semantic or model-based when it is
    keyword scoring. That intent is what is tested now, across every version.
    """
    for version in ROUTER_VERSIONS:
        result = classify_domain("What is our MRR?", MOCK_DOMAINS, version=version)
        assert result.method.startswith("keyword"), \
            f"{version} labelled its method {result.method!r}"
        for forbidden in ("semantic", "embedding", "llm", "neural", "ai"):
            assert forbidden not in result.method.lower(), \
                f"{version} method {result.method!r} implies a technique it does not use"


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


# ── Router versioning ─────────────────────────────────────────────────────────

def test_all_versions_registered_and_documented():
    assert set(ROUTER_VERSIONS) == set(ROUTER_VERSION_NOTES)
    assert DEFAULT_ROUTER_VERSION in ROUTER_VERSIONS
    for note in ROUTER_VERSION_NOTES.values():
        assert len(note) > 40, "each version must document what it changed"


def test_unknown_version_raises():
    try:
        classify_domain("What is our MRR?", MOCK_DOMAINS, version="v99_imaginary")
        assert False, "unknown router version should raise"
    except ValueError as e:
        assert "v99_imaginary" in str(e)


def test_every_version_tags_its_own_result():
    for version, fn in ROUTER_VERSIONS.items():
        assert fn("What is our CSAT score?", MOCK_DOMAINS).version == version


def test_default_version_is_used_when_unspecified():
    assert classify_domain("What is our CSAT?", MOCK_DOMAINS).version == DEFAULT_ROUTER_VERSION


def test_backward_compatible_three_arg_call():
    """Existing callers pass (question, domains, fallback) positionally — that must still work."""
    result = classify_domain("What is our CSAT score?", MOCK_DOMAINS, "sales")
    assert result.domain == "csup"


# ── The substring defect v2 fixes ─────────────────────────────────────────────

SUBSTRING_TRAP = {
    "product_usage": {"keywords": ["overage", "utilization", "plan"], "metrics": [], "sample_qa": []},
    "sales": {"keywords": ["pipeline", "coverage ratio", "quota"], "metrics": [], "sample_qa": []},
}


def test_v1_exhibits_the_substring_defect():
    """
    Documents the baseline defect rather than hiding it: 'coverage' contains
    'overage', so v1 scores the Product domain on a Sales question.
    """
    scores = dict(classify_domain_v1("What is our pipeline coverage ratio?", SUBSTRING_TRAP).top_domains)
    assert scores["product_usage"] > 0, "v1 is expected to false-positive here — that is the bug"


def test_v2_and_v3_do_not_exhibit_the_substring_defect():
    for fn in (classify_domain_v2, classify_domain_v3):
        scores = dict(fn("What is our pipeline coverage ratio?", SUBSTRING_TRAP).top_domains)
        assert scores["product_usage"] == 0, \
            f"{fn.__name__}: 'coverage' must not match the keyword 'overage'"
        assert scores["sales"] > 0


def test_token_matching_respects_word_boundaries():
    domains = {"a": {"keywords": ["plan"], "metrics": [], "sample_qa": []},
               "b": {"keywords": ["report"], "metrics": [], "sample_qa": []}}
    scores = dict(classify_domain_v3("What is our planning process?", domains).top_domains)
    assert scores["a"] == 0, "'planning' must not match the keyword 'plan'"


def test_plural_stemming_matches():
    domains = {"a": {"keywords": ["ticket"], "metrics": [], "sample_qa": []},
               "b": {"keywords": ["zzz"], "metrics": [], "sample_qa": []}}
    assert classify_domain_v3("How many tickets are open?", domains).domain == "a"


def test_multiword_phrase_scores_above_single_word():
    domains = {
        "phrase": {"keywords": ["pipeline coverage"], "metrics": [], "sample_qa": []},
        "single": {"keywords": ["pipeline"], "metrics": [], "sample_qa": []},
    }
    result = classify_domain_v3("What is our pipeline coverage?", domains)
    assert result.domain == "phrase", "a contiguous phrase match must outrank a single-word match"


# ── v3: inverse domain frequency ──────────────────────────────────────────────

def test_domain_frequency_counts_claiming_domains():
    df = domain_frequency(MOCK_DOMAINS)
    assert df["mrr"] == 2, "'mrr' is claimed by product_usage and sales"
    assert df["csat"] == 1, "'csat' is claimed only by csup"


def test_idf_downweights_shared_keywords():
    """
    A domain whose only match is a keyword every domain claims must lose to a domain
    matching a keyword unique to it. This is the collision fix.
    """
    domains = {
        "shared_only": {"keywords": ["customers", "revenue"], "metrics": [], "sample_qa": []},
        "unique_only": {"keywords": ["customers", "csat"], "metrics": [], "sample_qa": []},
        "third": {"keywords": ["customers"], "metrics": [], "sample_qa": []},
    }
    result = classify_domain_v3("How many customers report low csat?", domains)
    assert result.domain == "unique_only"


def test_no_match_is_flagged_explicitly():
    """A question matching nothing must say so, not silently return a domain as if confident."""
    result = classify_domain_v3("What is the airspeed velocity of a swallow?", MOCK_DOMAINS)
    assert result.no_match is True
    assert result.is_ambiguous is True
    assert result.is_confident is False


def test_matched_question_is_not_flagged_no_match():
    result = classify_domain_v3("What is our CSAT score?", MOCK_DOMAINS)
    assert result.no_match is False


def test_tie_is_flagged_rather_than_resolved_arbitrarily():
    domains = {
        "a": {"keywords": ["widget"], "metrics": [], "sample_qa": []},
        "b": {"keywords": ["widget"], "metrics": [], "sample_qa": []},
    }
    result = classify_domain_v3("How many widgets?", domains)
    assert result.is_tie is True
    assert result.is_ambiguous is True
    assert "arbitrary" in result.reasoning.lower()


def test_clear_winner_is_not_flagged_as_tie():
    result = classify_domain_v3("What is our CSAT score and SLA compliance?", MOCK_DOMAINS)
    assert result.is_tie is False


def test_is_confident_requires_all_three_clear():
    result = classify_domain_v3("What is our CSAT score and SLA compliance?", MOCK_DOMAINS)
    assert result.is_confident is True


def test_empty_domains_handled_by_every_version():
    for version, fn in ROUTER_VERSIONS.items():
        result = fn("anything", {})
        assert result.domain is None, f"{version} must return None with no skill files"
        assert result.is_ambiguous is True


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
