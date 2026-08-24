"""
tests/test_failure_taxonomy.py — Taxonomy integrity and label normalisation.

The taxonomy is the shared vocabulary between three independent evaluators. If
normalisation silently coerces an unknown label into a neighbouring category, the
failure distribution on the dashboard becomes fiction. These tests pin that down.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from failure_taxonomy import (
    FAILURE_MODES,
    SEVERITY_ORDER,
    all_modes,
    distribution,
    get_failure_mode,
    is_critical,
    label_of,
    normalize_failure_mode,
    severity_distribution,
    severity_of,
)


# ── structure ─────────────────────────────────────────────────────────────────

def test_every_mode_has_required_fields():
    required = ["label", "severity", "is_critical", "description",
                "example", "likely_cause", "remediation", "detected_by"]
    for key, entry in FAILURE_MODES.items():
        for field in required:
            assert field in entry, f"{key} missing '{field}'"


def test_every_severity_is_valid():
    for key, entry in FAILURE_MODES.items():
        assert entry["severity"] in SEVERITY_ORDER, f"{key} has invalid severity {entry['severity']}"


def test_critical_flag_matches_critical_severity():
    """is_critical and severity=='critical' must not drift apart — both drive the dashboard."""
    for key, entry in FAILURE_MODES.items():
        assert entry["is_critical"] == (entry["severity"] == "critical"), \
            f"{key}: is_critical={entry['is_critical']} but severity={entry['severity']}"


def test_none_mode_is_not_a_failure():
    assert FAILURE_MODES["none"]["severity"] == "none"
    assert not is_critical("none")


def test_detected_by_values_are_known_tiers():
    for key, entry in FAILURE_MODES.items():
        for tier in entry["detected_by"]:
            assert tier in ("deterministic", "human", "llm_judge"), f"{key}: unknown tier {tier}"


def test_all_modes_excludes_none_by_default():
    assert "none" not in all_modes()
    assert "none" in all_modes(include_none=True)


def test_all_modes_sorted_by_severity():
    severities = [severity_of(m) for m in all_modes()]
    indices = [SEVERITY_ORDER.index(s) for s in severities]
    assert indices == sorted(indices), "modes must be ordered most-severe-first"


# ── normalisation ─────────────────────────────────────────────────────────────

def test_canonical_keys_pass_through():
    for key in FAILURE_MODES:
        assert normalize_failure_mode(key) == key


def test_aliases_map_to_canonical():
    assert normalize_failure_mode("hallucination") == "hallucinated_number"
    assert normalize_failure_mode("unsupported_inference") == "unsupported_claim"
    assert normalize_failure_mode("over_refusal") == "unnecessary_refusal"
    assert normalize_failure_mode("routing_error") == "wrong_domain"


def test_normalisation_is_case_and_separator_insensitive():
    for variant in ["Hallucinated Number", "HALLUCINATED-NUMBER", "hallucinated  number",
                    "  hallucinated_number  ", "Hallucinated__Number"]:
        assert normalize_failure_mode(variant) == "hallucinated_number", f"failed on {variant!r}"


def test_empty_and_none_map_to_no_failure():
    for value in [None, "", "none", "N/A", "no failure", "pass"]:
        assert normalize_failure_mode(value) == "none", f"failed on {value!r}"


def test_unknown_label_becomes_unclassified_not_coerced():
    """
    The important one. A judge emitting an off-taxonomy label must surface as
    'unclassified', never be silently folded into a real category.
    """
    assert normalize_failure_mode("tone_problem") == "unclassified"
    assert normalize_failure_mode("model_was_rude") == "unclassified"
    assert normalize_failure_mode("hallucinated_vibes") == "unclassified"


def test_unclassified_entry_is_usable():
    entry = get_failure_mode("something_invented")
    assert entry["key"] == "unclassified"
    assert entry["severity"] in SEVERITY_ORDER
    assert entry["is_critical"] is False
    assert label_of("something_invented") == "Unclassified"


def test_lookup_never_raises():
    for value in [None, "", 0, "  ", "???", "none", "unsafe"]:
        entry = get_failure_mode(value)
        assert "severity" in entry and "label" in entry


# ── aggregation ───────────────────────────────────────────────────────────────

def test_distribution_excludes_non_failures():
    modes = ["none", "none", "hallucinated_number", "unsupported_claim", "hallucinated_number"]
    d = distribution(modes)
    assert "none" not in d
    assert d["hallucinated_number"] == 2
    assert d["unsupported_claim"] == 1


def test_distribution_normalises_before_counting():
    """Aliases and canonical keys must land in the same bucket, not two."""
    d = distribution(["hallucination", "hallucinated_number", "Hallucinated Number"])
    assert d == {"hallucinated_number": 3}


def test_distribution_orders_critical_first():
    d = distribution(["incomplete_answer", "incomplete_answer", "incomplete_answer",
                      "hallucinated_number"])
    keys = list(d.keys())
    assert keys[0] == "hallucinated_number", \
        "critical severity must outrank raw frequency in the ordering"


def test_distribution_of_empty_list():
    assert distribution([]) == {}
    assert distribution(["none", "none"]) == {}


def test_severity_distribution_counts_all_including_none():
    s = severity_distribution(["none", "none", "hallucinated_number", "incomplete_answer"])
    assert s["none"] == 2
    assert s["critical"] == 1
    assert s["medium"] == 1


def test_severity_of_unknown_is_not_none_severity():
    """An unclassified failure is still a failure — it must not read as 'no failure'."""
    assert severity_of("totally_made_up") != "none"
