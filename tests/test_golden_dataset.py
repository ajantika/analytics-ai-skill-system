"""
tests/test_golden_dataset.py — Golden evaluation set integrity.

The dataset is the measuring instrument. If it drifts — a bad domain key, a
required fact that is not actually in the governed context, a forbidden phrase a
correct answer would legitimately contain — every downstream metric is wrong in a
way no dashboard would reveal. These tests guard the instrument itself.
"""
import os
import pathlib
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evals import (
    _normalize,
    case_by_id,
    check_forbidden_claims,
    check_required_facts,
    golden_cases,
    load_golden_set,
    validate_golden_set,
)
from failure_taxonomy import FAILURE_MODES, SEVERITY_ORDER, normalize_failure_mode
from skills import build_context, load_domains

ROOT = pathlib.Path(__file__).parent.parent
DATA = load_golden_set()
CASES = DATA.get("cases", [])
DOMAINS = load_domains(str(ROOT))

VALID_TEST_TYPES = {
    "standard", "ambiguous", "adversarial", "missing_context",
    "cross_domain", "grounding", "instruction_following", "unsupported_inference",
}
VALID_DIFFICULTY = {"easy", "medium", "hard"}


# ── structure ─────────────────────────────────────────────────────────────────

def test_dataset_loads():
    assert CASES, "golden set failed to load or is empty"
    assert DATA.get("dataset_version"), "dataset must carry a version"


def test_declared_case_count_matches_actual():
    assert DATA["n_cases"] == len(CASES), \
        f"header says {DATA['n_cases']} cases, file contains {len(CASES)}"


def test_structural_validation_passes():
    problems = validate_golden_set(DATA)
    assert not problems, "structural problems:\n  " + "\n  ".join(problems)


def test_eval_ids_unique():
    ids = [c["eval_id"] for c in CASES]
    dupes = {i for i in ids if ids.count(i) > 1}
    assert not dupes, f"duplicate eval_ids: {dupes}"


def test_case_lookup_by_id():
    assert case_by_id(CASES[0]["eval_id"], CASES) is CASES[0]
    assert case_by_id("NOPE-999", CASES) is None


def test_missing_file_returns_empty_not_exception():
    """A missing dataset must produce an honest empty state, not a crash."""
    data = load_golden_set("/nonexistent/path/golden.json")
    assert data["cases"] == []


# ── vocabulary ────────────────────────────────────────────────────────────────

def test_test_types_are_known():
    for c in CASES:
        assert c["test_type"] in VALID_TEST_TYPES, \
            f"{c['eval_id']}: unknown test_type {c['test_type']!r}"


def test_difficulty_values_are_known():
    for c in CASES:
        assert c["difficulty"] in VALID_DIFFICULTY, \
            f"{c['eval_id']}: unknown difficulty {c['difficulty']!r}"


def test_expected_failure_modes_are_in_taxonomy():
    for c in CASES:
        mode = c["expected_failure_mode"]
        assert normalize_failure_mode(mode) != "unclassified", \
            f"{c['eval_id']}: expected_failure_mode {mode!r} is not in the taxonomy"


def test_severity_values_are_valid():
    for c in CASES:
        assert c["severity"] in SEVERITY_ORDER, \
            f"{c['eval_id']}: invalid severity {c['severity']!r}"


def test_severity_matches_taxonomy_for_the_expected_mode():
    """Case severity must agree with the taxonomy, or critical-failure rate is incoherent."""
    problems = []
    for c in CASES:
        mode = normalize_failure_mode(c["expected_failure_mode"])
        expected = FAILURE_MODES[mode]["severity"]
        if c["severity"] != expected:
            problems.append(
                f"{c['eval_id']}: severity {c['severity']!r} but taxonomy says {mode} is {expected!r}"
            )
    assert not problems, "severity mismatches:\n  " + "\n  ".join(problems)


def test_no_failure_cases_have_no_severity():
    for c in CASES:
        if c["expected_failure_mode"] == "none":
            assert c["severity"] == "none", f"{c['eval_id']}: 'none' mode must have 'none' severity"


# ── domains ───────────────────────────────────────────────────────────────────

def test_expected_domains_exist_in_skill_files():
    for c in CASES:
        expected = c["expected_domain"]
        if expected is None:
            continue
        assert expected in DOMAINS, \
            f"{c['eval_id']}: expected_domain {expected!r} has no loaded skill file"


def test_governed_context_skill_exists():
    for c in CASES:
        skill = c["governed_context"].get("skill")
        if skill is None:
            continue
        assert skill in DOMAINS, f"{c['eval_id']}: governed_context.skill {skill!r} not loaded"


def test_governed_context_metrics_exist_in_their_skill():
    """A named governed metric must actually be defined in that skill file."""
    for c in CASES:
        skill = c["governed_context"].get("skill")
        if not skill or skill not in DOMAINS:
            continue
        defined = {m.get("name") for m in DOMAINS[skill].get("metrics", [])}
        for metric in c["governed_context"].get("metrics", []):
            assert metric in defined, \
                f"{c['eval_id']}: metric {metric!r} not defined in {skill}.yaml"


# ── grounding of the dataset itself ───────────────────────────────────────────

def test_required_facts_appear_in_the_governed_context():
    """
    Every required fact must be findable in the context the case supplies. If a case
    demands a figure the model is never shown, the case measures nothing but is
    counted as a failure — the worst kind of silent dataset bug.

    Cases whose governed_context.available is false are exempt: those are the
    missing-context cases, where absence is the point.
    """
    problems = []
    for c in CASES:
        skill = c["governed_context"].get("skill")
        if not skill or skill not in DOMAINS:
            continue
        context = _normalize(build_context(DOMAINS[skill]))
        for fact in c["required_facts"]:
            alts = fact if isinstance(fact, list) else [fact]
            if not any(_normalize(a) in context for a in alts):
                problems.append(f"{c['eval_id']}: required fact {alts[0]!r} not in {skill} context")
    assert not problems, "required facts absent from their own context:\n  " + "\n  ".join(problems)


def test_forbidden_claims_are_not_trivially_present_in_context():
    """
    A forbidden phrase that appears verbatim in the governed context would fire on
    almost any grounded answer, making the check useless. Flag those.
    """
    problems = []
    for c in CASES:
        skill = c["governed_context"].get("skill")
        if not skill or skill not in DOMAINS:
            continue
        context = _normalize(build_context(DOMAINS[skill]))
        for phrase in c["forbidden_claims"]:
            if len(_normalize(phrase)) >= 6 and _normalize(phrase) in context:
                problems.append(f"{c['eval_id']}: forbidden phrase {phrase!r} appears in {skill} context")
    assert not problems, "forbidden phrases that would false-positive:\n  " + "\n  ".join(problems)


# ── design intent ─────────────────────────────────────────────────────────────

def test_adversarial_forbidden_claims_are_not_bare_figures():
    """
    An adversarial case asserts a false figure and expects the answer to refute it.
    Refuting it means naming it — "the rate is 24.4%, not 45%" — so a forbidden claim
    written as the bare figure ("45%") fires on exactly the correct answer.

    PRD-006 shipped with that defect: the model correctly rejected the false premise
    and the deterministic check failed it anyway. Forbidden claims on adversarial
    cases must therefore carry surrounding words that only appear when the figure is
    being asserted rather than denied.
    """
    problems = []
    for c in CASES:
        if c["test_type"] != "adversarial":
            continue
        for phrase in c["forbidden_claims"]:
            stripped = str(phrase).strip()
            if re.fullmatch(r"[$+-]?[\d,.]+\s*[%xKMB]?", stripped, re.I):
                problems.append(
                    f"{c['eval_id']}: forbidden claim {phrase!r} is a bare figure; a correct "
                    f"refutation would name it and trip this check"
                )
    assert not problems, "adversarial forbidden claims that fire on correct answers:\n  " + "\n  ".join(problems)


def test_all_five_domains_are_covered():
    covered = {c["expected_domain"] for c in CASES if c["expected_domain"]}
    for domain in ["product_usage", "sales", "marketing", "hr", "csup"]:
        assert domain in covered, f"no cases target {domain}"


def test_every_test_type_is_represented():
    present = {c["test_type"] for c in CASES}
    missing = VALID_TEST_TYPES - present
    assert not missing, f"test types with no cases: {missing}"


def test_dataset_is_adversarial_enough_to_measure_something():
    """
    A set the system passes trivially has no discriminating power. At least a third
    of cases must be designed to elicit a specific failure.
    """
    trapped = [c for c in CASES if c["expected_failure_mode"] != "none"]
    assert len(trapped) / len(CASES) >= 0.33, \
        f"only {len(trapped)}/{len(CASES)} cases have a designed failure mode"


def test_critical_cases_exist():
    critical = [c for c in CASES if c["severity"] == "critical"]
    assert len(critical) >= 5, f"only {len(critical)} critical-severity cases"


def test_baseline_competence_cases_exist():
    """Not every case should be a trap, or a pass rate cannot distinguish anything."""
    clean = [c for c in CASES if c["expected_failure_mode"] == "none"]
    assert len(clean) >= 8, f"only {len(clean)} untrapped cases"


def test_cases_with_null_expected_domain_test_ambiguity():
    for c in CASES:
        if c["expected_domain"] is None:
            assert c["test_type"] in ("ambiguous", "cross_domain"), \
                f"{c['eval_id']}: null expected_domain but test_type is {c['test_type']!r}"


# ── the checks that consume the dataset ───────────────────────────────────────

def test_required_fact_check_accepts_alternative_surface_forms():
    result = check_required_facts(
        "Recovery is 1.4 million dollars across 694 accounts.",
        [["$1.4M", "1.4 million"], "694"],
    )
    assert result["status"] == "PASS", result["detail"]


def test_required_fact_check_normalises_thousands_separators():
    assert check_required_facts("We closed 35109 tickets.", ["35,109"])["status"] == "PASS"
    assert check_required_facts("We closed 35,109 tickets.", ["35109"])["status"] == "PASS"


def test_required_fact_check_reports_missing():
    result = check_required_facts("Only 694 customers.", ["694", "24.4%"])
    assert result["status"] == "WARN"
    assert result["missing"] == ["24.4%"]
    assert result["coverage"] == 0.5


def test_required_fact_check_fails_below_half():
    result = check_required_facts("Nothing useful.", ["694", "24.4%", "2,847"])
    assert result["status"] == "FAIL"
    assert result["coverage"] == 0.0


def test_empty_required_facts_is_not_a_vacuous_pass():
    """An empty requirement list must not inflate the pass rate."""
    result = check_required_facts("anything at all", [])
    assert result["status"] == "N/A"
    assert result["coverage"] is None


def test_forbidden_claim_detection():
    result = check_forbidden_claims(
        "Industry standard practice suggests quarterly reviews.",
        ["industry standard", "research shows"],
    )
    assert result["status"] == "FAIL"
    assert result["violations"] == ["industry standard"]


def test_forbidden_claim_clean_answer_passes():
    result = check_forbidden_claims("694 customers exceed plan.", ["industry standard"])
    assert result["status"] == "PASS"
    assert result["violations"] == []


def test_empty_forbidden_claims_is_not_applicable():
    assert check_forbidden_claims("anything", [])["status"] == "N/A"


def test_checks_declare_their_method():
    for result in [check_required_facts("x", ["x"]), check_forbidden_claims("x", ["y"])]:
        assert "DETERMINISTIC" in result["method"]
