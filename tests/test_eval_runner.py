"""
tests/test_eval_runner.py — Run configuration, summarisation and regression comparison.

No network calls. The comparison logic is what decides whether a prompt or router
change shipped, so its handling of direction, missing metrics and unchanged values
is tested explicitly.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dataclasses import asdict

from eval_runner import (
    COMPARISON_METRICS,
    RunConfig,
    _config_diff,
    _dig,
    _slim_alignment,
    compare_runs,
    new_run_config,
    summarise_run,
)
from human_evals import DIMENSIONS, make_annotation
from llm import DEFAULT_MODEL, DEFAULT_PROMPT_VERSION, PROMPT_VERSIONS
from router import DEFAULT_ROUTER_VERSION, ROUTER_VERSIONS


def _run(run_id="run-a", **summary_overrides):
    summary = {
        "generation": {"n_cases": 10, "n_responses": 10, "n_errors": 0,
                       "mean_latency_seconds": 1.0, "mean_answer_chars": 500},
        "routing": {"accuracy": 0.80, "correct": 8, "total": 10,
                    "n_excluded_ambiguous": 0, "no_match_count": 0,
                    "tie_count": 0, "silent_misroutes": 2},
        "deterministic": {"n": 10, "verdict_pass_rate": 0.70,
                          "required_facts_pass_rate": 0.60,
                          "numeric_grounding_pass_rate": 0.80,
                          "unsupported_claim_rate": 0.10,
                          "forbidden_claims_violation_rate": 0.05},
        "judge": {"n": 10, "n_parsed": 10, "parse_success_rate": 1.0,
                  "mean_overall_score": 3.5, "pass_rate": 0.60,
                  "critical_failure_rate": 0.20,
                  "by_dimension": {d: 3.5 for d in DIMENSIONS},
                  "arithmetic_error_rate": 0.0},
        "alignment": {"human_only": {"n": 5, "human_mean_score": 3.4,
                                     "judge_mean_score": 3.6, "within_1_agreement": 0.80,
                                     "pass_agreement": 0.80, "human_pass_rate": 0.6,
                                     "judge_pass_rate": 0.6},
                      "demo_only": {"n": 10}, "combined": {"n": 10}},
        "coverage": {"total_cases": 10, "human_annotated": 5, "demo_annotated": 10,
                     "judge_evaluated": 10},
        "failure_distribution": {"judge": {}, "human": {}, "demo_profiles": {}},
    }
    for path, value in summary_overrides.items():
        keys = path.split(".")
        target = summary
        for k in keys[:-1]:
            target = target[k]
        target[keys[-1]] = value
    cfg = asdict(RunConfig(run_id=run_id, timestamp="2026-08-24T00:00:00+00:00"))
    return {"config": cfg, "summary": summary, "deterministic_records": []}


# ── configuration ─────────────────────────────────────────────────────────────

def test_new_run_config_has_unique_ids():
    assert new_run_config().run_id != new_run_config().run_id


def test_new_run_config_defaults_match_module_defaults():
    cfg = new_run_config()
    assert cfg.model_version == DEFAULT_MODEL
    assert cfg.system_prompt_version == DEFAULT_PROMPT_VERSION
    assert cfg.router_version == DEFAULT_ROUTER_VERSION


def test_config_records_dataset_version():
    assert new_run_config().dataset_version is not None


def test_config_overrides_applied():
    cfg = new_run_config(label="baseline", router_version="v1_substring",
                         system_prompt_version="sysprompt-v1")
    assert cfg.label == "baseline"
    assert cfg.router_version == "v1_substring"


def test_none_overrides_do_not_clobber_defaults():
    """argparse passes None for unset flags; those must not wipe the defaults."""
    cfg = new_run_config(model_version=None, router_version=None)
    assert cfg.model_version == DEFAULT_MODEL
    assert cfg.router_version == DEFAULT_ROUTER_VERSION


def test_config_versions_are_real():
    cfg = new_run_config()
    assert cfg.router_version in ROUTER_VERSIONS
    assert cfg.system_prompt_version in PROMPT_VERSIONS


def test_artifact_kind_defaults_to_real():
    """Runs are real model executions; nothing is labelled simulated unless it is."""
    assert new_run_config().artifact_kind == "real_model_run"


# ── metric extraction ─────────────────────────────────────────────────────────

def test_dig_reads_nested_paths():
    assert _dig({"a": {"b": {"c": 5}}}, ["a", "b", "c"]) == 5


def test_dig_returns_none_for_missing_path():
    assert _dig({"a": {}}, ["a", "b"]) is None
    assert _dig({}, ["a", "b", "c"]) is None


def test_dig_returns_none_for_non_numeric():
    assert _dig({"a": "text"}, ["a"]) is None
    assert _dig({"a": None}, ["a"]) is None


def test_every_comparison_metric_declares_a_direction():
    for label, path, fmt, better in COMPARISON_METRICS:
        assert better in ("up", "down"), f"{label} has no direction"
        assert fmt in ("percent", "score", "count"), f"{label} has format {fmt}"


def test_comparison_metric_labels_are_unique():
    labels = [m[0] for m in COMPARISON_METRICS]
    assert len(labels) == len(set(labels))


# ── comparison ────────────────────────────────────────────────────────────────

def test_improvement_detected_for_up_metrics():
    cmp = compare_runs(_run("a"), _run("b", **{"routing.accuracy": 0.95}))
    row = next(r for r in cmp["rows"] if r["metric"] == "Routing accuracy")
    assert row["verdict"] == "improved"
    assert abs(row["delta"] - 0.15) < 1e-9


def test_regression_detected_for_up_metrics():
    cmp = compare_runs(_run("a"), _run("b", **{"routing.accuracy": 0.50}))
    row = next(r for r in cmp["rows"] if r["metric"] == "Routing accuracy")
    assert row["verdict"] == "regressed"


def test_lower_is_better_metrics_invert_correctly():
    """Critical failure rate falling is an improvement, not a regression."""
    cmp = compare_runs(_run("a"), _run("b", **{"judge.critical_failure_rate": 0.05}))
    row = next(r for r in cmp["rows"] if r["metric"] == "Critical failure rate")
    assert row["verdict"] == "improved"
    assert row["delta"] < 0


def test_lower_is_better_metric_rising_is_a_regression():
    cmp = compare_runs(_run("a"), _run("b", **{"deterministic.unsupported_claim_rate": 0.40}))
    row = next(r for r in cmp["rows"] if r["metric"] == "Unsupported claim rate")
    assert row["verdict"] == "regressed"


def test_silent_misroutes_falling_is_an_improvement():
    cmp = compare_runs(_run("a"), _run("b", **{"routing.silent_misroutes": 0}))
    row = next(r for r in cmp["rows"] if r["metric"] == "Silent misroutes")
    assert row["verdict"] == "improved"


def test_identical_runs_are_all_unchanged():
    cmp = compare_runs(_run("a"), _run("b"))
    assert cmp["n_improved"] == 0
    assert cmp["n_regressed"] == 0
    assert cmp["n_unchanged"] > 0


def test_missing_metric_is_not_comparable_not_zero():
    """A metric absent from one run must never be treated as a value of zero."""
    baseline = _run("a")
    del baseline["summary"]["judge"]["pass_rate"]
    cmp = compare_runs(baseline, _run("b"))
    row = next(r for r in cmp["rows"] if r["metric"] == "AI judge pass rate")
    assert row["verdict"] == "not comparable"
    assert row["delta"] is None
    assert cmp["n_not_comparable"] >= 1


def test_none_valued_metric_is_not_comparable():
    baseline = _run("a")
    baseline["summary"]["alignment"]["human_only"]["within_1_agreement"] = None
    cmp = compare_runs(baseline, _run("b"))
    row = next(r for r in cmp["rows"] if "±1" in r["metric"])
    assert row["verdict"] == "not comparable"


def test_counts_sum_to_total_rows():
    cmp = compare_runs(_run("a"), _run("b", **{"routing.accuracy": 0.9}))
    total = cmp["n_improved"] + cmp["n_regressed"] + cmp["n_unchanged"] + cmp["n_not_comparable"]
    assert total == len(cmp["rows"])


def test_comparison_carries_both_configs():
    cmp = compare_runs(_run("a"), _run("b"))
    assert cmp["baseline_config"]["run_id"] == "a"
    assert cmp["current_config"]["run_id"] == "b"


# ── config diff ───────────────────────────────────────────────────────────────

def test_config_diff_reports_the_independent_variable():
    a = asdict(RunConfig("a", "t", system_prompt_version="sysprompt-v1"))
    b = asdict(RunConfig("b", "t", system_prompt_version="sysprompt-v2"))
    diff = _config_diff(a, b)
    assert len(diff) == 1
    assert diff[0]["field"] == "system_prompt_version"
    assert diff[0]["baseline"] == "sysprompt-v1"


def test_config_diff_ignores_run_id_and_timestamp():
    """Those always differ and are not the experiment's variable."""
    a = asdict(RunConfig("a", "t1"))
    b = asdict(RunConfig("b", "t2"))
    assert _config_diff(a, b) == []


def test_config_diff_catches_multiple_changes():
    a = asdict(RunConfig("a", "t", router_version="v1_substring", system_prompt_version="sysprompt-v1"))
    b = asdict(RunConfig("b", "t", router_version="v3_idf_weighted", system_prompt_version="sysprompt-v2"))
    assert {d["field"] for d in _config_diff(a, b)} == {"router_version", "system_prompt_version"}


# ── slim alignment ────────────────────────────────────────────────────────────

def test_slim_alignment_of_empty_is_zero_n():
    assert _slim_alignment({})["n"] == 0
    assert _slim_alignment({"n": 0})["n"] == 0


def test_slim_alignment_keeps_dimension_detail():
    from alignment import alignment_metrics, build_pairs

    def scores(**kw):
        base = {d: 4 for d in DIMENSIONS}
        base.update(kw)
        return base

    ann = make_annotation("A", "aj", scores(), rater_type="human")
    judge = {"eval_id": "A", "scores": scores(), "overall_score": 4.0, "pass": True,
             "critical_failure": False, "failure_mode": "none", "confidence": 0.8,
             "parse_ok": True}
    slim = _slim_alignment(alignment_metrics(build_pairs([ann], [judge])))
    assert slim["n"] == 1
    assert set(slim["by_dimension"]) == set(DIMENSIONS)
    assert "bias_direction" in slim


# ── summarisation ─────────────────────────────────────────────────────────────

def test_summarise_run_with_no_judge_or_human_data():
    """An empty evaluation must summarise honestly, not crash and not fabricate."""
    from skills import load_domains

    root = os.path.join(os.path.dirname(__file__), "..")
    domains = load_domains(root)
    cases = [{
        "eval_id": "T-1", "question": "What is our CSAT score?", "domain": "csup",
        "expected_domain": "csup", "test_type": "standard", "difficulty": "easy",
        "governed_context": {"skill": "csup", "metrics": [], "available": True},
        "expected_answer_summary": "", "required_facts": [], "forbidden_claims": [],
        "expected_behavior": "", "expected_failure_mode": "none", "severity": "none",
    }]
    cfg = RunConfig("t", "2026-08-24T00:00:00+00:00")
    summary = summarise_run(cfg, cases, [], [], [], [], domains)

    assert summary["generation"]["n_responses"] == 0
    assert summary["judge"]["n"] == 0
    assert summary["alignment"]["human_only"]["n"] == 0
    assert summary["coverage"]["human_annotated"] == 0
    assert summary["routing"]["accuracy"] == 1.0, "routing is computable without any model call"
