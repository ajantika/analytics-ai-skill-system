"""
tests/test_human_evals.py — Rubric, annotation records, and rater calibration.

Two things are load-bearing here and both are tested hard:
  1. The pass rule is a written rule applied identically by every evaluator. If it
     drifts, human-vs-judge pass agreement measures the drift, not the models.
  2. Provenance (human vs demo_profile) must never be flattened, or the app would
     be presenting scripted annotations as human judgement.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from human_evals import (
    DIMENSIONS,
    PASS_RULE,
    RUBRIC,
    SCALE,
    applies_pass_rule,
    by_rater_type,
    calibration_summary,
    consensus_by_eval,
    find_disagreements,
    load_annotations,
    make_annotation,
    overall_score,
    rater_agreement,
    rubric_markdown,
    save_annotations,
    upsert_annotation,
    validate_annotation,
)


def _scores(**kwargs) -> dict:
    """Full score dict defaulting to 4s, overridden per dimension."""
    base = {d: 4 for d in DIMENSIONS}
    base.update(kwargs)
    return base


def _ann(eval_id, rater, rater_type="human", **score_overrides) -> dict:
    return make_annotation(eval_id, rater, _scores(**score_overrides), rater_type=rater_type)


# ── rubric integrity ──────────────────────────────────────────────────────────

def test_rubric_covers_every_dimension():
    assert set(RUBRIC) == set(DIMENSIONS)


def test_every_dimension_has_all_five_anchors():
    for dim, spec in RUBRIC.items():
        for score in (1, 2, 3, 4, 5):
            assert score in spec["anchors"], f"{dim} missing anchor {score}"
            assert spec["anchors"][score].strip(), f"{dim} anchor {score} is empty"


def test_every_dimension_states_what_not_to_score():
    """The 'ignore' field is what makes two evaluators separable — it must exist."""
    for dim, spec in RUBRIC.items():
        assert spec.get("ignore", "").strip(), f"{dim} does not say what to exclude"
        assert spec.get("scored_on", "").strip(), f"{dim} does not say what to score"


def test_scale_is_one_to_five():
    assert sorted(SCALE) == [1, 2, 3, 4, 5]


def test_rubric_markdown_renders_everything():
    md = rubric_markdown()
    for dim in DIMENSIONS:
        assert RUBRIC[dim]["label"] in md
    assert PASS_RULE in md


# ── scoring ───────────────────────────────────────────────────────────────────

def test_overall_score_is_unweighted_mean():
    assert overall_score(_scores()) == 4.0
    assert overall_score(_scores(relevance=5, clarity=5)) == (5 + 4 + 4 + 4 + 4 + 5) / 6


def test_overall_score_ignores_unscored_dimensions():
    """A skipped dimension must be excluded, not treated as zero."""
    partial = {"relevance": 5, "groundedness": 3}
    assert overall_score(partial) == 4.0


def test_overall_score_of_nothing_is_none():
    assert overall_score({}) is None
    assert overall_score({d: None for d in DIMENSIONS}) is None


# ── pass rule ─────────────────────────────────────────────────────────────────

def test_pass_rule_passes_a_solid_answer():
    assert applies_pass_rule(_scores()) is True


def test_pass_rule_gates_on_groundedness():
    assert applies_pass_rule(_scores(groundedness=2)) is False
    assert applies_pass_rule(_scores(groundedness=3)) is True


def test_pass_rule_gates_on_correctness_and_relevance():
    assert applies_pass_rule(_scores(correctness=2)) is False
    assert applies_pass_rule(_scores(relevance=2)) is False


def test_pass_rule_ignores_non_gate_dimensions():
    """Low helpfulness or clarity degrades quality but does not fail the answer."""
    assert applies_pass_rule(_scores(helpfulness=2, clarity=2)) is True


def test_any_score_of_one_is_an_automatic_fail():
    assert applies_pass_rule(_scores(clarity=1)) is False
    assert applies_pass_rule(_scores(helpfulness=1)) is False


def test_critical_failure_mode_forces_fail_regardless_of_scores():
    """Perfect scores plus a critical mode is a contradiction the rule resolves as FAIL."""
    assert applies_pass_rule(_scores(**{d: 5 for d in DIMENSIONS}), "hallucinated_number") is False
    assert applies_pass_rule(_scores(**{d: 5 for d in DIMENSIONS}), "incomplete_answer") is True


def test_pass_rule_on_empty_scores_is_fail():
    assert applies_pass_rule({}) is False


# ── annotation records ────────────────────────────────────────────────────────

def test_make_annotation_derives_pass_and_critical():
    a = make_annotation("PRD-001", "rater_x", _scores(groundedness=1), failure_mode="hallucinated_number")
    assert a["pass"] is False
    assert a["critical_failure"] is True
    assert a["failure_mode"] == "hallucinated_number"


def test_make_annotation_normalises_the_failure_mode():
    a = make_annotation("PRD-001", "r", _scores(), failure_mode="Hallucination")
    assert a["failure_mode"] == "hallucinated_number"


def test_make_annotation_allows_explicit_override():
    """A human may overrule the derived pass/fail; the record must respect that."""
    a = make_annotation("PRD-001", "r", _scores(groundedness=2), passed=True)
    assert a["pass"] is True


def test_make_annotation_fills_every_dimension_key():
    a = make_annotation("PRD-001", "r", {"relevance": 4})
    assert set(a["scores"]) == set(DIMENSIONS)
    assert a["scores"]["clarity"] is None


def test_annotation_carries_rater_type():
    assert _ann("PRD-001", "aj")["rater_type"] == "human"
    assert _ann("PRD-001", "demo_strict", "demo_profile")["rater_type"] == "demo_profile"


# ── validation ────────────────────────────────────────────────────────────────

def test_valid_annotation_has_no_problems():
    assert validate_annotation(_ann("PRD-001", "aj")) == []


def test_validation_catches_out_of_range_scores():
    bad = _ann("PRD-001", "aj")
    bad["scores"]["relevance"] = 7
    assert any("relevance" in p for p in validate_annotation(bad))


def test_validation_catches_missing_identifiers():
    problems = validate_annotation({"scores": {"relevance": 4}, "rater_type": "human"})
    assert any("eval_id" in p for p in problems)
    assert any("evaluator_id" in p for p in problems)


def test_validation_rejects_unknown_rater_type():
    bad = _ann("PRD-001", "aj")
    bad["rater_type"] = "contractor"
    assert any("rater_type" in p for p in validate_annotation(bad))


def test_validation_rejects_empty_scores():
    empty = make_annotation("PRD-001", "aj", {})
    assert any("no dimensions scored" in p for p in validate_annotation(empty))


def test_validation_rejects_out_of_range_confidence():
    bad = _ann("PRD-001", "aj")
    bad["evaluator_confidence"] = 1.4
    assert any("confidence" in p for p in validate_annotation(bad))


# ── storage ───────────────────────────────────────────────────────────────────

def test_load_missing_file_returns_empty_list():
    assert load_annotations("/nonexistent/annotations.json") == []


def test_save_and_load_roundtrip(tmp_path=None):
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ann.json")
        records = [_ann("PRD-001", "aj"), _ann("PRD-002", "aj")]
        assert save_annotations(records, path) is True
        loaded = load_annotations(path)
        assert len(loaded) == 2
        assert {r["eval_id"] for r in loaded} == {"PRD-001", "PRD-002"}


def test_save_returns_false_on_unwritable_path():
    """Streamlit Cloud has a read-only filesystem; this must degrade, not crash."""
    assert save_annotations([_ann("PRD-001", "aj")], "/proc/nope/ann.json") is False


def test_upsert_replaces_same_rater_same_case():
    existing = [_ann("PRD-001", "aj", relevance=2)]
    updated = upsert_annotation(_ann("PRD-001", "aj", relevance=5), existing)
    assert len(updated) == 1
    assert updated[0]["scores"]["relevance"] == 5


def test_upsert_keeps_other_raters_on_the_same_case():
    existing = [_ann("PRD-001", "aj"), _ann("PRD-001", "demo_strict", "demo_profile")]
    updated = upsert_annotation(_ann("PRD-001", "aj", clarity=5), existing)
    assert len(updated) == 2


# ── views ─────────────────────────────────────────────────────────────────────

def test_by_rater_type_separates_provenance():
    pool = [_ann("A", "aj"), _ann("A", "demo_strict", "demo_profile")]
    assert len(by_rater_type(pool, "human")) == 1
    assert len(by_rater_type(pool, "demo_profile")) == 1


def test_consensus_averages_across_raters():
    pool = [_ann("A", "r1", relevance=2), _ann("A", "r2", relevance=4)]
    c = consensus_by_eval(pool)
    assert c["A"]["scores"]["relevance"] == 3.0
    assert c["A"]["n_raters"] == 2


def test_consensus_can_be_restricted_to_one_provenance():
    pool = [_ann("A", "aj", relevance=5), _ann("A", "demo_strict", "demo_profile", relevance=1)]
    human_only = consensus_by_eval(pool, rater_type="human")
    assert human_only["A"]["scores"]["relevance"] == 5
    assert human_only["A"]["n_raters"] == 1


def test_consensus_pass_tie_resolves_to_fail():
    """An unresolved split on acceptability must not be recorded as acceptable."""
    pool = [
        make_annotation("A", "r1", _scores(), passed=True),
        make_annotation("A", "r2", _scores(), passed=False),
    ]
    assert consensus_by_eval(pool)["A"]["pass"] is False


def test_consensus_critical_failure_is_any_rater():
    pool = [
        make_annotation("A", "r1", _scores(), critical_failure=False),
        make_annotation("A", "r2", _scores(), critical_failure=True),
    ]
    assert consensus_by_eval(pool)["A"]["critical_failure"] is True


# ── rater agreement ───────────────────────────────────────────────────────────

def _two_rater_pool() -> list[dict]:
    pool = []
    for i, (a_rel, b_rel) in enumerate([(5, 5), (4, 4), (3, 5), (2, 4), (5, 4)]):
        pool.append(_ann(f"C{i}", "rater_a", relevance=a_rel))
        pool.append(_ann(f"C{i}", "rater_b", relevance=b_rel))
    return pool


def test_rater_agreement_uses_only_shared_cases():
    pool = _two_rater_pool() + [_ann("SOLO", "rater_a")]
    ag = rater_agreement(pool, "rater_a", "rater_b")
    assert ag["n_shared"] == 5
    assert "SOLO" not in ag["eval_ids"]


def test_rater_agreement_reports_every_dimension():
    ag = rater_agreement(_two_rater_pool(), "rater_a", "rater_b")
    assert set(ag["by_dimension"]) == set(DIMENSIONS)


def test_rater_agreement_detects_systematic_severity():
    """rater_a scored relevance lower on average; mean_diff must be positive."""
    ag = rater_agreement(_two_rater_pool(), "rater_a", "rater_b")
    rel = ag["by_dimension"]["relevance"]
    assert rel["mean_diff_b_minus_a"] > 0, "rater_b is the more lenient rater here"
    assert rel["exact"] == 0.4


def test_rater_agreement_with_no_shared_cases():
    pool = [_ann("A", "r1"), _ann("B", "r2")]
    ag = rater_agreement(pool, "r1", "r2")
    assert ag["n_shared"] == 0
    assert ag["by_dimension"] == {}


def test_rater_agreement_records_provenance_of_both_raters():
    pool = [_ann("A", "aj"), _ann("A", "demo_strict", "demo_profile")]
    ag = rater_agreement(pool, "aj", "demo_strict")
    assert ag["rater_a_type"] == "human"
    assert ag["rater_b_type"] == "demo_profile"


# ── disagreements ─────────────────────────────────────────────────────────────

def test_disagreements_found_above_threshold():
    d = find_disagreements(_two_rater_pool(), "rater_a", "rater_b", threshold=2)
    assert len(d) == 2, "two cases differ by >= 2 on relevance"
    assert d[0]["max_gap"] >= 2


def test_disagreements_ignore_small_gaps():
    pool = [_ann("A", "r1", relevance=4), _ann("A", "r2", relevance=5)]
    assert find_disagreements(pool, "r1", "r2", threshold=2) == []


def test_pass_conflict_surfaces_even_without_a_large_gap():
    """A one-point difference straddling the pass gate must still be surfaced."""
    pool = [
        _ann("A", "r1", groundedness=3),
        _ann("A", "r2", groundedness=2),
    ]
    d = find_disagreements(pool, "r1", "r2", threshold=2)
    assert len(d) == 1
    assert d[0]["pass_conflict"] is True


def test_disagreements_carry_reason_and_clarification():
    d = find_disagreements(_two_rater_pool(), "rater_a", "rater_b", threshold=2)
    for entry in d:
        assert entry["likely_reason"].strip()
        assert entry["rubric_clarification"].strip()


def test_disagreements_sorted_worst_first():
    pool = [
        _ann("SMALL", "r1", clarity=5), _ann("SMALL", "r2", clarity=3),
        _ann("BIG", "r1", clarity=5), _ann("BIG", "r2", clarity=1),
    ]
    d = find_disagreements(pool, "r1", "r2", threshold=2)
    assert d[0]["eval_id"] == "BIG"


# ── calibration summary ───────────────────────────────────────────────────────

def test_calibration_summary_tags_pair_provenance():
    pool = [
        _ann("A", "aj"), _ann("A", "demo_strict", "demo_profile"),
        _ann("A", "demo_lenient", "demo_profile"),
    ]
    summary = calibration_summary(pool)
    provenances = {p["pair_provenance"] for p in summary["pairs"]}
    assert provenances == {"human-demo", "demo-demo"}
    assert summary["n_human_raters"] == 1
    assert summary["n_demo_raters"] == 2


def test_calibration_summary_skips_pairs_with_no_overlap():
    pool = [_ann("A", "r1"), _ann("B", "r2")]
    assert calibration_summary(pool)["pairs"] == []


def test_calibration_summary_on_empty_corpus():
    summary = calibration_summary([])
    assert summary["raters"] == []
    assert summary["pairs"] == []
    assert summary["n_human_raters"] == 0
