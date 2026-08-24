"""
tests/test_alignment.py — Human vs LLM-judge agreement.

The load-bearing property tested here is that provenance never leaks: a statistic
labelled "human raters only" must contain no demo-profile annotation, and vice
versa. Everything else on the alignment dashboard is downstream of that.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from alignment import (
    DISAGREEMENT_CAUSES,
    MATERIAL_GAP,
    agreement_by_failure_category,
    alignment_by_provenance,
    alignment_metrics,
    build_pairs,
    classify_disagreement,
    coverage,
    dimension_alignment,
    disagreement_cause_distribution,
    find_alignment_disagreements,
    judge_bias,
)
from human_evals import DIMENSIONS, make_annotation


def _scores(**kw):
    base = {d: 4 for d in DIMENSIONS}
    base.update(kw)
    return base


def _ann(eval_id, rater="aj", rater_type="human", **kw):
    return make_annotation(eval_id, rater, _scores(**kw), rater_type=rater_type)


def _judge(eval_id, parse_ok=True, failure_mode="none", **kw):
    scores = _scores(**kw)
    return {
        "eval_id": eval_id,
        "scores": scores,
        "overall_score": sum(scores.values()) / len(scores),
        "pass": all(scores[g] >= 3 for g in ("groundedness", "correctness", "relevance"))
                and min(scores.values()) > 1,
        "critical_failure": failure_mode == "hallucinated_number",
        "failure_mode": failure_mode,
        "confidence": 0.8,
        "reasoning_summary": "test",
        "parse_ok": parse_ok,
    }


# ── pairing ───────────────────────────────────────────────────────────────────

def test_pairs_join_on_eval_id():
    pairs = build_pairs([_ann("A"), _ann("B")], [_judge("A"), _judge("C")])
    assert [p["eval_id"] for p in pairs] == ["A"]


def test_unparsed_judge_records_are_not_paired():
    """A judge failure cannot contribute a score to an agreement statistic."""
    pairs = build_pairs([_ann("A")], [_judge("A", parse_ok=False)])
    assert pairs == []


def test_pairs_filter_by_rater_type():
    anns = [_ann("A", "aj", "human"), _ann("A", "demo_strict", "demo_profile"),
            _ann("B", "demo_strict", "demo_profile")]
    judges = [_judge("A"), _judge("B")]
    assert len(build_pairs(anns, judges, "human")) == 1
    assert len(build_pairs(anns, judges, "demo_profile")) == 2
    assert len(build_pairs(anns, judges, None)) == 2


def test_human_only_pairs_contain_no_demo_annotations():
    """The provenance guarantee. If this breaks, the app is misrepresenting its data."""
    anns = [_ann("A", "aj", "human", relevance=5),
            _ann("A", "demo_strict", "demo_profile", relevance=1)]
    pairs = build_pairs(anns, [_judge("A")], "human")
    assert pairs[0]["human"]["scores"]["relevance"] == 5
    assert pairs[0]["rater_types"] == ["human"]


def test_pairs_record_how_many_humans_contributed():
    anns = [_ann("A", "r1"), _ann("A", "r2")]
    assert build_pairs(anns, [_judge("A")])[0]["n_human_raters"] == 2


# ── coverage ──────────────────────────────────────────────────────────────────

def test_coverage_counts_each_population_separately():
    anns = [_ann("A", "aj", "human"), _ann("B", "demo_strict", "demo_profile")]
    c = coverage(anns, [_judge("A"), _judge("B"), _judge("C")], total_cases=60)
    assert c["human_annotated"] == 1
    assert c["demo_annotated"] == 1
    assert c["judge_evaluated"] == 3
    assert abs(c["human_coverage"] - 1 / 60) < 1e-9
    assert c["human_and_judge"] == 1


def test_coverage_with_zero_cases_returns_none_not_divide_by_zero():
    assert coverage([], [], 0)["human_coverage"] is None


# ── dimension alignment ───────────────────────────────────────────────────────

def test_dimension_alignment_on_empty_input():
    d = dimension_alignment([], "relevance")
    assert d["n"] == 0 and d["exact"] is None and d["within_1"] is None


def test_dimension_alignment_perfect_agreement():
    pairs = build_pairs([_ann("A"), _ann("B")], [_judge("A"), _judge("B")])
    d = dimension_alignment(pairs, "relevance")
    assert d["exact"] == 1.0 and d["within_1"] == 1.0
    assert d["mean_diff"] == 0.0


def test_dimension_alignment_reports_judge_direction():
    """Positive mean_diff means the judge scored higher than the human."""
    pairs = build_pairs([_ann("A", relevance=2)], [_judge("A", relevance=5)])
    assert dimension_alignment(pairs, "relevance")["mean_diff"] == 3.0


def test_dimension_alignment_states_its_rounding():
    pairs = build_pairs([_ann("A")], [_judge("A")])
    assert "round" in dimension_alignment(pairs, "relevance")["note"].lower()


def test_dimension_skipped_by_one_side_is_excluded():
    ann = make_annotation("A", "aj", {"relevance": 4}, rater_type="human")
    pairs = build_pairs([ann], [_judge("A")])
    assert dimension_alignment(pairs, "clarity")["n"] == 0
    assert dimension_alignment(pairs, "relevance")["n"] == 1


# ── overall metrics ───────────────────────────────────────────────────────────

def test_metrics_on_empty_pairs():
    m = alignment_metrics([], label="empty")
    assert m["n"] == 0
    assert set(m["by_dimension"]) == set(DIMENSIONS)


def test_metrics_label_is_preserved():
    assert alignment_metrics([], label="Human raters only")["label"] == "Human raters only"


def test_metrics_perfect_agreement():
    pairs = build_pairs([_ann("A"), _ann("B"), _ann("C")],
                        [_judge("A"), _judge("B"), _judge("C")])
    m = alignment_metrics(pairs)
    assert m["exact_agreement"] == 1.0
    assert m["pass_agreement"] == 1.0
    assert m["disagreement_rate"] == 0.0
    assert m["score_gap"] == 0.0


def test_metrics_detect_pass_disagreement():
    pairs = build_pairs([_ann("A", groundedness=2)], [_judge("A", groundedness=5)])
    m = alignment_metrics(pairs)
    assert m["human_pass_rate"] == 0.0
    assert m["judge_pass_rate"] == 1.0
    assert m["pass_agreement"] == 0.0
    assert m["disagreement_rate"] == 1.0


def test_confusion_matrix_counts_false_passes():
    pairs = build_pairs([_ann("A", groundedness=2)], [_judge("A", groundedness=5)])
    assert alignment_metrics(pairs)["pass_rates"]["false_pass"] == 1


def test_metrics_include_every_dimension():
    pairs = build_pairs([_ann("A")], [_judge("A")])
    assert set(alignment_metrics(pairs)["by_dimension"]) == set(DIMENSIONS)


def test_alignment_by_provenance_returns_three_labelled_views():
    anns = [_ann("A", "aj", "human"), _ann("A", "demo_strict", "demo_profile")]
    result = alignment_by_provenance(anns, [_judge("A")])
    assert set(result) == {"human_only", "demo_only", "combined"}
    assert "not human" in result["demo_only"]["label"].lower()
    assert result["human_only"]["n"] == 1
    assert result["demo_only"]["n"] == 1


# ── bias ──────────────────────────────────────────────────────────────────────

def test_bias_undefined_without_data():
    b = judge_bias({d: dimension_alignment([], d) for d in DIMENSIONS}, None, None)
    assert b["direction"] == "not defined"


def test_bias_detects_leniency():
    pairs = build_pairs([_ann(f"C{i}", relevance=2, groundedness=2) for i in range(4)],
                        [_judge(f"C{i}", relevance=5, groundedness=5) for i in range(4)])
    m = alignment_metrics(pairs)
    assert m["bias"]["direction"] == "lenient"
    assert m["bias"]["mean_gap"] > 0
    assert "relevance" in [d["dimension"] for d in m["bias"]["lenient_dimensions"]]


def test_bias_detects_over_severity():
    pairs = build_pairs([_ann(f"C{i}", relevance=5, helpfulness=5) for i in range(4)],
                        [_judge(f"C{i}", relevance=2, helpfulness=2) for i in range(4)])
    m = alignment_metrics(pairs)
    assert m["bias"]["direction"] == "severe"
    assert m["bias"]["mean_gap"] < 0


def test_bias_reports_calibrated_when_close():
    pairs = build_pairs([_ann(f"C{i}") for i in range(4)], [_judge(f"C{i}") for i in range(4)])
    assert alignment_metrics(pairs)["bias"]["direction"] == "calibrated"


def test_bias_statement_is_human_readable():
    pairs = build_pairs([_ann("A", relevance=2)], [_judge("A", relevance=5)])
    assert len(alignment_metrics(pairs)["bias"]["statement"]) > 30


# ── disagreement classification ───────────────────────────────────────────────

def test_every_cause_key_has_a_description():
    pairs = build_pairs([_ann("A", groundedness=2)], [_judge("A", groundedness=5)])
    analysis = classify_disagreement(pairs[0])
    assert analysis["primary_cause"] in DISAGREEMENT_CAUSES
    assert analysis["cause_description"] == DISAGREEMENT_CAUSES[analysis["primary_cause"]]


def test_unsupported_inference_accepted_is_detected():
    """Human penalised groundedness, judge did not — the classic judge failure."""
    pairs = build_pairs([_ann("A", groundedness=2)], [_judge("A", groundedness=5)])
    assert classify_disagreement(pairs[0])["primary_cause"] == "unsupported_inference_accepted"


def test_verbosity_bias_is_detected():
    """Judge rewards helpfulness and clarity the human did not, with grounding agreed."""
    pairs = build_pairs([_ann("A", helpfulness=2, clarity=2)],
                        [_judge("A", helpfulness=5, clarity=5)])
    assert classify_disagreement(pairs[0])["primary_cause"] == "model_verbosity_bias"


def test_missing_context_handling_is_detected():
    case = {"test_type": "missing_context", "governed_context": {"available": False}}
    pairs = build_pairs([_ann("A", groundedness=2, correctness=2)], [_judge("A")])
    causes = classify_disagreement(pairs[0], case)["all_causes"]
    assert "missing_context_handling" in causes


def test_human_inconsistency_is_flagged_before_blaming_the_judge():
    """When human raters disagree with each other, the judge is not the problem."""
    anns = [
        make_annotation("A", "r1", _scores(groundedness=2), rater_type="human",
                        failure_mode="hallucinated_number"),
        make_annotation("A", "r2", _scores(groundedness=2), rater_type="human",
                        failure_mode="incomplete_answer"),
    ]
    pairs = build_pairs(anns, [_judge("A", groundedness=5)])
    analysis = classify_disagreement(pairs[0])
    assert analysis["primary_cause"] == "human_annotation_inconsistency"
    assert "not a judge defect" in analysis["recommended_judge_improvement"].lower()


def test_every_classification_carries_an_improvement():
    pairs = build_pairs([_ann("A", relevance=2)], [_judge("A", relevance=5)])
    assert len(classify_disagreement(pairs[0])["recommended_judge_improvement"]) > 40


def test_agreement_pair_has_no_material_gaps():
    pairs = build_pairs([_ann("A")], [_judge("A")])
    assert classify_disagreement(pairs[0])["dimension_gaps"] == []


# ── disagreement listing ──────────────────────────────────────────────────────

def test_disagreements_exclude_agreeing_pairs():
    pairs = build_pairs([_ann("A"), _ann("B", relevance=2)],
                        [_judge("A"), _judge("B", relevance=5)])
    d = find_alignment_disagreements(pairs)
    assert [x["eval_id"] for x in d] == ["B"]


def test_disagreements_below_threshold_excluded():
    pairs = build_pairs([_ann("A", relevance=4)], [_judge("A", relevance=5)])
    assert find_alignment_disagreements(pairs, threshold=MATERIAL_GAP) == []


def test_pass_conflicts_sort_first():
    """
    GAP  has a 3-point clarity gap but both sides pass — clarity is not a pass gate.
    FLIP has only a 1-point gap, but it straddles the groundedness>=3 gate, so the
         human fails the response and the judge passes it.

    A pass/fail inversion is the more serious finding and must sort first even
    though its numeric gap is smaller.
    """
    pairs = build_pairs(
        [_ann("GAP", clarity=2), _ann("FLIP", groundedness=2)],
        [_judge("GAP", clarity=5), _judge("FLIP", groundedness=3)],
    )
    d = find_alignment_disagreements(pairs)
    assert {x["eval_id"] for x in d} == {"GAP", "FLIP"}
    assert d[0]["eval_id"] == "FLIP", "a pass/fail inversion outranks a larger dimension gap"
    assert d[0]["analysis"]["max_gap"] < d[1]["analysis"]["max_gap"]


def test_cause_distribution_counts_primary_causes():
    pairs = build_pairs([_ann("A", groundedness=2), _ann("B", groundedness=2)],
                        [_judge("A", groundedness=5), _judge("B", groundedness=5)])
    dist = disagreement_cause_distribution(find_alignment_disagreements(pairs))
    assert dist["unsupported_inference_accepted"] == 2


def test_cause_distribution_of_nothing():
    assert disagreement_cause_distribution([]) == {}


# ── by failure category ───────────────────────────────────────────────────────

def test_agreement_grouped_by_human_failure_mode():
    anns = [
        make_annotation("A", "aj", _scores(groundedness=1), rater_type="human",
                        failure_mode="hallucinated_number"),
        make_annotation("B", "aj", _scores(), rater_type="human", failure_mode="none"),
    ]
    result = agreement_by_failure_category(build_pairs(anns, [_judge("A"), _judge("B")]))
    assert result["hallucinated_number"]["n"] == 1
    assert result["hallucinated_number"]["severity"] == "critical"
    assert result["none"]["n"] == 1


def test_failure_category_reports_the_gap():
    anns = [make_annotation("A", "aj", _scores(groundedness=1), rater_type="human",
                            failure_mode="hallucinated_number")]
    result = agreement_by_failure_category(build_pairs(anns, [_judge("A", groundedness=5)]))
    assert result["hallucinated_number"]["mean_gap"] > 0, "judge scored this failure higher than the human"


def test_failure_category_on_empty_pairs():
    assert agreement_by_failure_category([]) == {}
