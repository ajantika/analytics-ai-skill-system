"""
alignment.py — Human vs LLM-judge agreement.

The point of this module is that the automated judge is itself under evaluation.
An LLM judge that disagrees with human raters is not a cheaper human rater; it is a
different measurement instrument, and shipping it without knowing the gap is how
teams end up optimising a metric nobody validated.

Every figure here is computed from evaluation records. Nothing is asserted, and
nothing is filled in when the data cannot support it — statistics that are
undefined return None and must render as "not defined", never as zero.

Provenance is never flattened. Human annotations and scripted demo-profile
annotations are aligned separately, and any combined view is labelled as such.
"""

from __future__ import annotations

import logging
from typing import Optional

from failure_taxonomy import normalize_failure_mode, severity_of
from human_evals import DIMENSIONS, consensus_by_eval
from stats_utils import (
    binary_rates,
    cohens_kappa,
    confusion_matrix,
    exact_agreement,
    mean,
    mean_signed_difference,
    pearson,
    spearman,
    within_n_agreement,
)

logger = logging.getLogger(__name__)

# A dimension gap of this size or more is a material disagreement worth reviewing.
MATERIAL_GAP = 2


# ── Pairing ───────────────────────────────────────────────────────────────────


def build_pairs(
    annotations: list[dict],
    judge_records: list[dict],
    rater_type: Optional[str] = None,
) -> list[dict]:
    """
    Join human annotations to judge records on eval_id.

    rater_type filters which annotations count as the human side:
      "human"        only annotations a person authored
      "demo_profile" only the scripted rubric profiles
      None           both, pooled — only for views explicitly labelled as combined

    Only judge records that parsed successfully are paired; a judge failure cannot
    contribute a score to an agreement statistic, though it is still counted in
    judge reliability elsewhere.
    """
    human = consensus_by_eval(annotations, rater_type=rater_type)
    judge = {r["eval_id"]: r for r in judge_records if r.get("parse_ok") and r.get("eval_id")}

    pairs = []
    for eval_id in sorted(set(human) & set(judge)):
        h, j = human[eval_id], judge[eval_id]
        pairs.append({
            "eval_id": eval_id,
            "human": h,
            "judge": j,
            "human_overall": h.get("overall_score"),
            "judge_overall": j.get("overall_score"),
            "human_pass": bool(h.get("pass")),
            "judge_pass": bool(j.get("pass")),
            "human_critical": bool(h.get("critical_failure")),
            "judge_critical": bool(j.get("critical_failure")),
            "rater_types": h.get("rater_types", []),
            "n_human_raters": h.get("n_raters", 0),
        })
    return pairs


def coverage(annotations: list[dict], judge_records: list[dict], total_cases: int) -> dict:
    """
    How much of the golden set each evaluator actually covered.

    Coverage is reported prominently because an agreement figure computed over 12 of
    60 cases means something very different from one computed over all 60, and a
    dashboard that hides the denominator is misleading by omission.
    """
    human_ids = {a["eval_id"] for a in annotations if a.get("rater_type") == "human"}
    demo_ids = {a["eval_id"] for a in annotations if a.get("rater_type") == "demo_profile"}
    judge_ids = {r["eval_id"] for r in judge_records if r.get("parse_ok")}

    return {
        "total_cases": total_cases,
        "human_annotated": len(human_ids),
        "demo_annotated": len(demo_ids),
        "judge_evaluated": len(judge_ids),
        "human_coverage": len(human_ids) / total_cases if total_cases else None,
        "demo_coverage": len(demo_ids) / total_cases if total_cases else None,
        "judge_coverage": len(judge_ids) / total_cases if total_cases else None,
        "human_and_judge": len(human_ids & judge_ids),
        "demo_and_judge": len(demo_ids & judge_ids),
    }


# ── Core metrics ──────────────────────────────────────────────────────────────


def dimension_alignment(pairs: list[dict], dimension: str) -> dict:
    """
    Agreement on one rubric dimension.

    Human scores are consensus means and may be fractional; judge scores are
    integers. Exact agreement therefore rounds the human side, and that rounding is
    stated in the returned record so the figure is not read as stricter than it is.
    """
    hv, jv = [], []
    for p in pairs:
        h = p["human"]["scores"].get(dimension)
        j = p["judge"]["scores"].get(dimension)
        if h is None or j is None:
            continue
        hv.append(h)
        jv.append(j)

    if not hv:
        return {
            "dimension": dimension, "n": 0, "exact": None, "within_1": None,
            "kappa_quadratic": None, "pearson": None, "spearman": None,
            "human_mean": None, "judge_mean": None, "mean_diff": None,
            "note": "no paired observations for this dimension",
        }

    hv_rounded = [int(round(v)) for v in hv]
    return {
        "dimension": dimension,
        "n": len(hv),
        "exact": exact_agreement(hv_rounded, jv),
        "within_1": within_n_agreement(hv, jv, 1),
        "kappa_quadratic": cohens_kappa(hv_rounded, jv, categories=[1, 2, 3, 4, 5], weights="quadratic"),
        "pearson": pearson(hv, jv),
        "spearman": spearman(hv, jv),
        "human_mean": mean(hv),
        "judge_mean": mean(jv),
        "mean_diff": mean_signed_difference(hv, jv),  # positive = judge more lenient
        "note": "exact agreement rounds the human consensus mean to the nearest integer",
    }


def alignment_metrics(pairs: list[dict], label: str = "") -> dict:
    """
    Full human-vs-judge alignment over a set of pairs.

    `label` records which population these pairs came from (e.g. "human raters only")
    so the caller cannot lose track of provenance downstream.
    """
    if not pairs:
        return {"label": label, "n": 0, "by_dimension": {d: dimension_alignment([], d) for d in DIMENSIONS}}

    h_overall = [p["human_overall"] for p in pairs if p["human_overall"] is not None]
    j_overall = [p["judge_overall"] for p in pairs if p["judge_overall"] is not None]
    both = [(p["human_overall"], p["judge_overall"]) for p in pairs
            if p["human_overall"] is not None and p["judge_overall"] is not None]
    hb = [a for a, _ in both]
    jb = [b for _, b in both]

    h_pass = [p["human_pass"] for p in pairs]
    j_pass = [p["judge_pass"] for p in pairs]

    by_dim = {d: dimension_alignment(pairs, d) for d in DIMENSIONS}

    # A material disagreement is any dimension gap >= MATERIAL_GAP or a pass/fail flip.
    disagreeing = sum(
        1 for p in pairs
        if p["human_pass"] != p["judge_pass"]
        or any(
            p["human"]["scores"].get(d) is not None
            and p["judge"]["scores"].get(d) is not None
            and abs(p["human"]["scores"][d] - p["judge"]["scores"][d]) >= MATERIAL_GAP
            for d in DIMENSIONS
        )
    )

    return {
        "label": label,
        "n": len(pairs),
        "responses_evaluated": len(pairs),

        "human_mean_score": mean(h_overall),
        "judge_mean_score": mean(j_overall),
        "score_gap": (mean(j_overall) - mean(h_overall))
                     if (h_overall and j_overall) else None,

        "human_pass_rate": sum(h_pass) / len(h_pass) if h_pass else None,
        "judge_pass_rate": sum(j_pass) / len(j_pass) if j_pass else None,

        "human_critical_rate": sum(1 for p in pairs if p["human_critical"]) / len(pairs),
        "judge_critical_rate": sum(1 for p in pairs if p["judge_critical"]) / len(pairs),

        "exact_agreement": exact_agreement([int(round(v)) for v in hb], [int(round(v)) for v in jb]),
        "within_1_agreement": within_n_agreement(hb, jb, 1),
        "pass_agreement": exact_agreement(h_pass, j_pass),
        "disagreement_rate": disagreeing / len(pairs),

        "kappa_pass": cohens_kappa(h_pass, j_pass, categories=[False, True]),
        "kappa_overall_quadratic": cohens_kappa(
            [int(round(v)) for v in hb], [int(round(v)) for v in jb],
            categories=[1, 2, 3, 4, 5], weights="quadratic",
        ),
        "pearson": pearson(hb, jb),
        "spearman": spearman(hb, jb),

        "pass_confusion": confusion_matrix(h_pass, j_pass, [False, True]),
        "pass_rates": binary_rates(h_pass, j_pass),

        "by_dimension": by_dim,
        "bias": judge_bias(by_dim, mean(h_overall), mean(j_overall)),
    }


def alignment_by_provenance(
    annotations: list[dict],
    judge_records: list[dict],
) -> dict:
    """
    The same alignment computed three ways, so the reader can see whether the
    headline figure rests on human judgement or on scripted profiles.
    """
    return {
        "human_only": alignment_metrics(
            build_pairs(annotations, judge_records, "human"),
            label="Human raters only",
        ),
        "demo_only": alignment_metrics(
            build_pairs(annotations, judge_records, "demo_profile"),
            label="Demo rubric profiles only (not human)",
        ),
        "combined": alignment_metrics(
            build_pairs(annotations, judge_records, None),
            label="Combined — human and demo profiles pooled",
        ),
    }


# ── Bias ──────────────────────────────────────────────────────────────────────


def judge_bias(by_dimension: dict, human_mean: Optional[float], judge_mean: Optional[float]) -> dict:
    """
    Which direction does the judge err, and where is it least reliable?

    Direction uses the mean signed difference; magnitude thresholds are stated
    rather than hidden so a reader can disagree with them. Reliability uses
    quadratic-weighted kappa, which is the appropriate ordinal statistic here.
    """
    gap = (judge_mean - human_mean) if (human_mean is not None and judge_mean is not None) else None

    if gap is None:
        direction, statement = "not defined", "Not enough paired evaluations to assess judge bias."
    elif gap > 0.25:
        direction = "lenient"
        statement = (
            f"The judge scores {gap:+.2f} points higher than human raters on average. "
            "It is more forgiving than the human reference, so its pass rate overstates quality."
        )
    elif gap < -0.25:
        direction = "severe"
        statement = (
            f"The judge scores {gap:+.2f} points lower than human raters on average. "
            "It is harsher than the human reference and will over-report failures."
        )
    else:
        direction = "calibrated"
        statement = (
            f"Mean scores differ by {gap:+.2f} points — the judge is broadly calibrated to the "
            "human reference on aggregate. Per-dimension agreement still varies."
        )

    scored = [d for d in by_dimension.values() if d["n"] > 0]

    lenient_dims = sorted(
        [d for d in scored if d["mean_diff"] is not None and d["mean_diff"] > 0.3],
        key=lambda d: -d["mean_diff"],
    )
    severe_dims = sorted(
        [d for d in scored if d["mean_diff"] is not None and d["mean_diff"] < -0.3],
        key=lambda d: d["mean_diff"],
    )
    with_kappa = [d for d in scored if d["kappa_quadratic"] is not None]
    weakest = sorted(with_kappa, key=lambda d: d["kappa_quadratic"])[:2]
    strongest = sorted(with_kappa, key=lambda d: -d["kappa_quadratic"])[:2]

    return {
        "direction": direction,
        "mean_gap": gap,
        "statement": statement,
        "lenient_dimensions": [{"dimension": d["dimension"], "gap": d["mean_diff"]} for d in lenient_dims],
        "severe_dimensions": [{"dimension": d["dimension"], "gap": d["mean_diff"]} for d in severe_dims],
        "least_reliable": [{"dimension": d["dimension"], "kappa": d["kappa_quadratic"]} for d in weakest],
        "most_reliable": [{"dimension": d["dimension"], "kappa": d["kappa_quadratic"]} for d in strongest],
    }


# ── Disagreement analysis ─────────────────────────────────────────────────────

DISAGREEMENT_CAUSES = {
    "judge_leniency": "The judge scored materially higher than the human on a dimension.",
    "judge_over_severity": "The judge scored materially lower than the human on a dimension.",
    "unsupported_inference_accepted": "The human penalised groundedness or correctness for an unmarked inference the judge accepted.",
    "ambiguous_rubric": "Both evaluators found a problem but recorded it on different dimensions — the rubric boundary is not decisive.",
    "ambiguous_reference_answer": "The case's expected behaviour is qualitative, leaving room for two defensible readings.",
    "human_annotation_inconsistency": "The human raters disagreed among themselves, so the consensus the judge is compared against is itself unstable.",
    "model_verbosity_bias": "The judge rewarded a long, fluent answer the human found padded or unhelpful.",
    "missing_context_handling": "The evaluators disagree on whether declining to answer was correct behaviour or an over-refusal.",
    "failure_mode_mismatch": "Both agree the response failed but classify the failure differently.",
    "uncategorised": "No structural cause identified from the record alone.",
}


def classify_disagreement(pair: dict, case: Optional[dict] = None) -> dict:
    """
    Attribute a human-judge disagreement to a structural cause.

    Rule-based and deliberately conservative. It reports what is visible in the two
    records and returns "uncategorised" rather than inventing an explanation — a
    plausible-sounding wrong diagnosis is worse than an honest blank.
    """
    h_scores = pair["human"]["scores"]
    j_scores = pair["judge"]["scores"]

    gaps = []
    for d in DIMENSIONS:
        hv, jv = h_scores.get(d), j_scores.get(d)
        if hv is None or jv is None:
            continue
        diff = jv - hv
        if abs(diff) >= MATERIAL_GAP:
            gaps.append({"dimension": d, "human": hv, "judge": jv, "diff": diff})
    gaps.sort(key=lambda g: -abs(g["diff"]))

    gap_dims = {g["dimension"] for g in gaps}
    judge_higher = [g for g in gaps if g["diff"] > 0]
    judge_lower = [g for g in gaps if g["diff"] < 0]

    causes = []
    test_type = (case or {}).get("test_type")
    ctx_available = (case or {}).get("governed_context", {}).get("available", True)

    # Human raters split among themselves — the reference itself is unstable.
    if pair.get("n_human_raters", 0) > 1:
        modes = {normalize_failure_mode(m) for m in pair["human"].get("failure_modes", [])}
        if len(modes) > 1:
            causes.append("human_annotation_inconsistency")

    if judge_higher and gap_dims & {"groundedness", "correctness"}:
        causes.append("unsupported_inference_accepted")

    if ctx_available is False and (pair["human_pass"] != pair["judge_pass"]):
        causes.append("missing_context_handling")

    if judge_higher and gap_dims & {"helpfulness", "clarity"} and not (gap_dims & {"groundedness", "correctness"}):
        causes.append("model_verbosity_bias")

    if judge_higher and judge_lower:
        causes.append("ambiguous_rubric")

    if test_type in ("ambiguous", "cross_domain") and not gaps:
        causes.append("ambiguous_reference_answer")

    h_mode = normalize_failure_mode(pair["human"].get("failure_modes", ["none"])[0]
                                    if pair["human"].get("failure_modes") else "none")
    j_mode = normalize_failure_mode(pair["judge"].get("failure_mode"))
    if h_mode != "none" and j_mode != "none" and h_mode != j_mode:
        causes.append("failure_mode_mismatch")

    if not causes:
        if judge_higher:
            causes.append("judge_leniency")
        elif judge_lower:
            causes.append("judge_over_severity")
        else:
            causes.append("uncategorised")

    primary = causes[0]
    return {
        "primary_cause": primary,
        "all_causes": causes,
        "cause_description": DISAGREEMENT_CAUSES[primary],
        "dimension_gaps": gaps,
        "largest_gap_dimension": gaps[0]["dimension"] if gaps else None,
        "max_gap": max((abs(g["diff"]) for g in gaps), default=0),
        "pass_conflict": pair["human_pass"] != pair["judge_pass"],
        "human_failure_mode": h_mode,
        "judge_failure_mode": j_mode,
        "recommended_judge_improvement": _judge_improvement(primary, gap_dims),
    }


def _judge_improvement(cause: str, gap_dims: set) -> str:
    """A concrete change to the judge prompt or rubric, not a generic aspiration."""
    if cause == "unsupported_inference_accepted":
        return (
            "Strengthen rule 3 in the judge prompt: require the judge to quote the specific span "
            "of governed context supporting each figure, and to score groundedness at 2 when an "
            "interpretive claim carries no such span."
        )
    if cause == "model_verbosity_bias":
        return (
            "Add an explicit anti-verbosity instruction and a worked example of a long fluent "
            "answer that scores 3 on helpfulness for restating the context."
        )
    if cause == "judge_leniency":
        dims = ", ".join(sorted(gap_dims)) or "the affected dimensions"
        return (
            f"The judge is scoring {dims} above the human reference. Add the human anchor text for "
            "those dimensions verbatim to the judge prompt and re-run the alignment."
        )
    if cause == "judge_over_severity":
        dims = ", ".join(sorted(gap_dims)) or "the affected dimensions"
        return (
            f"The judge over-penalises {dims}. Add a worked example of an acceptable answer at "
            "anchor 4 so the judge stops reserving 4 and 5 for perfect responses."
        )
    if cause == "ambiguous_rubric":
        return (
            "The judge moved score between dimensions rather than agreeing on total quality. "
            "Add the ordering rule — absence of a figure is groundedness, misapplication is "
            "correctness — to both the human rubric and the judge prompt."
        )
    if cause == "missing_context_handling":
        return (
            "Add two contrasting worked examples to the judge prompt: a correct refusal when the "
            "context lacks the figure, and an over-refusal when the context contains it."
        )
    if cause == "human_annotation_inconsistency":
        return (
            "This is not a judge defect. Resolve the human disagreement first — the judge is being "
            "compared against an unstable reference."
        )
    if cause == "failure_mode_mismatch":
        return (
            "Constrain the judge's failure_mode field to an enum and add the reviewer-priority rule "
            "('the failure a reviewer would fix first') to the taxonomy section of the prompt."
        )
    if cause == "ambiguous_reference_answer":
        return (
            "This is a dataset issue, not a judge issue. Sharpen the case's expected_behavior so "
            "only one reading is defensible, or accept the case as genuinely open."
        )
    return "Review this case manually — no structural cause was identified."


def find_alignment_disagreements(
    pairs: list[dict],
    cases_by_id: Optional[dict] = None,
    threshold: int = MATERIAL_GAP,
) -> list[dict]:
    """
    Every case where human and judge differ materially, with an attributed cause.
    Sorted so pass/fail inversions and the largest gaps appear first.
    """
    cases_by_id = cases_by_id or {}
    out = []
    for pair in pairs:
        case = cases_by_id.get(pair["eval_id"])
        analysis = classify_disagreement(pair, case)
        material = [g for g in analysis["dimension_gaps"] if abs(g["diff"]) >= threshold]
        if not material and not analysis["pass_conflict"]:
            continue
        out.append({**pair, "analysis": analysis, "case": case})
    return sorted(
        out,
        key=lambda d: (-int(d["analysis"]["pass_conflict"]), -d["analysis"]["max_gap"]),
    )


def disagreement_cause_distribution(disagreements: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for d in disagreements:
        cause = d["analysis"]["primary_cause"]
        counts[cause] = counts.get(cause, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def agreement_by_failure_category(pairs: list[dict]) -> dict:
    """
    Where does the judge track humans well, and where does it break down?

    Grouped by the human-assigned failure mode, because that is the reviewer's view
    of what went wrong. Reveals patterns like a judge that agrees on hallucinated
    numbers but not on unsupported claims.
    """
    grouped: dict[str, list[dict]] = {}
    for p in pairs:
        modes = p["human"].get("failure_modes") or ["none"]
        mode = normalize_failure_mode(modes[0])
        grouped.setdefault(mode, []).append(p)

    out = {}
    for mode, group in grouped.items():
        h = [p["human_overall"] for p in group if p["human_overall"] is not None]
        j = [p["judge_overall"] for p in group if p["judge_overall"] is not None]
        hp = [p["human_pass"] for p in group]
        jp = [p["judge_pass"] for p in group]
        out[mode] = {
            "failure_mode": mode,
            "severity": severity_of(mode),
            "n": len(group),
            "human_mean": mean(h),
            "judge_mean": mean(j),
            "mean_gap": (mean(j) - mean(h)) if (h and j) else None,
            "pass_agreement": exact_agreement(hp, jp),
            "eval_ids": [p["eval_id"] for p in group],
        }
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["n"]))
