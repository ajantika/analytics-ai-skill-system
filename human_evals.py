"""
human_evals.py — Human evaluation rubric, annotation storage, and rater calibration.

Humans are the reference standard in this system. Deterministic checks are faster
and the LLM judge is cheaper, but neither defines what "good" means — the rubric
below does, and human application of it is what the judge is measured against.

Annotation provenance is tracked explicitly and never flattened:

  rater_type = "human"        A person applied the rubric in the Human Evaluation UI.
  rater_type = "demo_profile" A scripted rubric profile. NOT a person. It reads the
                              real model response and applies deterministic anchor
                              rules with a fixed leniency bias, so that the
                              calibration workflow has two differing raters to
                              reason about. Included to demonstrate methodology.

Every agreement statistic in this application reports human-only, demo-only and
combined populations separately. No statistic silently pools the two.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
from typing import Optional

from failure_taxonomy import normalize_failure_mode, is_critical
from stats_utils import (
    cohens_kappa,
    exact_agreement,
    mean,
    mean_signed_difference,
    paired,
    within_n_agreement,
)

logger = logging.getLogger(__name__)

DATA_DIR = pathlib.Path(__file__).parent / "data"
ANNOTATIONS_PATH = DATA_DIR / "human_annotations.json"

# ── Rubric ────────────────────────────────────────────────────────────────────

SCALE = {
    1: "Poor",
    2: "Major problems",
    3: "Acceptable",
    4: "Good",
    5: "Excellent",
}

SCALE_MIN, SCALE_MAX = 1, 5

# Ordered as they appear in the annotation UI and on every dashboard.
DIMENSIONS: list[str] = [
    "relevance",
    "groundedness",
    "correctness",
    "instruction_following",
    "helpfulness",
    "clarity",
]

DIMENSION_LABELS = {
    "relevance": "Relevance",
    "groundedness": "Groundedness",
    "correctness": "Correctness",
    "instruction_following": "Instruction following",
    "helpfulness": "Helpfulness",
    "clarity": "Clarity",
}

# Anchors are written so that two evaluators reading only this table would apply
# comparable standards. Each anchor names an observable property of the text, not
# a feeling about it.
RUBRIC: dict[str, dict] = {
    "relevance": {
        "label": "Relevance",
        "question": "Does the answer address the question that was actually asked?",
        "scored_on": "The match between the user's information need and what the answer delivers.",
        "ignore": "Whether the facts are correct — that is Correctness. Judge relevance even of a wrong answer.",
        "anchors": {
            5: "Answers exactly what was asked, including every clause of a multi-part question.",
            4: "Answers the question; includes some material that was not asked for but does not obscure the answer.",
            3: "Answers the general topic but misses a clause, or answers a narrower question than the one asked.",
            2: "Addresses the domain but not the question — the reader would have to ask again.",
            1: "Does not address the question at all, or answers a different question entirely.",
        },
    },
    "groundedness": {
        "label": "Groundedness",
        "question": "Is every claim traceable to the supplied governed context?",
        "scored_on": "Whether each figure and factual assertion can be located in the context, or is explicitly marked as derived or as inference.",
        "ignore": "Whether the claim is useful or well written. An elegantly written ungrounded claim scores 1.",
        "anchors": {
            5: "Every figure appears verbatim in the context; any derivation or inference is explicitly labelled as such.",
            4: "All figures are grounded; one mild interpretive statement goes slightly beyond the context without being flagged.",
            3: "Figures are grounded but the answer adds unmarked interpretation, or reuses a real figure in a loosely supported way.",
            2: "Contains a claim or comparison the context does not support, or an external benchmark not present in the context.",
            1: "Contains a figure that does not appear in the context and cannot be derived from it. Any hallucinated number caps this dimension at 1.",
        },
    },
    "correctness": {
        "label": "Correctness",
        "question": "Are the figures attached to the right metrics, and are the conclusions logically valid?",
        "scored_on": "Metric-to-figure mapping, arithmetic, direction of comparisons, and whether conclusions follow from the stated evidence.",
        "ignore": "Whether the figure was in the context — that is Groundedness. A grounded figure attached to the wrong metric is a Correctness failure.",
        "anchors": {
            5: "All figures map to the correct metric definitions; every comparison and conclusion holds.",
            4: "Substantively correct; one minor imprecision that would not change a decision.",
            3: "Core answer correct but contains a secondary error — a mislabelled comparison or a rounding claim that does not hold.",
            2: "A figure is attached to the wrong metric, or a stated conclusion does not follow from the evidence given.",
            1: "The central claim is wrong and would lead a reader to the opposite decision.",
        },
    },
    "instruction_following": {
        "label": "Instruction following",
        "question": "Were explicit instructions in the question obeyed?",
        "scored_on": "Only constraints the user stated — length, format, scope, 'no recommendation', 'numbers only', 'one sentence'.",
        "ignore": "House style. If the user gave no formatting instruction, score 5 unless the output is unusable.",
        "anchors": {
            5: "Every explicit instruction followed exactly.",
            4: "All instructions followed with a trivial deviation (e.g. two sentences where one was requested).",
            3: "One instruction partially followed.",
            2: "An explicit instruction ignored.",
            1: "Multiple explicit instructions ignored, or the answer does the opposite of what was asked.",
        },
    },
    "helpfulness": {
        "label": "Helpfulness",
        "question": "Would a business reader be able to act on this, or does it just restate the context?",
        "scored_on": "Whether the answer synthesises — surfaces the comparison, the outlier, the implication — rather than echoing definitions back.",
        "ignore": "Whether it is grounded or correct. A useless-but-accurate restatement scores low here and high on Groundedness.",
        "anchors": {
            5: "Surfaces the decision-relevant point the reader would otherwise have had to work out, with a proportionate next step.",
            4: "Useful and specific; the implication is present but the reader does some work.",
            3: "Accurate but largely a restatement of the context with little synthesis.",
            2: "Generic advice that would apply to any company with any numbers.",
            1: "No usable information conveyed.",
        },
    },
    "clarity": {
        "label": "Clarity",
        "question": "Is it readable, well structured, and free of padding?",
        "scored_on": "Structure, concision, and whether figures are presented so the reader can follow them.",
        "ignore": "Length alone. A long answer that earns its length is clear; a short one that omits the units is not.",
        "anchors": {
            5: "Tight, well structured, every sentence carries weight.",
            4: "Clear, with minor redundancy.",
            3: "Understandable but padded, repetitive, or loosely organised.",
            2: "Requires rereading; structure fights the content.",
            1: "Confusing or self-contradictory.",
        },
    },
}

# Written down so pass/fail is a rule, not a mood. Applied identically by the
# human UI default, the demo profiles, and the LLM judge prompt.
PASS_RULE = (
    "PASS requires all of: groundedness >= 3, correctness >= 3, relevance >= 3, "
    "and no critical failure mode. Any single dimension at 1 is an automatic FAIL."
)

CRITICAL_RULE = (
    "CRITICAL FAILURE is set when the answer would lead a reader to a wrong business "
    "decision or an unsafe action: a hallucinated figure, a figure attached to the wrong "
    "metric, or a consequential recommendation about a named individual."
)

CONFIDENCE_RULE = (
    "Evaluator confidence records how sure the evaluator is of their own scores (0-1). "
    "Low confidence marks cases where the rubric was hard to apply — these are the cases "
    "that drive rubric revision, so they are recorded rather than resolved by guessing."
)


def rubric_markdown() -> str:
    """Render the full rubric for the collapsible panel in the annotation UI."""
    out = [
        "**Scale** — " + " · ".join(f"`{k}` {v}" for k, v in SCALE.items()),
        "",
    ]
    for dim in DIMENSIONS:
        r = RUBRIC[dim]
        out.append(f"#### {r['label']}")
        out.append(f"*{r['question']}*")
        out.append("")
        out.append(f"**Score on:** {r['scored_on']}")
        out.append(f"**Do not score on:** {r['ignore']}")
        out.append("")
        for score in (5, 4, 3, 2, 1):
            out.append(f"- **{score} — {SCALE[score]}:** {r['anchors'][score]}")
        out.append("")
    out.append("---")
    out.append(f"**Pass rule** — {PASS_RULE}")
    out.append("")
    out.append(f"**Critical failure** — {CRITICAL_RULE}")
    out.append("")
    out.append(f"**Confidence** — {CONFIDENCE_RULE}")
    return "\n".join(out)


# ── Scoring ───────────────────────────────────────────────────────────────────


def overall_score(scores: dict) -> Optional[float]:
    """
    Unweighted mean of the dimensions actually scored.

    Unweighted on purpose: a weighted composite would bury the dimension-level
    disagreement that the alignment analysis exists to surface. Dimensions are
    compared individually everywhere it matters.
    """
    vals = [scores.get(d) for d in DIMENSIONS if scores.get(d) is not None]
    return sum(vals) / len(vals) if vals else None


def applies_pass_rule(scores: dict, failure_mode: Optional[str] = None) -> bool:
    """Deterministic application of PASS_RULE. Used as the UI default and by the demo profiles."""
    if failure_mode and is_critical(failure_mode):
        return False
    present = [scores.get(d) for d in DIMENSIONS if scores.get(d) is not None]
    if not present:
        return False
    if any(v <= 1 for v in present):
        return False
    for gate in ("groundedness", "correctness", "relevance"):
        v = scores.get(gate)
        if v is not None and v < 3:
            return False
    return True


def validate_annotation(ann: dict) -> list[str]:
    """Return a list of problems with an annotation record. Empty list means valid."""
    problems = []
    if not ann.get("eval_id"):
        problems.append("missing eval_id")
    if not ann.get("evaluator_id"):
        problems.append("missing evaluator_id")
    if ann.get("rater_type") not in ("human", "demo_profile"):
        problems.append(f"rater_type must be 'human' or 'demo_profile', got {ann.get('rater_type')!r}")
    scores = ann.get("scores") or {}
    for dim in DIMENSIONS:
        v = scores.get(dim)
        if v is None:
            continue
        if not isinstance(v, int) or not (SCALE_MIN <= v <= SCALE_MAX):
            problems.append(f"{dim} must be an int in {SCALE_MIN}-{SCALE_MAX}, got {v!r}")
    if not any(scores.get(d) is not None for d in DIMENSIONS):
        problems.append("no dimensions scored")
    conf = ann.get("evaluator_confidence")
    if conf is not None and not (0.0 <= float(conf) <= 1.0):
        problems.append(f"evaluator_confidence must be 0-1, got {conf!r}")
    return problems


def make_annotation(
    eval_id: str,
    evaluator_id: str,
    scores: dict,
    rater_type: str = "human",
    passed: Optional[bool] = None,
    critical_failure: Optional[bool] = None,
    failure_mode: Optional[str] = None,
    evaluator_confidence: float = 0.8,
    notes: str = "",
    timestamp: Optional[str] = None,
    run_id: Optional[str] = None,
    response_fingerprint: Optional[str] = None,
) -> dict:
    """
    Build a normalised annotation record. Pass/critical default to the written rules.

    run_id and response_fingerprint record *which* generated response was actually in
    front of the evaluator. Without them a rating is just a score attached to a case id,
    and pairing it against a judge verdict silently assumes both saw the same text —
    an assumption that failed once already when a baseline run replaced the responses
    the annotation UI was serving.
    """
    mode = normalize_failure_mode(failure_mode)
    crit = is_critical(mode) if critical_failure is None else bool(critical_failure)
    ok = applies_pass_rule(scores, mode) if passed is None else bool(passed)
    return {
        "eval_id": eval_id,
        "evaluator_id": evaluator_id,
        "rater_type": rater_type,
        "scores": {d: scores.get(d) for d in DIMENSIONS},
        "overall_score": overall_score(scores),
        "pass": ok,
        "critical_failure": crit,
        "failure_mode": mode,
        "evaluator_confidence": float(evaluator_confidence),
        "notes": notes,
        "timestamp": timestamp,
        "run_id": run_id,
        "response_fingerprint": response_fingerprint,
    }


def fingerprint(text: str) -> str:
    """Short stable digest of a response, used to prove a rating and a judgement saw the same text."""
    import hashlib
    return hashlib.sha256((text or "").strip().encode()).hexdigest()[:12]


# ── Storage ───────────────────────────────────────────────────────────────────


def load_annotations(path: Optional[str] = None) -> list[dict]:
    """Load all annotations. Returns [] when the file is absent — an empty corpus, not an error."""
    p = pathlib.Path(path) if path else ANNOTATIONS_PATH
    if not p.exists():
        return []
    try:
        with open(p, "r") as f:
            data = json.load(f)
        records = data.get("annotations", data) if isinstance(data, dict) else data
        return [r for r in records if isinstance(r, dict)]
    except Exception as e:
        logger.error(f"Failed to load annotations from {p}: {e}")
        return []


def save_annotations(annotations: list[dict], path: Optional[str] = None) -> bool:
    """
    Persist annotations, replacing any existing record with the same
    (eval_id, evaluator_id). Returns False if the filesystem is read-only, which
    is the normal case on Streamlit Community Cloud.
    """
    p = pathlib.Path(path) if path else ANNOTATIONS_PATH
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "note": (
                "rater_type 'human' records were authored by a person in the Human Evaluation UI. "
                "rater_type 'demo_profile' records were produced by a scripted rubric profile and "
                "are not human annotations."
            ),
            "annotations": annotations,
        }
        tmp = p.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, p)
        return True
    except OSError as e:
        logger.warning(f"Could not persist annotations to {p}: {e}")
        return False


def upsert_annotation(annotation: dict, existing: Optional[list[dict]] = None) -> list[dict]:
    """Insert or replace by (eval_id, evaluator_id). Pure — does not touch disk."""
    records = list(existing if existing is not None else load_annotations())
    key = (annotation.get("eval_id"), annotation.get("evaluator_id"))
    records = [r for r in records if (r.get("eval_id"), r.get("evaluator_id")) != key]
    records.append(annotation)
    return records


# ── Views over the corpus ─────────────────────────────────────────────────────


def by_rater_type(annotations: list[dict], rater_type: str) -> list[dict]:
    return [a for a in annotations if a.get("rater_type") == rater_type]


def raters(annotations: list[dict]) -> list[dict]:
    """Distinct raters with their type and annotation count, human raters first."""
    seen: dict[str, dict] = {}
    for a in annotations:
        rid = a.get("evaluator_id")
        if rid not in seen:
            seen[rid] = {"evaluator_id": rid, "rater_type": a.get("rater_type"), "count": 0}
        seen[rid]["count"] += 1
    return sorted(seen.values(), key=lambda r: (r["rater_type"] != "human", r["evaluator_id"]))


def consensus_by_eval(annotations: list[dict], rater_type: Optional[str] = None) -> dict[str, dict]:
    """
    Collapse multiple raters into one reference score per eval_id.

    Mean for ordinal dimensions; majority for pass (ties resolved to FAIL, because
    an unresolved disagreement about whether an answer is acceptable should not be
    recorded as acceptable); any-rater for critical failure.
    """
    pool = [a for a in annotations if rater_type is None or a.get("rater_type") == rater_type]
    grouped: dict[str, list[dict]] = {}
    for a in pool:
        grouped.setdefault(a.get("eval_id"), []).append(a)

    out = {}
    for eval_id, anns in grouped.items():
        dim_means = {}
        for d in DIMENSIONS:
            vals = [a["scores"].get(d) for a in anns if a.get("scores", {}).get(d) is not None]
            dim_means[d] = sum(vals) / len(vals) if vals else None
        passes = [bool(a.get("pass")) for a in anns]
        out[eval_id] = {
            "eval_id": eval_id,
            "n_raters": len(anns),
            "rater_ids": [a.get("evaluator_id") for a in anns],
            "rater_types": sorted({a.get("rater_type") for a in anns}),
            "scores": dim_means,
            "overall_score": mean([v for v in dim_means.values() if v is not None]),
            "pass": sum(passes) > len(passes) / 2,
            "critical_failure": any(bool(a.get("critical_failure")) for a in anns),
            "failure_modes": [a.get("failure_mode") for a in anns],
            "confidence": mean([a.get("evaluator_confidence") for a in anns]),
        }
    return out


# ── Human-vs-human calibration ────────────────────────────────────────────────


def rater_agreement(
    annotations: list[dict],
    rater_a: str,
    rater_b: str,
) -> dict:
    """
    Agreement between two named raters across every eval both scored.

    Per dimension: exact agreement, within-1 agreement, quadratic-weighted kappa
    (the right kappa for ordinal 1-5 scores), and mean signed difference showing
    which rater is systematically more severe.
    """
    a_recs = {r["eval_id"]: r for r in annotations if r.get("evaluator_id") == rater_a}
    b_recs = {r["eval_id"]: r for r in annotations if r.get("evaluator_id") == rater_b}
    shared = sorted(set(a_recs) & set(b_recs))

    result = {
        "rater_a": rater_a,
        "rater_b": rater_b,
        "rater_a_type": next((r.get("rater_type") for r in a_recs.values()), None),
        "rater_b_type": next((r.get("rater_type") for r in b_recs.values()), None),
        "n_shared": len(shared),
        "eval_ids": shared,
        "by_dimension": {},
        "overall": {},
        "pass_fail": {},
    }
    if not shared:
        return result

    for dim in DIMENSIONS:
        xa = {e: a_recs[e]["scores"].get(dim) for e in shared}
        xb = {e: b_recs[e]["scores"].get(dim) for e in shared}
        _, va, vb = paired(xa, xb)
        result["by_dimension"][dim] = {
            "n": len(va),
            "exact": exact_agreement(va, vb),
            "within_1": within_n_agreement(va, vb, 1),
            "kappa_quadratic": cohens_kappa(va, vb, categories=[1, 2, 3, 4, 5], weights="quadratic"),
            "mean_diff_b_minus_a": mean_signed_difference(va, vb),
            "a_mean": mean(va),
            "b_mean": mean(vb),
        }

    oa = [a_recs[e].get("overall_score") for e in shared]
    ob = [b_recs[e].get("overall_score") for e in shared]
    result["overall"] = {
        "a_mean": mean(oa),
        "b_mean": mean(ob),
        "within_1": within_n_agreement(
            [v for v in oa if v is not None], [v for v in ob if v is not None], 1
        ),
        "mean_diff_b_minus_a": mean_signed_difference(
            [v for v in oa if v is not None], [v for v in ob if v is not None]
        ),
    }

    pa = [bool(a_recs[e].get("pass")) for e in shared]
    pb = [bool(b_recs[e].get("pass")) for e in shared]
    result["pass_fail"] = {
        "agreement": exact_agreement(pa, pb),
        "kappa": cohens_kappa(pa, pb, categories=[False, True]),
        "a_pass_rate": sum(pa) / len(pa),
        "b_pass_rate": sum(pb) / len(pb),
    }
    return result


def find_disagreements(
    annotations: list[dict],
    rater_a: str,
    rater_b: str,
    threshold: int = 2,
) -> list[dict]:
    """
    Cases where two raters differ by >= threshold on any dimension, or disagree on
    pass/fail. Sorted by severity of disagreement so the worst calibration gaps
    surface first.
    """
    a_recs = {r["eval_id"]: r for r in annotations if r.get("evaluator_id") == rater_a}
    b_recs = {r["eval_id"]: r for r in annotations if r.get("evaluator_id") == rater_b}

    out = []
    for eval_id in sorted(set(a_recs) & set(b_recs)):
        ra, rb = a_recs[eval_id], b_recs[eval_id]
        gaps = []
        for dim in DIMENSIONS:
            va, vb = ra["scores"].get(dim), rb["scores"].get(dim)
            if va is None or vb is None:
                continue
            if abs(va - vb) >= threshold:
                gaps.append({"dimension": dim, "a": va, "b": vb, "gap": abs(va - vb)})
        pass_conflict = bool(ra.get("pass")) != bool(rb.get("pass"))
        if not gaps and not pass_conflict:
            continue
        out.append({
            "eval_id": eval_id,
            "rater_a": rater_a,
            "rater_b": rater_b,
            "a_record": ra,
            "b_record": rb,
            "dimension_gaps": sorted(gaps, key=lambda g: -g["gap"]),
            "pass_conflict": pass_conflict,
            "max_gap": max([g["gap"] for g in gaps], default=0),
            "likely_reason": _disagreement_reason(ra, rb, gaps, pass_conflict),
            "rubric_clarification": _rubric_clarification(gaps, pass_conflict),
        })
    return sorted(out, key=lambda d: (-int(d["pass_conflict"]), -d["max_gap"]))


def _disagreement_reason(ra: dict, rb: dict, gaps: list[dict], pass_conflict: bool) -> str:
    """
    Diagnose a rater disagreement from the record itself.

    Deliberately rule-based and conservative: it names the structural reason two
    raters could land differently, and says so when it cannot tell.
    """
    if gaps:
        dims = {g["dimension"] for g in gaps}
        if "groundedness" in dims and "correctness" in dims:
            return (
                "Both raters flagged a problem but split it across dimensions — one recorded it "
                "as ungrounded, the other as incorrect. The rubric boundary between 'the figure "
                "is not in the context' and 'the figure is in the context but misapplied' was not "
                "decisive here."
            )
        if "helpfulness" in dims:
            return (
                "Helpfulness is the least anchored dimension: raters differ on whether restating "
                "the governed context counts as useful to a business reader."
            )
        if "groundedness" in dims:
            return (
                "Disagreement over whether an interpretive statement counts as an unmarked "
                "inference (anchor 3) or a claim beyond the context (anchor 2)."
            )
        if "clarity" in dims:
            return "Clarity is stylistic; raters weight concision against completeness differently."
        if "instruction_following" in dims:
            return (
                "Raters differ on whether the fixed three-section response template constitutes "
                "the model ignoring the user's formatting instruction."
            )
        if "correctness" in dims:
            return "Raters differ on whether a secondary imprecision changes the decision the answer supports."
    if pass_conflict:
        conf = [ra.get("evaluator_confidence"), rb.get("evaluator_confidence")]
        if any(c is not None and c < 0.6 for c in conf):
            return (
                "Pass/fail split with at least one rater reporting low confidence — the case sits "
                "on the boundary of the pass rule rather than being genuinely contested."
            )
        return (
            "Dimension scores are close but fall on opposite sides of a pass-rule gate "
            "(groundedness / correctness / relevance >= 3), which turns a one-point difference "
            "into a pass/fail inversion."
        )
    return "No structural cause identified from the record alone."


def _rubric_clarification(gaps: list[dict], pass_conflict: bool) -> str:
    dims = {g["dimension"] for g in gaps}
    if "helpfulness" in dims:
        return (
            "Add a worked example to the Helpfulness anchors distinguishing a 3 (restates the "
            "context) from a 4 (surfaces the comparison the reader would have had to compute)."
        )
    if "groundedness" in dims and "correctness" in dims:
        return (
            "State an ordering rule: if a figure is absent from the context, score it under "
            "Groundedness; if present but attached to the wrong metric, score it under Correctness. "
            "Never both."
        )
    if "groundedness" in dims:
        return "Define 'unmarked inference' with two contrasting examples at anchors 2 and 3."
    if "instruction_following" in dims:
        return (
            "Clarify that the system's default template is not a user instruction, so its presence "
            "is only an Instruction-following failure when the user asked for something else."
        )
    if "clarity" in dims:
        return "Cap Clarity's influence by noting that completeness is scored under Relevance, not Clarity."
    if pass_conflict:
        return (
            "The pass rule is a hard gate at 3. Record borderline cases with confidence < 0.6 so "
            "gate-boundary cases can be reviewed as a batch rather than resolved ad hoc."
        )
    return "Review the anchors for the affected dimension with both raters."


def calibration_summary(annotations: list[dict]) -> dict:
    """
    All pairwise rater agreement, with each pair tagged by provenance so
    human-human, human-demo and demo-demo comparisons are never conflated.
    """
    rater_list = raters(annotations)
    pairs = []
    for i, ra in enumerate(rater_list):
        for rb in rater_list[i + 1:]:
            ag = rater_agreement(annotations, ra["evaluator_id"], rb["evaluator_id"])
            if ag["n_shared"] == 0:
                continue
            types = {ra["rater_type"], rb["rater_type"]}
            ag["pair_provenance"] = (
                "human-human" if types == {"human"}
                else "demo-demo" if types == {"demo_profile"}
                else "human-demo"
            )
            pairs.append(ag)
    return {
        "raters": rater_list,
        "n_human_raters": sum(1 for r in rater_list if r["rater_type"] == "human"),
        "n_demo_raters": sum(1 for r in rater_list if r["rater_type"] == "demo_profile"),
        "pairs": pairs,
    }
