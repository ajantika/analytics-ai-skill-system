"""
ui/page_dashboard.py — AI Quality Dashboard.

The executive view. Every figure on this page is computed from stored evaluation
records at render time; none is written into the page. Where a statistic cannot be
computed from the data present, the tile shows an em-dash and the footnote says why.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from failure_taxonomy import distribution, label_of, severity_of
from human_evals import DIMENSIONS
from ui import data as D
from ui import theme as T


def _headline_tiles(state: D.EvalState) -> None:
    align = state.best_alignment()
    judge = state.judge_stats
    det = (state.latest_run or {}).get("summary", {}).get("deterministic", {})
    cov = state.coverage

    n_pairs = align.get("n", 0)
    provenance = (
        "human raters" if state.has_human
        else "demo profiles — not human" if state.has_demo
        else "no annotations"
    )

    human_mean = align.get("human_mean_score")
    judge_mean = judge.get("mean_overall_score")
    within1 = align.get("within_1_agreement")

    tiles = [
        T.metric_card(
            "Responses evaluated",
            T.num(judge.get("n_parsed")) if judge.get("n_parsed") else T.DASH,
            footnote=f"of {len(state.cases)} golden cases · {T.pct(cov.get('judge_coverage'))} judge coverage",
            accent=T.ACCENT,
        ),
        T.metric_card(
            "Human quality score",
            f"{T.score(human_mean, 2)}/5" if human_mean is not None else T.DASH,
            footnote=(f"mean across {n_pairs} rated responses · {provenance}"
                      if n_pairs else "no responses rated yet"),
            accent=T.GOOD if state.has_human else T.WARN,
            value_color=T.score_color(human_mean),
        ),
        T.metric_card(
            "AI judge quality score",
            f"{T.score(judge_mean, 2)}/5" if judge_mean is not None else T.DASH,
            delta=(T.signed(align.get("score_gap"))
                   + (" vs human" if state.has_human else " vs demo"))
                  if align.get("score_gap") is not None else None,
            delta_good=(abs(align["score_gap"]) < 0.25) if align.get("score_gap") is not None else None,
            footnote=f"mean across {judge.get('n_parsed', 0)} judged responses",
            accent=T.WARN,
            value_color=T.score_color(judge_mean),
        ),
        T.metric_card(
            "Human ↔ AI agreement",
            T.pct(within1),
            footnote=(f"within ±1 point over {n_pairs} paired evaluations · {provenance}"
                      if n_pairs else "requires both human and judge evaluations"),
            accent=T.ACCENT_DEEP,
            value_color=T.rate_color(within1, 0.85, 0.70),
        ),
    ]
    T.metric_row(tiles)

    st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)

    critical = judge.get("critical_failure_rate")
    grounding = det.get("numeric_grounding_pass_rate")
    relevance = det.get("relevance_pass_rate")
    unsupported = det.get("unsupported_claim_rate")

    tiles2 = [
        T.metric_card(
            "Critical failure rate",
            T.pct(critical),
            footnote="hallucinated figure, wrong metric, or policy breach · AI judge",
            target="Target 0%",
            accent=T.BAD,
            value_color=T.rate_color(critical, 0.95, 0.90, lower_is_better=True),
        ),
        T.metric_card(
            "Groundedness pass rate",
            T.pct(grounding),
            footnote=f"deterministic figure matching · n={det.get('n', 0)}",
            target="Target ≥90%",
            accent=T.GOOD,
            value_color=T.rate_color(grounding, 0.90, 0.75),
        ),
        T.metric_card(
            "Relevance pass rate",
            T.pct(relevance),
            footnote="heuristic — term overlap and non-answer detection",
            target="Target ≥90%",
            accent=T.INFO,
            value_color=T.rate_color(relevance, 0.90, 0.75),
        ),
        T.metric_card(
            "Unsupported claim rate",
            T.pct(unsupported),
            footnote="heuristic — generalisation phrase matching",
            target="Target ≤5%",
            accent=T.WARN,
            value_color=T.rate_color(unsupported, 0.95, 0.90, lower_is_better=True),
        ),
    ]
    T.metric_row(tiles2)

    st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)

    routing = (state.latest_run or {}).get("summary", {}).get("routing", {})
    gen = (state.latest_run or {}).get("summary", {}).get("generation", {})

    tiles3 = [
        T.metric_card(
            "Routing accuracy",
            T.pct(routing.get("accuracy")),
            footnote=(f"{routing.get('correct', 0)}/{routing.get('total', 0)} labelled cases · "
                      f"{routing.get('n_excluded_ambiguous', 0)} ambiguous excluded"),
            accent=T.ACCENT,
            value_color=T.rate_color(routing.get("accuracy"), 0.85, 0.70),
        ),
        T.metric_card(
            "Silent misroutes",
            T.num(routing.get("silent_misroutes")),
            footnote="wrong domain with no ambiguity flag — the costly routing error",
            target="Target 0",
            accent=T.BAD,
            value_color=T.GOOD if routing.get("silent_misroutes") == 0 else T.BAD,
        ),
        T.metric_card(
            "Mean response latency",
            f"{T.score(gen.get('mean_latency_seconds'), 2)}s" if gen.get("mean_latency_seconds") else T.DASH,
            footnote=f"generation only · {gen.get('n_errors', 0)} model errors",
            accent=T.INFO,
        ),
        T.metric_card(
            "Human evaluation coverage",
            T.pct(cov.get("human_coverage")),
            footnote=(f"{cov.get('human_annotated', 0)} of {cov.get('total_cases', 0)} cases "
                      f"rated by a person"),
            accent=T.GOOD if state.has_human else T.FAINT,
            value_color=T.rate_color(cov.get("human_coverage"), 0.25, 0.10),
        ),
    ]
    T.metric_row(tiles3)


def _dimension_chart(state: D.EvalState) -> None:
    align = state.best_alignment()
    judge_dims = state.judge_stats.get("by_dimension", {})
    by_dim = align.get("by_dimension", {})

    labels = [T.dim_label(d) for d in DIMENSIONS]
    human_vals = [by_dim.get(d, {}).get("human_mean") for d in DIMENSIONS]
    judge_vals = [judge_dims.get(d) for d in DIMENSIONS]

    if not any(v is not None for v in judge_vals):
        T.empty_state(
            "No dimension scores yet",
            "Rubric dimension means are computed from judge and human evaluation records.",
            D.GENERATE_COMMAND,
        )
        return

    fig = go.Figure()
    if any(v is not None for v in human_vals):
        fig.add_bar(
            name="Human" if state.has_human else "Demo profiles (not human)",
            x=labels, y=human_vals,
            marker_color=T.GOOD if state.has_human else T.WARN,
            marker_line_width=0,
            hovertemplate="%{x}<br>%{y:.2f}/5<extra>Human</extra>",
        )
    fig.add_bar(
        name="AI judge", x=labels, y=judge_vals,
        marker_color=T.ACCENT, marker_line_width=0,
        hovertemplate="%{x}<br>%{y:.2f}/5<extra>AI judge</extra>",
    )
    fig.update_yaxes(range=[0, 5], dtick=1, title_text="Mean score")
    fig.update_layout(barmode="group", bargap=0.28, bargroupgap=0.08)
    st.plotly_chart(T.style_fig(fig, height=290, showlegend=True), use_container_width=True)


def _failure_chart(state: D.EvalState) -> None:
    modes = [r.get("failure_mode") for r in state.judge_results if r.get("parse_ok")]
    dist = distribution(modes)

    if not dist:
        if state.has_judge:
            T.note("The AI judge recorded no failures across the evaluated responses.", "good")
        else:
            T.empty_state("No failure data yet", D.NO_RUN_EXPLANATION, D.GENERATE_COMMAND)
        return

    from failure_taxonomy import SEVERITY_COLORS

    items = list(dist.items())[::-1]
    fig = go.Figure(go.Bar(
        x=[c for _, c in items],
        y=[label_of(m) for m, _ in items],
        orientation="h",
        marker_color=[SEVERITY_COLORS.get(severity_of(m), T.DIM) for m, _ in items],
        marker_line_width=0,
        text=[str(c) for _, c in items],
        textposition="outside",
        textfont=dict(color=T.MUTED, size=11),
        hovertemplate="%{y}<br>%{x} responses<extra></extra>",
    ))
    fig.update_xaxes(title_text="Responses", dtick=T.count_tick(max(c for _, c in items)))
    st.plotly_chart(T.style_fig(fig, height=max(220, 34 * len(items) + 60)), use_container_width=True)


def _score_distribution(state: D.EvalState) -> None:
    scores = [r["overall_score"] for r in state.judge_results
              if r.get("parse_ok") and r.get("overall_score") is not None]
    if len(scores) < 3:
        return

    fig = go.Figure(go.Histogram(
        x=scores, xbins=dict(start=1, end=5.25, size=0.25),
        marker_color=T.ACCENT, marker_line_width=0, opacity=0.85,
        hovertemplate="Score %{x}<br>%{y} responses<extra></extra>",
    ))
    fig.add_vline(x=3.0, line_dash="dash", line_color=T.WARN, line_width=1,
                  annotation_text="pass gate", annotation_font_color=T.WARN,
                  annotation_font_size=10)
    fig.update_xaxes(title_text="Overall score (AI judge)", range=[1, 5.25])
    fig.update_yaxes(title_text="Responses")
    st.plotly_chart(T.style_fig(fig, height=250), use_container_width=True)


def _test_type_breakdown(state: D.EvalState) -> None:
    """Where the system is weak, by what each case was designed to test."""
    rows = []
    for case in state.cases:
        judge = state.judge_for(case["eval_id"])
        if not judge or not judge.get("parse_ok"):
            continue
        rows.append({
            "Test type": D.TEST_TYPE_LABELS.get(case["test_type"], case["test_type"]),
            "score": judge["overall_score"],
            "passed": bool(judge["pass"]),
            "critical": bool(judge["critical_failure"]),
        })

    if not rows:
        return

    df = pd.DataFrame(rows)
    agg = df.groupby("Test type").agg(
        n=("score", "size"),
        mean_score=("score", "mean"),
        pass_rate=("passed", "mean"),
        critical_rate=("critical", "mean"),
    ).reset_index().sort_values("mean_score")

    fig = go.Figure(go.Bar(
        x=agg["mean_score"], y=agg["Test type"], orientation="h",
        marker_color=[T.score_color(v) for v in agg["mean_score"]],
        marker_line_width=0,
        customdata=agg[["n", "pass_rate"]].values,
        text=[f"{v:.2f}" for v in agg["mean_score"]],
        textposition="outside", textfont=dict(color=T.MUTED, size=11),
        hovertemplate="%{y}<br>mean %{x:.2f}/5<br>n=%{customdata[0]}<br>"
                      "pass rate %{customdata[1]:.0%}<extra></extra>",
    ))
    fig.add_vline(x=3.0, line_dash="dash", line_color=T.WARN, line_width=1)
    fig.update_xaxes(title_text="Mean AI judge score", range=[0, 5.4], dtick=1)
    st.plotly_chart(T.style_fig(fig, height=max(230, 32 * len(agg) + 70)), use_container_width=True)


def _key_findings(state: D.EvalState) -> None:
    """
    Findings derived from the records, not authored. Each is a conditional statement
    over computed values, so it cannot describe something the data does not show.
    """
    findings = []

    judge = state.judge_stats
    align = state.best_alignment()
    routing = (state.latest_run or {}).get("summary", {}).get("routing", {})
    det = (state.latest_run or {}).get("summary", {}).get("deterministic", {})

    if judge.get("critical_failure_rate"):
        n_crit = round(judge["critical_failure_rate"] * judge.get("n_parsed", 0))
        findings.append((
            T.BAD,
            f"{T.pct(judge['critical_failure_rate'])} critical failure rate",
            f"{n_crit} of {judge.get('n_parsed')} responses carry a failure that would lead a "
            f"reader to a wrong decision — a hallucinated figure, a figure attached to the wrong "
            f"metric, or a recommendation about a named individual.",
        ))

    if align.get("n") and align.get("bias", {}).get("direction") in ("lenient", "severe"):
        bias = align["bias"]
        findings.append((
            T.WARN,
            f"The AI judge is {bias['direction']} relative to human raters",
            bias["statement"] + " The Alignment page breaks this down by dimension.",
        ))

    if routing.get("silent_misroutes") == 0 and routing.get("accuracy") is not None:
        findings.append((
            T.GOOD,
            "No silent misroutes",
            f"Routing accuracy is {T.pct(routing['accuracy'])}. Every remaining misroute is "
            f"flagged as ambiguous rather than answered confidently from the wrong governed "
            f"context — the failure mode that produces a plausible, auditable-looking wrong answer.",
        ))
    elif routing.get("silent_misroutes"):
        findings.append((
            T.BAD,
            f"{routing['silent_misroutes']} silent misroute(s)",
            "These questions were answered confidently from the wrong domain's governed context, "
            "with no ambiguity signal shown to the user.",
        ))

    if judge.get("off_taxonomy_rate"):
        findings.append((
            T.WARN,
            f"{T.pct(judge['off_taxonomy_rate'])} of judge outputs used an off-taxonomy label",
            "The judge invented a failure category outside the taxonomy. These are counted "
            "separately as 'unclassified' rather than folded into a neighbouring category.",
        ))

    if judge.get("pass_rule_error_rate"):
        findings.append((
            T.WARN,
            f"Judge misapplied its own pass rule on {T.pct(judge['pass_rule_error_rate'])} of responses",
            "The judge's self-reported pass/fail disagreed with the written rule applied to its "
            "own dimension scores. The stored record uses the rule, not the judge's claim.",
        ))

    if det.get("required_facts_pass_rate") is not None and det["required_facts_pass_rate"] < 0.8:
        findings.append((
            T.WARN,
            f"Required-fact coverage is {T.pct(det['required_facts_pass_rate'])}",
            f"Across {det.get('required_facts_n', 0)} cases that specify required facts, responses "
            f"are omitting figures the reference answer requires.",
        ))

    if not state.has_human:
        findings.append((
            T.DIM,
            "No human annotations yet",
            "The AI judge is currently unvalidated — its agreement with human judgement is the "
            "only evidence that its scores mean anything. Rate cases on the Human Evaluation page.",
        ))

    if not findings:
        return

    for color, title, detail in findings[:6]:
        st.markdown(
            T.panel(
                T.label_text(title, color) + T.body_text(detail, T.TEXT, "0.79rem"),
                accent=color,
            ),
            unsafe_allow_html=True,
        )


def _run_banner(state: D.EvalState) -> None:
    run = state.latest_run
    if not run:
        return
    cfg = run["config"]
    is_real = cfg.get("artifact_kind") == "real_model_run"
    kind = "Live model run" if is_real else cfg.get("artifact_kind", "unknown")
    kind_color = T.GOOD if is_real else T.BAD
    kind_bg = "rgba(34,197,94,0.12)" if is_real else "rgba(248,113,113,0.15)"
    st.markdown(
        f'<div style="background:{T.SURFACE};border:1px solid {T.BORDER};border-radius:9px;'
        f'padding:9px 14px;margin-bottom:14px;display:flex;flex-wrap:wrap;gap:6px;'
        f'align-items:center">'
        + T.chip(kind, kind_color, kind_bg)
        + T.chip(f"run {cfg['run_id']}", T.MUTED)
        + T.chip(f"model {cfg['model_version']}", T.MUTED)
        + T.chip(f"prompt {cfg['system_prompt_version']}", T.MUTED)
        + T.chip(f"router {cfg['router_version']}", T.MUTED)
        + T.chip(f"judge {cfg.get('judge_version', '—')}", T.MUTED)
        + T.chip(f"dataset v{cfg.get('dataset_version', '—')}", T.MUTED)
        + T.chip(cfg["timestamp"][:16].replace("T", " ") + " UTC", T.FAINT)
        + "</div>",
        unsafe_allow_html=True,
    )
    if not is_real:
        T.note(
            f"<strong>This run is not a live model run.</strong> Its artifact_kind is "
            f"<code>{cfg.get('artifact_kind')}</code>. Every figure below is computed from these "
            f"records, but the records themselves did not come from real model calls — so they "
            f"say nothing about model quality. Run "
            f"<code>python eval_runner.py --all</code> with a Groq key to replace them.",
            "warn",
        )


def render(state: D.EvalState) -> None:
    T.page_header(
        "AI Quality Dashboard",
        "Response quality measured by three independent evaluators over a 60-case adversarial "
        "golden set. Every figure below is computed from stored evaluation records — nothing on "
        "this page is written in.",
        eyebrow="Evaluation overview",
    )

    if not state.has_responses:
        T.empty_state(
            "No evaluation run found",
            D.NO_RUN_EXPLANATION,
            D.GENERATE_COMMAND,
        )
        T.section("What this page will show", top_rule=True)
        T.evaluator_tier_legend()
        return

    _run_banner(state)
    _headline_tiles(state)

    if not state.has_human:
        T.note(
            "<strong>Human quality figures are unavailable.</strong> No responses have been rated "
            "by a person yet, so human-versus-judge agreement cannot be computed. Where a "
            "human-side figure appears below sourced from demo profiles, it is labelled "
            "<em>demo profiles — not human</em>. Demo profiles are scripted rubric applications, "
            "not annotators.",
            "warn",
        )

    T.section(
        "Quality by rubric dimension",
        "The six dimensions are scored independently. Averaging them into a single number hides "
        "exactly the disagreement worth looking at, so they are compared individually.",
    )
    _dimension_chart(state)

    left, right = st.columns([1, 1], gap="medium")
    with left:
        T.section("Failure mode distribution", "AI judge classification, ordered by severity.")
        _failure_chart(state)
    with right:
        T.section("Quality by test type", "What each case was designed to probe.")
        _test_type_breakdown(state)

    T.section(
        "Score distribution",
        "Where responses cluster relative to the pass gate. A tight distribution just above the "
        "gate usually means the rubric is not discriminating, not that quality is uniform.",
    )
    _score_distribution(state)

    T.section("Key findings", "Derived from the evaluation records on every render.")
    _key_findings(state)

    T.section(
        "Evaluator hierarchy",
        "Three evaluators with different cost, coverage and authority. The architecture depends "
        "on not confusing them.",
    )
    T.evaluator_tier_legend()

    with st.expander("What these numbers do and do not establish"):
        st.markdown(f"""
**Computed from records.** Every figure on this page is derived at render time from
`data/evaluation_runs.json`, `data/judge_results.json` and `data/human_annotations.json`.
No percentage on this page is a constant in the source.

**Denominators are small.** The golden set is {len(state.cases)} cases. That is enough to
expose failure modes and to compare configurations against each other. It is not a
statistically powered benchmark, and a difference of one or two cases moves several of
these rates by more than a percentage point.

**The golden set is not held out.** It was authored against the same governed YAML files
the system answers from, by the person who built the system. Overfitting is possible and
should be assumed until an independently authored set exists.

**Deterministic checks verify presence, not correctness.** Numeric grounding confirms a
figure appears in the governed context. It does not confirm the figure was used correctly
— a real number attached to the wrong metric passes this check and is caught, if at all,
by the human rubric or the judge.

**The AI judge is not a source of truth.** It is a scalable approximation of human
judgement whose agreement with humans is measured on the Alignment page. Reading judge
scores without reading that page is the specific mistake this application exists to argue
against.

**No production traffic.** All figures in the governed layer are illustrative data
embedded in YAML skill files. Nothing here reflects real users, real customers or real
employees.
""")
