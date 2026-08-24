"""
ui/page_judge.py — AI Judge.

What the automated judge is, how it is prompted, and — the part usually missing — how
reliable it is as an instrument. A judge's own defects are measured here: whether it
can produce valid structured output, whether its arithmetic matches its scores, whether
it applies its own pass rule, and whether it stays inside the failure taxonomy.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from failure_taxonomy import label_of
from human_evals import DIMENSIONS, PASS_RULE
from llm_judge import JUDGE_MODEL_CHAIN, JUDGE_SYSTEM_PROMPT, JUDGE_TEMPERATURE, JUDGE_VERSION
from ui import data as D
from ui import theme as T


def _reliability_tiles(state: D.EvalState) -> None:
    j = state.judge_stats
    tiles = [
        T.metric_card(
            "Structured output success",
            T.pct(j.get("parse_success_rate")),
            footnote=f"{j.get('n_parsed', 0)} of {j.get('n', 0)} responses returned valid JSON "
                     f"after at most one repair retry",
            target="Target 100%",
            accent=T.GOOD,
            value_color=T.rate_color(j.get("parse_success_rate"), 0.98, 0.90),
        ),
        T.metric_card(
            "Arithmetic error rate",
            T.pct(j.get("arithmetic_error_rate")),
            footnote="judge's own overall_score disagreed with the mean of its dimension scores",
            target="Target 0%",
            accent=T.WARN,
            value_color=T.rate_color(j.get("arithmetic_error_rate"), 0.98, 0.90, lower_is_better=True),
        ),
        T.metric_card(
            "Pass-rule error rate",
            T.pct(j.get("pass_rule_error_rate")),
            footnote="judge's self-reported pass/fail disagreed with the written rule applied "
                     "to its own scores",
            target="Target 0%",
            accent=T.WARN,
            value_color=T.rate_color(j.get("pass_rule_error_rate"), 0.98, 0.90, lower_is_better=True),
        ),
        T.metric_card(
            "Off-taxonomy label rate",
            T.pct(j.get("off_taxonomy_rate")),
            footnote="judge invented a failure category outside the taxonomy",
            target="Target 0%",
            accent=T.BAD,
            value_color=T.rate_color(j.get("off_taxonomy_rate"), 0.98, 0.90, lower_is_better=True),
        ),
    ]
    T.metric_row(tiles)


def _quality_tiles(state: D.EvalState) -> None:
    j = state.judge_stats
    tiles = [
        T.metric_card("Mean overall score", f"{T.score(j.get('mean_overall_score'))}/5",
                      footnote=f"across {j.get('n_parsed', 0)} judged responses",
                      accent=T.ACCENT, value_color=T.score_color(j.get("mean_overall_score"))),
        T.metric_card("Pass rate", T.pct(j.get("pass_rate")),
                      footnote="recomputed from dimension scores under the written pass rule",
                      accent=T.ACCENT, value_color=T.rate_color(j.get("pass_rate"), 0.8, 0.6)),
        T.metric_card("Critical failure rate", T.pct(j.get("critical_failure_rate")),
                      footnote="derived from the taxonomy severity of the assigned failure mode",
                      accent=T.BAD,
                      value_color=T.rate_color(j.get("critical_failure_rate"), 0.95, 0.90,
                                               lower_is_better=True)),
        T.metric_card("Mean judge confidence", T.score(j.get("mean_confidence")),
                      footnote="the judge's own stated confidence, 0 to 1",
                      accent=T.INFO),
    ]
    T.metric_row(tiles)


def _dimension_profile(state: D.EvalState) -> None:
    dims = state.judge_stats.get("by_dimension", {})
    values = [dims.get(d) for d in DIMENSIONS]
    if not any(v is not None for v in values):
        return

    fig = go.Figure(go.Bar(
        x=[T.dim_label(d) for d in DIMENSIONS],
        y=values,
        marker_color=[T.DIMENSION_COLORS[d] for d in DIMENSIONS],
        marker_line_width=0,
        text=[T.score(v) for v in values],
        textposition="outside", textfont=dict(color=T.MUTED, size=11),
        hovertemplate="%{x}<br>%{y:.2f}/5<extra></extra>",
    ))
    fig.add_hline(y=3.0, line_dash="dash", line_color=T.WARN, line_width=1,
                  annotation_text="pass gate", annotation_font_color=T.WARN,
                  annotation_font_size=10)
    fig.update_yaxes(range=[0, 5.4], dtick=1, title_text="Mean judge score")
    st.plotly_chart(T.style_fig(fig, height=270), use_container_width=True)


def _confidence_vs_score(state: D.EvalState) -> None:
    """Does the judge know when it is unsure? A flat relationship means its confidence is noise."""
    points = [
        (r["confidence"], r["overall_score"], r["eval_id"])
        for r in state.judge_results
        if r.get("parse_ok") and r.get("confidence") is not None and r.get("overall_score") is not None
    ]
    if len(points) < 5:
        return

    from stats_utils import pearson

    conf = [p[0] for p in points]
    scores = [p[1] for p in points]
    corr = pearson(conf, scores)

    fig = go.Figure(go.Scatter(
        x=conf, y=scores, mode="markers",
        marker=dict(size=9, color=T.ACCENT, opacity=0.72,
                    line=dict(width=1, color="rgba(255,255,255,0.18)")),
        text=[p[2] for p in points],
        hovertemplate="%{text}<br>confidence %{x:.2f}<br>score %{y:.2f}<extra></extra>",
    ))
    fig.update_xaxes(title_text="Judge confidence", range=[0, 1.05])
    fig.update_yaxes(title_text="Overall score", range=[0.5, 5.5])
    st.plotly_chart(T.style_fig(fig, height=260), use_container_width=True)

    if corr is not None:
        if abs(corr) < 0.2:
            verdict = (
                "Essentially no relationship. The judge's stated confidence carries little "
                "information about the score it produced, which means it should not be used to "
                "triage which cases need human review."
            )
        elif corr > 0:
            verdict = (
                "The judge reports higher confidence on responses it scores well. That is the "
                "expected direction, but it also means low-scoring cases are exactly where its "
                "confidence is lowest — so confidence-based triage would skip the hard cases."
            )
        else:
            verdict = (
                "The judge reports higher confidence on responses it scores poorly. Clear "
                "failures are easier to call than borderline ones, which is a coherent pattern."
            )
        T.note(f"Pearson correlation between confidence and score: <strong>{corr:+.2f}</strong>. "
               f"{verdict}", "info")


def _failure_labels(state: D.EvalState) -> None:
    from failure_taxonomy import distribution

    dist = distribution([r.get("failure_mode") for r in state.judge_results if r.get("parse_ok")])
    if not dist:
        return
    rows = "".join(
        f'<tr><td style="padding:5px 0;color:{T.TEXT};width:44%">{label_of(m)}</td>'
        f'<td style="padding:5px 0;width:22%">{T.severity_chip(__import__("failure_taxonomy").severity_of(m))}</td>'
        f'<td style="padding:5px 0;color:{T.MUTED};font-weight:600">{c}</td></tr>'
        for m, c in dist.items()
    )
    st.markdown(
        f'<table style="width:100%;border-collapse:collapse;font-size:0.8rem">'
        f'<tr><th style="text-align:left;padding-bottom:5px">Failure mode</th>'
        f'<th style="text-align:left;padding-bottom:5px">Severity</th>'
        f'<th style="text-align:left;padding-bottom:5px">Count</th></tr>{rows}</table>',
        unsafe_allow_html=True,
    )


def _browse(state: D.EvalState) -> None:
    parsed = [r for r in state.judge_results if r.get("parse_ok")]
    if not parsed:
        return

    col1, col2, col3 = st.columns([2, 1, 1])
    verdict_filter = col1.selectbox(
        "Show", ["All", "Failures only", "Critical failures only", "Passes only"], index=0,
        key="judge_verdict_filter",
    )
    sort_by = col2.selectbox("Sort by", ["Lowest score", "Highest score", "Case ID"], index=0,
                             key="judge_sort")
    limit = col3.number_input("Show up to", min_value=3, max_value=60, value=10, step=1,
                              key="judge_limit")

    rows = parsed
    if verdict_filter == "Failures only":
        rows = [r for r in rows if not r["pass"]]
    elif verdict_filter == "Critical failures only":
        rows = [r for r in rows if r["critical_failure"]]
    elif verdict_filter == "Passes only":
        rows = [r for r in rows if r["pass"]]

    if sort_by == "Lowest score":
        rows = sorted(rows, key=lambda r: (r["overall_score"] is None, r["overall_score"]))
    elif sort_by == "Highest score":
        rows = sorted(rows, key=lambda r: -(r["overall_score"] or 0))
    else:
        rows = sorted(rows, key=lambda r: r["eval_id"])

    if not rows:
        T.note("No judged responses match this filter.", "info")
        return

    st.markdown(
        f'<div style="color:{T.FAINT};font-size:0.72rem;margin-bottom:8px">'
        f'Showing {min(len(rows), int(limit))} of {len(rows)} matching · '
        f'{len(parsed)} judged in total</div>',
        unsafe_allow_html=True,
    )

    for record in rows[: int(limit)]:
        case = state.case_for(record["eval_id"])
        response = state.response_for(record["eval_id"])
        question = case["question"] if case else record["eval_id"]

        header = (
            T.chip(record["eval_id"], T.ACCENT, "rgba(99,102,241,0.13)")
            + T.pass_chip(record["pass"])
            + T.chip(f"overall {T.score(record['overall_score'])}/5",
                     T.score_color(record["overall_score"]))
            + (T.chip(label_of(record["failure_mode"]), T.BAD, "rgba(248,113,113,0.13)")
               if record["failure_mode"] != "none" else "")
            + (T.chip("CRITICAL", T.BAD, "rgba(248,113,113,0.2)") if record["critical_failure"] else "")
        )

        with st.expander(f"{record['eval_id']} — {question[:78]}"):
            st.markdown(f'<div style="margin-bottom:9px">{header}</div>', unsafe_allow_html=True)

            score_chips = "".join(
                f'<div style="display:flex;justify-content:space-between;align-items:center;'
                f'padding:3px 0;font-size:0.77rem">'
                f'<span style="color:{T.DIMENSION_COLORS[d]}">{T.dim_label(d)}</span>'
                f'{T.score_bar(record["scores"].get(d))}</div>'
                for d in DIMENSIONS
            )
            left, right = st.columns([1, 1], gap="medium")
            left.markdown(
                T.panel(T.label_text("Judge scores") + score_chips), unsafe_allow_html=True
            )
            right.markdown(
                T.panel(
                    T.label_text("Judge rationale", T.WARN)
                    + T.body_text(record.get("reasoning_summary") or "(none)", T.TEXT, "0.79rem")
                    + f'<div style="margin-top:8px;color:{T.FAINT};font-size:0.68rem">'
                    f'confidence {T.score(record.get("confidence"))} · '
                    f'{record.get("judge_model", "—")} · '
                    f'{T.score(record.get("latency_seconds"), 2)}s'
                    f'{" · " + str(record.get("retries", 0)) + " retry" if record.get("retries") else ""}'
                    f'</div>',
                    accent=T.WARN,
                ),
                unsafe_allow_html=True,
            )

            flags = []
            if record.get("arithmetic_error"):
                flags.append(
                    f"Judge reported overall_score {T.score(record['overall_score_reported'])} but "
                    f"its own dimension scores average {T.score(record['overall_score'])}. "
                    f"The stored record uses the computed value."
                )
            if record.get("pass_rule_error"):
                flags.append(
                    f"Judge self-reported {'PASS' if record['pass_reported'] else 'FAIL'} but the "
                    f"written rule gives {'PASS' if record['pass'] else 'FAIL'} for its scores. "
                    f"The stored record uses the rule."
                )
            if record.get("off_taxonomy_label"):
                flags.append(
                    f"Judge returned the off-taxonomy label {record.get('failure_mode_raw')!r}, "
                    f"counted as 'unclassified'."
                )
            for flag in flags:
                T.note(f"<strong>Judge self-consistency:</strong> {flag}", "warn")

            if response and not response.get("error"):
                st.markdown(
                    T.panel(
                        T.label_text("Response that was judged", T.DIM)
                        + T.body_text(T.response_html(response["answer"]), T.TEXT, "0.78rem"),
                        accent=T.DIM,
                    ),
                    unsafe_allow_html=True,
                )

            annotations = state.annotations_for(record["eval_id"])
            if annotations:
                st.markdown(
                    f'<div style="color:{T.DIM};font-size:0.7rem;font-weight:700;'
                    f'text-transform:uppercase;letter-spacing:0.08em;margin:8px 0 4px">'
                    f'How raters scored the same response</div>',
                    unsafe_allow_html=True,
                )
                for a in annotations:
                    gap = (
                        (record["overall_score"] - a["overall_score"])
                        if a.get("overall_score") is not None and record.get("overall_score") is not None
                        else None
                    )
                    st.markdown(
                        f'<div style="padding:5px 0;font-size:0.76rem">'
                        + T.provenance_chip(a.get("rater_type", ""))
                        + T.chip(a["evaluator_id"], T.MUTED)
                        + T.chip(f"overall {T.score(a.get('overall_score'))}/5", T.MUTED)
                        + T.pass_chip(a.get("pass"))
                        + (T.chip(f"judge {T.signed(gap)} vs this rater",
                                  T.BAD if gap and abs(gap) >= 1 else T.MUTED)
                           if gap is not None else "")
                        + "</div>",
                        unsafe_allow_html=True,
                    )


def render(state: D.EvalState) -> None:
    T.page_header(
        "AI Judge",
        "A model scoring model output against the same rubric a human applies. It is the only "
        "evaluator that can cover the whole set cheaply, and the only one whose own reliability "
        "has to be established before its scores mean anything.",
        eyebrow="Automated evaluation",
    )

    if not state.has_judge:
        T.empty_state("No judge evaluations found", D.NO_RUN_EXPLANATION, D.GENERATE_COMMAND)
        with st.expander("The judge prompt used when a run is executed", expanded=True):
            st.code(JUDGE_SYSTEM_PROMPT, language="text")
        return

    T.note(
        f"Judge version <code>{JUDGE_VERSION}</code> · temperature {JUDGE_TEMPERATURE} "
        f"(deterministic — a judge that drifts between runs is not a measurement) · "
        f"models actually used: {', '.join(state.judge_stats.get('models_used') or ['—'])}",
        "info",
    )

    T.section("Judge reliability", "Defects in the instrument itself, before any quality claim.",
              top_rule=False)
    _reliability_tiles(state)

    with st.expander("Why these four reliability metrics exist"):
        st.markdown(f"""
The judge reports an `overall_score`, a `pass` flag and a `failure_mode`. None of the
three is taken at face value.

**Structured output success** — the judge is asked for strict JSON. When it returns
prose, a fence, or a truncated object, it is retried once with an explicit repair
instruction. A second failure is recorded as a judge failure rather than dropped, because
a judge that cannot produce parseable output on some inputs has a coverage gap, not a
quality result.

**Arithmetic error rate** — `overall_score` is recomputed as the mean of the six dimension
scores. Where the judge's own figure differs by more than 0.15, the discrepancy is
recorded and the computed value is stored. Language models are unreliable at arithmetic;
trusting a self-reported average would import that unreliability into every downstream
metric.

**Pass-rule error rate** — pass/fail is recomputed by applying the written rule
(*{PASS_RULE}*) to the judge's own dimension scores. Where its self-report disagrees, the
rule wins. This keeps human and judge pass rates comparable: both come from the same rule
applied to their respective scores, not from two different notions of "acceptable".

**Off-taxonomy label rate** — a judge that invents categories makes the failure
distribution meaningless. Unrecognised labels are counted as `unclassified` and surfaced,
never mapped onto the nearest real category.
""")

    T.section("Judge quality output", "What the judge concluded about the responses.")
    _quality_tiles(state)

    left, right = st.columns([3, 2], gap="medium")
    with left:
        T.section("Score by rubric dimension", top_rule=False)
        _dimension_profile(state)
    with right:
        T.section("Failure labels assigned", top_rule=False)
        _failure_labels(state)

    T.section(
        "Is the judge's confidence informative?",
        "The judge reports its own confidence. Whether that number carries any signal determines "
        "whether it can be used to route cases to human review.",
    )
    _confidence_vs_score(state)

    T.section("Browse judge evaluations", "Every judgement, with its rationale and the response it scored.")
    _browse(state)

    T.section("Judge configuration", "The exact prompt and settings that produced these scores.")
    with st.expander("Judge system prompt"):
        T.note(
            "Note what this prompt does <strong>not</strong> ask for: private chain-of-thought. "
            "The judge returns scores and a short justification of the rating. Requesting hidden "
            "reasoning would add cost and latency without making the scores more auditable.",
            "info",
        )
        st.code(JUDGE_SYSTEM_PROMPT, language="text")

    with st.expander("Model selection and self-preference bias"):
        generator = (state.latest_run or {}).get("config", {}).get("model_version", "unknown")
        used = state.judge_stats.get("models_used") or []
        same = any(m == generator for m in used)
        st.markdown(f"""
The judge tries these models in order and records the one that actually ran:

{chr(10).join(f"{i + 1}. `{m}`" for i, m in enumerate(JUDGE_MODEL_CHAIN))}

**Generator model for the current run:** `{generator}`
**Judge model(s) actually used:** {", ".join(f"`{m}`" for m in used) or "—"}

A model evaluating its own output exhibits self-preference bias — it tends to rate text
produced by the same model more favourably than a third party would. The default chain
puts a larger, different model first for exactly this reason.
""")
        if same:
            T.note(
                "<strong>The judge and the generator are the same model in this run.</strong> "
                "Judge scores here should be read as an upper bound on quality: self-preference "
                "bias is unmeasured but present. Human-versus-judge agreement is the check that "
                "matters, and it is on the Alignment page.",
                "warn",
            )
        else:
            T.note(
                "The judge and generator are different models, which avoids the most direct form "
                "of self-preference bias. It does not eliminate shared-family bias.",
                "good",
            )
