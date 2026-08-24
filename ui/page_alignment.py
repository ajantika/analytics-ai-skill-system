"""
ui/page_alignment.py — Human ↔ AI Alignment.

The page the whole architecture exists to make possible. An automated judge is only
worth its cost to the extent it reproduces human judgement, and that is an empirical
question with an answer, not an assumption.

Provenance is selectable and always labelled. The human-rater view is the one that
matters; the demo-profile view exists so the methodology is demonstrable before any
human has rated anything, and it is never presented as human judgement.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from alignment import build_pairs
from human_evals import DIMENSIONS
from stats_utils import kappa_interpretation
from ui import data as D
from ui import theme as T


def _population_selector(state: D.EvalState) -> tuple[str, dict, list[dict]]:
    """Choose which annotation population the judge is compared against."""
    options = []
    if state.has_human:
        options.append(("human_only", f"Human raters ({state.coverage['human_annotated']} cases)"))
    if state.has_demo:
        options.append(("demo_only", f"Demo profiles — not human ({state.coverage['demo_annotated']} cases)"))
    if state.has_human and state.has_demo:
        options.append(("combined", "Combined (human + demo pooled)"))

    if not options:
        return "human_only", state.alignment["human_only"], []

    keys = [k for k, _ in options]
    labels = {k: v for k, v in options}
    default = 0

    choice = st.radio(
        "Compare the AI judge against",
        options=keys,
        index=default,
        format_func=lambda k: labels[k],
        horizontal=True,
        key="alignment_population",
    )

    rater_type = {"human_only": "human", "demo_only": "demo_profile", "combined": None}[choice]
    pairs = build_pairs(state.annotations, state.judge_results, rater_type)
    return choice, state.alignment[choice], pairs


def _provenance_banner(choice: str, metrics: dict) -> None:
    if choice == "human_only":
        T.note(
            f"<strong>{metrics['label']}.</strong> These figures compare the AI judge against "
            f"ratings authored by a person applying the rubric. This is the measurement that "
            f"determines whether the judge can be trusted.",
            "good",
        )
    elif choice == "demo_only":
        T.note(
            "<strong>Demo profiles are not human annotators.</strong> They are scripted rubric "
            "applications that read the real generated responses and apply explicit anchor rules "
            "with a fixed leniency bias. They exist so the calibration workflow is demonstrable "
            "before human ratings exist. Agreement figures below measure the judge against a "
            "rule-based rater, not against human judgement — read them as a methodology "
            "demonstration, not as validation of the judge.",
            "warn",
        )
    else:
        T.note(
            "<strong>Pooled population.</strong> Human annotations and scripted demo profiles are "
            "combined here, which mixes two different kinds of rater. Use the human-only view for "
            "any claim about whether the judge matches human judgement.",
            "warn",
        )


def _headline(metrics: dict, choice: str) -> None:
    n = metrics.get("n", 0)
    tiles = [
        T.metric_card("Responses compared", T.num(n),
                      footnote="cases with both a rater score and a parsed judge score",
                      accent=T.ACCENT),
        T.metric_card("Exact agreement", T.pct(metrics.get("exact_agreement")),
                      footnote="identical overall score after rounding the rater consensus",
                      accent=T.INFO,
                      value_color=T.rate_color(metrics.get("exact_agreement"), 0.6, 0.4)),
        T.metric_card("Agreement within ±1", T.pct(metrics.get("within_1_agreement")),
                      footnote="the figure that matters for an ordinal rubric",
                      target="Target ≥85%", accent=T.GOOD,
                      value_color=T.rate_color(metrics.get("within_1_agreement"), 0.85, 0.70)),
        T.metric_card("Pass/fail agreement", T.pct(metrics.get("pass_agreement")),
                      footnote="same accept/reject decision on the same response",
                      target="Target ≥90%", accent=T.ACCENT_DEEP,
                      value_color=T.rate_color(metrics.get("pass_agreement"), 0.90, 0.75)),
    ]
    T.metric_row(tiles)

    st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)

    kappa = metrics.get("kappa_pass")
    tiles2 = [
        T.metric_card("Cohen's κ (pass/fail)",
                      T.score(kappa) if kappa is not None else T.DASH,
                      footnote=f"chance-corrected · {kappa_interpretation(kappa)}",
                      accent=T.WARN,
                      value_color=T.rate_color(kappa, 0.6, 0.4) if kappa is not None else T.DIM),
        T.metric_card("Cohen's κ (scores, weighted)",
                      T.score(metrics.get("kappa_overall_quadratic")),
                      footnote="quadratic weighting — the right statistic for ordinal 1-5 scores",
                      accent=T.WARN),
        T.metric_card("Pearson r", T.score(metrics.get("pearson")),
                      footnote="linear correlation of overall scores",
                      accent=T.INFO),
        T.metric_card("Spearman ρ", T.score(metrics.get("spearman")),
                      footnote="rank correlation — does the judge order responses as humans do?",
                      accent=T.INFO),
    ]
    T.metric_row(tiles2)

    if n and n < 15:
        T.note(
            f"<strong>n = {n}.</strong> Chance-corrected and correlation statistics are unstable "
            f"at this sample size — a single case can move Cohen's κ by more than 0.1. Treat these "
            f"as directional. Rating more cases on the Human Evaluation page is the fix.",
            "warn",
        )


def _scatter(pairs: list[dict], choice: str) -> None:
    if len(pairs) < 3:
        T.note("At least three paired evaluations are needed to plot a relationship.", "info")
        return

    human = [p["human_overall"] for p in pairs if p["human_overall"] is not None]
    judge = [p["judge_overall"] for p in pairs if p["judge_overall"] is not None]
    ids = [p["eval_id"] for p in pairs
           if p["human_overall"] is not None and p["judge_overall"] is not None]

    colors = []
    for p in pairs:
        if p["human_overall"] is None or p["judge_overall"] is None:
            continue
        if p["human_pass"] != p["judge_pass"]:
            colors.append(T.BAD)
        elif abs(p["judge_overall"] - p["human_overall"]) >= 1:
            colors.append(T.WARN)
        else:
            colors.append(T.GOOD)

    human_label = "Demo profile score" if choice == "demo_only" else "Human score"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[1, 5], y=[1, 5], mode="lines",
        line=dict(color="rgba(255,255,255,0.16)", dash="dash", width=1),
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=human, y=judge, mode="markers",
        marker=dict(size=11, color=colors, opacity=0.82,
                    line=dict(width=1, color="rgba(255,255,255,0.22)")),
        text=ids,
        hovertemplate="%{text}<br>" + human_label.lower() + " %{x:.2f}<br>"
                      "judge %{y:.2f}<extra></extra>",
        showlegend=False,
    ))
    fig.update_xaxes(title_text=f"{human_label} (overall, 1-5)", range=[0.7, 5.3], dtick=1)
    fig.update_yaxes(title_text="AI judge score (overall, 1-5)", range=[0.7, 5.3], dtick=1)
    st.plotly_chart(T.style_fig(fig, height=380), use_container_width=True)

    st.markdown(
        f'<div style="font-size:0.72rem;color:{T.DIM};margin-top:-10px">'
        + T.chip("agree within 1 point", T.GOOD, "rgba(34,197,94,0.13)")
        + T.chip("differ by 1 point or more", T.WARN, "rgba(234,179,8,0.13)")
        + T.chip("pass/fail inversion", T.BAD, "rgba(248,113,113,0.13)")
        + f' &nbsp; Points above the diagonal are cases the judge scored higher than the rater.'
        f'</div>',
        unsafe_allow_html=True,
    )


def _confusion(metrics: dict, choice: str) -> None:
    matrix = metrics.get("pass_confusion")
    rates = metrics.get("pass_rates", {})
    if not matrix:
        return

    human_label = "Demo profile" if choice == "demo_only" else "Human"
    cells = [
        # (row, col, label, value, colour, note)
        (0, 0, f"{human_label} FAIL / Judge FAIL", matrix[0][0], T.GOOD, "agreed rejection"),
        (0, 1, f"{human_label} FAIL / Judge PASS", matrix[0][1], T.BAD,
         "the costly error — judge accepts a response the rater rejected"),
        (1, 0, f"{human_label} PASS / Judge FAIL", matrix[1][0], T.WARN,
         "judge rejects a response the rater accepted"),
        (1, 1, f"{human_label} PASS / Judge PASS", matrix[1][1], T.GOOD, "agreed acceptance"),
    ]

    cols = st.columns(2, gap="small")
    for i, (_, _, label, value, color, note_text) in enumerate(cells):
        cols[i % 2].markdown(
            f'<div style="background:{T.SURFACE};border:1px solid {T.BORDER};'
            f'border-left:3px solid {color};border-radius:10px;padding:12px 14px;'
            f'margin-bottom:8px">'
            f'<div style="color:{T.DIM};font-size:0.68rem;font-weight:600">{label}</div>'
            f'<div style="color:{color};font-size:1.5rem;font-weight:800;margin:3px 0">{value}</div>'
            f'<div style="color:{T.FAINT};font-size:0.66rem;line-height:1.5">{note_text}</div></div>',
            unsafe_allow_html=True,
        )

    if rates.get("false_pass"):
        T.note(
            f"<strong>{rates['false_pass']} false pass(es).</strong> The judge accepted responses "
            f"the rater rejected. In an automated quality gate these are the failures that ship: "
            f"a false fail costs a review cycle, a false pass costs a wrong answer in front of a "
            f"user. Every one is listed on the Disagreements page.",
            "warn",
        )


def _by_dimension(metrics: dict, choice: str) -> None:
    by_dim = metrics.get("by_dimension", {})
    scored = [d for d in DIMENSIONS if by_dim.get(d, {}).get("n", 0) > 0]
    if not scored:
        return

    human_label = "Demo profile" if choice == "demo_only" else "Human"

    fig = go.Figure()
    fig.add_bar(
        name="Agreement within ±1",
        x=[T.dim_label(d) for d in scored],
        y=[by_dim[d]["within_1"] for d in scored],
        marker_color=[T.DIMENSION_COLORS[d] for d in scored],
        marker_line_width=0,
        text=[T.pct(by_dim[d]["within_1"]) for d in scored],
        textposition="outside", textfont=dict(color=T.MUTED, size=11),
        hovertemplate="%{x}<br>within ±1: %{y:.0%}<extra></extra>",
    )
    fig.update_yaxes(range=[0, 1.15], tickformat=".0%", title_text="Agreement within ±1")
    st.plotly_chart(T.style_fig(fig, height=280), use_container_width=True)

    rows = "".join(
        f'<tr>'
        f'<td style="padding:6px 0;color:{T.DIMENSION_COLORS[d]};font-weight:600;width:20%">'
        f'{T.dim_label(d)}</td>'
        f'<td style="padding:6px 0;color:{T.MUTED};width:10%">{by_dim[d]["n"]}</td>'
        f'<td style="padding:6px 0;color:{T.MUTED};width:12%">{T.pct(by_dim[d]["exact"])}</td>'
        f'<td style="padding:6px 0;color:{T.MUTED};width:12%">{T.pct(by_dim[d]["within_1"])}</td>'
        f'<td style="padding:6px 0;color:{T.MUTED};width:14%">{T.score(by_dim[d]["kappa_quadratic"])}</td>'
        f'<td style="padding:6px 0;color:{T.MUTED};width:11%">{T.score(by_dim[d]["human_mean"])}</td>'
        f'<td style="padding:6px 0;color:{T.MUTED};width:11%">{T.score(by_dim[d]["judge_mean"])}</td>'
        f'<td style="padding:6px 0;font-weight:600;'
        f'color:{T.BAD if abs(by_dim[d]["mean_diff"] or 0) >= 0.5 else T.MUTED}">'
        f'{T.signed(by_dim[d]["mean_diff"])}</td>'
        f'</tr>'
        for d in scored
    )
    st.markdown(
        f'<table style="width:100%;border-collapse:collapse;font-size:0.76rem;margin-top:6px">'
        f'<tr style="border-bottom:1px solid {T.BORDER}">'
        f'<th style="text-align:left;padding-bottom:5px;color:{T.DIM}">Dimension</th>'
        f'<th style="text-align:left;padding-bottom:5px;color:{T.DIM}">n</th>'
        f'<th style="text-align:left;padding-bottom:5px;color:{T.DIM}">Exact</th>'
        f'<th style="text-align:left;padding-bottom:5px;color:{T.DIM}">Within ±1</th>'
        f'<th style="text-align:left;padding-bottom:5px;color:{T.DIM}">κ (weighted)</th>'
        f'<th style="text-align:left;padding-bottom:5px;color:{T.DIM}">{human_label}</th>'
        f'<th style="text-align:left;padding-bottom:5px;color:{T.DIM}">Judge</th>'
        f'<th style="text-align:left;padding-bottom:5px;color:{T.DIM}">Judge − rater</th>'
        f'</tr>{rows}</table>',
        unsafe_allow_html=True,
    )


def _bias(metrics: dict) -> None:
    bias = metrics.get("bias", {})
    if not bias or bias.get("direction") == "not defined":
        T.note("Not enough paired evaluations to assess judge bias.", "info")
        return

    direction = bias["direction"]
    color = {"lenient": T.WARN, "severe": T.INFO, "calibrated": T.GOOD}.get(direction, T.DIM)

    st.markdown(
        T.panel(
            T.label_text(f"Overall: the judge is {direction}", color)
            + T.body_text(bias["statement"], T.TEXT, "0.84rem"),
            accent=color,
        ),
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2, gap="medium")

    with col1:
        lenient = bias.get("lenient_dimensions", [])
        severe = bias.get("severe_dimensions", [])
        html = ""
        if lenient:
            html += T.label_text("Judge scores higher than raters", T.WARN)
            html += "".join(
                f'<div style="color:{T.TEXT};font-size:0.78rem;padding:2px 0">'
                f'{T.dim_label(d["dimension"])} <strong style="color:{T.WARN}">'
                f'{T.signed(d["gap"])}</strong></div>' for d in lenient
            )
        if severe:
            html += ('<div style="height:8px"></div>' if lenient else "")
            html += T.label_text("Judge scores lower than raters", T.INFO)
            html += "".join(
                f'<div style="color:{T.TEXT};font-size:0.78rem;padding:2px 0">'
                f'{T.dim_label(d["dimension"])} <strong style="color:{T.INFO}">'
                f'{T.signed(d["gap"])}</strong></div>' for d in severe
            )
        if not html:
            html = (T.label_text("Directional bias by dimension")
                    + T.body_text("No dimension shows a mean gap above 0.3 points.", T.TEXT, "0.78rem"))
        st.markdown(T.panel(html, accent=T.WARN), unsafe_allow_html=True)

    with col2:
        weak = bias.get("least_reliable", [])
        strong = bias.get("most_reliable", [])
        html = ""
        if weak:
            html += T.label_text("Least reliable dimensions", T.BAD)
            html += "".join(
                f'<div style="color:{T.TEXT};font-size:0.78rem;padding:2px 0">'
                f'{T.dim_label(d["dimension"])} <strong style="color:{T.BAD}">'
                f'κ {T.score(d["kappa"])}</strong> '
                f'<span style="color:{T.FAINT}">({kappa_interpretation(d["kappa"])})</span></div>'
                for d in weak
            )
        if strong:
            html += '<div style="height:8px"></div>'
            html += T.label_text("Most reliable dimensions", T.GOOD)
            html += "".join(
                f'<div style="color:{T.TEXT};font-size:0.78rem;padding:2px 0">'
                f'{T.dim_label(d["dimension"])} <strong style="color:{T.GOOD}">'
                f'κ {T.score(d["kappa"])}</strong> '
                f'<span style="color:{T.FAINT}">({kappa_interpretation(d["kappa"])})</span></div>'
                for d in strong
            )
        if not html:
            html = (T.label_text("Reliability by dimension")
                    + T.body_text("Chance-corrected agreement is not defined at this sample size.",
                                  T.TEXT, "0.78rem"))
        st.markdown(T.panel(html, accent=T.INFO), unsafe_allow_html=True)

    weak = bias.get("least_reliable", [])
    if weak and weak[0].get("kappa") is not None and weak[0]["kappa"] < 0.4:
        dim = T.dim_label(weak[0]["dimension"])
        T.note(
            f"<strong>Practical implication.</strong> Judge agreement on <em>{dim}</em> is "
            f"{kappa_interpretation(weak[0]['kappa'])}. Judge scores for that dimension should "
            f"not be used to gate releases on their own — either route those cases to human "
            f"review, or fix the anchors for that dimension in both the rubric and the judge "
            f"prompt and re-measure.",
            "warn",
        )


def render(state: D.EvalState) -> None:
    T.page_header(
        "Human ↔ AI Alignment",
        "How closely the automated judge reproduces rater judgement on the same responses. An "
        "automated evaluator that has not been checked against human judgement is an unvalidated "
        "instrument, and every quality number downstream of it inherits that.",
        eyebrow="Judge validation",
    )

    if not state.has_judge:
        T.empty_state("No judge evaluations found", D.NO_RUN_EXPLANATION, D.GENERATE_COMMAND)
        return

    if not state.has_any_annotations:
        T.empty_state("No rater annotations found", D.NO_HUMAN_EXPLANATION)
        return

    choice, metrics, pairs = _population_selector(state)
    _provenance_banner(choice, metrics)

    if not metrics.get("n"):
        T.empty_state(
            "No overlapping evaluations",
            "There are annotations and there are judge results, but no case has both. Agreement "
            "can only be computed where the same response was evaluated by both.",
        )
        return

    _headline(metrics, choice)

    T.section(
        "Score-for-score comparison",
        "Each point is one response. Perfect agreement lies on the diagonal.",
    )
    _scatter(pairs, choice)

    T.section(
        "Pass/fail confusion matrix",
        "The accept/reject decision, which is what an automated quality gate would act on.",
    )
    _confusion(metrics, choice)

    T.section(
        "Agreement by rubric dimension",
        "Aggregate agreement hides the useful signal. Judges typically track humans well on "
        "objective dimensions and poorly on subjective ones — and that difference determines "
        "which dimensions can be automated.",
    )
    _by_dimension(metrics, choice)

    T.section(
        "Judge bias analysis",
        "Direction and magnitude of the judge's systematic error, and where it is least reliable.",
    )
    _bias(metrics)

    with st.expander("How each statistic here is computed, and why that one"):
        st.markdown("""
**Rater consensus.** Where several raters scored the same response, their scores are
averaged per dimension before comparison. Pass/fail uses a majority, with ties resolved
to FAIL — an unresolved disagreement about whether an answer is acceptable should not be
recorded as acceptable.

**Exact agreement** rounds the rater consensus mean to the nearest integer, because judge
scores are integers and consensus means are not. This is stated on every dimension record
rather than left implicit, since it makes exact agreement look slightly better than a
strict comparison would.

**Within ±1** is the headline agreement figure. On a 1-5 ordinal rubric, a 4-versus-5
disagreement is close to noise while a 2-versus-5 is a substantive conflict, and exact
agreement treats those identically.

**Cohen's κ** corrects for agreement that would occur by chance given each rater's
marginal distribution. Two evaluators who both pass 90% of responses will agree 82% of the
time while knowing nothing about each other; κ removes that floor. Pass/fail uses nominal
κ; the 1-5 scores use quadratic weighting, which penalises a 1-versus-5 disagreement
sixteen times more than a 4-versus-5.

**Pearson and Spearman** answer different questions. Pearson asks whether the judge's
scores track human scores linearly. Spearman asks whether it *ranks* responses the same
way, which is what matters if the judge is used to triage the worst cases for review.

**Undefined statistics render as an em-dash.** κ is undefined when both raters used a
single category; correlation is undefined when either series is constant or n < 3. None
of these is reported as zero, because zero means "no agreement beyond chance" and that is
a different claim from "not enough data to say".
""")
