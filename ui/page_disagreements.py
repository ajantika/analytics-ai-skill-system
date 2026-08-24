"""
ui/page_disagreements.py — Human ↔ AI Disagreements, and human rater calibration.

Two calibration problems live here, and they are different problems:

  Human ↔ AI     Where the automated judge and the raters reach different conclusions
                 about the same response, with an attributed cause and a concrete
                 change to the judge prompt.

  Human ↔ Human  Where raters disagree with each other. When this is large, the judge
                 is being compared against an unstable reference and no amount of judge
                 tuning will fix it. Diagnosing that before blaming the judge is the
                 point of keeping the two analyses on one page.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from alignment import (
    DISAGREEMENT_CAUSES,
    build_pairs,
    disagreement_cause_distribution,
    find_alignment_disagreements,
)
from failure_taxonomy import label_of
from human_evals import DIMENSIONS, find_disagreements
from ui import data as D
from ui import theme as T


def _cause_chart(disagreements: list[dict]) -> None:
    dist = disagreement_cause_distribution(disagreements)
    if not dist:
        return
    items = list(dist.items())[::-1]
    fig = go.Figure(go.Bar(
        x=[c for _, c in items],
        y=[k.replace("_", " ").title() for k, _ in items],
        orientation="h", marker_color=T.WARN, marker_line_width=0,
        text=[str(c) for _, c in items], textposition="outside",
        textfont=dict(color=T.MUTED, size=11),
        hovertemplate="%{y}<br>%{x} cases<extra></extra>",
    ))
    fig.update_xaxes(title_text="Cases", dtick=T.count_tick(max(c for _, c in items)))
    st.plotly_chart(T.style_fig(fig, height=max(200, 32 * len(items) + 60)), use_container_width=True)


def _disagreement_card(entry: dict, state: D.EvalState) -> None:
    analysis = entry["analysis"]
    case = entry.get("case") or {}
    response = state.response_for(entry["eval_id"])
    judge = entry["judge"]
    human = entry["human"]

    gap_summary = ""
    if analysis["dimension_gaps"]:
        top = analysis["dimension_gaps"][0]
        gap_summary = (
            f"{T.dim_label(top['dimension'])} — rater {top['human']:.0f}/5 vs judge {top['judge']}/5"
        )
    elif analysis["pass_conflict"]:
        gap_summary = "pass/fail inversion with no large dimension gap"

    header = (
        T.chip(entry["eval_id"], T.ACCENT, "rgba(99,102,241,0.13)")
        + (T.chip("PASS/FAIL INVERSION", T.BAD, "rgba(248,113,113,0.18)")
           if analysis["pass_conflict"] else "")
        + T.chip(analysis["primary_cause"].replace("_", " "), T.WARN, "rgba(234,179,8,0.13)")
        + (T.chip(f"max gap {analysis['max_gap']}", T.BAD) if analysis["max_gap"] else "")
    )

    title = f"{entry['eval_id']} — {gap_summary}"
    with st.expander(title):
        st.markdown(f'<div style="margin-bottom:10px">{header}</div>', unsafe_allow_html=True)

        st.markdown(
            T.panel(
                T.label_text("Question")
                + T.body_text(case.get("question", "(case not found)"), T.INK, "0.88rem"),
                accent=T.ACCENT,
            ),
            unsafe_allow_html=True,
        )

        if response and not response.get("error"):
            st.markdown(
                T.panel(
                    T.label_text("Model response", T.DIM)
                    + T.body_text(T.response_html(response["answer"]), T.TEXT, "0.79rem"),
                    accent=T.DIM,
                ),
                unsafe_allow_html=True,
            )

        left, right = st.columns(2, gap="medium")

        rater_types = human.get("rater_types", [])
        rater_kind = ("Human raters" if rater_types == ["human"]
                      else "Demo profiles — not human" if rater_types == ["demo_profile"]
                      else "Mixed raters")
        rater_color = T.GOOD if rater_types == ["human"] else T.WARN

        with left:
            rows = "".join(
                f'<div style="display:flex;justify-content:space-between;align-items:center;'
                f'padding:3px 0;font-size:0.77rem">'
                f'<span style="color:{T.DIMENSION_COLORS[d]}">{T.dim_label(d)}</span>'
                f'{T.score_bar(human["scores"].get(d))}</div>'
                for d in DIMENSIONS
            )
            notes = ""
            rater_notes = [
                a for a in state.annotations_for(entry["eval_id"]) if a.get("notes")
            ]
            if rater_notes:
                notes = (
                    '<div style="margin-top:9px;padding-top:7px;'
                    'border-top:1px solid rgba(255,255,255,0.06)">'
                    + T.label_text("Rater rationale", T.DIM)
                    + "".join(
                        T.body_text(f'<em>{a["evaluator_id"]}:</em> {a["notes"][:340]}',
                                    T.FAINT, "0.71rem")
                        for a in rater_notes[:3]
                    )
                    + "</div>"
                )
            st.markdown(
                T.panel(
                    T.label_text(rater_kind, rater_color)
                    + f'<div style="margin-bottom:7px">{T.pass_chip(entry["human_pass"])}'
                    + T.chip(f"overall {T.score(entry['human_overall'])}/5", T.MUTED)
                    + T.chip(f"{human.get('n_raters', 0)} rater(s)", T.FAINT) + "</div>"
                    + rows + notes,
                    accent=rater_color,
                ),
                unsafe_allow_html=True,
            )

        with right:
            rows = "".join(
                f'<div style="display:flex;justify-content:space-between;align-items:center;'
                f'padding:3px 0;font-size:0.77rem">'
                f'<span style="color:{T.DIMENSION_COLORS[d]}">{T.dim_label(d)}</span>'
                f'{T.score_bar(judge["scores"].get(d))}</div>'
                for d in DIMENSIONS
            )
            st.markdown(
                T.panel(
                    T.label_text("AI judge", T.WARN)
                    + f'<div style="margin-bottom:7px">{T.pass_chip(entry["judge_pass"])}'
                    + T.chip(f"overall {T.score(entry['judge_overall'])}/5", T.MUTED)
                    + T.chip(f"confidence {T.score(judge.get('confidence'))}", T.FAINT) + "</div>"
                    + rows
                    + '<div style="margin-top:9px;padding-top:7px;'
                      'border-top:1px solid rgba(255,255,255,0.06)">'
                    + T.label_text("Judge rationale", T.DIM)
                    + T.body_text(judge.get("reasoning_summary") or "(none)", T.FAINT, "0.71rem")
                    + "</div>",
                    accent=T.WARN,
                ),
                unsafe_allow_html=True,
            )

        modes_html = ""
        if analysis["human_failure_mode"] != analysis["judge_failure_mode"]:
            modes_html = (
                f'<div style="margin-top:7px">'
                f'{T.chip("rater: " + label_of(analysis["human_failure_mode"]), T.GOOD)}'
                f'{T.chip("judge: " + label_of(analysis["judge_failure_mode"]), T.WARN)}</div>'
            )

        st.markdown(
            T.panel(
                T.label_text("Likely cause", T.BAD)
                + T.body_text(analysis["cause_description"], T.TEXT, "0.8rem")
                + modes_html,
                accent=T.BAD,
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            T.panel(
                T.label_text("Recommended judge improvement", T.GOOD)
                + T.body_text(analysis["recommended_judge_improvement"], T.TEXT, "0.8rem"),
                accent=T.GOOD,
            ),
            unsafe_allow_html=True,
        )

        if case.get("expected_behavior"):
            st.markdown(
                f'<div style="color:{T.FAINT};font-size:0.72rem;line-height:1.6;margin-top:4px">'
                f'<strong style="color:{T.DIM}">Reference expected behaviour:</strong> '
                f'{case["expected_behavior"]}</div>',
                unsafe_allow_html=True,
            )


def _human_calibration(state: D.EvalState) -> None:
    calibration = state.calibration
    pairs = calibration.get("pairs", [])

    if not pairs:
        T.empty_state(
            "No rater pairs with overlapping cases",
            "Inter-rater agreement needs at least two raters who scored the same case. Rate some "
            "cases on the Human Evaluation page — the demo profiles have already rated every "
            "response, so your first rating creates a human-versus-demo pair immediately.",
        )
        return

    human_pairs = [p for p in pairs if p["pair_provenance"] == "human-human"]
    mixed_pairs = [p for p in pairs if p["pair_provenance"] == "human-demo"]
    demo_pairs = [p for p in pairs if p["pair_provenance"] == "demo-demo"]

    T.note(
        f"<strong>{calibration['n_human_raters']} human rater(s)</strong> and "
        f"<strong>{calibration['n_demo_raters']} demo profile(s)</strong>. "
        + ("Human-versus-human agreement is the figure that describes rubric quality. "
           if human_pairs else
           "No two humans have rated the same case yet, so human-versus-human agreement cannot "
           "be computed. Pairs involving a demo profile measure the rubric's mechanical "
           "applicability, not inter-annotator reliability. ")
        + "Demo profiles are scripted, not people.",
        "good" if human_pairs else "warn",
    )

    for group, title, kind in (
        (human_pairs, "Human ↔ Human", "good"),
        (mixed_pairs, "Human ↔ Demo profile", "info"),
        (demo_pairs, "Demo profile ↔ Demo profile", "info"),
    ):
        if not group:
            continue
        st.markdown(
            f'<div style="color:{T.DIM};font-size:0.7rem;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:0.08em;margin:14px 0 6px">{title}</div>',
            unsafe_allow_html=True,
        )
        for pair in group:
            pf = pair["pass_fail"]
            overall = pair["overall"]
            tiles = [
                T.metric_card("Shared cases", T.num(pair["n_shared"]),
                              footnote=f"{pair['rater_a']} ↔ {pair['rater_b']}", accent=T.ACCENT),
                T.metric_card("Overall within ±1", T.pct(overall.get("within_1")),
                              footnote="mean rubric score agreement", accent=T.GOOD,
                              value_color=T.rate_color(overall.get("within_1"), 0.85, 0.7)),
                T.metric_card("Pass/fail agreement", T.pct(pf.get("agreement")),
                              footnote=f"κ {T.score(pf.get('kappa'))}", accent=T.ACCENT_DEEP,
                              value_color=T.rate_color(pf.get("agreement"), 0.9, 0.75)),
                T.metric_card("Severity gap", T.signed(overall.get("mean_diff_b_minus_a")),
                              footnote=f"{pair['rater_b']} minus {pair['rater_a']} — "
                                       f"positive means {pair['rater_b']} is more lenient",
                              accent=T.WARN),
            ]
            T.metric_row(tiles)

            disagreements = find_disagreements(
                state.annotations, pair["rater_a"], pair["rater_b"], threshold=2
            )
            if not disagreements:
                T.note(
                    f"{pair['rater_a']} and {pair['rater_b']} never differ by 2 or more points on "
                    f"any dimension and never disagree on pass/fail across "
                    f"{pair['n_shared']} shared cases.",
                    "good",
                )
                continue

            st.markdown(
                f'<div style="color:{T.FAINT};font-size:0.73rem;margin:6px 0 4px">'
                f'{len(disagreements)} case(s) where these raters materially disagree</div>',
                unsafe_allow_html=True,
            )

            for d in disagreements[:6]:
                case = state.case_for(d["eval_id"]) or {}
                gap_desc = ", ".join(
                    f"{T.dim_label(g['dimension'])} {g['a']} vs {g['b']}"
                    for g in d["dimension_gaps"][:2]
                ) or "pass/fail only"

                with st.expander(f"{d['eval_id']} — {gap_desc}"):
                    st.markdown(
                        T.panel(
                            T.label_text("Question")
                            + T.body_text(case.get("question", ""), T.INK, "0.85rem"),
                            accent=T.ACCENT,
                        ),
                        unsafe_allow_html=True,
                    )
                    response = state.response_for(d["eval_id"])
                    if response and not response.get("error"):
                        with st.expander("Model response"):
                            st.markdown(response["answer"])

                    c1, c2 = st.columns(2, gap="medium")
                    for col, record, other_id in (
                        (c1, d["a_record"], pair["rater_a"]),
                        (c2, d["b_record"], pair["rater_b"]),
                    ):
                        rows = "".join(
                            f'<div style="display:flex;justify-content:space-between;'
                            f'padding:2px 0;font-size:0.76rem">'
                            f'<span style="color:{T.DIMENSION_COLORS[dim]}">{T.dim_label(dim)}</span>'
                            f'{T.score_bar(record["scores"].get(dim))}</div>'
                            for dim in DIMENSIONS
                        )
                        col.markdown(
                            T.panel(
                                T.provenance_chip(record.get("rater_type", ""))
                                + T.chip(other_id, T.MUTED)
                                + T.pass_chip(record.get("pass"))
                                + T.chip(f"conf {T.score(record.get('evaluator_confidence'))}", T.FAINT)
                                + f'<div style="margin-top:7px">{rows}</div>'
                                + (T.body_text(record.get("notes", "")[:280], T.FAINT, "0.7rem")
                                   if record.get("notes") else ""),
                                accent=T.GOOD if record.get("rater_type") == "human" else T.WARN,
                            ),
                            unsafe_allow_html=True,
                        )

                    st.markdown(
                        T.panel(
                            T.label_text("Likely reason", T.BAD)
                            + T.body_text(d["likely_reason"], T.TEXT, "0.79rem"),
                            accent=T.BAD,
                        ),
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        T.panel(
                            T.label_text("Suggested rubric clarification", T.GOOD)
                            + T.body_text(d["rubric_clarification"], T.TEXT, "0.79rem"),
                            accent=T.GOOD,
                        ),
                        unsafe_allow_html=True,
                    )


def render(state: D.EvalState) -> None:
    T.page_header(
        "Human ↔ AI Disagreements",
        "Where the automated judge and the raters reach different conclusions about the same "
        "response — and where raters disagree with each other. Aggregate agreement tells you "
        "whether there is a problem; only the individual cases tell you what it is.",
        eyebrow="Calibration",
    )

    tab_ai, tab_human = st.tabs(["Human ↔ AI judge", "Human evaluator calibration"])

    with tab_ai:
        if not state.has_judge or not state.has_any_annotations:
            T.empty_state(
                "Not enough evaluations to compare",
                "Disagreement analysis needs both judge results and rater annotations on the same "
                "responses.",
                D.GENERATE_COMMAND,
            )
        else:
            population = st.radio(
                "Rater population",
                options=["human", "demo_profile", None],
                index=0 if state.has_human else 1,
                format_func=lambda k: {
                    "human": "Human raters",
                    "demo_profile": "Demo profiles (not human)",
                    None: "Combined",
                }[k],
                horizontal=True,
                key="disagree_population",
            )
            pairs = build_pairs(state.annotations, state.judge_results, population)
            disagreements = find_alignment_disagreements(pairs, state.cases_by_id)

            if not pairs:
                T.empty_state(
                    "No overlapping evaluations for this population",
                    "No case has both a judge result and an annotation from this rater group.",
                )
            elif not disagreements:
                T.note(
                    f"No material disagreements across {len(pairs)} paired evaluations. Every "
                    f"dimension gap is under 2 points and no pass/fail decision is inverted. At "
                    f"this sample size that is worth treating with suspicion rather than "
                    f"satisfaction — check that the rubric is actually discriminating.",
                    "good",
                )
            else:
                inversions = sum(1 for d in disagreements if d["analysis"]["pass_conflict"])
                tiles = [
                    T.metric_card("Paired evaluations", T.num(len(pairs)),
                                  footnote="cases with both a rater and a judge score", accent=T.ACCENT),
                    T.metric_card("Material disagreements", T.num(len(disagreements)),
                                  footnote=f"{T.pct(len(disagreements) / len(pairs))} of paired cases",
                                  accent=T.WARN,
                                  value_color=T.rate_color(len(disagreements) / len(pairs),
                                                           0.85, 0.7, lower_is_better=True)),
                    T.metric_card("Pass/fail inversions", T.num(inversions),
                                  footnote="opposite accept/reject decision on the same response",
                                  accent=T.BAD, value_color=T.GOOD if not inversions else T.BAD),
                    T.metric_card("Distinct causes", T.num(len(disagreement_cause_distribution(disagreements))),
                                  footnote="structural reasons identified across these cases",
                                  accent=T.INFO),
                ]
                T.metric_row(tiles)

                T.section("Disagreement causes",
                          "Attributed from the two records by rule. Where no structural cause is "
                          "visible, the case is marked uncategorised rather than given a "
                          "plausible-sounding guess.")
                _cause_chart(disagreements)

                with st.expander("What each cause means"):
                    for key, desc in DISAGREEMENT_CAUSES.items():
                        st.markdown(f"**{key.replace('_', ' ').title()}** — {desc}")

                T.section("Individual disagreements",
                          "Sorted with pass/fail inversions first, then by the size of the largest "
                          "dimension gap.")
                for entry in disagreements[:12]:
                    _disagreement_card(entry, state)

                if len(disagreements) > 12:
                    st.markdown(
                        f'<div style="color:{T.FAINT};font-size:0.73rem">'
                        f'Showing 12 of {len(disagreements)} disagreements.</div>',
                        unsafe_allow_html=True,
                    )

    with tab_human:
        T.section(
            "Human evaluator calibration",
            "Subjective quality requires calibration. Two evaluators applying the same written "
            "rubric to the same response will still disagree, and where they disagree is a "
            "property of the rubric rather than of the evaluators.",
            top_rule=False,
        )
        _human_calibration(state)

        with st.expander("Why this analysis comes before judge tuning"):
            st.markdown("""
Human-versus-human agreement is the ceiling on human-versus-judge agreement. If two
raters applying the same rubric agree only 70% of the time, a judge that agrees with
either of them 70% of the time is performing as well as a human — and tuning it toward
higher agreement would be tuning it toward one rater's idiosyncrasies.

So the order of operations matters:

1. **Measure inter-rater agreement first.** Low agreement means the rubric is
   underspecified, not that the raters are careless.
2. **Fix the rubric where raters diverge.** Each disagreement above yields a specific
   clarification — an ordering rule, a worked example at a contested anchor, a statement
   of what a dimension excludes.
3. **Only then compare the judge against the reference.** With a stable reference,
   judge-versus-human gaps are attributable to the judge.

The `evaluator_confidence` field exists for this loop. Cases rated with low confidence
are where the rubric was hard to apply, and reviewing them as a batch is more productive
than resolving them one at a time as they arise.
""")
