"""
ui/page_failures.py — Failure Analysis.

A failure count is not actionable; a failure with a root cause and a named fix is. Each
failure here carries the full chain: question, response, rater verdict, judge verdict,
category, root cause, and the change that would address it.

The taxonomy itself is documented on this page, because a shared vocabulary is what makes
three independent evaluators comparable at all.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from alignment import agreement_by_failure_category, build_pairs
from failure_taxonomy import (
    FAILURE_MODES,
    SEVERITY_COLORS,
    SEVERITY_ORDER,
    all_modes,
    distribution,
    get_failure_mode,
    label_of,
    normalize_failure_mode,
    severity_distribution,
    severity_of,
)
from human_evals import DIMENSIONS
from ui import data as D
from ui import theme as T


def _collect_failures(state: D.EvalState) -> list[dict]:
    """One row per failing response, combining every evaluator's view of it."""
    rows = []
    for case in state.cases:
        judge = state.judge_for(case["eval_id"])
        annotations = state.annotations_for(case["eval_id"])
        det = state.deterministic_for(case["eval_id"])

        judge_mode = normalize_failure_mode(judge.get("failure_mode")) if judge and judge.get("parse_ok") else None
        human_modes = [normalize_failure_mode(a.get("failure_mode")) for a in annotations]
        human_failed = [m for m in human_modes if m != "none"]

        judge_failed = bool(judge_mode and judge_mode != "none")
        det_failed = bool(det and det["verdict"] == "FAIL")

        if not (judge_failed or human_failed or det_failed):
            continue

        primary = judge_mode if judge_failed else (human_failed[0] if human_failed else None)
        if not primary and det_failed:
            primary = "wrong_domain" if "expected_domain" in det["hard_failures"] else "incomplete_answer"

        rows.append({
            "eval_id": case["eval_id"],
            "case": case,
            "judge": judge,
            "annotations": annotations,
            "deterministic": det,
            "primary_mode": primary or "unclassified",
            "severity": severity_of(primary),
            "judge_mode": judge_mode,
            "human_modes": human_modes,
            "critical": bool(judge and judge.get("critical_failure")) or any(
                a.get("critical_failure") for a in annotations
            ),
            "detected_by": [
                tier for tier, hit in (
                    ("Deterministic", det_failed),
                    ("Human", any(a.get("rater_type") == "human" and
                                  normalize_failure_mode(a.get("failure_mode")) != "none"
                                  for a in annotations)),
                    ("Demo profile", any(a.get("rater_type") == "demo_profile" and
                                         normalize_failure_mode(a.get("failure_mode")) != "none"
                                         for a in annotations)),
                    ("AI judge", judge_failed),
                ) if hit
            ],
            "agreement": _agreement_status(judge_mode, human_modes),
        })
    return rows


def _agreement_status(judge_mode, human_modes) -> str:
    human_failed = [m for m in human_modes if m and m != "none"]
    if judge_mode is None or not human_modes:
        return "single evaluator"
    if bool(judge_mode != "none") != bool(human_failed):
        return "disputed"
    if human_failed and judge_mode != "none" and judge_mode not in human_failed:
        return "category mismatch"
    return "agreed"


def _summary_tiles(state: D.EvalState, failures: list[dict]) -> None:
    n_cases = len([c for c in state.cases if state.response_for(c["eval_id"])])
    critical = [f for f in failures if f["critical"]]
    disputed = [f for f in failures if f["agreement"] in ("disputed", "category mismatch")]

    tiles = [
        T.metric_card("Total failures", T.num(len(failures)),
                      footnote=f"responses with a failure recorded by at least one evaluator, "
                               f"of {n_cases} evaluated",
                      accent=T.WARN),
        T.metric_card("Critical failures", T.num(len(critical)),
                      footnote="would lead a reader to a wrong decision or an unsafe action",
                      target="Target 0", accent=T.BAD,
                      value_color=T.GOOD if not critical else T.BAD),
        T.metric_card("Failure rate", T.pct(len(failures) / n_cases if n_cases else None),
                      footnote="share of evaluated responses with any recorded failure",
                      accent=T.WARN,
                      value_color=T.rate_color(len(failures) / n_cases if n_cases else None,
                                               0.7, 0.5, lower_is_better=True)),
        T.metric_card("Disputed classifications", T.num(len(disputed)),
                      footnote="evaluators disagree on whether it failed, or on which category",
                      accent=T.INFO),
    ]
    T.metric_row(tiles)


def _category_chart(failures: list[dict]) -> None:
    dist = distribution([f["primary_mode"] for f in failures])
    if not dist:
        return
    items = list(dist.items())[::-1]
    fig = go.Figure(go.Bar(
        x=[c for _, c in items], y=[label_of(m) for m, _ in items], orientation="h",
        marker_color=[SEVERITY_COLORS.get(severity_of(m), T.DIM) for m, _ in items],
        marker_line_width=0,
        text=[str(c) for _, c in items], textposition="outside",
        textfont=dict(color=T.MUTED, size=11),
        hovertemplate="%{y}<br>%{x} responses<extra></extra>",
    ))
    fig.update_xaxes(title_text="Responses", dtick=T.count_tick(max(c for _, c in items)))
    st.plotly_chart(T.style_fig(fig, height=max(200, 32 * len(items) + 60)), use_container_width=True)


def _severity_chart(failures: list[dict]) -> None:
    counts = severity_distribution([f["primary_mode"] for f in failures])
    counts = {k: v for k, v in counts.items() if v and k != "none"}
    if not counts:
        return
    order = [s for s in SEVERITY_ORDER if s in counts]
    fig = go.Figure(go.Bar(
        x=[s.title() for s in order], y=[counts[s] for s in order],
        marker_color=[SEVERITY_COLORS[s] for s in order], marker_line_width=0,
        text=[str(counts[s]) for s in order], textposition="outside",
        textfont=dict(color=T.MUTED, size=11),
        hovertemplate="%{x}<br>%{y} responses<extra></extra>",
    ))
    fig.update_yaxes(title_text="Responses", dtick=T.count_tick(max(counts[s] for s in order)))
    st.plotly_chart(T.style_fig(fig, height=250), use_container_width=True)


def _agreement_by_category(state: D.EvalState) -> None:
    population = "human" if state.has_human else "demo_profile"
    pairs = build_pairs(state.annotations, state.judge_results, population)
    if not pairs:
        return
    grouped = agreement_by_failure_category(pairs)
    if not grouped:
        return

    label = "Human" if population == "human" else "Demo profile"
    rows = "".join(
        f'<tr>'
        f'<td style="padding:6px 0;color:{T.TEXT};width:28%">{label_of(m)}</td>'
        f'<td style="padding:6px 0;width:14%">{T.severity_chip(v["severity"])}</td>'
        f'<td style="padding:6px 0;color:{T.MUTED};width:8%">{v["n"]}</td>'
        f'<td style="padding:6px 0;color:{T.MUTED};width:13%">{T.score(v["human_mean"])}</td>'
        f'<td style="padding:6px 0;color:{T.MUTED};width:13%">{T.score(v["judge_mean"])}</td>'
        f'<td style="padding:6px 0;width:12%;font-weight:600;'
        f'color:{T.BAD if abs(v["mean_gap"] or 0) >= 0.75 else T.MUTED}">{T.signed(v["mean_gap"])}</td>'
        f'<td style="padding:6px 0;color:{T.MUTED}">{T.pct(v["pass_agreement"])}</td>'
        f'</tr>'
        for m, v in grouped.items()
    )
    st.markdown(
        f'<table style="width:100%;border-collapse:collapse;font-size:0.76rem">'
        f'<tr style="border-bottom:1px solid {T.BORDER}">'
        f'<th style="text-align:left;padding-bottom:5px;color:{T.DIM}">Failure category ({label.lower()}-assigned)</th>'
        f'<th style="text-align:left;padding-bottom:5px;color:{T.DIM}">Severity</th>'
        f'<th style="text-align:left;padding-bottom:5px;color:{T.DIM}">n</th>'
        f'<th style="text-align:left;padding-bottom:5px;color:{T.DIM}">{label}</th>'
        f'<th style="text-align:left;padding-bottom:5px;color:{T.DIM}">Judge</th>'
        f'<th style="text-align:left;padding-bottom:5px;color:{T.DIM}">Gap</th>'
        f'<th style="text-align:left;padding-bottom:5px;color:{T.DIM}">Pass agreement</th>'
        f'</tr>{rows}</table>',
        unsafe_allow_html=True,
    )
    T.note(
        "A large positive gap means the judge rates that failure category more favourably than "
        "raters do — the categories where the judge is least likely to catch a real problem.",
        "info",
    )


def _failure_card(failure: dict, state: D.EvalState) -> None:
    case = failure["case"]
    judge = failure["judge"]
    det = failure["deterministic"]
    entry = get_failure_mode(failure["primary_mode"])
    response = state.response_for(case["eval_id"])

    header = (
        T.chip(case["eval_id"], T.ACCENT, "rgba(99,102,241,0.13)")
        + T.severity_chip(failure["severity"])
        + T.chip(entry["label"], T.BAD, "rgba(248,113,113,0.13)")
        + (T.chip("CRITICAL", T.BAD, "rgba(248,113,113,0.2)") if failure["critical"] else "")
        + T.chip(failure["agreement"], T.WARN if failure["agreement"] != "agreed" else T.GOOD)
        + "".join(T.chip(t, T.FAINT) for t in failure["detected_by"])
    )

    with st.expander(f"{case['eval_id']} — {entry['label']} — {case['question'][:66]}"):
        st.markdown(f'<div style="margin-bottom:10px">{header}</div>', unsafe_allow_html=True)

        st.markdown(
            T.panel(T.label_text("Question") + T.body_text(case["question"], T.INK, "0.88rem"),
                    accent=T.ACCENT),
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

        cols = st.columns(2, gap="medium")
        with cols[0]:
            if failure["annotations"]:
                for a in failure["annotations"]:
                    rows = "".join(
                        f'<div style="display:flex;justify-content:space-between;padding:2px 0;'
                        f'font-size:0.75rem">'
                        f'<span style="color:{T.DIMENSION_COLORS[d]}">{T.dim_label(d)}</span>'
                        f'{T.score_bar(a["scores"].get(d), width=54)}</div>'
                        for d in DIMENSIONS
                    )
                    st.markdown(
                        T.panel(
                            T.provenance_chip(a.get("rater_type", ""))
                            + T.chip(a["evaluator_id"], T.MUTED)
                            + T.pass_chip(a.get("pass"))
                            + T.chip(label_of(a.get("failure_mode")), T.BAD
                                     if a.get("failure_mode") != "none" else T.GOOD)
                            + f'<div style="margin-top:7px">{rows}</div>',
                            accent=T.GOOD if a.get("rater_type") == "human" else T.WARN,
                        ),
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    T.panel(T.label_text("Rater evaluation", T.DIM)
                            + T.body_text("Not rated.", T.FAINT, "0.78rem"), accent=T.DIM),
                    unsafe_allow_html=True,
                )

        with cols[1]:
            if judge and judge.get("parse_ok"):
                rows = "".join(
                    f'<div style="display:flex;justify-content:space-between;padding:2px 0;'
                    f'font-size:0.75rem">'
                    f'<span style="color:{T.DIMENSION_COLORS[d]}">{T.dim_label(d)}</span>'
                    f'{T.score_bar(judge["scores"].get(d), width=54)}</div>'
                    for d in DIMENSIONS
                )
                st.markdown(
                    T.panel(
                        T.label_text("AI judge", T.WARN)
                        + T.pass_chip(judge["pass"])
                        + T.chip(label_of(judge["failure_mode"]),
                                 T.BAD if judge["failure_mode"] != "none" else T.GOOD)
                        + f'<div style="margin-top:7px">{rows}</div>'
                        + T.body_text(judge.get("reasoning_summary", ""), T.FAINT, "0.71rem"),
                        accent=T.WARN,
                    ),
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    T.panel(T.label_text("AI judge", T.DIM)
                            + T.body_text("Not evaluated.", T.FAINT, "0.78rem"), accent=T.DIM),
                    unsafe_allow_html=True,
                )

        if det and det["hard_failures"]:
            checks = "".join(
                f'<div style="font-size:0.75rem;color:{T.TEXT};padding:2px 0">'
                f'{T.status_chip(det["checks"][name]["status"])} '
                f'<span style="color:{T.FAINT}">{det["checks"][name]["detail"]}</span></div>'
                for name in det["hard_failures"] if name in det["checks"]
            )
            st.markdown(
                T.panel(T.label_text("Deterministic checks failed", T.BAD) + checks, accent=T.BAD),
                unsafe_allow_html=True,
            )

        st.markdown(
            T.panel(
                T.label_text("Root cause", T.BAD)
                + T.body_text(entry["likely_cause"], T.TEXT, "0.8rem"),
                accent=T.BAD,
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            T.panel(
                T.label_text("Recommended fix", T.GOOD)
                + T.body_text(entry["remediation"], T.TEXT, "0.8rem"),
                accent=T.GOOD,
            ),
            unsafe_allow_html=True,
        )

        designed = normalize_failure_mode(case.get("expected_failure_mode"))
        if designed != "none":
            caught = designed == failure["primary_mode"]
            T.note(
                f"This case was designed to elicit <strong>{label_of(designed)}</strong>. "
                + ("The evaluators classified it as designed — the case did its job."
                   if caught else
                   f"The evaluators classified it as <strong>{entry['label']}</strong> instead. "
                   f"Either the response failed differently than anticipated, or the case's "
                   f"expected failure mode needs revisiting."),
                "good" if caught else "warn",
            )


def _taxonomy_reference() -> None:
    for severity in SEVERITY_ORDER:
        modes = [m for m in all_modes() if FAILURE_MODES[m]["severity"] == severity]
        if not modes:
            continue
        st.markdown(
            f'<div style="margin:14px 0 6px">{T.severity_chip(severity)}'
            f'<span style="color:{T.FAINT};font-size:0.72rem;margin-left:6px">'
            f'{len(modes)} categor{"y" if len(modes) == 1 else "ies"}</span></div>',
            unsafe_allow_html=True,
        )
        for mode in modes:
            entry = FAILURE_MODES[mode]
            tiers = "".join(T.chip(t.replace("_", " "), T.FAINT) for t in entry["detected_by"])
            st.markdown(
                T.panel(
                    f'<div style="display:flex;align-items:baseline;gap:8px;flex-wrap:wrap;'
                    f'margin-bottom:6px">'
                    f'<span style="color:{T.INK};font-size:0.85rem;font-weight:700">{entry["label"]}</span>'
                    f'<code style="font-size:0.68rem">{mode}</code>{tiers}</div>'
                    + T.body_text(entry["description"], T.TEXT, "0.79rem")
                    + (f'<div style="margin-top:7px;color:{T.FAINT};font-size:0.74rem;'
                       f'line-height:1.6"><strong style="color:{T.DIM}">Example:</strong> '
                       f'{entry["example"]}</div>' if entry["example"] else "")
                    + (f'<div style="margin-top:5px;color:{T.FAINT};font-size:0.74rem;'
                       f'line-height:1.6"><strong style="color:{T.DIM}">Likely cause:</strong> '
                       f'{entry["likely_cause"]}</div>' if entry["likely_cause"] else "")
                    + (f'<div style="margin-top:5px;color:{T.FAINT};font-size:0.74rem;'
                       f'line-height:1.6"><strong style="color:{T.DIM}">Remediation:</strong> '
                       f'{entry["remediation"]}</div>' if entry["remediation"] else ""),
                    accent=SEVERITY_COLORS.get(severity, T.DIM),
                ),
                unsafe_allow_html=True,
            )


def render(state: D.EvalState) -> None:
    T.page_header(
        "Failure Analysis",
        "Every recorded failure with its category, root cause and the specific change that would "
        "address it. A failure count without a cause is a status report; this page is the input "
        "to the next iteration.",
        eyebrow="Failure taxonomy",
    )

    tab_analysis, tab_taxonomy = st.tabs(["Failure analysis", "Taxonomy reference"])

    with tab_taxonomy:
        T.section(
            "Failure mode taxonomy",
            "The shared vocabulary. Deterministic checks, the human rubric and the AI judge all "
            "classify into these categories, which is what makes their outputs comparable. "
            "Severity is a property of the category, so critical-failure rate means the same "
            "thing regardless of which evaluator found it.",
            top_rule=False,
        )
        _taxonomy_reference()

    with tab_analysis:
        if not state.has_responses:
            T.empty_state("No evaluation run found", D.NO_RUN_EXPLANATION, D.GENERATE_COMMAND)
            return

        failures = _collect_failures(state)

        if not failures:
            T.note(
                "No failures recorded by any evaluator across the evaluated responses. On a set "
                "designed to be adversarial, that is a reason to check the evaluators before "
                "celebrating the model.",
                "good",
            )
            return

        _summary_tiles(state, failures)

        T.section("Filters", top_rule=True)
        c1, c2, c3, c4 = st.columns(4)
        domains = sorted({f["case"]["expected_domain"] or f["case"]["domain"] for f in failures})
        severities = [s for s in SEVERITY_ORDER if any(f["severity"] == s for f in failures)]
        modes = sorted({f["primary_mode"] for f in failures})
        agreements = sorted({f["agreement"] for f in failures})

        sel_domain = c1.multiselect("Domain", domains, default=[],
                                    format_func=D.domain_label, key="fail_domain")
        sel_severity = c2.multiselect("Severity", severities, default=[],
                                      format_func=str.title, key="fail_severity")
        sel_mode = c3.multiselect("Failure type", modes, default=[],
                                  format_func=label_of, key="fail_mode")
        sel_agreement = c4.multiselect("Evaluator agreement", agreements, default=[],
                                       key="fail_agreement")

        filtered = failures
        if sel_domain:
            filtered = [f for f in filtered
                        if (f["case"]["expected_domain"] or f["case"]["domain"]) in sel_domain]
        if sel_severity:
            filtered = [f for f in filtered if f["severity"] in sel_severity]
        if sel_mode:
            filtered = [f for f in filtered if f["primary_mode"] in sel_mode]
        if sel_agreement:
            filtered = [f for f in filtered if f["agreement"] in sel_agreement]

        left, right = st.columns([3, 2], gap="medium")
        with left:
            T.section("Failure categories", top_rule=False)
            _category_chart(filtered)
        with right:
            T.section("Severity distribution", top_rule=False)
            _severity_chart(filtered)

        T.section(
            "Judge agreement by failure category",
            "Where the automated judge tracks rater judgement, and where it does not.",
        )
        _agreement_by_category(state)

        T.section(
            f"Failure detail — {len(filtered)} case(s)",
            "Question → response → rater evaluation → judge evaluation → category → root cause → fix.",
        )
        order = {s: i for i, s in enumerate(SEVERITY_ORDER)}
        for failure in sorted(filtered, key=lambda f: (order.get(f["severity"], 9),
                                                       not f["critical"], f["eval_id"])):
            _failure_card(failure, state)
