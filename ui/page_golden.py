"""
ui/page_golden.py — Golden Dataset explorer.

The measuring instrument, made inspectable. Anyone reading a quality number should be
able to see exactly what was tested, what a correct answer was defined to contain, and
how each evaluator scored it.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from evals import load_golden_set, validate_golden_set
from failure_taxonomy import SEVERITY_COLORS, SEVERITY_ORDER, label_of, normalize_failure_mode
from human_evals import DIMENSIONS
from skills import build_context
from ui import data as D
from ui import theme as T


def _composition_charts(cases: list[dict]) -> None:
    left, right = st.columns(2, gap="medium")

    with left:
        counts = {}
        for c in cases:
            key = c["test_type"]
            counts[key] = counts.get(key, 0) + 1
        items = sorted(counts.items(), key=lambda kv: kv[1])
        fig = go.Figure(go.Bar(
            x=[v for _, v in items],
            y=[D.TEST_TYPE_LABELS.get(k, k) for k, _ in items],
            orientation="h", marker_color=T.ACCENT, marker_line_width=0,
            text=[str(v) for _, v in items], textposition="outside",
            textfont=dict(color=T.MUTED, size=11),
            hovertemplate="%{y}<br>%{x} cases<extra></extra>",
        ))
        fig.update_xaxes(title_text="Cases", dtick=T.count_tick(max(v for _, v in items)))
        st.markdown(
            f'<div style="color:{T.DIM};font-size:0.72rem;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:0.08em;margin-bottom:4px">By test type</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(T.style_fig(fig, height=260), use_container_width=True)

    with right:
        counts = {}
        for c in cases:
            key = c["expected_domain"] or "cross_domain"
            counts[key] = counts.get(key, 0) + 1
        items = sorted(counts.items(), key=lambda kv: kv[1])
        fig = go.Figure(go.Bar(
            x=[v for _, v in items], y=[D.domain_label(k) for k, _ in items],
            orientation="h", marker_color=T.INFO, marker_line_width=0,
            text=[str(v) for _, v in items], textposition="outside",
            textfont=dict(color=T.MUTED, size=11),
            hovertemplate="%{y}<br>%{x} cases<extra></extra>",
        ))
        fig.update_xaxes(title_text="Cases", dtick=T.count_tick(max(v for _, v in items)))
        st.markdown(
            f'<div style="color:{T.DIM};font-size:0.72rem;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:0.08em;margin-bottom:4px">By expected domain</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(T.style_fig(fig, height=260), use_container_width=True)


def _summary_tiles(cases: list[dict], data: dict) -> None:
    trapped = [c for c in cases if c["expected_failure_mode"] != "none"]
    critical = [c for c in cases if c["severity"] == "critical"]
    unavailable = [c for c in cases if c["governed_context"].get("available") is False]

    tiles = [
        T.metric_card("Golden cases", T.num(len(cases)),
                      footnote=f"dataset version {data.get('dataset_version', '—')}",
                      accent=T.ACCENT),
        T.metric_card("Cases with a designed trap", T.num(len(trapped)),
                      footnote=f"{T.pct(len(trapped) / len(cases))} of the set is built to elicit "
                               f"a specific failure",
                      accent=T.WARN),
        T.metric_card("Critical-severity cases", T.num(len(critical)),
                      footnote="failure here would invert a business decision",
                      accent=T.BAD),
        T.metric_card("Answer absent by design", T.num(len(unavailable)),
                      footnote="governed context deliberately lacks the answer — correct "
                               "behaviour is to say so",
                      accent=T.INFO),
    ]
    T.metric_row(tiles)


def _case_row(case: dict, state: D.EvalState) -> None:
    judge = state.judge_for(case["eval_id"])
    annotations = state.annotations_for(case["eval_id"])
    response = state.response_for(case["eval_id"])
    det = state.deterministic_for(case["eval_id"])

    human_scores = [a["overall_score"] for a in annotations
                    if a.get("rater_type") == "human" and a.get("overall_score") is not None]
    demo_scores = [a["overall_score"] for a in annotations
                   if a.get("rater_type") == "demo_profile" and a.get("overall_score") is not None]
    rater_mean = (sum(human_scores) / len(human_scores)) if human_scores else (
        (sum(demo_scores) / len(demo_scores)) if demo_scores else None
    )
    rater_kind = "human" if human_scores else ("demo" if demo_scores else None)
    judge_score = judge["overall_score"] if judge and judge.get("parse_ok") else None

    if rater_mean is not None and judge_score is not None:
        gap = abs(judge_score - rater_mean)
        status = ("agreed" if gap < 1 else "disagreed")
        status_color = T.GOOD if gap < 1 else T.BAD
    else:
        status, status_color = "not compared", T.FAINT

    chips = (
        T.chip(case["eval_id"], T.ACCENT, "rgba(99,102,241,0.13)")
        + T.chip(D.domain_label(case["expected_domain"] or case["domain"]), T.MUTED)
        + T.chip(D.TEST_TYPE_LABELS.get(case["test_type"], case["test_type"]), T.INFO,
                 "rgba(56,189,248,0.11)")
        + T.chip(case["difficulty"], T.FAINT)
        + (T.severity_chip(case["severity"]) if case["severity"] != "none" else "")
        + (T.chip(f"trap: {label_of(case['expected_failure_mode'])}", T.WARN,
                  "rgba(234,179,8,0.11)") if case["expected_failure_mode"] != "none" else "")
    )

    scores_line = (
        f'<span style="color:{T.FAINT};font-size:0.72rem">'
        f'{("human" if rater_kind == "human" else "demo profile" if rater_kind else "rater")} '
        f'{T.score(rater_mean)} · judge {T.score(judge_score)} · '
        f'<span style="color:{status_color}">{status}</span></span>'
    )

    with st.expander(f"{case['eval_id']} — {case['question'][:80]}"):
        st.markdown(f'<div style="margin-bottom:8px">{chips}</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="margin-bottom:10px">{scores_line}</div>', unsafe_allow_html=True)

        st.markdown(
            T.panel(T.label_text("Question") + T.body_text(case["question"], T.INK, "0.9rem"),
                    accent=T.ACCENT),
            unsafe_allow_html=True,
        )

        left, right = st.columns(2, gap="medium")

        with left:
            st.markdown(
                T.panel(
                    T.label_text("Expected answer", T.GOOD)
                    + T.body_text(case["expected_answer_summary"], T.TEXT, "0.79rem"),
                    accent=T.GOOD,
                ),
                unsafe_allow_html=True,
            )
            st.markdown(
                T.panel(
                    T.label_text("Expected behaviour", T.GOOD)
                    + T.body_text(case["expected_behavior"], T.TEXT, "0.78rem"),
                    accent=T.GOOD,
                ),
                unsafe_allow_html=True,
            )

        with right:
            required = case.get("required_facts") or []
            forbidden = case.get("forbidden_claims") or []
            html = ""
            if required:
                html += T.label_text("Required facts", T.GOOD)
                html += "".join(
                    T.chip(" | ".join(str(x) for x in f) if isinstance(f, list) else str(f),
                           T.GOOD, "rgba(34,197,94,0.11)")
                    for f in required
                )
            else:
                html += (T.label_text("Required facts", T.DIM)
                         + T.body_text("None — this case is judged qualitatively.", T.FAINT, "0.75rem"))
            if forbidden:
                html += '<div style="height:9px"></div>' + T.label_text("Forbidden claims", T.BAD)
                html += "".join(T.chip(f, T.BAD, "rgba(248,113,113,0.11)") for f in forbidden)
            st.markdown(T.panel(html, accent=T.INFO), unsafe_allow_html=True)

            gc = case.get("governed_context", {})
            if gc.get("available") is False:
                T.note(
                    "The governed context does not contain what this question asks for. Saying so "
                    "is the correct answer.",
                    "warn",
                )
            if gc.get("metrics"):
                st.markdown(
                    f'<div style="color:{T.FAINT};font-size:0.72rem;line-height:1.7">'
                    f'<strong style="color:{T.DIM}">Governed metrics:</strong> '
                    f'{", ".join(gc["metrics"])} <em>(from {gc.get("skill")}.yaml)</em></div>',
                    unsafe_allow_html=True,
                )

        if case.get("notes"):
            T.note(f"<strong>Design note.</strong> {case['notes']}", "info")

        if response and not response.get("error"):
            with st.expander("Model response"):
                st.markdown(response["answer"])
                st.markdown(
                    f'<div style="color:{T.FAINT};font-size:0.68rem;margin-top:6px">'
                    f'routed to {D.domain_label(response.get("routed_domain"))} · '
                    f'{response.get("router_version")} · {response.get("prompt_version")} · '
                    f'{T.score(response.get("latency_seconds"), 2)}s</div>',
                    unsafe_allow_html=True,
                )

        if det:
            with st.expander("Deterministic checks"):
                rows = "".join(
                    f'<tr><td style="padding:4px 0;color:{T.DIM};width:30%">'
                    f'{name.replace("_", " ").title()}</td>'
                    f'<td style="padding:4px 0;width:13%">{T.status_chip(check["status"])}</td>'
                    f'<td style="padding:4px 0;color:{T.FAINT};font-size:0.73rem">{check["detail"]}</td></tr>'
                    for name, check in det["checks"].items()
                )
                st.markdown(
                    f'<table style="width:100%;border-collapse:collapse;font-size:0.77rem">{rows}</table>',
                    unsafe_allow_html=True,
                )

        if judge and judge.get("parse_ok"):
            with st.expander("Evaluator scores"):
                cols = st.columns(2, gap="medium")
                rows = "".join(
                    f'<div style="display:flex;justify-content:space-between;padding:2px 0;'
                    f'font-size:0.75rem">'
                    f'<span style="color:{T.DIMENSION_COLORS[d]}">{T.dim_label(d)}</span>'
                    f'{T.score_bar(judge["scores"].get(d), width=54)}</div>'
                    for d in DIMENSIONS
                )
                cols[1].markdown(
                    T.panel(
                        T.label_text("AI judge", T.WARN) + T.pass_chip(judge["pass"])
                        + T.chip(label_of(judge["failure_mode"]), T.MUTED)
                        + f'<div style="margin-top:6px">{rows}</div>'
                        + T.body_text(judge.get("reasoning_summary", ""), T.FAINT, "0.71rem"),
                        accent=T.WARN,
                    ),
                    unsafe_allow_html=True,
                )
                if annotations:
                    for a in annotations[:2]:
                        rows_a = "".join(
                            f'<div style="display:flex;justify-content:space-between;padding:2px 0;'
                            f'font-size:0.75rem">'
                            f'<span style="color:{T.DIMENSION_COLORS[d]}">{T.dim_label(d)}</span>'
                            f'{T.score_bar(a["scores"].get(d), width=54)}</div>'
                            for d in DIMENSIONS
                        )
                        cols[0].markdown(
                            T.panel(
                                T.provenance_chip(a.get("rater_type", ""))
                                + T.chip(a["evaluator_id"], T.MUTED) + T.pass_chip(a.get("pass"))
                                + f'<div style="margin-top:6px">{rows_a}</div>',
                                accent=T.GOOD if a.get("rater_type") == "human" else T.WARN,
                            ),
                            unsafe_allow_html=True,
                        )
                else:
                    cols[0].markdown(
                        T.panel(T.label_text("Rater evaluation", T.DIM)
                                + T.body_text("Not rated yet.", T.FAINT, "0.78rem"), accent=T.DIM),
                        unsafe_allow_html=True,
                    )


def render(state: D.EvalState) -> None:
    data = load_golden_set()
    cases = state.cases

    T.page_header(
        "Golden Dataset",
        "The evaluation set, made inspectable. Every quality number in this application is "
        "computed over these cases, so what they test — and what a correct answer was defined to "
        "contain — determines what those numbers mean.",
        eyebrow="Measuring instrument",
    )

    if not cases:
        T.empty_state("Golden set not found", "data/golden_eval_set.json could not be loaded.")
        return

    _summary_tiles(cases, data)

    problems = validate_golden_set(data)
    if problems:
        T.note("<strong>Dataset validation problems:</strong><br>" + "<br>".join(problems[:8]), "warn")

    T.note(f"<strong>Provenance.</strong> {data.get('provenance', '')}", "info")
    T.note(f"<strong>Design.</strong> {data.get('design_note', '')}", "info")

    T.section("Composition", "What the set actually covers.")
    _composition_charts(cases)

    T.section("Filter and inspect", top_rule=True)
    c1, c2, c3, c4 = st.columns(4)
    domains = sorted({c["expected_domain"] or "cross_domain" for c in cases})
    test_types = sorted({c["test_type"] for c in cases})
    difficulties = ["easy", "medium", "hard"]
    modes = sorted({c["expected_failure_mode"] for c in cases})

    sel_domain = c1.multiselect("Domain", domains, format_func=D.domain_label, key="g_domain")
    sel_type = c2.multiselect("Test type", test_types,
                              format_func=lambda t: D.TEST_TYPE_LABELS.get(t, t), key="g_type")
    sel_diff = c3.multiselect("Difficulty", [d for d in difficulties
                                             if any(c["difficulty"] == d for c in cases)],
                              format_func=str.title, key="g_diff")
    sel_mode = c4.multiselect("Expected failure mode", modes,
                              format_func=lambda m: "No trap" if m == "none" else label_of(m),
                              key="g_mode")

    c5, c6 = st.columns([1, 3])
    outcome = c5.selectbox(
        "Evaluation outcome",
        ["All", "Judge failed", "Judge passed", "Rater/judge disagreed", "Not yet evaluated"],
        key="g_outcome",
    )

    filtered = cases
    if sel_domain:
        filtered = [c for c in filtered if (c["expected_domain"] or "cross_domain") in sel_domain]
    if sel_type:
        filtered = [c for c in filtered if c["test_type"] in sel_type]
    if sel_diff:
        filtered = [c for c in filtered if c["difficulty"] in sel_diff]
    if sel_mode:
        filtered = [c for c in filtered if c["expected_failure_mode"] in sel_mode]

    def _outcome_ok(case: dict) -> bool:
        judge = state.judge_for(case["eval_id"])
        parsed = judge and judge.get("parse_ok")
        if outcome == "Judge failed":
            return bool(parsed and not judge["pass"])
        if outcome == "Judge passed":
            return bool(parsed and judge["pass"])
        if outcome == "Not yet evaluated":
            return not parsed
        if outcome == "Rater/judge disagreed":
            annotations = state.annotations_for(case["eval_id"])
            scores = [a["overall_score"] for a in annotations if a.get("overall_score") is not None]
            if not parsed or not scores:
                return False
            return abs(judge["overall_score"] - (sum(scores) / len(scores))) >= 1
        return True

    if outcome != "All":
        filtered = [c for c in filtered if _outcome_ok(c)]

    st.markdown(
        f'<div style="color:{T.FAINT};font-size:0.75rem;margin:6px 0 10px">'
        f'{len(filtered)} of {len(cases)} cases match</div>',
        unsafe_allow_html=True,
    )

    if not filtered:
        T.note("No cases match these filters.", "info")
        return

    rows = []
    for case in filtered:
        judge = state.judge_for(case["eval_id"])
        annotations = state.annotations_for(case["eval_id"])
        human = [a["overall_score"] for a in annotations
                 if a.get("rater_type") == "human" and a.get("overall_score") is not None]
        demo = [a["overall_score"] for a in annotations
                if a.get("rater_type") == "demo_profile" and a.get("overall_score") is not None]
        rows.append({
            "ID": case["eval_id"],
            "Question": case["question"][:64] + ("…" if len(case["question"]) > 64 else ""),
            "Domain": D.domain_label(case["expected_domain"] or case["domain"]),
            "Test type": D.TEST_TYPE_LABELS.get(case["test_type"], case["test_type"]),
            "Difficulty": case["difficulty"],
            "Designed trap": ("—" if case["expected_failure_mode"] == "none"
                              else label_of(case["expected_failure_mode"])),
            "Human": round(sum(human) / len(human), 2) if human else None,
            "Demo": round(sum(demo) / len(demo), 2) if demo else None,
            "Judge": round(judge["overall_score"], 2) if judge and judge.get("parse_ok") else None,
            "Judge verdict": ("PASS" if judge and judge.get("parse_ok") and judge["pass"]
                              else "FAIL" if judge and judge.get("parse_ok") else "—"),
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=340)

    T.section("Case detail", "Expand any case to see the reference definition and every evaluator's view.")
    for case in filtered[:20]:
        _case_row(case, state)
    if len(filtered) > 20:
        st.markdown(
            f'<div style="color:{T.FAINT};font-size:0.73rem">Showing detail for the first 20 of '
            f'{len(filtered)} matching cases. Narrow the filters to see others.</div>',
            unsafe_allow_html=True,
        )

    with st.expander("How this dataset was built, and its limitations"):
        st.markdown(f"""
**Every required fact exists in the governed context.** A test in `tests/test_golden_dataset.py`
loads each skill file and asserts that every required fact for every case appears in the
context that case supplies. A dataset that demands a figure the model is never shown would
manufacture failures and count them as model defects.

**Forbidden claims are checked for triviality.** A second test asserts that no forbidden
phrase appears verbatim in its own governed context, since such a phrase would fire on
almost any grounded answer.

**Severity is enforced against the taxonomy.** Each case's severity must match the
severity the taxonomy assigns to its expected failure mode, so critical-failure rate is
coherent no matter which evaluator produced the classification.

**Three genuine inconsistencies in the governed layer are included deliberately** rather
than corrected: total MRR differs between the Product and Sales skills; new-customer
counts coincide between Sales and Marketing without the layer stating whether they refer
to the same population; and one figure exists only in a skill's sample Q&A rather than in
its governed metric definitions. A governance layer with conflicts is the normal condition,
and an evaluation suite that quietly avoids them is not testing governance.

---

**Limitations, stated plainly.**

*Not held out.* The set was authored against the same YAML files the system answers from,
by the person who built the system. Overfitting is possible. An independently authored set
would be the single highest-value addition.

*Small.* {len(cases)} cases is enough to expose failure modes and compare configurations.
It is not enough for a confident absolute quality claim, and per-slice figures (a domain,
a test type) often rest on five or six cases.

*Reference answers are summaries, not gold text.* `expected_answer_summary` describes what
a correct answer contains; it is not the only correct phrasing. That is deliberate — scoring
against a single gold string would penalise valid variation — but it does mean the human
rubric and the judge, rather than string matching, carry most of the evaluative weight.

*Forbidden-claim detection is precision-oriented.* Literal phrase matching catches the
obvious surface form. A model expressing the same forbidden idea in different words is
caught by the rubric and the judge, or not at all.
""")
