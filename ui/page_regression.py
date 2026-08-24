"""
ui/page_regression.py — Regression Testing and the quality improvement loop.

Two things live here, and the second is the point of the first.

  Run comparison       Two stored runs, metric by metric, with improvement and
                       regression marked against each metric's known good direction.

  Improvement loop     Concrete cases where an evaluation finding produced a specific
                       change, and the change was re-measured. The router comparison
                       recomputes live on every render against the current skill files,
                       so the before-and-after is evidence rather than narration.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from eval_runner import compare_runs
from evals import run_golden_routing_eval
from llm import PROMPT_VERSION_NOTES
from router import ROUTER_VERSION_NOTES, ROUTER_VERSIONS
from ui import data as D
from ui import theme as T


# ── Live router regression ────────────────────────────────────────────────────


@st.cache_data(show_spinner=False)
def _router_comparison(_domains_key: int) -> list[dict]:
    """
    Recompute routing accuracy for every router version against the current golden set.

    Cheap, deterministic, and needs no model call — so the router half of the
    improvement loop is measured on every page load rather than read from a stored
    claim. The argument exists only to key the cache.
    """
    from evals import golden_cases
    from skills import load_domains

    domains = load_domains(str(D.ROOT))
    cases = golden_cases()
    rows = []
    for version, fn in ROUTER_VERSIONS.items():
        result = run_golden_routing_eval(lambda q, d, _fn=fn: _fn(q, d), domains, cases)
        routed = [fn(c["question"], domains) for c in cases]
        rows.append({
            "version": version,
            "accuracy": result["accuracy"],
            "correct": result["correct"],
            "total": result["total"],
            "no_match": sum(1 for r in routed if r.no_match),
            "ties": sum(1 for r in routed if r.is_tie),
            "silent_misroutes": sum(1 for r in result["results"]
                                    if not r["correct"] and not r["is_ambiguous"]),
            "failures": result["failures"],
        })
    return rows


def _router_regression() -> None:
    rows = _router_comparison(1)
    baseline, latest = rows[0], rows[-1]

    tiles = []
    for row in rows:
        is_current = row is latest
        delta = None
        delta_good = None
        if not is_current and row is not baseline:
            delta = None
        if is_current and baseline["accuracy"] is not None and row["accuracy"] is not None:
            diff = row["accuracy"] - baseline["accuracy"]
            delta = T.pp(diff)
            delta_good = diff > 0
        tiles.append(T.metric_card(
            row["version"],
            T.pct(row["accuracy"], 1),
            delta=delta, delta_good=delta_good,
            footnote=f"{row['correct']}/{row['total']} correct · "
                     f"{row['silent_misroutes']} silent misroute(s) · "
                     f"{row['no_match']} no-match · {row['ties']} tie(s)",
            accent=T.GOOD if is_current else T.FAINT,
            value_color=T.rate_color(row["accuracy"], 0.85, 0.70),
        ))
    T.metric_row(tiles)

    fig = go.Figure()
    fig.add_bar(
        name="Routing accuracy",
        x=[r["version"] for r in rows],
        y=[r["accuracy"] for r in rows],
        marker_color=[T.FAINT] * (len(rows) - 1) + [T.GOOD],
        marker_line_width=0,
        text=[T.pct(r["accuracy"], 1) for r in rows],
        textposition="outside", textfont=dict(color=T.MUTED, size=11),
        yaxis="y", hovertemplate="%{x}<br>accuracy %{y:.1%}<extra></extra>",
    )
    fig.add_trace(go.Scatter(
        name="Silent misroutes",
        x=[r["version"] for r in rows],
        y=[r["silent_misroutes"] for r in rows],
        mode="lines+markers",
        line=dict(color=T.BAD, width=2),
        marker=dict(size=10, color=T.BAD),
        yaxis="y2", hovertemplate="%{x}<br>%{y} silent misroutes<extra></extra>",
    ))
    fig.update_layout(
        yaxis=dict(title="Routing accuracy", range=[0, 1.15], tickformat=".0%",
                   gridcolor="rgba(255,255,255,0.05)"),
        yaxis2=dict(title="Silent misroutes", overlaying="y", side="right",
                    range=[0, max(r["silent_misroutes"] for r in rows) + 2],
                    gridcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(T.style_fig(fig, height=300, showlegend=True), use_container_width=True)

    for row in rows:
        st.markdown(
            T.panel(
                T.label_text(row["version"], T.GOOD if row is latest else T.DIM)
                + T.body_text(ROUTER_VERSION_NOTES[row["version"]], T.TEXT, "0.79rem"),
                accent=T.GOOD if row is latest else T.FAINT,
            ),
            unsafe_allow_html=True,
        )

    if baseline["accuracy"] is not None and latest["accuracy"] is not None:
        diff = latest["accuracy"] - baseline["accuracy"]
        silent_diff = latest["silent_misroutes"] - baseline["silent_misroutes"]
        T.note(
            f"<strong>Measured outcome.</strong> Routing accuracy moved from "
            f"{T.pct(baseline['accuracy'], 1)} to {T.pct(latest['accuracy'], 1)} "
            f"({T.pp(diff)}). Silent misroutes — wrong domain answered confidently with no "
            f"ambiguity signal — went from {baseline['silent_misroutes']} to "
            f"{latest['silent_misroutes']}. The second number matters more than the first: a "
            f"flagged uncertain route lets the user correct it, a silent one produces a "
            f"confident answer grounded in the wrong governed context.",
            "good" if diff >= 0 and silent_diff <= 0 else "warn",
        )

    with st.expander(f"Remaining routing failures under {latest['version']} "
                     f"({len(latest['failures'])} cases)"):
        T.note(
            "These are shown rather than tuned away. Adding keywords until this list empties "
            "would be fitting the router to the evaluation set — the exact overfitting this "
            "project is meant to argue against.",
            "info",
        )
        for f in latest["failures"]:
            flag = ("flagged ambiguous" if f["is_ambiguous"] else "answered confidently")
            st.markdown(
                f'<div style="padding:7px 0;border-bottom:1px solid rgba(255,255,255,0.04);'
                f'font-size:0.78rem">'
                f'<span style="color:{T.MUTED}">{f["eval_id"]}</span> · '
                f'<span style="color:{T.TEXT}">{f["question"][:70]}</span><br>'
                f'<span style="color:{T.FAINT};font-size:0.72rem">'
                f'expected {D.domain_label(f["expected"])} → got {D.domain_label(f["predicted"])} · '
                f'{D.TEST_TYPE_LABELS.get(f["test_type"], f["test_type"])} · '
                f'<span style="color:{T.GOOD if f["is_ambiguous"] else T.BAD}">{flag}</span>'
                f'</span></div>',
                unsafe_allow_html=True,
            )


# ── Stored run comparison ─────────────────────────────────────────────────────


def _run_selector(state: D.EvalState) -> tuple[dict, dict] | None:
    runs = state.runs
    labels = {
        r["config"]["run_id"]: (
            f"{r['config']['run_id']} · {r['config'].get('label') or 'unlabelled'} · "
            f"{r['config']['system_prompt_version']} / {r['config']['router_version']}"
        )
        for r in runs
    }
    ids = [r["config"]["run_id"] for r in runs]

    col1, col2 = st.columns(2, gap="medium")
    baseline_id = col1.selectbox("Baseline run", ids, index=0,
                                 format_func=lambda i: labels[i], key="reg_baseline")
    current_id = col2.selectbox("Current run", ids, index=len(ids) - 1,
                                format_func=lambda i: labels[i], key="reg_current")

    if baseline_id == current_id:
        T.note("Select two different runs to compare.", "info")
        return None

    baseline = next(r for r in runs if r["config"]["run_id"] == baseline_id)
    current = next(r for r in runs if r["config"]["run_id"] == current_id)
    return baseline, current


def _comparison_table(comparison: dict) -> None:
    diff = comparison["config_diff"]
    if diff:
        st.markdown(
            T.panel(
                T.label_text("What changed between these runs", T.ACCENT)
                + "".join(
                    f'<div style="color:{T.TEXT};font-size:0.8rem;padding:2px 0">'
                    f'<code>{c["field"]}</code>: '
                    f'<span style="color:{T.FAINT}">{c["baseline"]}</span> → '
                    f'<strong style="color:{T.INK}">{c["current"]}</strong></div>'
                    for c in diff
                ),
                accent=T.ACCENT,
            ),
            unsafe_allow_html=True,
        )
    else:
        T.note(
            "The configuration is identical between these two runs. Any difference below is "
            "run-to-run variance, not the effect of a change — which is itself worth knowing, "
            "since it bounds how large a difference has to be before it means anything.",
            "warn",
        )

    tiles = [
        T.metric_card("Improved", T.num(comparison["n_improved"]), accent=T.GOOD,
                      value_color=T.GOOD, footnote="metrics that moved in the good direction"),
        T.metric_card("Regressed", T.num(comparison["n_regressed"]), accent=T.BAD,
                      value_color=T.BAD if comparison["n_regressed"] else T.MUTED,
                      footnote="metrics that moved in the bad direction"),
        T.metric_card("Unchanged", T.num(comparison["n_unchanged"]), accent=T.FAINT,
                      footnote="identical between runs"),
        T.metric_card("Not comparable", T.num(comparison["n_not_comparable"]), accent=T.WARN,
                      footnote="metric absent or undefined in one run — never treated as zero"),
    ]
    T.metric_row(tiles)

    rows_html = ""
    for row in comparison["rows"]:
        if row["verdict"] == "not comparable":
            rows_html += (
                f'<tr><td style="padding:6px 0;color:{T.DIM}">{row["metric"]}</td>'
                f'<td colspan="4" style="padding:6px 0;color:{T.FAINT};font-style:italic;'
                f'font-size:0.73rem">not comparable — {row["reason"]}</td></tr>'
            )
            continue

        fmt = row["format"]
        fmt_value = (lambda v: T.pct(v, 1) if fmt == "percent"
                     else T.score(v) if fmt == "score" else T.num(v))
        delta = row["delta"]
        delta_str = (T.pp(delta) if fmt == "percent"
                     else T.signed(delta) if fmt == "score" else f"{delta:+g}")

        colors = {"improved": T.GOOD, "regressed": T.BAD, "unchanged": T.FAINT}
        marks = {"improved": "▲", "regressed": "▼", "unchanged": "="}
        color = colors[row["verdict"]]

        rows_html += (
            f'<tr style="border-bottom:1px solid rgba(255,255,255,0.04)">'
            f'<td style="padding:7px 0;color:{T.TEXT};width:34%">{row["metric"]}'
            f'<span style="color:{T.FAINT};font-size:0.66rem"> '
            f'({"higher is better" if row["better"] == "up" else "lower is better"})</span></td>'
            f'<td style="padding:7px 0;color:{T.MUTED};width:14%">{fmt_value(row["baseline"])}</td>'
            f'<td style="padding:7px 0;color:{T.INK};font-weight:600;width:14%">'
            f'{fmt_value(row["current"])}</td>'
            f'<td style="padding:7px 0;color:{color};font-weight:700;width:14%">{delta_str}</td>'
            f'<td style="padding:7px 0;color:{color};font-size:0.76rem">'
            f'{marks[row["verdict"]]} {row["verdict"]}</td></tr>'
        )

    st.markdown(
        f'<table style="width:100%;border-collapse:collapse;font-size:0.79rem;margin-top:10px">'
        f'<tr style="border-bottom:1px solid {T.BORDER}">'
        f'<th style="text-align:left;padding-bottom:6px;color:{T.DIM}">Metric</th>'
        f'<th style="text-align:left;padding-bottom:6px;color:{T.DIM}">Baseline</th>'
        f'<th style="text-align:left;padding-bottom:6px;color:{T.DIM}">Current</th>'
        f'<th style="text-align:left;padding-bottom:6px;color:{T.DIM}">Delta</th>'
        f'<th style="text-align:left;padding-bottom:6px;color:{T.DIM}">Verdict</th>'
        f'</tr>{rows_html}</table>',
        unsafe_allow_html=True,
    )

    if comparison["n_regressed"]:
        T.note(
            f"<strong>{comparison['n_regressed']} metric(s) regressed.</strong> A change that "
            f"improves an average while regressing groundedness or critical-failure rate is not "
            f"an improvement. The decision to accept or reject belongs to whoever owns the "
            f"quality bar — this page supplies the evidence, not the verdict.",
            "warn",
        )


def _trend_charts(state: D.EvalState) -> None:
    runs = state.runs
    if len(runs) < 2:
        return

    labels = [r["config"]["run_id"][-9:] for r in runs]

    series = [
        ("AI judge mean score", [r["summary"].get("judge", {}).get("mean_overall_score") for r in runs],
         T.ACCENT, "y", ".2f"),
        ("Critical failure rate",
         [r["summary"].get("judge", {}).get("critical_failure_rate") for r in runs], T.BAD, "y2", ".0%"),
    ]

    fig = go.Figure()
    for name, values, color, axis, _ in series:
        fig.add_trace(go.Scatter(
            name=name, x=labels, y=values, mode="lines+markers",
            line=dict(color=color, width=2), marker=dict(size=9, color=color),
            yaxis=axis, connectgaps=False,
        ))
    fig.update_layout(
        yaxis=dict(title="Mean score (1-5)", range=[0, 5], gridcolor="rgba(255,255,255,0.05)"),
        yaxis2=dict(title="Critical failure rate", overlaying="y", side="right",
                    tickformat=".0%", range=[0, 1], gridcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(T.style_fig(fig, height=290, showlegend=True), use_container_width=True)

    fig2 = go.Figure()
    for name, path, color in (
        ("Routing accuracy", ("routing", "accuracy"), T.GOOD),
        ("Deterministic gate pass", ("deterministic", "verdict_pass_rate"), T.INFO),
        ("Human ↔ AI agreement (±1)", ("alignment", "human_only", "within_1_agreement"), T.WARN),
    ):
        values = []
        for r in runs:
            node = r["summary"]
            for key in path:
                node = node.get(key, {}) if isinstance(node, dict) else None
                if node is None:
                    break
            values.append(node if isinstance(node, (int, float)) else None)
        fig2.add_trace(go.Scatter(
            name=name, x=labels, y=values, mode="lines+markers",
            line=dict(color=color, width=2), marker=dict(size=9, color=color), connectgaps=False,
        ))
    fig2.update_yaxes(range=[0, 1.05], tickformat=".0%", title_text="Rate")
    st.plotly_chart(T.style_fig(fig2, height=290, showlegend=True), use_container_width=True)


# ── The improvement loop ──────────────────────────────────────────────────────


def _improvement_loop(state: D.EvalState) -> None:
    rows = _router_comparison(1)
    by_version = {r["version"]: r for r in rows}
    v1, v2, v3 = by_version["v1_substring"], by_version["v2_token_aware"], by_version["v3_idf_weighted"]

    st.markdown(
        f'<div style="background:{T.SURFACE};border:1px solid {T.BORDER};border-radius:11px;'
        f'padding:14px 18px;margin-bottom:16px;text-align:center;font-size:0.78rem;'
        f'color:{T.MUTED};line-height:2.2">'
        f'<span style="color:{T.BAD};font-weight:600">Failure detected</span> → '
        f'<span style="color:{T.WARN};font-weight:600">root cause identified</span> → '
        f'<span style="color:{T.ACCENT};font-weight:600">change made</span> → '
        f'<span style="color:{T.INFO};font-weight:600">golden set re-run</span> → '
        f'<span style="color:{T.GOOD};font-weight:600">quality compared</span> → '
        f'<span style="color:{T.INK};font-weight:600">accepted or rejected</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    cases = [
        {
            "n": 1,
            "title": "Substring keyword matching misroutes on shared word fragments",
            "detected": (
                "Golden case SLS-002 — <em>“What is our pipeline coverage ratio?”</em> — routed to "
                "Product Analytics. The answer was grounded in over-utilization metrics and never "
                "mentioned pipeline."
            ),
            "cause": (
                "The v1 router scored a keyword whenever it appeared anywhere in the question "
                "string. The Product skill claims the keyword <code>overage</code>, and "
                "<code>coverage</code> contains it. The match had nothing to do with meaning."
            ),
            "change": (
                "Added <code>v2_token_aware</code>: keywords must align to word boundaries, with "
                "light plural stemming and a bonus for contiguous multi-word phrases. v1 was kept "
                "callable rather than deleted, so the before-and-after stays measurable."
            ),
            "result": (
                f"Substring false positives eliminated. Routing accuracy "
                f"{T.pct(v1['accuracy'], 1)} → {T.pct(v2['accuracy'], 1)}; silent misroutes "
                f"{v1['silent_misroutes']} → {v2['silent_misroutes']}."
            ),
            "verdict": (
                "Accepted with a caveat. Accuracy barely moved: the substring bug was real but not "
                "the dominant error. Fixing it exposed the actual problem, which is the next case. "
                "A fix that removes a real defect without moving the headline metric is still the "
                "right change — and reporting it as a win would have been the wrong call."
            ),
            "verdict_good": None,
        },
        {
            "n": 2,
            "title": "Keyword collisions between domains resolved by arbitrary tie-breaking",
            "detected": (
                "After the v2 fix, most remaining misroutes were ties. <code>customers</code> is "
                "claimed by both Product and Sales; <code>pipeline</code> by both Sales and "
                "Marketing. Tied scores were resolved by dictionary iteration order — a coin flip "
                "presented to the user as a confident routing decision."
            ),
            "cause": (
                "Every keyword carried equal weight regardless of how many domains claimed it. A "
                "token shared across domains carries almost no routing signal, but scored the same "
                "as one unique to a single skill."
            ),
            "change": (
                "Added <code>v3_idf_weighted</code>: each keyword token is weighted by the inverse "
                "of how many domains claim it. Partial credit is capped at once per question token. "
                "No-match and tied outcomes now set explicit flags instead of being resolved "
                "silently."
            ),
            "result": (
                f"Routing accuracy {T.pct(v2['accuracy'], 1)} → {T.pct(v3['accuracy'], 1)}; "
                f"silent misroutes {v2['silent_misroutes']} → {v3['silent_misroutes']}; "
                f"{v3['ties']} case(s) now flagged as tied rather than answered arbitrarily."
            ),
            "verdict": (
                "Accepted. This is the change that mattered. The accuracy gain is real, but the "
                "reduction in silent misroutes is the more important result: a flagged uncertain "
                "route lets the user pick a domain, while a silent one produces a fluent, "
                "well-formatted answer built on the wrong governed context."
            ),
            "verdict_good": True,
        },
        {
            "n": 3,
            "title": "Skill files advertised routing keywords their metrics did not support",
            "detected": (
                "Five golden cases matched no keyword in any skill file and fell through to an "
                "arbitrary fallback domain — including SLS-005 (<em>NRR</em>), MKT-005 "
                "(<em>Paid Social</em>) and SUP-005 (<em>reopen rate</em>). Every one of those "
                "terms names a figure the governed metrics actually contain."
            ),
            "cause": (
                "A governance gap rather than a code defect. The Sales skill defines Net Revenue "
                "Retention but never listed <code>nrr</code> as a keyword; Support defines a reopen "
                "rate but listed only <code>reopened</code>. The routing vocabulary had drifted "
                "from the metric definitions it is supposed to reach."
            ),
            "change": (
                "Added the missing terms to <code>sales.yaml</code>, <code>marketing.yaml</code> "
                "and <code>csup.yaml</code>. A data change, versioned with the skill files, not a "
                "code change."
            ),
            "result": (
                f"Three of five zero-match cases resolved. The remaining two ask about named "
                f"individuals (<em>“Is Sarah M underperforming?”</em>) and carry no domain "
                f"vocabulary at all — a genuine limit of keyword routing, not a coverage gap. They "
                f"are left failing and visible above."
            ),
            "verdict": (
                "Accepted. Also the clearest argument for keeping an evaluation set adversarial: "
                "these gaps were invisible to every functional test because each skill file was "
                "internally consistent. Only questions phrased the way a user would phrase them "
                "exposed the drift."
            ),
            "verdict_good": True,
        },
        {
            "n": 4,
            "title": "The response template overrode explicit user formatting instructions",
            "detected": (
                "Every <code>instruction_following</code> case in the golden set is designed to "
                "conflict with the default output format — <em>“Answer with a single number and "
                "nothing else”</em>, <em>“as a markdown table with no prose”</em>, <em>“in one "
                "sentence with no recommendation”</em>."
            ),
            "cause": (
                "The v1 system prompt reads <em>“Structure every response in exactly this format — "
                "use these exact headers”</em> and requires an Insight and a Recommended action "
                "for every question. A user instruction had no way to win against that."
            ),
            "change": (
                "Added <code>sysprompt-v2</code>: the three-section template becomes the default "
                "for open-ended questions and explicitly yields to a user format instruction. The "
                "same revision permits an explicit <em>not in the governed layer</em> outcome and "
                "adds premise-checking, since the same forced-output pressure was producing "
                "fabricated figures on missing-context cases."
            ),
            "result": None,   # requires two runs; filled in below
            "verdict": None,
            "verdict_good": None,
        },
    ]

    for case in cases:
        st.markdown(
            f'<div style="color:{T.ACCENT};font-size:0.66rem;font-weight:700;'
            f'text-transform:uppercase;letter-spacing:0.1em;margin:18px 0 6px">'
            f'Case {case["n"]}</div>'
            f'<div style="color:{T.INK};font-size:1.0rem;font-weight:700;margin-bottom:10px">'
            f'{case["title"]}</div>',
            unsafe_allow_html=True,
        )

        steps = [
            ("Failure detected", case["detected"], T.BAD),
            ("Root cause", case["cause"], T.WARN),
            ("Change made", case["change"], T.ACCENT),
        ]
        if case["result"]:
            steps.append(("Regression result", case["result"], T.INFO))
        if case["verdict"]:
            colour = T.GOOD if case["verdict_good"] else T.WARN
            steps.append(("Accepted or rejected", case["verdict"], colour))

        for label, text, colour in steps:
            st.markdown(
                T.panel(T.label_text(label, colour) + T.body_text(text, T.TEXT, "0.8rem"),
                        accent=colour),
                unsafe_allow_html=True,
            )

        if case["n"] == 4:
            _prompt_case_status(state)


def _prompt_case_status(state: D.EvalState) -> None:
    """Case 4 needs two runs to be measurable. Say so rather than asserting a result."""
    versions = {r["config"]["system_prompt_version"] for r in state.runs}

    for version, notes in PROMPT_VERSION_NOTES.items():
        st.markdown(
            f'<div style="color:{T.FAINT};font-size:0.75rem;line-height:1.65;padding:3px 0">'
            f'<code>{version}</code> — {notes}</div>',
            unsafe_allow_html=True,
        )

    if len(versions) >= 2:
        T.note(
            "Both prompt versions have been run. Select one run of each above to see the measured "
            "effect on instruction-following and missing-context handling.",
            "good",
        )
        return

    T.note(
        "<strong>Not yet measured.</strong> This change alters generated text, so unlike the "
        "router cases it cannot be recomputed from stored records — it needs a model run under "
        "each prompt version. Until both exist, no before-and-after is claimed here. "
        "Run the two commands below and compare them above.",
        "warn",
    )
    st.code(
        'export GROQ_API_KEY="..."\n'
        'python eval_runner.py --all --prompt-version sysprompt-v1 --label "baseline prompt"\n'
        'python eval_runner.py --all --prompt-version sysprompt-v2 --label "revised prompt"',
        language="bash",
    )


def render(state: D.EvalState) -> None:
    T.page_header(
        "Regression Testing",
        "Whether a change to the prompt, the router or the model actually improved quality — "
        "measured against the same golden set, with each metric judged against its own known "
        "good direction.",
        eyebrow="Prompt and model iteration",
    )

    tab_loop, tab_runs = st.tabs(["Quality improvement loop", "Run comparison"])

    with tab_loop:
        T.section(
            "From failure to fix to re-measurement",
            "Four changes driven by specific golden-set findings. The router cases recompute on "
            "every page load against the current skill files, so the numbers below are evidence "
            "rather than a stored claim.",
            top_rule=False,
        )
        _improvement_loop(state)

        T.section(
            "Router versions, measured live",
            "All three router implementations run against the current golden set on every render. "
            "Earlier versions are kept callable precisely so that an improvement can be shown "
            "rather than asserted.",
        )
        _router_regression()

    with tab_runs:
        if not state.runs:
            T.empty_state(
                "No evaluation runs stored",
                "Run comparison needs at least two stored runs with different configurations. "
                "Each run records its model, prompt version, router version, judge version and "
                "dataset version, so a quality difference can be attributed to a specific change.",
                D.GENERATE_COMMAND,
            )
            return

        T.section("Stored runs", f"{len(state.runs)} run(s) recorded.", top_rule=False)
        for run in state.runs:
            cfg, summary = run["config"], run["summary"]
            st.markdown(
                f'<div style="background:{T.SURFACE};border:1px solid {T.BORDER};'
                f'border-radius:9px;padding:10px 14px;margin-bottom:7px">'
                f'<div style="display:flex;flex-wrap:wrap;gap:5px;align-items:center;'
                f'margin-bottom:5px">'
                + T.chip(cfg["run_id"], T.ACCENT, "rgba(99,102,241,0.13)")
                + T.chip(cfg.get("label") or "unlabelled", T.MUTED)
                + T.chip(cfg["model_version"], T.FAINT)
                + T.chip(cfg["system_prompt_version"], T.FAINT)
                + T.chip(cfg["router_version"], T.FAINT)
                + T.chip(f"dataset v{cfg.get('dataset_version')}", T.FAINT)
                + f'</div>'
                f'<div style="color:{T.DIM};font-size:0.72rem">'
                f'{cfg["timestamp"][:16].replace("T", " ")} UTC · '
                f'{cfg["n_cases"]} cases · '
                f'routing {T.pct(summary.get("routing", {}).get("accuracy"), 1)} · '
                f'judge mean {T.score(summary.get("judge", {}).get("mean_overall_score"))}/5'
                + (f' · <em>{cfg["notes"]}</em>' if cfg.get("notes") else "")
                + f'</div></div>',
                unsafe_allow_html=True,
            )

        if len(state.runs) < 2:
            T.note(
                "Only one run is stored, so there is nothing to compare against. Run the pipeline "
                "again with a different configuration — a different prompt version or router "
                "version — to produce a comparison.",
                "info",
            )
            st.code(
                'python eval_runner.py --all --prompt-version sysprompt-v1 --label "baseline"',
                language="bash",
            )
            return

        T.section("Compare two runs", top_rule=True)
        selection = _run_selector(state)
        if not selection:
            return
        baseline, current = selection
        _comparison_table(compare_runs(baseline, current))

        T.section("Quality across runs", "Every stored run, in chronological order.")
        _trend_charts(state)

        with st.expander("How to read a regression comparison"):
            st.markdown("""
**One variable at a time.** The configuration diff at the top of the comparison names what
actually changed. If it lists three fields, the comparison cannot attribute the outcome to
any one of them.

**Direction is per metric.** Critical failure rate falling is an improvement; groundedness
falling is a regression. Each metric declares its own good direction rather than assuming
higher is better.

**Not comparable is not zero.** When a metric is absent or undefined in one run — no human
annotations, no judge results, a statistic undefined at that sample size — it is reported
as not comparable. Treating it as zero would manufacture a regression out of missing data.

**Run-to-run variance is real.** Generation runs at temperature 0.2, so two runs with
identical configuration will not produce identical text or identical scores. Comparing two
identically configured runs is the cheapest way to estimate how large a difference has to
be before it means anything, and it is worth doing before trusting a small delta.

**The tool does not decide.** It reports what moved and in which direction. Whether a
change ships is a judgement about which metrics matter, and that belongs to a person.
""")
