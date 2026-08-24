"""
ui/page_human_eval.py — Human Evaluation.

The annotation workstation. A person applies the six-dimension rubric to one response
at a time, with the governed context, the reference answer, the required facts and the
forbidden claims all visible — because a rating made without the reference is an
opinion, not an evaluation.

Persistence degrades honestly. On a writable filesystem annotations are saved to
data/human_annotations.json. On Streamlit Community Cloud, where the filesystem is
read-only, they are held in session state and offered as a download, and the page says
so rather than silently discarding them.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import streamlit as st

from failure_taxonomy import all_modes, get_failure_mode, label_of
from human_evals import (
    CRITICAL_RULE,
    DIMENSIONS,
    PASS_RULE,
    RUBRIC,
    SCALE,
    applies_pass_rule,
    load_annotations,
    make_annotation,
    rubric_markdown,
    save_annotations,
    upsert_annotation,
)
from skills import build_context
from ui import data as D
from ui import theme as T

SESSION_ANNOTATIONS = "session_annotations"
RATER_ID = "rater_id"
FLASH = "he_flash"


def _init() -> None:
    st.session_state.setdefault(RATER_ID, "")
    st.session_state.setdefault(SESSION_ANNOTATIONS, [])
    st.session_state.setdefault("he_index", 0)
    st.session_state.setdefault("he_filter_unrated", True)
    st.session_state.setdefault("he_persist_failed", False)


def _all_annotations(state: D.EvalState) -> list[dict]:
    """Disk annotations plus anything written this session that could not be persisted."""
    on_disk = state.annotations
    session = st.session_state.get(SESSION_ANNOTATIONS, [])
    if not session:
        return on_disk
    keyed = {(a["eval_id"], a["evaluator_id"]): a for a in on_disk}
    for a in session:
        keyed[(a["eval_id"], a["evaluator_id"])] = a
    return list(keyed.values())


def _rubric_panel() -> None:
    with st.expander("Evaluation rubric — read before rating", expanded=False):
        st.markdown(rubric_markdown())


def _rater_bar(state: D.EvalState) -> str:
    col_id, col_stats = st.columns([1, 2], gap="medium")

    rater_id = col_id.text_input(
        "Your evaluator ID",
        value=st.session_state.get(RATER_ID, ""),
        placeholder="e.g. ajantika",
        key="rater_id_input",
        help="Your annotations are stored under this ID with rater_type='human'.",
    )
    st.session_state[RATER_ID] = rater_id.strip()

    annotations = _all_annotations(state)
    mine = [a for a in annotations if a.get("evaluator_id") == st.session_state[RATER_ID]]
    demo = [a for a in annotations if a.get("rater_type") == "demo_profile"]
    human = [a for a in annotations if a.get("rater_type") == "human"]

    with col_stats:
        st.markdown('<div style="height:26px"></div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="display:flex;gap:6px;flex-wrap:wrap">'
            + T.chip(f"{len(mine)} rated by you", T.GOOD if mine else T.FAINT,
                     "rgba(34,197,94,0.13)" if mine else "rgba(148,163,184,0.08)")
            + T.chip(f"{len(human)} human annotations total", T.MUTED)
            + T.chip(f"{len(demo)} demo-profile records across "
                     f"{len({a['eval_id'] for a in demo})} cases — not human",
                     T.WARN, "rgba(234,179,8,0.11)")
            + T.chip(f"{len(state.cases)} cases in the set", T.FAINT)
            + "</div>",
            unsafe_allow_html=True,
        )

    return st.session_state[RATER_ID]


def _case_reference(case: dict, state: D.EvalState) -> None:
    """
    Everything the evaluator needs to judge against, before they see any score.

    Stacked in a single column: this renders inside one half of a side-by-side
    layout, so it cannot split itself further without the panels becoming unreadably
    narrow.
    """
    st.markdown(
        T.panel(
            T.label_text("User question")
            + T.body_text(case["question"], T.INK, "0.92rem"),
            accent=T.ACCENT,
        ),
        unsafe_allow_html=True,
    )

    response = state.response_for(case["eval_id"])
    if response and not response.get("error"):
        st.markdown(
            T.panel(
                T.label_text("Model response — score this", T.WARN)
                + T.body_text(T.response_html(response["answer"]), T.TEXT, "0.82rem"),
                accent=T.WARN,
            ),
            unsafe_allow_html=True,
        )
    else:
        T.empty_state(
            "No stored response for this case",
            "This case has no generated response yet, so there is nothing to rate.",
            D.GENERATE_COMMAND,
        )

    st.markdown(
        T.panel(
            T.label_text("Expected answer — your answer key", T.GOOD)
            + T.body_text(case["expected_answer_summary"], T.TEXT, "0.79rem")
            + f'<div style="margin-top:8px;padding-top:7px;'
              f'border-top:1px solid rgba(255,255,255,0.06)">'
            + T.label_text("Expected behaviour", T.GOOD)
            + T.body_text(case["expected_behavior"], T.TEXT, "0.76rem")
            + "</div>",
            accent=T.GOOD,
        ),
        unsafe_allow_html=True,
    )

    required = case.get("required_facts") or []
    forbidden = case.get("forbidden_claims") or []
    if required or forbidden:
        html = ""
        if required:
            html += T.label_text("Required facts", T.GOOD) + "".join(
                T.chip(" | ".join(str(x) for x in f) if isinstance(f, list) else str(f),
                       T.GOOD, "rgba(34,197,94,0.11)")
                for f in required
            )
        if forbidden:
            html += ('<div style="height:7px"></div>' if required else "")
            html += T.label_text("Forbidden claims", T.BAD) + "".join(
                T.chip(f, T.BAD, "rgba(248,113,113,0.11)") for f in forbidden
            )
        st.markdown(T.panel(html, accent=T.INFO), unsafe_allow_html=True)

    with st.expander("Governed context supplied to the model"):
        skill = case.get("governed_context", {}).get("skill")
        available = case.get("governed_context", {}).get("available", True)
        if available is False:
            T.note(
                "<strong>This case is designed so the governed context does not contain what was "
                "asked.</strong> The correct behaviour is to say so. Supplying a figure anyway is "
                "a failure, however plausible it looks.",
                "warn",
            )
        if skill and skill in state.domains:
            st.code(build_context(state.domains[skill]), language="text")
        else:
            st.markdown(f"*No single governed skill declared for this case.*")

    with st.expander("Deterministic check results (objective checks only)"):
        det = state.deterministic_for(case["eval_id"])
        if not det:
            st.markdown("*No deterministic record stored for this case.*")
        else:
            rows = "".join(
                f'<tr><td style="padding:5px 0;color:{T.DIM};width:32%">'
                f'{name.replace("_", " ").title()}</td>'
                f'<td style="padding:5px 0;width:14%">{T.status_chip(check["status"])}</td>'
                f'<td style="padding:5px 0;color:{T.FAINT};font-size:0.73rem">{check["detail"]}</td></tr>'
                for name, check in det["checks"].items()
            )
            st.markdown(
                f'<table style="width:100%;border-collapse:collapse;font-size:0.78rem">{rows}</table>',
                unsafe_allow_html=True,
            )
            T.note(
                "These are objective checks. They do not determine your rating — a response can "
                "pass every deterministic check and still be unhelpful or wrong.",
                "info",
            )


def _score_controls(case: dict, existing: dict | None, compact: bool = True) -> tuple[dict, dict]:
    """
    The rating form. Returns (scores, meta).

    Compact by default. The full anchor text for every dimension makes the form about
    three screens tall, which forces the evaluator to scroll away from the question to
    reach the submit button and then scroll back to read the next one — a bad loop to
    repeat twenty times. The selected anchor is shown as one short line; the full
    rubric stays one click away in the collapsible panel.
    """
    prior = (existing or {}).get("scores", {})
    scores = {}

    st.markdown(
        f'<p style="color:{T.FAINT};font-size:0.63rem;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:0.08em;margin:2px 0 2px">Rubric scores — 1 Poor to 5 Excellent</p>',
        unsafe_allow_html=True,
    )

    for dim in DIMENSIONS:
        spec = RUBRIC[dim]
        st.markdown(
            f'<div style="color:{T.DIMENSION_COLORS[dim]};font-size:0.76rem;font-weight:700;'
            f'margin:2px 0 -14px">{spec["label"]}'
            f'<span style="color:{T.FAINT};font-weight:400;font-size:0.68rem"> — '
            f'{spec["question"]}</span></div>',
            unsafe_allow_html=True,
        )
        value = st.slider(
            spec["label"],
            min_value=1, max_value=5,
            value=int(prior.get(dim) or 4),
            key=f"score_{case['eval_id']}_{dim}",
            label_visibility="collapsed",
        )
        scores[dim] = value
        anchor = spec["anchors"][value]
        if compact and len(anchor) > 88:
            anchor = anchor[:85].rstrip() + "…"
        st.markdown(
            f'<div style="color:{T.FAINT};font-size:0.67rem;line-height:1.45;margin-top:-14px">'
            f'<strong style="color:{T.MUTED}">{value} — {SCALE[value]}:</strong> {anchor}</div>',
            unsafe_allow_html=True,
        )

    st.markdown(f'<hr style="margin:8px 0 10px">', unsafe_allow_html=True)

    derived_pass = applies_pass_rule(scores)
    col1, col2, col3 = st.columns([1, 1, 1], gap="medium")

    with col1:
        st.markdown(
            f'<div style="color:{T.MUTED};font-size:0.74rem;font-weight:600;margin-bottom:2px">'
            f'Overall verdict</div>'
            f'<div style="color:{T.FAINT};font-size:0.66rem;line-height:1.45;margin-bottom:4px">'
            f'Rule gives <strong style="color:{T.GOOD if derived_pass else T.BAD}">'
            f'{"PASS" if derived_pass else "FAIL"}</strong>. Override if you disagree.'
            f'</div>',
            unsafe_allow_html=True,
        )
        verdict = st.radio(
            "verdict",
            options=["Apply the rule", "Force PASS", "Force FAIL"],
            index=0,
            key=f"verdict_{case['eval_id']}",
            label_visibility="collapsed",
        )
        passed = derived_pass if verdict == "Apply the rule" else (verdict == "Force PASS")

    with col2:
        st.markdown(
            f'<div style="color:{T.MUTED};font-size:0.74rem;font-weight:600;margin-bottom:2px">'
            f'Failure mode</div>'
            f'<div style="color:{T.FAINT};font-size:0.66rem;line-height:1.45;margin-bottom:4px">'
            f'The one a reviewer would fix first.</div>',
            unsafe_allow_html=True,
        )
        options = ["none"] + all_modes()
        prior_mode = (existing or {}).get("failure_mode", "none")
        failure_mode = st.selectbox(
            "failure mode",
            options=options,
            index=options.index(prior_mode) if prior_mode in options else 0,
            format_func=lambda m: "No failure" if m == "none" else label_of(m),
            key=f"mode_{case['eval_id']}",
            label_visibility="collapsed",
        )
        entry = get_failure_mode(failure_mode)
        if failure_mode != "none":
            st.markdown(
                f'<div style="margin-top:4px">{T.severity_chip(entry["severity"])}</div>'
                f'<div style="color:{T.FAINT};font-size:0.68rem;line-height:1.5;margin-top:4px">'
                f'{entry["description"]}</div>',
                unsafe_allow_html=True,
            )

    with col3:
        st.markdown(
            f'<div style="color:{T.MUTED};font-size:0.74rem;font-weight:600;margin-bottom:2px">'
            f'Your confidence</div>'
            f'<div style="color:{T.FAINT};font-size:0.66rem;line-height:1.45;margin-bottom:4px">'
            f'How sure are you? Low confidence flags cases worth reviewing later.'
            f'</div>',
            unsafe_allow_html=True,
        )
        confidence = st.slider(
            "confidence", 0.0, 1.0,
            value=float((existing or {}).get("evaluator_confidence", 0.8)),
            step=0.05, key=f"conf_{case['eval_id']}", label_visibility="collapsed",
        )
        critical = st.checkbox(
            "Critical failure",
            value=bool((existing or {}).get("critical_failure", entry["is_critical"])),
            key=f"crit_{case['eval_id']}",
            help=CRITICAL_RULE,
        )

    notes = st.text_area(
        "Notes — what drove your lowest score? (optional)",
        value=(existing or {}).get("notes", ""),
        placeholder="Name the specific claim or figure.",
        key=f"notes_{case['eval_id']}",
        height=68,
    )

    return scores, {
        "pass": passed,
        "critical_failure": critical,
        "failure_mode": failure_mode,
        "evaluator_confidence": confidence,
        "notes": notes,
    }


def _persist(annotation: dict, state: D.EvalState) -> None:
    """
    Write through to disk where possible; hold in session and warn where not.

    The confirmation is stashed in session state rather than rendered here, because
    the caller reruns immediately afterwards to advance to the next case — which
    would wipe a message written now before the evaluator ever saw it. Submitting a
    rating and getting no visible response reads as a broken button.
    """
    session = st.session_state.get(SESSION_ANNOTATIONS, [])
    session = [a for a in session
               if (a["eval_id"], a["evaluator_id"]) != (annotation["eval_id"], annotation["evaluator_id"])]
    session.append(annotation)
    st.session_state[SESSION_ANNOTATIONS] = session

    merged = upsert_annotation(annotation, load_annotations())
    n_mine = sum(1 for a in merged
                 if a.get("evaluator_id") == annotation["evaluator_id"]
                 and a.get("rater_type") == "human")

    if save_annotations(merged):
        st.session_state["he_persist_failed"] = False
        st.session_state[FLASH] = (
            "good",
            f"Saved <strong>{annotation['eval_id']}</strong> — overall "
            f"{annotation['overall_score']:.2f}/5, "
            f"{'PASS' if annotation['pass'] else 'FAIL'}. "
            f"That is <strong>{n_mine}</strong> case(s) you have rated.",
        )
    else:
        st.session_state["he_persist_failed"] = True
        st.session_state[FLASH] = (
            "warn",
            f"Recorded <strong>{annotation['eval_id']}</strong> in this session, but the "
            "filesystem is read-only (expected on Streamlit Community Cloud). Use the download "
            "button below and commit the file to keep these ratings.",
        )


def _show_flash() -> None:
    """Render and clear the confirmation left by the previous submit."""
    flash = st.session_state.pop(FLASH, None)
    if flash:
        kind, message = flash
        T.note(message, kind)


def _now_rating_strip(case: dict, index: int, total: int) -> None:
    """
    Restate the case being rated, immediately above the button row.

    Streamlit preserves scroll position across a rerun, so after submitting from the
    bottom of a long form the evaluator sees a fresh set of sliders that look exactly
    like the ones they just moved, with the new question off-screen above. Scrolling
    the parent page from a component is not possible — Streamlit sandboxes component
    iframes into an opaque origin — so the case is restated here instead, where the
    evaluator is already looking.
    """
    st.markdown(
        f'<div style="background:{T.SURFACE};border:1px solid {T.BORDER};'
        f'border-left:3px solid {T.ACCENT};border-radius:9px;padding:10px 14px;'
        f'margin:4px 0 10px">'
        f'<div style="color:{T.DIM};font-size:0.62rem;font-weight:700;'
        f'text-transform:uppercase;letter-spacing:0.09em;margin-bottom:3px">'
        f'Now rating &nbsp;·&nbsp; {index + 1} of {total} in queue</div>'
        f'<div style="color:{T.INK};font-size:0.86rem;line-height:1.5">'
        f'<strong style="color:{T.ACCENT}">{case["eval_id"]}</strong> &nbsp; '
        f'{case["question"]}</div></div>',
        unsafe_allow_html=True,
    )


def _download_block() -> None:
    session = st.session_state.get(SESSION_ANNOTATIONS, [])
    if not session:
        return
    payload = json.dumps(
        {
            "schema_version": 1,
            "note": "Human annotations exported from the Human Evaluation page.",
            "annotations": session,
        },
        indent=2,
    )
    st.download_button(
        f"Download {len(session)} annotation(s) from this session",
        data=payload,
        file_name="human_annotations_session.json",
        mime="application/json",
        use_container_width=False,
    )


def _progress_strip(state: D.EvalState, rater_id: str, queue: list[dict], index: int) -> None:
    annotations = _all_annotations(state)
    rated_ids = {a["eval_id"] for a in annotations if a.get("evaluator_id") == rater_id}
    ratable = [c for c in state.cases if state.response_for(c["eval_id"])]
    done = len(rated_ids & {c["eval_id"] for c in ratable})
    total = len(ratable)

    bar = int((done / total) * 100) if total else 0
    st.markdown(
        f'<div style="margin:2px 0 12px">'
        f'<div style="display:flex;justify-content:space-between;font-size:0.72rem;'
        f'color:{T.DIM};margin-bottom:4px">'
        f'<span>{done} of {total} responses rated by <strong style="color:{T.MUTED}">'
        f'{rater_id or "—"}</strong></span>'
        f'<span>case {index + 1} of {len(queue)} in queue</span></div>'
        f'<div style="height:4px;border-radius:3px;background:rgba(255,255,255,0.07);overflow:hidden">'
        f'<div style="width:{bar}%;height:100%;background:{T.GOOD}"></div></div></div>',
        unsafe_allow_html=True,
    )


def render(state: D.EvalState) -> None:
    _init()

    T.page_header(
        "Human Evaluation",
        "Humans are the reference standard in this system. The automated judge is measured "
        "against these ratings, not the other way round — so the rubric, the reference answer and "
        "the governed context are all shown before any score is entered.",
        eyebrow="Annotation workstation",
    )

    if not state.has_responses:
        T.empty_state("No responses to rate", D.NO_RUN_EXPLANATION, D.GENERATE_COMMAND)
        _rubric_panel()
        return

    rater_id = _rater_bar(state)
    _rubric_panel()

    if not rater_id:
        T.note(
            "Enter an evaluator ID above to begin. Your ratings are stored with "
            "<code>rater_type=\"human\"</code> and are reported separately from the demo profiles "
            "everywhere in this application.",
            "info",
        )
        return

    T.note(
        f"<strong>Pass rule.</strong> {PASS_RULE}<br>"
        f"<strong>Critical failure.</strong> {CRITICAL_RULE}",
        "info",
    )

    annotations = _all_annotations(state)
    mine = {a["eval_id"] for a in annotations if a.get("evaluator_id") == rater_id}

    col_f1, col_f2 = st.columns([1, 3])
    only_unrated = col_f1.checkbox("Hide cases I have rated", value=st.session_state["he_filter_unrated"])
    st.session_state["he_filter_unrated"] = only_unrated

    ratable = [c for c in state.cases if state.response_for(c["eval_id"])
               and not state.response_for(c["eval_id"]).get("error")]
    queue = [c for c in ratable if c["eval_id"] not in mine] if only_unrated else ratable

    if not queue:
        st.success(
            f"You have rated every response available ({len(mine)} cases). "
            "Uncheck the filter above to revise an earlier rating."
        )
        _download_block()
        return

    index = min(st.session_state["he_index"], len(queue) - 1)
    case = queue[index]

    _progress_strip(state, rater_id, queue, index)

    meta_chips = (
        T.chip(case["eval_id"], T.ACCENT, "rgba(99,102,241,0.13)")
        + T.chip(D.domain_label(case["expected_domain"] or case["domain"]), T.MUTED)
        + T.chip(D.TEST_TYPE_LABELS.get(case["test_type"], case["test_type"]), T.INFO,
                 "rgba(56,189,248,0.11)")
        + T.chip(case["difficulty"], T.FAINT)
    )
    st.markdown(f'<div style="margin-bottom:8px">{meta_chips}</div>', unsafe_allow_html=True)

    existing = next((a for a in annotations
                     if a["eval_id"] == case["eval_id"] and a["evaluator_id"] == rater_id), None)

    # Reference on the left, scoring on the right, so the question stays on screen
    # while the evaluator scores it. Stacking them vertically meant scrolling away
    # from the question to reach submit, then back up to read the next one.
    col_ref, col_score = st.columns([1, 1], gap="large")

    with col_ref:
        _case_reference(case, state)

    with col_score:
        if existing:
            T.note(
                f"You already rated this case (overall "
                f"{T.score(existing.get('overall_score'))}/5). Submitting replaces it.",
                "info",
            )
        scores, meta = _score_controls(case, existing)

    _show_flash()
    _now_rating_strip(case, index, len(queue))

    col_prev, col_submit, col_skip, _ = st.columns([1, 1.4, 1, 3])

    if col_prev.button("← Previous", disabled=index == 0, use_container_width=True):
        st.session_state["he_index"] = max(0, index - 1)
        st.rerun()

    if col_submit.button("Submit rating", type="primary", use_container_width=True):
        annotation = make_annotation(
            eval_id=case["eval_id"],
            evaluator_id=rater_id,
            scores=scores,
            rater_type="human",
            passed=meta["pass"],
            critical_failure=meta["critical_failure"],
            failure_mode=meta["failure_mode"],
            evaluator_confidence=meta["evaluator_confidence"],
            notes=meta["notes"],
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        _persist(annotation, state)
        st.session_state["he_index"] = index if only_unrated else min(index + 1, len(queue) - 1)
        st.rerun()

    if col_skip.button("Skip →", use_container_width=True):
        st.session_state["he_index"] = min(index + 1, len(queue) - 1)
        st.rerun()

    _download_block()

    other = [a for a in annotations
             if a["eval_id"] == case["eval_id"] and a["evaluator_id"] != rater_id]
    if other:
        with st.expander(f"How other raters scored this case ({len(other)})"):
            T.note(
                "Shown after you have seen the case so it cannot anchor your own rating. "
                "Demo profiles are scripted rubric applications, not people.",
                "info",
            )
            for a in other:
                chips = "".join(
                    T.chip(f"{T.dim_label(d)} {a['scores'].get(d) or '—'}", T.MUTED)
                    for d in DIMENSIONS
                )
                st.markdown(
                    T.panel(
                        f'<div style="margin-bottom:6px">'
                        f'{T.provenance_chip(a.get("rater_type", ""))}'
                        f'{T.chip(a["evaluator_id"], T.MUTED)}'
                        f'{T.pass_chip(a.get("pass"))}'
                        f'{T.chip("overall " + T.score(a.get("overall_score")), T.MUTED)}</div>'
                        f'<div style="margin-bottom:6px">{chips}</div>'
                        + (T.body_text(a.get("notes", ""), T.FAINT, "0.73rem") if a.get("notes") else ""),
                        accent=T.GOOD if a.get("rater_type") == "human" else T.WARN,
                    ),
                    unsafe_allow_html=True,
                )
