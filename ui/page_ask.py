"""
ui/page_ask.py — Ask AI.

The original product surface, preserved: route a question, ground it in the governed
semantic layer, return a structured answer, and show the deterministic checks inline.

Added here: the answer can be sent to the LLM judge on demand, and any answer can be
carried into the Human Evaluation page as an ad-hoc case. That closes the loop between
using the assistant and evaluating it.
"""

from __future__ import annotations

import re

import streamlit as st

from evals import evaluate_response
from llm import DEFAULT_PROMPT_VERSION, ask_groq, get_api_key, parse_structured_answer
from router import AMBIGUITY_THRESHOLD, RoutingResult, classify_domain
from skills import build_context, get_all_metric_names, get_metric_display, get_follow_up_questions
from ui import data as D
from ui import theme as T

DOMAIN_UI = {
    "product_usage": {
        "icon": "📊", "label": "Product", "subtitle": "Usage · Adoption · Monetization",
        "questions": [
            "How many customers are over-utilizing their plans?",
            "What is the MRR recovery opportunity from right-sizing?",
            "Which regions have the highest over-utilization?",
            "What is our product margin by region?",
        ],
    },
    "marketing": {
        "icon": "📣", "label": "Marketing", "subtitle": "Campaigns · Pipeline · Conversion",
        "questions": [
            "Which campaign brought the highest number of customers?",
            "How are our MQL to SQL conversion rates trending?",
            "What is the ACV from each marketing channel?",
            "How many opportunities were closed last quarter?",
        ],
    },
    "sales": {
        "icon": "💰", "label": "Sales", "subtitle": "Revenue · ARR · Discounts",
        "questions": [
            "Which sales rep gives the highest discounts?",
            "What is our MRR breakdown by customer type?",
            "What is our pipeline coverage ratio?",
            "What is our average contract value by segment?",
        ],
    },
    "hr": {
        "icon": "👥", "label": "People", "subtitle": "Attrition · Hiring · Retention",
        "questions": [
            "Which teams have the highest attrition?",
            "What is our regrettable attrition this quarter?",
            "Are we on track with our hiring plan?",
            "Which teams have the lowest eNPS?",
        ],
    },
    "csup": {
        "icon": "🎧", "label": "Support", "subtitle": "CSAT · SLA · Tickets",
        "questions": [
            "What is our CSAT score?",
            "Who are the top performing support agents?",
            "Are we meeting our SLA targets?",
            "How many tickets were closed in 2026?",
        ],
    },
}

PREFERRED_ORDER = ["product_usage", "sales", "marketing", "hr", "csup"]

HERO_QUESTIONS = [
    "How many customers are over-utilizing their plans?",
    "What is the MRR recovery opportunity from right-sizing?",
    "What is our pipeline coverage ratio?",
    "Which teams have the highest attrition?",
    "What's driving our low CSAT?",
]

GREETINGS = {
    "hi", "hello", "hey", "howdy", "hiya", "sup", "yo",
    "good morning", "good afternoon", "good evening", "good night",
    "how are you", "how are you doing", "how do you do",
    "what's up", "whats up", "what is up", "how's it going", "hows it going",
}


def _init_state() -> None:
    for key, value in {
        "ask_active_domain": "product_usage",
        "ask_pending_question": "",
        "ask_last": None,
    }.items():
        st.session_state.setdefault(key, value)


def _confidence_label(conf: float) -> tuple[str, str]:
    if conf >= 0.65:
        return "High", T.GOOD
    if conf >= 0.40:
        return "Medium", T.WARN
    return "Low", T.BAD


def _headline_kpi(insight: str) -> str:
    first = insight[:110]
    for pattern in (r"\$[\d,]+\.?\d*[KMBkm]?", r"\b\d+\.?\d*x\b"):
        m = re.search(pattern, first)
        if m:
            return m.group()
    m = re.search(r"\b(\d+\.?\d*)%", first)
    if m and float(m.group(1)) >= 10:
        return f"{float(m.group(1)):g}%"
    return ""


def _hero() -> None:
    st.markdown(f"""
<div style="text-align:center;padding:1.4rem 0 0.6rem">
  <h1 style="color:{T.INK};font-size:2.0rem;font-weight:800;margin:0 0 10px;line-height:1.15;
  letter-spacing:-0.03em">Analytics AI Skill System</h1>
  <p style="color:{T.MUTED};font-size:0.95rem;margin:0 0 14px;line-height:1.6;max-width:560px;
  display:inline-block">
    Ask business questions across five analytics domains using governed metric definitions.
  </p>
  <div style="display:inline-flex;align-items:center;gap:7px;margin-bottom:12px;flex-wrap:wrap;
  justify-content:center;background:rgba(99,102,241,0.06);border:1px solid rgba(99,102,241,0.16);
  border-radius:8px;padding:7px 16px">
    <span style="color:{T.ACCENT_DEEP};font-size:0.72rem;font-weight:600">Routed</span>
    <span style="color:{T.FAINT};font-size:0.72rem">→</span>
    <span style="color:{T.ACCENT_DEEP};font-size:0.72rem;font-weight:600">Governed context</span>
    <span style="color:{T.FAINT};font-size:0.72rem">→</span>
    <span style="color:{T.ACCENT_DEEP};font-size:0.72rem;font-weight:600">Answer</span>
    <span style="color:{T.FAINT};font-size:0.72rem">→</span>
    <span style="color:{T.ACCENT_DEEP};font-size:0.72rem;font-weight:600">Evaluated</span>
  </div><br>
  <span style="background:rgba(234,179,8,0.07);border:1px solid rgba(234,179,8,0.20);
  border-radius:5px;padding:3px 12px;font-size:0.64rem;color:#a16207;letter-spacing:0.03em">
    Portfolio demo · Illustrative data only · No production or customer data
  </span>
</div>
""", unsafe_allow_html=True)


def _domain_cards(domains: dict) -> None:
    order = [d for d in PREFERRED_ORDER if d in domains]
    order += [d for d in domains if d not in order]
    if not order:
        return

    cols = st.columns(len(order))
    for col, key in zip(cols, order):
        cfg = DOMAIN_UI.get(key, {"icon": "◈", "label": key.title(), "subtitle": ""})
        if col.button(f"{cfg['icon']}\n{cfg['label']}", key=f"dom_{key}",
                      use_container_width=True, help=cfg.get("subtitle", "")):
            st.session_state["ask_active_domain"] = key
            st.session_state["ask_forced_domain"] = key
            qs = cfg.get("questions", [])
            st.session_state["ask_pending_question"] = qs[0] if qs else ""
            st.rerun()
        if cfg.get("subtitle"):
            col.markdown(
                f'<div style="text-align:center;font-size:0.57rem;color:{T.FAINT};'
                f'margin-top:-6px;line-height:1.3">{cfg["subtitle"]}</div>',
                unsafe_allow_html=True,
            )


def _examples(domains: dict) -> None:
    active = st.session_state.get("ask_active_domain", "product_usage")
    if active not in domains:
        active = next(iter(domains), "product_usage")
    cfg = DOMAIN_UI.get(active, {"icon": "◈", "label": active, "questions": []})

    st.markdown(
        f'<p style="color:{T.FAINT};font-size:0.65rem;font-weight:600;text-transform:uppercase;'
        f'letter-spacing:0.08em;margin:12px 0 5px">Try an example — {cfg["icon"]} {cfg["label"]}</p>',
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    for i, q in enumerate(cfg.get("questions", [])):
        if cols[i % 2].button(q, key=f"ex_{active}_{i}", use_container_width=True):
            st.session_state["ask_pending_question"] = q
            st.session_state["ask_forced_domain"] = active
            st.rerun()


def _routing_strip(routing: RoutingResult, label: str) -> None:
    conf_label, conf_color = _confidence_label(routing.confidence)
    flags = ""
    if routing.no_match:
        flags += T.chip("no keyword match", T.BAD, "rgba(248,113,113,0.13)")
    if routing.is_tie:
        flags += T.chip("tied signal", T.WARN, "rgba(234,179,8,0.13)")
    st.markdown(
        f'<div style="margin:8px 0 12px;display:flex;align-items:center;gap:8px;flex-wrap:wrap">'
        f'<span style="background:rgba(99,102,241,0.12);border:1px solid rgba(99,102,241,0.3);'
        f'color:{T.ACCENT};font-size:0.74rem;font-weight:600;padding:3px 11px;'
        f'border-radius:20px">→ {label}</span>'
        f'<span style="color:{conf_color};font-size:0.71rem;font-weight:600">{conf_label} confidence</span>'
        f'<span style="color:{T.FAINT};font-size:0.69rem;font-style:italic">'
        f'{routing.method} · {routing.version}</span>{flags}</div>',
        unsafe_allow_html=True,
    )


def _render_answer(parsed: dict) -> None:
    insight = parsed.get("insight", "")
    why = parsed.get("why_it_matters", "")
    action = parsed.get("recommended_action", "")

    if not (insight or why or action):
        st.markdown(
            T.panel(T.label_text("Answer") + T.body_text(parsed.get("raw", ""), T.INK, "0.88rem")),
            unsafe_allow_html=True,
        )
        return

    html = ""
    if insight:
        kpi = _headline_kpi(insight)
        kpi_html = (
            f'<div style="text-align:center;margin:8px 0 10px">'
            f'<span style="color:{T.ACCENT};font-size:2.5rem;font-weight:800;'
            f'letter-spacing:-0.03em;line-height:1">{kpi}</span></div>' if kpi else ""
        )
        html += (f'<div style="margin-bottom:16px">{T.label_text("Insight")}{kpi_html}'
                 f'{T.body_text(insight, T.INK, "0.88rem")}</div>')
    if why:
        html += (f'<div style="margin-bottom:16px">{T.label_text("Why it matters")}'
                 f'{T.body_text(why, T.TEXT, "0.85rem")}</div>')
    if action:
        html += (f'<div>{T.label_text("Recommended action")}'
                 f'{T.body_text(action, T.TEXT, "0.85rem")}</div>')
    st.markdown(T.panel(html), unsafe_allow_html=True)


def _trust_badges(ev) -> None:
    def mark(status):
        return "✓" if status in ("PASS", "NONE") else "⚠" if status == "WARN" else "✗"

    def colour(status):
        return T.GOOD if status in ("PASS", "NONE") else T.WARN if status == "WARN" else T.BAD

    items = [
        ("Numeric grounding", ev.groundedness),
        ("Metric recognised", ev.metric_validity),
        ("Relevant answer", ev.relevance),
        ("No unsupported claims", ev.unsupported_claims),
    ]
    inner = "".join(
        f'<span style="color:{colour(s)};font-weight:600;margin-right:16px">'
        f'{mark(s)} {name}</span>'
        for name, s in items
    )
    st.markdown(
        f'<div style="display:flex;flex-wrap:wrap;margin:4px 0 10px;font-size:0.74rem">{inner}</div>',
        unsafe_allow_html=True,
    )


def _eval_detail(ev, routing, domain_name, domain_data, answer_text) -> None:
    label = D.domain_label(domain_name)
    metric_names = get_all_metric_names(domain_data)

    with st.expander("Metric definitions and evaluation detail"):
        rows = "".join(
            f'<tr><td style="padding:5px 0;color:{T.DIM};width:44%">{name}</td>'
            f'<td style="padding:5px 0">{T.status_chip(status)}</td>'
            f'<td style="padding:5px 0 5px 8px;color:{T.FAINT};font-size:0.7rem">{method}</td></tr>'
            for name, status, method in [
                ("Numeric grounding", ev.groundedness, "Deterministic — figure matching"),
                ("Metric recognition", ev.metric_validity, "Deterministic — metric lookup"),
                ("Answer relevance", ev.relevance, "Heuristic — term overlap"),
                ("Unsupported claims", ev.unsupported_claims, "Heuristic — phrase matching"),
            ]
        )
        st.markdown(
            f'<div style="font-size:0.78rem;color:{T.DIM};margin-bottom:10px">'
            f'Domain <strong style="color:{T.MUTED}">{label}</strong> · '
            f'router <strong style="color:{T.MUTED}">{routing.version}</strong> · '
            f'{len(metric_names)} governed metrics in this skill'
            f'</div>'
            f'<table style="width:100%;border-collapse:collapse;font-size:0.78rem">{rows}</table>'
            f'<div style="margin-top:10px;padding-top:8px;border-top:1px solid rgba(255,255,255,0.05);'
            f'font-size:0.7rem;color:{T.FAINT};line-height:1.7;font-style:italic">'
            f'These are the deterministic and heuristic checks only. They detect common failure '
            f'modes; they do not verify that a figure was used correctly. Subjective quality is '
            f'assessed by the human rubric and the AI judge.</div>',
            unsafe_allow_html=True,
        )

        answer_lower = answer_text.lower()
        metrics = domain_data.get("metrics", [])
        relevant = [
            m for m in metrics
            if any(w in answer_lower for w in m.get("name", "").lower().split() if len(w) > 4)
        ] or metrics[:2]
        for m in relevant[:3]:
            st.markdown(
                f'<div style="margin-top:8px;padding:9px 13px;background:rgba(99,102,241,0.05);'
                f'border:1px solid rgba(99,102,241,0.13);border-radius:7px">'
                f'{T.label_text(m.get("name", ""))}'
                f'{T.body_text(m.get("definition") or m.get("business_definition", ""), T.TEXT, "0.78rem")}'
                f'</div>',
                unsafe_allow_html=True,
            )


def _live_judge(question: str, answer: str, context: str, routed: str) -> None:
    """Run the LLM judge on an ad-hoc answer, using the same prompt as the batch runs."""
    from llm_judge import judge_response

    st.markdown(
        f'<p style="color:{T.FAINT};font-size:0.65rem;font-weight:600;text-transform:uppercase;'
        f'letter-spacing:0.08em;margin:14px 0 5px">Evaluate this answer</p>',
        unsafe_allow_html=True,
    )
    col_a, col_b = st.columns([1, 3])
    if not col_a.button("Run AI judge", key="live_judge", use_container_width=True):
        col_b.markdown(
            f'<div style="color:{T.FAINT};font-size:0.73rem;padding-top:10px">'
            f'Scores this answer against the same six-dimension rubric used across the golden set.'
            f'</div>',
            unsafe_allow_html=True,
        )
        return

    ad_hoc_case = {
        "eval_id": "ad-hoc",
        "question": question,
        "expected_behavior": (
            "Answer the question using only the governed context. If the context does not "
            "contain what was asked, say so rather than supplying a figure."
        ),
        "expected_answer_summary": "(no reference answer — this is an ad-hoc question)",
        "required_facts": [],
        "forbidden_claims": [],
        "governed_context": {"skill": routed, "metrics": [], "available": True},
        "test_type": "standard",
    }

    with st.spinner("Judging…"):
        record = judge_response(ad_hoc_case, answer, context)

    if not record.get("parse_ok"):
        T.note(f"The judge did not return usable output: {record.get('parse_error')}", "warn")
        return

    T.note(
        "This is a <strong>reference-free</strong> judgement — an ad-hoc question has no golden "
        "reference answer, required facts or forbidden claims. Golden-set judgements are "
        "reference-based and considerably more reliable.",
        "info",
    )

    from human_evals import DIMENSIONS
    cols = st.columns(6, gap="small")
    for col, dim in zip(cols, DIMENSIONS):
        value = record["scores"].get(dim)
        col.markdown(
            T.metric_card(T.dim_label(dim), f"{value}/5" if value else T.DASH,
                          accent=T.DIMENSION_COLORS.get(dim, T.ACCENT),
                          value_color=T.score_color(value)),
            unsafe_allow_html=True,
        )

    st.markdown(
        T.panel(
            T.label_text("Judge rationale")
            + T.body_text(record.get("reasoning_summary") or "(none returned)", T.TEXT, "0.8rem")
            + f'<div style="margin-top:9px">{T.pass_chip(record["pass"])}'
            + T.chip(f"overall {T.score(record['overall_score'])}/5", T.MUTED)
            + T.chip(f"confidence {T.score(record.get('confidence'))}", T.MUTED)
            + (T.chip(f"failure: {record['failure_mode']}", T.BAD, "rgba(248,113,113,0.13)")
               if record["failure_mode"] != "none" else "")
            + "</div>",
            accent=T.WARN,
        ),
        unsafe_allow_html=True,
    )


def render(state: D.EvalState) -> None:
    _init_state()
    domains = state.domains

    _hero()

    if not domains:
        st.error("No domain skill files found. Verify the YAML files are present in the repo root.")
        return

    _domain_cards(domains)
    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

    if st.session_state.get("ask_pending_question"):
        st.session_state["ask_question_input"] = st.session_state["ask_pending_question"]
        st.session_state["ask_pending_question"] = ""

    question = st.text_input(
        "question",
        placeholder="e.g. What is our biggest revenue opportunity?",
        label_visibility="collapsed",
        key="ask_question_input",
    )

    _examples(domains)

    if not question:
        return

    normalised = question.strip().lower().rstrip("!?.,")
    if normalised in GREETINGS or any(normalised.startswith(g + " ") for g in ("hi", "hello", "hey")):
        st.markdown(
            T.panel(
                T.label_text("Hello", T.ACCENT)
                + T.body_text(
                    "I answer business questions across five analytics domains — Product, Sales, "
                    "Marketing, People and Support — grounded in governed metric definitions. Try "
                    "<em>What is our MRR recovery opportunity?</em> or pick an example above.",
                    T.TEXT, "0.84rem",
                ),
            ),
            unsafe_allow_html=True,
        )
        return

    if not get_api_key():
        T.note(
            "<strong>No GROQ_API_KEY configured.</strong> Answer generation needs a Groq key in "
            "the environment or in <code>.streamlit/secrets.toml</code>. Everything else in this "
            "application reads stored evaluation records and works without one.",
            "warn",
        )
        return

    forced = st.session_state.pop("ask_forced_domain", None)
    if forced and forced in domains:
        routing = RoutingResult(
            domain=forced, confidence=0.85, method="manual_selection",
            top_domains=[(forced, 5.0)], is_ambiguous=False,
            reasoning=f"Domain selected directly from the {D.domain_label(forced)} examples.",
            version="manual",
        )
    else:
        routing = classify_domain(
            question, domains,
            fallback_domain=st.session_state.get("ask_active_domain", ""),
        )

    if routing.no_match:
        st.markdown(
            T.panel(
                T.label_text("Outside the governed knowledge base", T.DIM)
                + T.body_text(
                    "No keyword in this question matched any domain skill file. Rather than "
                    "answering from whichever domain scored least badly, the router reports that "
                    "it has no basis for a decision. Try business terms from Product, Sales, "
                    "Marketing, People or Support — or pick a domain above.",
                    T.TEXT, "0.83rem",
                ),
                accent=T.DIM,
            ),
            unsafe_allow_html=True,
        )
        return

    if routing.is_tie:
        top = routing.top_domains[:2]
        T.note(
            "<strong>Tied routing signal.</strong> "
            f"{D.domain_label(top[0][0])} and {D.domain_label(top[1][0])} scored effectively "
            "equally, so this selection is arbitrary. The answer below uses "
            f"{D.domain_label(routing.domain)} — select a domain above to override it.",
            "warn",
        )
    elif routing.is_ambiguous and routing.confidence < AMBIGUITY_THRESHOLD:
        T.note(
            "<strong>Low routing confidence.</strong> This question does not map cleanly onto one "
            "domain. The answer below may be grounded in the wrong governed context — select a "
            "domain above to be explicit.",
            "warn",
        )

    domain_name = routing.domain
    if domain_name not in domains:
        st.error("Could not identify a domain — try rephrasing with more specific business terms.")
        return

    domain_data = domains[domain_name]
    _routing_strip(routing, D.domain_label(domain_name))

    with st.spinner("Analysing…"):
        context = build_context(domain_data)
        result = ask_groq(question, context, prompt_version=DEFAULT_PROMPT_VERSION)

    if result["error"]:
        kind = result.get("error_kind", "unknown")
        heading = {
            "quota_exhausted": "Daily model quota reached",
            "rate_limited": "Model is rate limited",
            "model_unavailable": "Model unavailable",
            "auth_failed": "API key rejected",
        }.get(kind, "Model call failed")
        T.note(f"<strong>{heading}.</strong> {result['answer']}", "warn")
        if kind == "quota_exhausted":
            st.markdown(
                T.panel(
                    T.label_text("The rest of the application still works", T.ACCENT)
                    + T.body_text(
                        "Answer generation is the only feature that calls the model live. The "
                        "evaluation pages read stored records from a completed run, so the "
                        "Quality Dashboard, AI Judge, Alignment, Failure Analysis, Golden "
                        "Dataset and Regression pages are all fully populated right now.",
                        T.TEXT, "0.82rem",
                    ),
                ),
                unsafe_allow_html=True,
            )
        return

    answer_text = result["answer"]
    _render_answer(parse_structured_answer(answer_text))

    ev = evaluate_response(
        question=question, answer=answer_text, context=context,
        domain_data=domain_data, routing_confidence=routing.confidence,
    )
    _trust_badges(ev)
    _eval_detail(ev, routing, domain_name, domain_data, answer_text)

    st.markdown(
        f'<div style="font-size:0.68rem;color:{T.FAINT};margin:2px 0 6px">'
        f'{result["model"]} · {result["prompt_version"]} · '
        f'{T.score(result.get("latency_seconds"), 2)}s</div>',
        unsafe_allow_html=True,
    )

    _live_judge(question, answer_text, context, domain_name)

    follow_ups = get_follow_up_questions(domain_data, question)
    if follow_ups:
        st.markdown(
            f'<p style="color:{T.FAINT};font-size:0.65rem;font-weight:600;text-transform:uppercase;'
            f'letter-spacing:0.08em;margin:16px 0 5px">Explore further</p>',
            unsafe_allow_html=True,
        )
        for fq in follow_ups:
            if st.button(f"→  {fq}", key=f"fu_{abs(hash(fq))}", use_container_width=True):
                st.session_state["ask_pending_question"] = fq
                st.rerun()
