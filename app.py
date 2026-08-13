"""
app.py — Analytics AI Skill System
UI orchestration layer. All domain logic lives in router.py, skills.py, llm.py, evals.py.
"""

import os
import pathlib
import logging
import streamlit as st

from router import classify_domain, get_ambiguous_domains, AMBIGUITY_THRESHOLD
from skills import (load_domains, build_context, get_follow_up_questions,
                    get_all_metric_names, get_metric_display)
from llm import ask_groq, parse_structured_answer
from evals import evaluate_response, run_routing_eval, EVAL_DATASET

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Analytics AI Skill System",
    page_icon="◈",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── GA4 ────────────────────────────────────────────────────────────────────────
GA_ID = "G-BEKZJV5CJJ"

def _inject_ga4():
    ga_script = (
        f'<script async src="https://www.googletagmanager.com/gtag/js?id={GA_ID}"></script>'
        f'<script>window.dataLayer=window.dataLayer||[];'
        f'function gtag(){{dataLayer.push(arguments);}}gtag("js",new Date());'
        f'gtag("config","{GA_ID}");</script>'
    )
    try:
        index_path = pathlib.Path(st.__file__).parent / "static" / "index.html"
        html = index_path.read_text()
        if GA_ID not in html:
            index_path.write_text(html.replace("</head>", ga_script + "</head>"))
    except Exception:
        pass

_inject_ga4()

# ── CSS ────────────────────────────────────────────────────────────────────────
# Using a theme.toml approach is more reliable than CSS injection for backgrounds,
# but we still patch key elements here for the dark enterprise look.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    background-color: #0d0b1e !important;
}

/* Main background */
.stApp {
    background: linear-gradient(135deg, #0d0b1e 0%, #111827 50%, #0d0b1e 100%) !important;
}

section[data-testid="stMain"] > div {
    background: transparent !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden !important; }
[data-testid="stHeader"] { background: transparent !important; height: 0 !important; }
[data-testid="stToolbar"] { display: none !important; }

/* Content width */
.block-container {
    max-width: 720px !important;
    padding: 2rem 1.5rem 3rem !important;
    margin: 0 auto !important;
}

/* Make all containers transparent */
[data-testid="stVerticalBlock"],
[data-testid="element-container"],
[data-testid="stHorizontalBlock"],
div[class*="stMarkdown"] {
    background: transparent !important;
}

/* ── All Streamlit buttons ── */
/* Target the button element inside stButton wrapper — avoids breaking click events */
div[data-testid="stButton"] > button {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    color: #94a3b8 !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    padding: 9px 14px !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: background 0.15s ease, border-color 0.15s ease, color 0.15s ease !important;
    pointer-events: auto !important;
}
div[data-testid="stButton"] > button:hover {
    background: rgba(99,102,241,0.15) !important;
    border-color: rgba(99,102,241,0.4) !important;
    color: #e2e8f0 !important;
}
div[data-testid="stButton"] > button:active {
    background: rgba(99,102,241,0.25) !important;
}

/* ── Text input ── */
div[data-testid="stTextInput"] > div > div {
    background: rgba(15,12,40,0.95) !important;
    border: 1px solid rgba(99,102,241,0.35) !important;
    border-radius: 10px !important;
}
div[data-testid="stTextInput"] input {
    background: transparent !important;
    color: #f1f5f9 !important;
    font-size: 0.95rem !important;
    padding: 14px 18px !important;
    caret-color: #818cf8 !important;
}
div[data-testid="stTextInput"] input::placeholder {
    color: #475569 !important;
}
div[data-testid="stTextInput"] input:focus {
    box-shadow: 0 0 0 2px rgba(129,140,248,0.2) !important;
}
div[data-testid="stTextInput"] label {
    color: #64748b !important;
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}

/* ── Expander ── */
div[data-testid="stExpander"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
    margin-bottom: 6px !important;
}
div[data-testid="stExpander"] summary {
    color: #64748b !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    padding: 10px 14px !important;
}
div[data-testid="stExpander"] summary:hover {
    color: #94a3b8 !important;
}

/* ── Spinner ── */
div[data-testid="stSpinner"] > div {
    border-top-color: #818cf8 !important;
}

/* ── Metrics ── */
div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
    padding: 12px 16px !important;
}
div[data-testid="metric-container"] label {
    color: #64748b !important;
    font-size: 0.72rem !important;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #e2e8f0 !important;
    font-size: 1.4rem !important;
    font-weight: 700 !important;
}

/* ── Tables (eval dashboard) ── */
table { color: #94a3b8 !important; }
th { color: #64748b !important; font-size: 0.75rem !important; }

/* Markdown text defaults */
p, li { color: #94a3b8 !important; }
h1 { color: #f1f5f9 !important; }
h2, h3, h4 { color: #e2e8f0 !important; }
code { background: rgba(99,102,241,0.12) !important; color: #a5b4fc !important; border-radius: 4px !important; }
pre { background: rgba(0,0,0,0.3) !important; border: 1px solid rgba(255,255,255,0.07) !important; border-radius: 8px !important; }
strong { color: #e2e8f0 !important; }
hr { border-color: rgba(255,255,255,0.07) !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
DOMAIN_UI = {
    "product_usage": {
        "icon": "◈", "label": "Product",
        "questions": [
            "Which customers are over-utilizing their plans?",
            "What is the MRR recovery opportunity from right-sizing?",
            "Which regions have the highest over-utilization?",
            "What is our product margin by region?",
        ]
    },
    "marketing": {
        "icon": "◉", "label": "Marketing",
        "questions": [
            "Which campaign brought the highest number of customers?",
            "How are our MQL to SQL conversion rates trending?",
            "What is the ACV from each marketing channel?",
            "How many opportunities were closed last quarter?",
        ]
    },
    "sales": {
        "icon": "◆", "label": "Sales",
        "questions": [
            "What is our pipeline coverage ratio?",
            "Which sales rep gives the highest discounts?",
            "What is our MRR breakdown by customer type?",
            "How many new customers did we add this quarter?",
        ]
    },
    "hr": {
        "icon": "◎", "label": "HR",
        "questions": [
            "Which teams have the highest attrition?",
            "What is our regrettable attrition this quarter?",
            "Are we on track with our hiring plan?",
            "What is our new hire 90-day retention rate?",
        ]
    },
    "csup": {
        "icon": "◇", "label": "Support",
        "questions": [
            "What is our CSAT score?",
            "Who are the top performing support agents?",
            "Are we meeting our SLA targets?",
            "How many tickets were closed in 2026?",
        ]
    },
}

PREFERRED_ORDER = ["product_usage", "sales", "marketing", "hr", "csup"]

DOMAIN_LABELS = {
    "product_usage": "Product Analytics",
    "marketing":     "Marketing Analytics",
    "sales":         "Sales Analytics",
    "hr":            "People Analytics",
    "csup":          "Support Analytics",
}

# Top-level example questions shown on the landing page (before any domain is selected)
HERO_QUESTIONS = [
    "Which customers are over-utilizing their plans?",
    "What is the MRR recovery opportunity from right-sizing?",
    "Which regions have the highest over-utilization?",
    "What's driving churn risk in our customer base?",
    "How is our sales pipeline performing?",
]


# ── Session state ──────────────────────────────────────────────────────────────
def _init():
    for k, v in {
        "active_domain": "product_usage",
        "pending_question": "",
        "show_page": "main",
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()


# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data
def cached_load_domains():
    return load_domains(".")

domains = cached_load_domains()


# ── Helpers ────────────────────────────────────────────────────────────────────
def api_key():
    return st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))


def _card(content_html: str, accent: str = "#6366f1") -> str:
    """Wrap HTML in a styled card."""
    return (
        f'<div style="background:rgba(255,255,255,0.025);border:1px solid rgba(255,255,255,0.07);'
        f'border-left:3px solid {accent};border-radius:12px;padding:20px 24px;margin-bottom:12px">'
        f'{content_html}</div>'
    )

def _label(text: str) -> str:
    return (f'<div style="color:#6366f1;font-size:0.65rem;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:0.1em;margin-bottom:6px">{text}</div>')

def _body(text: str, muted: bool = False) -> str:
    color = "#94a3b8" if muted else "#e2e8f0"
    return f'<div style="color:{color};font-size:0.9rem;line-height:1.75">{text}</div>'


# ── Rendering functions ────────────────────────────────────────────────────────

def render_hero():
    st.markdown("""
<div style="text-align:center;padding:2.5rem 0 1.5rem">
  <div style="display:inline-flex;gap:8px;flex-wrap:wrap;justify-content:center;margin-bottom:20px">
    <span style="background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.25);
    color:#818cf8;font-size:0.68rem;font-weight:600;padding:4px 12px;border-radius:20px;
    letter-spacing:0.06em">Multi-domain</span>
    <span style="background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.25);
    color:#818cf8;font-size:0.68rem;font-weight:600;padding:4px 12px;border-radius:20px;
    letter-spacing:0.06em">Governed metrics</span>
    <span style="background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.25);
    color:#818cf8;font-size:0.68rem;font-weight:600;padding:4px 12px;border-radius:20px;
    letter-spacing:0.06em">Grounded answers</span>
    <span style="background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.25);
    color:#818cf8;font-size:0.68rem;font-weight:600;padding:4px 12px;border-radius:20px;
    letter-spacing:0.06em">Evaluated responses</span>
  </div>
  <h1 style="color:#f1f5f9;font-size:2.1rem;font-weight:700;margin:0 0 12px;line-height:1.2;
  letter-spacing:-0.02em">Analytics AI Skill System</h1>
  <p style="color:#64748b;font-size:0.95rem;margin:0 0 20px;line-height:1.6;max-width:500px;
  display:inline-block">
    Ask business questions in plain English.<br>
    Get governed, explainable analytics answers.
  </p>
  <div style="display:inline-block;background:rgba(234,179,8,0.06);
  border:1px solid rgba(234,179,8,0.18);border-radius:6px;padding:5px 16px;
  font-size:0.7rem;color:#92400e;letter-spacing:0.04em">
    Demo environment · Illustrative data only · No production or customer data
  </div>
</div>
""", unsafe_allow_html=True)


def render_input_section():
    st.markdown("""
<p style="color:#475569;font-size:0.7rem;font-weight:600;text-transform:uppercase;
letter-spacing:0.08em;margin-bottom:6px">Ask a business question</p>
""", unsafe_allow_html=True)

    question = st.text_input(
        "question",
        placeholder="e.g. What is our biggest revenue opportunity?",
        label_visibility="collapsed",
        key="question_input"
    )

    return question


def render_example_questions():
    st.markdown("""
<p style="color:#374151;font-size:0.68rem;font-weight:600;text-transform:uppercase;
letter-spacing:0.08em;margin:16px 0 8px">Try an analysis</p>
""", unsafe_allow_html=True)

    for q in HERO_QUESTIONS:
        if st.button(q, key=f"hero_{hash(q)}", use_container_width=True):
            st.session_state["pending_question"] = q
            st.rerun()


def render_domain_explorer():
    with st.expander("Explore by domain", expanded=False):
        display_order = [d for d in PREFERRED_ORDER if d in domains]
        display_order += [d for d in domains if d not in display_order]

        dcols = st.columns(len(display_order))
        for i, dk in enumerate(display_order):
            cfg = DOMAIN_UI.get(dk, {"icon": "◈", "label": dk.replace("_", " ").title()})
            if dcols[i].button(f"{cfg['icon']} {cfg['label']}", key=f"dom_{dk}",
                               use_container_width=True):
                st.session_state["active_domain"] = dk
                qs = cfg.get("questions", [])
                st.session_state["pending_question"] = qs[0] if qs else ""
                st.rerun()

        active = st.session_state.get("active_domain", display_order[0] if display_order else "")
        if active not in domains and display_order:
            active = display_order[0]
        active_cfg = DOMAIN_UI.get(active, {"icon": "◈", "label": active, "questions": []})

        st.markdown(f"""
<p style="color:#374151;font-size:0.68rem;font-weight:600;text-transform:uppercase;
letter-spacing:0.08em;margin:14px 0 6px">{active_cfg['icon']} {active_cfg['label']} — example questions</p>
""", unsafe_allow_html=True)

        qcols = st.columns(2)
        for i, q in enumerate(active_cfg.get("questions", [])):
            if qcols[i % 2].button(q, key=f"dq_{active}_{i}", use_container_width=True):
                st.session_state["question_input"] = q
                st.rerun()


def render_routing_indicator(routing, domain_label):
    pct = int(routing.confidence * 100)
    conf_color = "#22c55e" if pct >= 70 else "#eab308" if pct >= 45 else "#f87171"
    st.markdown(f"""
<div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;
margin:14px 0 16px;padding:10px 16px;
background:rgba(99,102,241,0.06);border:1px solid rgba(99,102,241,0.15);
border-radius:8px">
  <span style="color:#6366f1;font-size:0.68rem;font-weight:700;text-transform:uppercase;
  letter-spacing:0.08em">Routed to</span>
  <span style="color:#e2e8f0;font-size:0.85rem;font-weight:600">{domain_label}</span>
  <span style="color:#1e293b">·</span>
  <span style="color:{conf_color};font-size:0.8rem;font-weight:600">{pct}% confidence</span>
  <span style="color:#1e293b">·</span>
  <span style="color:#334155;font-size:0.75rem;font-style:italic">Keyword-based router</span>
</div>
""", unsafe_allow_html=True)


def render_answer(parsed):
    insight = parsed.get("insight", "")
    why     = parsed.get("why_it_matters", "")
    action  = parsed.get("recommended_action", "")
    raw     = parsed.get("raw", "")

    if insight or why or action:
        sections = ""
        if insight:
            sections += (
                f'<div style="margin-bottom:18px">'
                f'{_label("Insight")}'
                f'{_body(insight)}'
                f'</div>'
            )
        if why:
            sections += (
                f'<div style="margin-bottom:18px">'
                f'{_label("Why it matters")}'
                f'{_body(why, muted=True)}'
                f'</div>'
            )
        if action:
            sections += (
                f'<div>'
                f'{_label("Recommended action")}'
                f'{_body(action, muted=True)}'
                f'</div>'
            )
        st.markdown(_card(sections), unsafe_allow_html=True)
    else:
        st.markdown(_card(
            f'{_label("Answer")}{_body(raw)}'
        ), unsafe_allow_html=True)


def render_eval_panel(ev, routing, domain_name):
    pct = int(ev.overall_quality * 100)
    ql_color = "#22c55e" if ev.quality_label == "HIGH" else "#eab308" if ev.quality_label == "MEDIUM" else "#f87171"
    label = DOMAIN_LABELS.get(domain_name, domain_name.replace("_", " ").title())

    def badge(status):
        if status in ("PASS", "NONE"):
            return '<span style="color:#22c55e;font-weight:600">Pass</span>'
        elif status == "WARN":
            return '<span style="color:#eab308;font-weight:600">Warn</span>'
        return '<span style="color:#f87171;font-weight:600">Fail</span>'

    with st.expander("Answer quality & trust", expanded=False):
        st.markdown(f"""
<div style="margin-bottom:14px">
  <span style="color:#475569;font-size:0.68rem;font-weight:600;text-transform:uppercase;
  letter-spacing:0.07em">Overall quality</span>
  <span style="color:{ql_color};font-size:1rem;font-weight:700;margin-left:10px">
    {ev.quality_label} · {pct}%
  </span>
</div>

<table style="width:100%;border-collapse:collapse;font-size:0.8rem">
  <tr>
    <td style="padding:6px 0;color:#64748b;width:42%">Numeric grounding check</td>
    <td style="padding:6px 0">{badge(ev.groundedness)}</td>
    <td style="padding:6px 0 6px 10px;color:#334155;font-size:0.7rem">{ev.methods['groundedness']}</td>
  </tr>
  <tr>
    <td style="padding:6px 0;color:#64748b">Metric definition check</td>
    <td style="padding:6px 0">{badge(ev.metric_validity)}</td>
    <td style="padding:6px 0 6px 10px;color:#334155;font-size:0.7rem">{ev.methods['metric_validity']}</td>
  </tr>
  <tr>
    <td style="padding:6px 0;color:#64748b">Answer relevance</td>
    <td style="padding:6px 0">{badge(ev.relevance)}</td>
    <td style="padding:6px 0 6px 10px;color:#334155;font-size:0.7rem">{ev.methods['relevance']}</td>
  </tr>
  <tr>
    <td style="padding:6px 0;color:#64748b">Unsupported claim check</td>
    <td style="padding:6px 0">{badge(ev.unsupported_claims)}</td>
    <td style="padding:6px 0 6px 10px;color:#334155;font-size:0.7rem">{ev.methods['unsupported_claims']}</td>
  </tr>
</table>

<div style="margin-top:14px;padding-top:12px;border-top:1px solid rgba(255,255,255,0.05);
font-size:0.75rem;color:#334155;line-height:1.9">
  Source: {label} skill &nbsp;·&nbsp; Governed YAML definitions &nbsp;·&nbsp;
  Illustrative dataset &nbsp;·&nbsp;
  {int(routing.confidence * 100)}% routing confidence (keyword method)
</div>
""", unsafe_allow_html=True)


def render_trust_panel(domain_name, domain_data, routing):
    label = DOMAIN_LABELS.get(domain_name, domain_name.replace("_", " ").title())
    pct = int(routing.confidence * 100)
    metric_names = get_all_metric_names(domain_data)
    metrics_str = ", ".join(metric_names[:3]) + ("…" if len(metric_names) > 3 else "")

    with st.expander("Why you can trust this answer", expanded=False):
        st.markdown(f"""
<div style="font-size:0.82rem;color:#475569;line-height:2.1">
  <span style="color:#22c55e">✓</span> &nbsp;<strong style="color:#94a3b8">{label}</strong> skill loaded<br>
  <span style="color:#22c55e">✓</span> &nbsp;Governed metric definitions: <em style="color:#64748b">{metrics_str}</em><br>
  <span style="color:#22c55e">✓</span> &nbsp;LLM constrained to supplied context — cannot invent KPI definitions<br>
  <span style="color:#22c55e">✓</span> &nbsp;Illustrative demo dataset — not production or customer data<br>
  <span style="color:#22c55e">✓</span> &nbsp;Routing confidence {pct}% &nbsp;·&nbsp; <em>Keyword-based — not semantic or embedding routing</em>
</div>
<div style="margin-top:12px;font-size:0.7rem;color:#1e293b;font-style:italic">
  Evaluation is deterministic figure-matching and heuristic phrase-detection — not LLM-based verification.
</div>
""", unsafe_allow_html=True)


def render_metric_definitions(domain_data, answer_text):
    metrics = domain_data.get("metrics", [])
    if not metrics:
        return
    answer_lower = answer_text.lower()
    relevant = [
        m for m in metrics
        if any(w in answer_lower for w in m.get("name", "").lower().split() if len(w) > 4)
    ]
    if not relevant:
        relevant = metrics[:2]
    for m in relevant[:3]:
        with st.expander(f"View metric definition: {m.get('name', '')}", expanded=False):
            st.markdown(get_metric_display(m))


def render_follow_ups(followups):
    if not followups:
        return
    st.markdown("""
<p style="color:#374151;font-size:0.68rem;font-weight:600;text-transform:uppercase;
letter-spacing:0.08em;margin:14px 0 6px">Explore further</p>
""", unsafe_allow_html=True)
    for fq in followups:
        if st.button(f"→  {fq}", key=f"fq_{hash(fq)}", use_container_width=True):
            st.session_state["question_input"] = fq
            st.rerun()


def render_ambiguity_ui(routing, domains):
    ambig_domains = get_ambiguous_domains(routing, domains)
    domain_lines = "\n".join(
        f"  {DOMAIN_UI.get(d, {}).get('icon','◈')} &nbsp;<strong style='color:#94a3b8'>"
        f"{DOMAIN_LABELS.get(d, d)}</strong>"
        for d in ambig_domains
    )
    st.markdown(f"""
<div style="background:rgba(234,179,8,0.05);border:1px solid rgba(234,179,8,0.2);
border-radius:10px;padding:16px 20px;margin:10px 0 14px">
  <div style="color:#92400e;font-size:0.8rem;font-weight:600;margin-bottom:8px">
    This question may involve multiple business signals
  </div>
  <div style="color:#64748b;font-size:0.8rem;line-height:2">
    Potential areas:<br>
    {domain_lines}
  </div>
  <div style="color:#374151;font-size:0.72rem;margin-top:10px">
    Routing confidence: {int(routing.confidence * 100)}% — below the automatic-routing threshold.<br>
    Choose an area to investigate first, or rephrase with more specific terms.
  </div>
</div>
""", unsafe_allow_html=True)

    cols = st.columns(len(ambig_domains))
    for i, d in enumerate(ambig_domains):
        cfg = DOMAIN_UI.get(d, {})
        label = DOMAIN_LABELS.get(d, d.replace("_", " ").title())
        if cols[i].button(f"{cfg.get('icon','◈')} {label}", key=f"amb_{d}",
                          use_container_width=True):
            st.session_state["active_domain"] = d
            st.session_state["pending_question"] = st.session_state.get("question_input", "")
            routing.domain = d
            routing.confidence = 0.6
            routing.is_ambiguous = False
            st.rerun()

    return routing


def render_how_it_works():
    with st.expander("How this system works", expanded=False):
        st.markdown("""
**System architecture**

```
User Question (plain English)
        ↓
Domain Router  ←  keyword scoring, labelled accurately
        ↓
Analytics Skill  ←  YAML file for the detected domain
        ↓
Governed Metric Layer  ←  definitions, formulas, owners, validation dates
        ↓
LLM  ←  Groq / Llama 3.1, constrained to supplied context
        ↓
Evaluation  ←  deterministic + heuristic checks
        ↓
Business Answer  ←  Insight / Why it matters / Recommended action
```

**Why YAML skill files?**
Domain knowledge is separated from application logic. Adding a new domain — Finance, Operations, Legal — requires only a new YAML file. No application rewrite. Metric definitions are owned by the analytics team, not inferred by the model.

**What the LLM does:** Reasons over the supplied skill context and formats the response into structured sections. It does not access live data, query a database, or invent metric definitions.

**What the evaluation layer checks:** Numeric figure matching against the governed context (deterministic), metric name recognition (deterministic), answer relevance via term overlap (heuristic), and unsupported generalisation phrase detection (heuristic). All methods are labelled.
""")


def render_prototype_vs_production():
    with st.expander("Prototype → Production", expanded=False):
        st.markdown("""
**What this portfolio implementation actually uses:**
- Streamlit (UI)
- YAML domain skill files (governed metric layer)
- Keyword-based domain routing (deterministic, not semantic)
- Groq / Llama 3.1 (LLM)
- Illustrative pre-built data embedded in skill files — no live database
- Deterministic + heuristic response evaluation

---

**What a production implementation would require:**

| Capability | Production approach |
|---|---|
| Data layer | Snowflake / data warehouse with live queries |
| Semantic layer | dbt semantic layer or Cube |
| Routing | Embedding-based or LLM-based semantic routing |
| Access control | RBAC, row-level security, PII masking |
| Observability | Latency, cost, token monitoring |
| Evaluation | Held-out eval datasets, automated regression |
| Feedback | Human rating loop → retraining signal |
| Security | Prompt injection testing, audit logging |
| Reliability | Caching, fallback models, graceful degradation |

**None of the production capabilities above exist in this demo.**
""")


def render_footer():
    st.markdown("""
<div style="border-top:1px solid rgba(255,255,255,0.05);margin-top:2rem;
padding:1rem 0 0.5rem;text-align:center;font-size:0.7rem;color:#1e293b">
  Built by <a href="https://ajantika.github.io"
  style="color:#6366f1;text-decoration:none;font-weight:600">Ajantika Paul</a>
  &nbsp;·&nbsp;
  <a href="https://github.com/ajantika/analytics-ai-skill-system"
  style="color:#6366f1;text-decoration:none">GitHub</a>
  &nbsp;·&nbsp; YAML Semantic Layer &nbsp;·&nbsp; Keyword routing &nbsp;·&nbsp; Groq / Llama 3.1
</div>
""", unsafe_allow_html=True)


# ── Navigation ─────────────────────────────────────────────────────────────────
_, nav_r = st.columns([4, 1])
with nav_r:
    if st.session_state.get("show_page") == "main":
        if st.button("System eval →", use_container_width=True):
            st.session_state["show_page"] = "eval"
            st.rerun()
    else:
        if st.button("← Main", use_container_width=True):
            st.session_state["show_page"] = "main"
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# EVAL DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.get("show_page") == "eval":
    st.markdown("""
<div style="padding:1.5rem 0 1rem">
  <h2 style="color:#f1f5f9;font-size:1.5rem;font-weight:700;margin:0 0 6px">System Evaluation</h2>
  <p style="color:#475569;font-size:0.82rem;margin:0">
    Routing accuracy and answer quality measured against a curated evaluation dataset.<br>
    A trustworthy AI analytics system must be evaluated — not just demoed.
  </p>
</div>
""", unsafe_allow_html=True)

    eval_results = run_routing_eval(classify_domain, domains)
    accuracy = eval_results["accuracy"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Routing accuracy", f"{int(accuracy * 100)}%")
    c2.metric("Eval questions", eval_results["total"])
    c3.metric("Correct routes", eval_results["correct"])
    c4.metric("Failures", len(eval_results["failures"]))

    st.markdown("""
<div style="background:rgba(99,102,241,0.05);border:1px solid rgba(99,102,241,0.12);
border-radius:8px;padding:12px 16px;margin:12px 0;font-size:0.78rem;color:#475569">
  <strong style="color:#6366f1">Evaluation method:</strong>
  Keyword-based routing is tested against 11 labelled questions across 5 domains.
  Routing confidence is computed from keyword score separation between the top and runner-up domains.
  2 intentionally ambiguous questions are excluded from accuracy calculation.
  This is not a held-out test set — a production evaluation would use a larger,
  separately curated dataset with human-labelled ground truth.
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Routing results")

    for r in eval_results["results"]:
        icon = "✓" if r["correct"] else "✗"
        color = "#22c55e" if r["correct"] else "#f87171"
        exp_label = DOMAIN_LABELS.get(r["expected"], r["expected"])
        pred_label = DOMAIN_LABELS.get(r["predicted"], r["predicted"])
        conf = int(r["confidence"] * 100)
        with st.expander(f"[{icon}] {r['question'][:75]}"):
            st.markdown(f"""
| | |
|---|---|
| Expected domain | {exp_label} |
| Predicted domain | {pred_label} |
| Confidence | {conf}% (keyword routing) |
| Result | <span style="color:{color};font-weight:600">{'Correct' if r['correct'] else 'Incorrect'}</span> |
| Note | {r['note']} |
""", unsafe_allow_html=True)

    if eval_results["failures"]:
        st.markdown("---")
        st.markdown("#### Failure cases")
        st.markdown("""
<div style="font-size:0.8rem;color:#475569;margin-bottom:10px">
  These questions were mis-routed by the keyword router. Failures are expected and informative —
  they identify where keyword scoring breaks down and where semantic routing would help.
  In a production system, failures trigger keyword list review or routing method upgrade.
</div>
""", unsafe_allow_html=True)
        for f in eval_results["failures"]:
            exp = DOMAIN_LABELS.get(f["expected"], f["expected"])
            pred = DOMAIN_LABELS.get(f["predicted"], f["predicted"])
            st.markdown(
                f'- **"{f["question"]}"** → expected `{exp}`, predicted `{pred}`'
            )
    else:
        st.markdown("""
<div style="background:rgba(34,197,94,0.05);border:1px solid rgba(34,197,94,0.15);
border-radius:8px;padding:12px 16px;margin:12px 0;font-size:0.78rem;color:#166534">
  No routing failures on this eval set. Note: 100% accuracy on a small keyword-matched dataset
  is expected — it does not mean the router is robust to arbitrary real-world questions.
  Edge cases and ambiguous questions remain failure modes.
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
#### Known limitations and edge cases

- **Ambiguous questions** (e.g. *"Why is revenue declining?"*) correctly surface low confidence
  rather than confidently mis-routing — this is by design.
- **Cross-domain questions** (e.g. *"How are we doing overall?"*) have no correct single-domain answer
  and expose the fundamental limitation of single-domain keyword routing.
- **Keyword collisions** (e.g. MRR appears in both Sales and Product) can route to the wrong domain
  when the question lacks additional disambiguating terms.
- A production router would use embedding similarity or an LLM classifier with a confidence threshold.
""")

    render_footer()
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PAGE
# ══════════════════════════════════════════════════════════════════════════════
render_hero()

st.markdown('<div style="margin-bottom:6px"></div>', unsafe_allow_html=True)

# Apply any pending question from button clicks BEFORE the text input renders
if st.session_state.get("pending_question"):
    st.session_state["question_input"] = st.session_state["pending_question"]
    st.session_state["pending_question"] = ""

question = render_input_section()

render_example_questions()

st.markdown('<div style="margin-top:6px"></div>', unsafe_allow_html=True)

render_domain_explorer()

st.markdown('<div style="margin-top:4px"></div>', unsafe_allow_html=True)

# ── Answer flow ────────────────────────────────────────────────────────────────
if question and domains:
    routing = classify_domain(
        question, domains,
        fallback_domain=st.session_state.get("active_domain", "")
    )

    if routing.is_ambiguous and routing.confidence < AMBIGUITY_THRESHOLD:
        routing = render_ambiguity_ui(routing, domains)

    if not routing.is_ambiguous or routing.confidence >= AMBIGUITY_THRESHOLD:
        domain_name = routing.domain
        if domain_name not in domains:
            st.error("Could not identify a domain — try rephrasing with more specific terms.")
        else:
            domain_data = domains[domain_name]
            domain_label = DOMAIN_LABELS.get(domain_name,
                           domain_name.replace("_", " ").title())

            render_routing_indicator(routing, domain_label)

            with st.spinner("Analysing…"):
                context   = build_context(domain_data)
                llm_resp  = ask_groq(question, context, api_key=api_key())

            answer_text = llm_resp["answer"]
            parsed      = parse_structured_answer(answer_text)

            render_answer(parsed)

            ev = evaluate_response(
                question=question,
                answer=answer_text,
                context=context,
                domain_data=domain_data,
                routing_confidence=routing.confidence
            )

            render_eval_panel(ev, routing, domain_name)
            render_trust_panel(domain_name, domain_data, routing)
            render_metric_definitions(domain_data, answer_text)

            followups = get_follow_up_questions(domain_data, question)
            render_follow_ups(followups)

elif question and not domains:
    st.error("No domain skill files found. Verify YAML files are present in the repository root.")

# ── Architecture expanders ─────────────────────────────────────────────────────
st.markdown('<div style="margin-top:1.5rem"></div>', unsafe_allow_html=True)
render_how_it_works()
render_prototype_vs_production()

render_footer()
