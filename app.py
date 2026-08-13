"""
app.py — Analytics AI Skill System
UI orchestration layer. All domain logic lives in router.py, skills.py, llm.py, evals.py.
"""

import os
import pathlib
import logging
import re
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
    max-width: 820px !important;
    padding: 2rem 1.5rem 3rem !important;
    margin: 0 auto !important;
}

/* Domain cards */
.domain-card-strip {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 8px;
    margin-bottom: 18px;
}
.domain-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 12px 10px;
    text-align: center;
    cursor: pointer;
    transition: background 0.15s, border-color 0.15s;
}
.domain-card:hover {
    background: rgba(99,102,241,0.1);
    border-color: rgba(99,102,241,0.3);
}
.domain-card.active {
    background: rgba(99,102,241,0.12);
    border-color: rgba(99,102,241,0.4);
}
.domain-card-icon { font-size: 1.2rem; margin-bottom: 4px; }
.domain-card-label { color: #e2e8f0; font-size: 0.75rem; font-weight: 600; }
.domain-card-sub { color: #475569; font-size: 0.62rem; margin-top: 3px; line-height: 1.4; }

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

/* ── Nav control ── */
div[data-testid="stHorizontalBlock"]:first-of-type div[data-testid="stButton"] > button {
    width: auto !important;
    padding: 4px 12px !important;
    font-size: 0.72rem !important;
    color: #6366f1 !important;
    border-color: rgba(99,102,241,0.25) !important;
    background: rgba(99,102,241,0.06) !important;
    white-space: nowrap !important;
}
div[data-testid="stHorizontalBlock"]:first-of-type div[data-testid="stButton"] > button:hover {
    background: rgba(99,102,241,0.12) !important;
    border-color: rgba(99,102,241,0.45) !important;
    color: #a5b4fc !important;
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
        "icon": "📊", "label": "Product", "subtitle": "Usage · Adoption · Monetization",
        "questions": [
            "How many customers are over-utilizing their plans?",
            "What is the MRR recovery opportunity from right-sizing?",
            "Which regions have the highest over-utilization?",
            "What is our product margin by region?",
        ]
    },
    "marketing": {
        "icon": "📣", "label": "Marketing", "subtitle": "Campaigns · Pipeline · Conversion",
        "questions": [
            "Which campaign brought the highest number of customers?",
            "How are our MQL to SQL conversion rates trending?",
            "What is the ACV from each marketing channel?",
            "How many opportunities were closed last quarter?",
        ]
    },
    "sales": {
        "icon": "💰", "label": "Sales", "subtitle": "Revenue · ARR · Discounts",
        "questions": [
            "Which sales rep gives the highest discounts?",
            "What is our MRR breakdown by customer type?",
            "How many new customers did we add this quarter?",
            "What is our average contract value by segment?",
        ]
    },
    "hr": {
        "icon": "👥", "label": "People", "subtitle": "Attrition · Hiring · Retention",
        "questions": [
            "Which teams have the highest attrition?",
            "What is our regrettable attrition this quarter?",
            "Are we on track with our hiring plan?",
            "What is our new hire 90-day retention rate?",
        ]
    },
    "csup": {
        "icon": "🎧", "label": "Support", "subtitle": "CSAT · SLA · Tickets",
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
    "How many customers are over-utilizing their plans?",
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
    app_dir = str(pathlib.Path(__file__).parent)
    return load_domains(app_dir)

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


def _confidence_label(conf: float) -> tuple[str, str]:
    """Convert raw confidence float to (label, color) for display. Never show raw % as if calibrated."""
    if conf >= 0.65:
        return "High", "#22c55e"
    if conf >= 0.40:
        return "Medium", "#eab308"
    return "Low", "#f87171"


def _extract_headline_kpi(insight: str) -> str:
    """Return the most prominent numeric KPI from an insight string, or '' if none."""
    first = insight[:100]
    m = re.search(r'\$[\d,]+\.?\d*[KMBkm]?', first)
    if m:
        return m.group()
    m = re.search(r'\b\d+\.?\d*x\b', first)
    if m:
        return m.group()
    m = re.search(r'\b(\d+\.?\d*)%', first)
    if m and float(m.group(1)) >= 10:
        return f"{int(round(float(m.group(1))))}%"
    return ""


def _failure_analysis(expected: str, predicted: str, question: str) -> dict:
    """Return structured failure analysis with cause type, cause, and candidate improvement."""
    q = question.lower()
    if expected == "marketing" and predicted == "sales" and "customer" in q:
        return {
            "cause_type": "Observed cause",
            "cause": "Shared customer terminology received greater keyword weight than the Marketing signal. The keyword 'customers' scores higher for Sales than 'campaign' scores for Marketing.",
            "improvement": "Benchmark token-aware weighting and semantic routing against the current keyword baseline.",
        }
    if expected == "sales" and predicted == "product_usage" and "coverage" in q:
        return {
            "cause_type": "Observed cause",
            "cause": "<code>coverage</code> contains <code>overage</code> as a substring, causing the Product keyword to trigger a false-positive match. The Sales score from 'pipeline' is outweighed by Product's substring score.",
            "improvement": "Whole-word or token-aware matching to prevent substring false positives.",
        }
    exp_lbl = DOMAIN_LABELS.get(expected, expected.replace("_", " ").title())
    pred_lbl = DOMAIN_LABELS.get(predicted, predicted.replace("_", " ").title())
    return {
        "cause_type": "Likely cause",
        "cause": f"Keyword overlap between {exp_lbl} and {pred_lbl} — the question lacked sufficient disambiguating terms for the keyword router.",
        "improvement": "Add domain-specific terminology, or evaluate whether semantic routing would better handle this case.",
    }


# ── Rendering functions ────────────────────────────────────────────────────────

def render_hero():
    st.markdown("""
<div style="text-align:center;padding:2rem 0 1rem">
  <h1 style="color:#f1f5f9;font-size:2.2rem;font-weight:800;margin:0 0 10px;line-height:1.15;
  letter-spacing:-0.03em">Analytics AI Skill System</h1>
  <p style="color:#94a3b8;font-size:1rem;margin:0 0 14px;line-height:1.6;max-width:520px;
  display:inline-block">
    Ask business questions across five analytics domains<br>using governed metrics and structured AI answers.
  </p>
  <div style="display:inline-flex;align-items:center;gap:8px;margin-bottom:14px;
  background:rgba(99,102,241,0.06);border:1px solid rgba(99,102,241,0.15);
  border-radius:8px;padding:7px 18px">
    <span style="color:#6366f1;font-size:0.75rem;font-weight:600">Auto-routed</span>
    <span style="color:#334155;font-size:0.75rem">→</span>
    <span style="color:#6366f1;font-size:0.75rem;font-weight:600">Governed metrics</span>
    <span style="color:#334155;font-size:0.75rem">→</span>
    <span style="color:#6366f1;font-size:0.75rem;font-weight:600">Structured AI answer</span>
    <span style="color:#334155;font-size:0.75rem">→</span>
    <span style="color:#6366f1;font-size:0.75rem;font-weight:600">Evaluated</span>
  </div>
  <br>
  <span style="background:rgba(234,179,8,0.06);border:1px solid rgba(234,179,8,0.18);
  border-radius:5px;padding:3px 12px;font-size:0.65rem;color:#78350f;letter-spacing:0.04em">
    Demo · Illustrative data only · No production or customer data
  </span>
</div>
""", unsafe_allow_html=True)


def render_input_section():
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
letter-spacing:0.08em;margin:14px 0 6px">Explore an analysis</p>
""", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    for i, q in enumerate(HERO_QUESTIONS):
        col = col_a if i % 2 == 0 else col_b
        if col.button(q, key=f"hero_{hash(q)}", use_container_width=True):
            st.session_state["pending_question"] = q
            st.rerun()


def render_domain_cards():
    """Render the 5 domain selector cards."""
    display_order = [d for d in PREFERRED_ORDER if d in domains]
    display_order += [d for d in domains if d not in display_order]

    if not display_order:
        return

    active = st.session_state.get("active_domain", display_order[0])
    if active not in domains:
        active = display_order[0]

    dcols = st.columns(len(display_order))
    for i, dk in enumerate(display_order):
        cfg = DOMAIN_UI.get(dk, {"icon": "◈", "label": dk.replace("_", " ").title(), "subtitle": ""})
        is_active = dk == active
        if dcols[i].button(
            f"{cfg['icon']}\n{cfg['label']}",
            key=f"dom_{dk}",
            use_container_width=True,
            help=cfg.get("subtitle", "")
        ):
            st.session_state["active_domain"] = dk
            qs = cfg.get("questions", [])
            st.session_state["pending_question"] = qs[0] if qs else ""
            st.rerun()
        sub = cfg.get("subtitle", "")
        if sub:
            dcols[i].markdown(
                f'<div style="text-align:center;font-size:0.58rem;color:#64748b;'
                f'margin-top:-8px;line-height:1.3">{sub}</div>',
                unsafe_allow_html=True
            )


def render_domain_samples():
    """Render sample questions for the active domain."""
    display_order = [d for d in PREFERRED_ORDER if d in domains]
    display_order += [d for d in domains if d not in display_order]
    if not display_order:
        return
    active = st.session_state.get("active_domain", display_order[0])
    if active not in domains:
        active = display_order[0]
    active_cfg = DOMAIN_UI.get(active, {"icon": "◈", "label": active, "questions": []})
    st.markdown(f"""
<p style="color:#374151;font-size:0.65rem;font-weight:600;text-transform:uppercase;
letter-spacing:0.08em;margin:14px 0 5px">Or try an example — {active_cfg['icon']} {active_cfg['label']}</p>
""", unsafe_allow_html=True)
    qcols = st.columns(2)
    for i, q in enumerate(active_cfg.get("questions", [])):
        if qcols[i % 2].button(q, key=f"dq_{active}_{i}", use_container_width=True):
            st.session_state["pending_question"] = q
            st.rerun()


def render_routing_indicator(routing, domain_label):
    conf_lbl, conf_color = _confidence_label(routing.confidence)
    st.markdown(f"""
<div style="margin:10px 0 12px;display:flex;align-items:center;gap:8px;flex-wrap:wrap">
  <span style="background:rgba(99,102,241,0.12);border:1px solid rgba(99,102,241,0.3);
  color:#818cf8;font-size:0.75rem;font-weight:600;padding:3px 11px;border-radius:20px">
    → {domain_label}
  </span>
  <span style="color:#334155;font-size:0.72rem">·</span>
  <span style="color:{conf_color};font-size:0.72rem;font-weight:600">{conf_lbl} confidence</span>
  <span style="color:#334155;font-size:0.72rem">·</span>
  <span style="color:#334155;font-size:0.7rem;font-style:italic">Keyword router</span>
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
            kpi = _extract_headline_kpi(insight)
            if kpi:
                sections += (
                    f'<div style="margin-bottom:18px">'
                    f'{_label("Insight")}'
                    f'<div style="text-align:center;margin:10px 0 12px">'
                    f'<span style="color:#818cf8;font-size:2.8rem;font-weight:800;'
                    f'letter-spacing:-0.03em;line-height:1">{kpi}</span>'
                    f'</div>'
                    f'{_body(insight)}'
                    f'</div>'
                )
            else:
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


def render_trust_badges(ev):
    """Compact inline quality signal row shown directly below the answer."""
    def check(status):
        return "✓" if status in ("PASS", "NONE") else "⚠" if status == "WARN" else "✗"
    def chk_color(status):
        return "#22c55e" if status in ("PASS", "NONE") else "#eab308" if status == "WARN" else "#f87171"

    ng_chk  = check(ev.groundedness)
    ng_col  = chk_color(ev.groundedness)
    mr_chk  = check(ev.metric_validity)
    mr_col  = chk_color(ev.metric_validity)
    rel_chk = check(ev.relevance)
    rel_col = chk_color(ev.relevance)

    st.markdown(f"""
<div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin:6px 0 4px;
font-size:0.75rem">
  <span style="color:{ng_col};font-weight:600">{ng_chk} Numeric grounding</span>
  <span style="color:{mr_col};font-weight:600">{mr_chk} Metric recognized</span>
  <span style="color:{rel_col};font-weight:600">{rel_chk} Relevant answer</span>
</div>
""", unsafe_allow_html=True)


def render_trust_eval_panel(ev, routing, domain_name, domain_data, answer_text=""):
    """Single expander combining routing context, metric provenance, quality checks, and metric definitions."""
    domain_label = DOMAIN_LABELS.get(domain_name, domain_name.replace("_", " ").title())
    conf_lbl, conf_color = _confidence_label(routing.confidence)
    metric_names = get_all_metric_names(domain_data)
    metrics_str = ", ".join(metric_names[:3]) + ("…" if len(metric_names) > 3 else "")
    pct = int(ev.overall_quality * 100)
    ql_color = "#22c55e" if ev.quality_label == "HIGH" else "#eab308" if ev.quality_label == "MEDIUM" else "#f87171"

    def badge(status):
        if status in ("PASS", "NONE"):
            return '<span style="color:#22c55e;font-weight:600">Pass</span>'
        elif status == "WARN":
            return '<span style="color:#eab308;font-weight:600">Warn</span>'
        return '<span style="color:#f87171;font-weight:600">Fail</span>'

    with st.expander("View metric definition & evaluation details", expanded=False):
        st.markdown(f"""
<div style="margin-bottom:12px;padding-bottom:10px;border-bottom:1px solid rgba(255,255,255,0.05)">
  <div style="display:grid;grid-template-columns:auto 1fr;gap:4px 20px;font-size:0.78rem;line-height:2.2">
    <span style="color:#475569">Domain</span>
    <span style="color:#94a3b8;font-weight:600">{domain_label}</span>
    <span style="color:#475569">Routing</span>
    <span><span style="color:{conf_color};font-weight:600">{conf_lbl} confidence</span>&nbsp;·&nbsp;<span style="color:#334155;font-style:italic">Keyword-based router</span></span>
    <span style="color:#475569">Metrics in domain</span>
    <span style="color:#64748b">{metrics_str}</span>
    <span style="color:#475569">Metric source</span>
    <span style="color:#64748b">{domain_label} YAML skill</span>
  </div>
</div>

<div style="margin-bottom:8px">
  <span style="color:#475569;font-size:0.68rem;font-weight:600;text-transform:uppercase;letter-spacing:0.07em">Overall quality</span>
  <span style="color:{ql_color};font-size:0.95rem;font-weight:700;margin-left:10px">{ev.quality_label} · {pct}%</span>
</div>

<table style="width:100%;border-collapse:collapse;font-size:0.78rem">
  <tr>
    <td style="padding:5px 0;color:#64748b;width:42%">Numeric grounding check</td>
    <td style="padding:5px 0">{badge(ev.groundedness)}</td>
    <td style="padding:5px 0 5px 10px;color:#334155;font-size:0.7rem">Deterministic figure matching</td>
  </tr>
  <tr>
    <td style="padding:5px 0;color:#64748b">Metric recognition</td>
    <td style="padding:5px 0">{badge(ev.metric_validity)}</td>
    <td style="padding:5px 0 5px 10px;color:#334155;font-size:0.7rem">Deterministic metric lookup</td>
  </tr>
  <tr>
    <td style="padding:5px 0;color:#64748b">Answer relevance heuristic</td>
    <td style="padding:5px 0">{badge(ev.relevance)}</td>
    <td style="padding:5px 0 5px 10px;color:#334155;font-size:0.7rem">Term overlap + non-answer phrase detection</td>
  </tr>
  <tr>
    <td style="padding:5px 0;color:#64748b">Unsupported claim heuristic</td>
    <td style="padding:5px 0">{badge(ev.unsupported_claims)}</td>
    <td style="padding:5px 0 5px 10px;color:#334155;font-size:0.7rem">Generalisation phrase matching</td>
  </tr>
</table>

<div style="margin-top:10px;padding-top:8px;border-top:1px solid rgba(255,255,255,0.05);
font-size:0.7rem;color:#334155;font-style:italic;line-height:1.8">
  These checks detect common failure modes but do not independently verify factual correctness.<br>
  Source: {domain_label} YAML skill &nbsp;·&nbsp; Illustrative dataset &nbsp;·&nbsp; Routing confidence: {conf_lbl} (keyword method)
</div>
""", unsafe_allow_html=True)

        # Metric definitions relevant to the answer
        metrics = domain_data.get("metrics", [])
        if metrics:
            answer_lower = answer_text.lower()
            relevant = [
                m for m in metrics
                if any(w in answer_lower for w in m.get("name", "").lower().split() if len(w) > 4)
            ]
            if not relevant:
                relevant = metrics[:2]
            for m in relevant[:3]:
                mname = m.get("name", "")
                mdef  = m.get("definition", "")
                st.markdown(
                    f'<div style="margin-top:8px;padding:8px 12px;'
                    f'background:rgba(99,102,241,0.05);border:1px solid rgba(99,102,241,0.12);'
                    f'border-radius:7px">'
                    f'<div style="color:#6366f1;font-size:0.65rem;font-weight:700;text-transform:uppercase;'
                    f'letter-spacing:0.08em;margin-bottom:4px">{mname}</div>'
                    f'<div style="color:#94a3b8;font-size:0.78rem;line-height:1.6">{mdef}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )


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
            st.session_state["pending_question"] = fq
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
    Routing confidence: {_confidence_label(routing.confidence)[0]} — below the automatic-routing threshold.<br>
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
_, nav_r = st.columns([6, 1])
with nav_r:
    if st.session_state.get("show_page") == "main":
        if st.button("◈ System Evaluation"):
            st.session_state["show_page"] = "eval"
            st.rerun()
    else:
        if st.button("← Back"):
            st.session_state["show_page"] = "main"
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# EVAL DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.get("show_page") == "eval":
    st.markdown("""
<div style="padding:1.2rem 0 0.6rem">
  <h2 style="color:#f1f5f9;font-size:1.6rem;font-weight:800;margin:0 0 6px;letter-spacing:-0.02em">
    System Evaluation</h2>
  <p style="color:#64748b;font-size:0.85rem;margin:0;line-height:1.7">
    Routing accuracy and answer quality — measured, not just claimed.
    Failures are shown, not hidden.
  </p>
</div>
""", unsafe_allow_html=True)

    eval_results = run_routing_eval(classify_domain, domains)
    accuracy = eval_results["accuracy"]

    # ── KPI row ───────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Routing accuracy", f"{int(accuracy * 100)}%")
    c2.metric("Labelled questions", eval_results["total"])
    c3.metric("Correct routes", eval_results["correct"])
    c4.metric("Routing failures", len(eval_results["failures"]))

    st.markdown("""
<div style="font-size:0.76rem;color:#475569;margin:8px 0 16px;line-height:1.7">
  Keyword-based router tested against 11 labelled questions across 5 domains.
  Confidence reflects keyword score separation — a routing heuristic, not a calibrated probability.
  2 intentionally ambiguous questions are excluded from accuracy calculation.
</div>
""", unsafe_allow_html=True)

    # ── Routing Failure Analysis ───────────────────────────────────────────────
    if eval_results["failures"]:
        st.markdown("""
<div style="border-top:1px solid rgba(255,255,255,0.07);padding-top:16px;margin-bottom:10px">
  <h3 style="color:#e2e8f0;font-size:0.95rem;font-weight:700;margin:0 0 4px">
    Routing Failure Analysis</h3>
  <p style="color:#475569;font-size:0.78rem;margin:0 0 12px;line-height:1.6">
    Understand why questions are misrouted and identify opportunities to improve the routing strategy.
  </p>
</div>
""", unsafe_allow_html=True)
        for f in eval_results["failures"]:
            exp  = DOMAIN_LABELS.get(f["expected"],  f["expected"])
            pred = DOMAIN_LABELS.get(f["predicted"], f["predicted"])
            fa   = _failure_analysis(f["expected"], f["predicted"], f["question"])
            st.markdown(f"""
<div style="background:rgba(248,113,113,0.04);border:1px solid rgba(248,113,113,0.15);
border-left:3px solid #f87171;border-radius:8px;padding:14px 16px;margin-bottom:10px">
  <div style="color:#e2e8f0;font-size:0.82rem;font-weight:500;margin-bottom:8px">
    "{f['question']}"
  </div>
  <div style="display:flex;gap:16px;margin-bottom:10px;font-size:0.75rem">
    <span style="color:#64748b">Expected: <strong style="color:#94a3b8">{exp}</strong></span>
    <span style="color:#334155">→</span>
    <span style="color:#64748b">Predicted: <strong style="color:#f87171">{pred}</strong></span>
  </div>
  <div style="margin-bottom:6px">
    <span style="color:#6366f1;font-size:0.65rem;font-weight:700;text-transform:uppercase;
    letter-spacing:0.08em">{fa['cause_type']}</span><br>
    <span style="color:#94a3b8;font-size:0.78rem;line-height:1.6">{fa['cause']}</span>
  </div>
  <div>
    <span style="color:#475569;font-size:0.65rem;font-weight:700;text-transform:uppercase;
    letter-spacing:0.08em">Candidate improvement</span><br>
    <span style="color:#64748b;font-size:0.78rem;line-height:1.6">{fa['improvement']}</span>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Answer Quality Evaluation ──────────────────────────────────────────────
    st.markdown("""
<div style="border-top:1px solid rgba(255,255,255,0.07);padding-top:16px;margin-bottom:10px">
  <h3 style="color:#e2e8f0;font-size:0.95rem;font-weight:700;margin:0 0 4px">
    Answer Quality Evaluation</h3>
  <p style="color:#475569;font-size:0.78rem;margin:0 0 12px;line-height:1.6">
    Generated answers are checked for grounding, metric usage, relevance, and potentially unsupported claims.
  </p>
</div>
""", unsafe_allow_html=True)

    eval_dims = [
        ("Numeric grounding",  "Deterministic", "Are figures in the answer present in the governed context?",            "#22c55e"),
        ("Metric recognition", "Deterministic", "Are referenced metrics defined in the selected domain skill?",          "#22c55e"),
        ("Answer relevance",   "Heuristic",     "Does the response address the business question?",                     "#eab308"),
        ("Unsupported claims", "Heuristic",     "Does the answer contain language that may go beyond the supplied context?", "#eab308"),
    ]
    dim_cols = st.columns(4)
    for i, (name, method, desc, color) in enumerate(eval_dims):
        dim_cols[i].markdown(
            f'<div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);'
            f'border-top:2px solid {color};border-radius:9px;padding:12px 12px 10px">'
            f'<div style="color:#e2e8f0;font-size:0.78rem;font-weight:700;margin-bottom:5px">{name}</div>'
            f'<div style="background:rgba(255,255,255,0.04);border-radius:4px;display:inline-block;'
            f'padding:1px 7px;color:#64748b;font-size:0.6rem;font-weight:700;letter-spacing:0.06em;'
            f'margin-bottom:7px">{method.upper()}</div>'
            f'<div style="color:#64748b;font-size:0.7rem;line-height:1.5">{desc}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown("""
<div style="margin-top:10px;font-size:0.74rem;color:#475569;line-height:1.7">
  <strong style="color:#64748b">Evaluation scope:</strong>
  These checks identify common grounding and relevance issues; they do not independently verify factual correctness.
</div>
""", unsafe_allow_html=True)

    # ── Collapsed sections ─────────────────────────────────────────────────────
    st.markdown('<div style="margin-top:16px"></div>', unsafe_allow_html=True)

    with st.expander("View all 11 routing test cases", expanded=False):
        for r in eval_results["results"]:
            icon  = "✓" if r["correct"] else "✗"
            color = "#22c55e" if r["correct"] else "#f87171"
            exp_label  = DOMAIN_LABELS.get(r["expected"],  r["expected"])
            pred_label = DOMAIN_LABELS.get(r["predicted"], r["predicted"])
            conf_lbl, _ = _confidence_label(r["confidence"])
            st.markdown(f"""
<div style="padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.04)">
  <span style="color:{color};font-weight:700;margin-right:8px">{icon}</span>
  <span style="color:#e2e8f0;font-size:0.82rem">{r['question']}</span><br>
  <span style="color:#475569;font-size:0.72rem">
    Expected: <strong style="color:#94a3b8">{exp_label}</strong>
    &nbsp;·&nbsp; Predicted: <strong style="color:#94a3b8">{pred_label}</strong>
    &nbsp;·&nbsp; Confidence: {conf_lbl} &nbsp;·&nbsp; {r['note']}
  </span>
</div>
""", unsafe_allow_html=True)

    with st.expander("Evaluation methodology", expanded=False):
        st.markdown("""
Routing accuracy is measured against 11 hand-labelled questions spanning 5 business domains.
Confidence labels (High / Medium / Low) reflect the keyword score separation between the top and runner-up domain — a routing heuristic, not a calibrated classifier probability.
2 intentionally ambiguous questions are included in the test set but excluded from accuracy calculation.

**Answer quality checks:**
Numeric grounding and metric recognition are deterministic — they match figures and metric names against the governed YAML context. Answer relevance and unsupported-claim checks are heuristic — they use term overlap and phrase matching and have known false-positive and false-negative failure modes.

These checks identify common failure modes. They do not independently verify factual correctness, eliminate hallucinations, or confirm the LLM's reasoning. Numeric grounding confirms a figure appeared in the supplied context — it does not confirm the figure was used correctly.

**This is not a held-out production benchmark.** Questions were reviewed during router development; overfitting is possible.
""")

    with st.expander("Evaluation limitations", expanded=False):
        st.markdown("""
- **Small, curated dataset** — 11 routing questions is not a statistically meaningful benchmark.
- **Not a held-out set** — questions were reviewed while building the keyword router; overfitting is possible.
- **Deterministic keyword baseline** — a semantic router (embedding or LLM-based) would likely outperform this on ambiguous questions, but is not implemented here.
- **Heuristic answer quality** — no ground-truth correct answers; heuristics have known failure modes.
- **No independent human evaluation** — answer correctness and usefulness have not been assessed by domain experts.
- **No calibrated routing probability** — confidence labels (High / Medium / Low) reflect keyword score separation, not a trained classifier's probability.
- **Illustrative data** — all figures are pre-embedded in YAML skill files, not live business data.
""")

    with st.expander("Next evaluation steps", expanded=False):
        st.markdown("""
*These are future improvements — none currently exist in this demo.*

1. **Benchmark semantic routing** — compare embedding-based or LLM-based routing against this keyword baseline to quantify the accuracy improvement on ambiguous questions.
2. **Build a larger held-out evaluation set** — independently label 100+ questions covering straightforward, ambiguous, and adversarial cases, with no overlap with the keyword lists.
3. **Add regression evaluations** — run the evaluation suite automatically whenever routing logic, prompts, skill files, or models change.
4. **Add human evaluation** — measure answer correctness, usefulness, and actionability via analyst or business-user review, not just automated heuristics.
""")

    render_footer()
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PAGE
# ══════════════════════════════════════════════════════════════════════════════
render_hero()

render_domain_cards()

st.markdown('<div style="margin-top:10px"></div>', unsafe_allow_html=True)

st.markdown("""
<p style="color:#475569;font-size:0.7rem;font-weight:600;text-transform:uppercase;
letter-spacing:0.08em;margin-bottom:6px">Ask a business question</p>
""", unsafe_allow_html=True)

# Apply any pending question from button clicks BEFORE the text input renders
if st.session_state.get("pending_question"):
    st.session_state["question_input"] = st.session_state["pending_question"]
    st.session_state["pending_question"] = ""

question = render_input_section()

render_domain_samples()

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

            render_trust_badges(ev)
            render_trust_eval_panel(ev, routing, domain_name, domain_data, answer_text)

            followups = get_follow_up_questions(domain_data, question)
            render_follow_ups(followups)

elif question and not domains:
    st.error("No domain skill files found. Verify YAML files are present in the repository root.")

# ── Architecture expanders ─────────────────────────────────────────────────────
st.markdown('<div style="margin-top:1.5rem"></div>', unsafe_allow_html=True)
render_how_it_works()
render_prototype_vs_production()

render_footer()
