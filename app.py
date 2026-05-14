import os
import yaml
import streamlit as st
from groq import Groq

st.set_page_config(page_title="Analytics AI Skill System", page_icon="🤖", layout="centered")

st.markdown("""
<style>
*, html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
}
[data-testid="stAppViewContainer"] {
    background: #0f0c29 !important;
    background-image: radial-gradient(ellipse at 60% 0%, #2d1f6e 0%, #0f0c29 60%) !important;
}
[data-testid="stHeader"] { background: transparent !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 1rem !important;
    padding-bottom: 0 !important;
    max-width: 740px !important;
    background: transparent !important;
}
[data-testid="stVerticalBlock"], [data-testid="element-container"],
div[class*="stMarkdown"], div[class*="stButton"],
[data-testid="stHorizontalBlock"] { background: transparent !important; }

div[data-testid="column"] .stButton > button {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.13) !important;
    border-radius: 10px !important;
    color: #c4b5fd !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    padding: 12px 8px !important;
    white-space: nowrap !important;
    transition: all 0.2s !important;
    width: 100% !important;
}
div[data-testid="column"] .stButton > button:hover {
    background: rgba(99,102,241,0.28) !important;
    border-color: #818cf8 !important;
    color: white !important;
    transform: translateY(-2px) !important;
}
.stButton > button {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 9px !important;
    color: #94a3b8 !important;
    font-size: 0.78rem !important;
    padding: 8px 12px !important;
    transition: all 0.15s !important;
    text-align: left !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: rgba(99,102,241,0.12) !important;
    border-color: rgba(99,102,241,0.3) !important;
    color: white !important;
}
.stTextInput > div > div { background: rgba(10,8,40,0.98) !important; border-radius: 12px !important; }
.stTextInput > div > div > input {
    background: rgba(10,8,40,0.98) !important;
    border: 1px solid rgba(129,140,248,0.4) !important;
    border-radius: 12px !important;
    color: #ffffff !important;
    caret-color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    font-size: 0.95rem !important;
    padding: 13px 18px !important;
}
.stTextInput > div > div > input:focus {
    border-color: #818cf8 !important;
    box-shadow: 0 0 0 3px rgba(129,140,248,0.15) !important;
}
.stTextInput > div > div > input::placeholder {
    color: #3d4a5c !important;
    -webkit-text-fill-color: #3d4a5c !important;
}
.stTextInput label {
    color: #64748b !important; font-size: 0.72rem !important;
    font-weight: 600 !important; text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
}
[data-testid="stSpinner"] > div { border-top-color: #818cf8 !important; }
</style>
""", unsafe_allow_html=True)


# ── Domain config ─────────────────────────────────────────────────────────────
DOMAIN_UI = {
    "product_usage": {
        "icon": "📊", "label": "Product",
        "questions": [
            "Which customers are over-utilizing their plans?",
            "What is the MRR recovery opportunity from right-sizing?",
            "Which regions have the highest over-utilization?",
            "What is our product margin by region?"
        ]
    },
    "marketing": {
        "icon": "📢", "label": "Marketing",
        "questions": [
            "Which campaign brought the highest number of customers?",
            "How many opportunities were closed last quarter?",
            "What is the ACV from each marketing channel?",
            "How are our MQL to SQL conversion rates trending?"
        ]
    },
    "sales": {
        "icon": "💰", "label": "Sales",
        "questions": [
            "Which sales rep gives the highest discounts?",
            "What is our MRR breakdown by customer type?",
            "How many new customers did we add this quarter?",
            "What is our pipeline coverage ratio?"
        ]
    },
    "hr": {
        "icon": "👥", "label": "HR",
        "questions": [
            "Which teams have the highest attrition?",
            "What is our regrettable attrition this quarter?",
            "Are we on track with our hiring plan?",
            "What is our new hire 90-day retention rate?"
        ]
    },
    "csup": {
        "icon": "🎧", "label": "Support",
        "questions": [
            "How many tickets were closed in 2026?",
            "What is our average response time?",
            "What is our CSAT score?",
            "Who are the top performing support agents?"
        ]
    }
}
PREFERRED_ORDER = ["product_usage", "marketing", "sales", "hr", "csup"]


# ── Load domains ──────────────────────────────────────────────────────────────
@st.cache_data
def load_domains():
    domains = {}
    for file in os.listdir("."):
        if file.endswith(".yaml"):
            try:
                with open(file, "r") as f:
                    data = yaml.safe_load(f)
                    if isinstance(data, dict) and "domain" in data:
                        key = str(data["domain"]).strip().lower().replace(" ", "_")
                        data["domain"] = key
                        domains[key] = data
            except Exception:
                pass
    return domains


def classify_domain(question, domains):
    if not domains:
        return None
    q = question.lower()
    scores = {}
    for dn, data in domains.items():
        score = 0
        for kw in data.get("keywords", []):
            kw_c = str(kw).lower().replace("-", " ").replace("_", " ")
            if kw_c in q:
                score += 2
            else:
                for w in kw_c.split():
                    if len(w) > 3 and w in q:
                        score += 1
        scores[dn] = score
    best = max(scores, key=scores.get)
    if scores[best] > 0:
        return best
    active = st.session_state.get("active_domain", "")
    return active if active in domains else list(domains.keys())[0]


def build_context(domain_data):
    ctx = f"Domain: {domain_data.get('domain','').upper()}\n"
    ctx += f"Description: {domain_data.get('description','')}\n\nKey Metrics:\n"
    for m in domain_data.get("metrics", []):
        ctx += f"  - {m.get('name','')}: {m.get('definition','')}\n"
    ctx += "\nSample Q&A:\n"
    for qa in domain_data.get("sample_qa", []):
        ctx += f"  Q: {qa.get('q','')}\n  A: {qa.get('a','')}\n\n"
    return ctx


def ask_groq(question, context):
    try:
        api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
        if not api_key:
            return "Add GROQ_API_KEY to Streamlit secrets."
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": f"""You are a senior analytics expert. The knowledge base below contains REAL DATA with specific numbers.
You MUST answer using the exact numbers, percentages and figures provided in the knowledge base.
Do NOT generate SQL queries. Do NOT say "calculate this". Do NOT give generic steps.
Give a direct, confident 3 to 4 sentence answer using the actual figures.

DOMAIN KNOWLEDGE:
{context}

QUESTION: {question}

Answer using ONLY the real numbers from the knowledge base above:"""}],
            max_tokens=350
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"


# ── Session state ─────────────────────────────────────────────────────────────
for key, val in [("active_domain", "product_usage"), ("prefill", ""), ("show_kb", False)]:
    if key not in st.session_state:
        st.session_state[key] = val

domains = load_domains()

# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:1.25rem 0 0.75rem">
  <div style="display:inline-flex;align-items:center;gap:8px;background:rgba(99,102,241,0.15);
  border:1px solid rgba(99,102,241,0.35);color:#a5b4fc;font-size:11px;font-weight:600;
  padding:5px 16px;border-radius:20px;margin-bottom:14px;letter-spacing:0.05em">
    🤖 AI-POWERED &nbsp;·&nbsp; GROQ + LLAMA 3.1
  </div>
  <h1 style="color:white;font-size:2rem;font-weight:700;margin:0 0 8px;line-height:1.2">
    Analytics AI <span style="color:#a78bfa">Skill System</span>
  </h1>
  <p style="color:#94a3b8;font-size:0.88rem;margin:0">
    Ask in plain English &nbsp;·&nbsp; AI routes to the right domain &nbsp;·&nbsp; Instant answer
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background:rgba(234,179,8,0.07);border:1px solid rgba(234,179,8,0.2);
border-radius:8px;padding:7px 16px;margin-bottom:1rem;text-align:center;
font-size:11px;color:#a16207;letter-spacing:0.02em">
  ⚠️ Demo environment &nbsp;·&nbsp; Illustrative data only, not production data
</div>
""", unsafe_allow_html=True)

# ── DOMAIN ROW ────────────────────────────────────────────────────────────────
st.markdown('<p style="color:#64748b;font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:0.4rem">Select a domain</p>', unsafe_allow_html=True)

display_order = [d for d in PREFERRED_ORDER if d in domains] + [d for d in domains if d not in PREFERRED_ORDER]
dcols = st.columns(max(len(display_order), 1))
for i, dk in enumerate(display_order):
    cfg = DOMAIN_UI.get(dk, {"icon": "📊", "label": dk.replace("_", " ").title()})
    if dcols[i].button(f"{cfg['icon']} {cfg['label']}", key=f"dom_{dk}", use_container_width=True):
        st.session_state["active_domain"] = dk
        qs = cfg.get("questions", [])
        st.session_state["prefill"] = qs[0] if qs else ""
        st.session_state["show_kb"] = False
        st.rerun()

active = st.session_state.get("active_domain", display_order[0] if display_order else "")
if active not in domains and display_order:
    active = display_order[0]
    st.session_state["active_domain"] = active
active_cfg = DOMAIN_UI.get(active, {"icon": "📊", "label": active.replace("_", " ").title(), "questions": []})

st.markdown('<hr style="border:none;border-top:1px solid rgba(255,255,255,0.07);margin:0.75rem 0 0.6rem"/>', unsafe_allow_html=True)

# ── EXAMPLE QUESTIONS ─────────────────────────────────────────────────────────
st.markdown(f'<p style="color:#64748b;font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:0.4rem">{active_cfg["icon"]} {active_cfg["label"]} — try a question</p>', unsafe_allow_html=True)
qcols = st.columns(2)
for i, q in enumerate(active_cfg.get("questions", [])):
    if qcols[i % 2].button(q, key=f"q_{active}_{i}", use_container_width=True):
        st.session_state["prefill"] = q
        st.session_state["show_kb"] = False
        st.rerun()

# ── INPUT ─────────────────────────────────────────────────────────────────────
st.markdown('<div style="margin-top:0.5rem"></div>', unsafe_allow_html=True)
question = st.text_input(
    "Or ask your own question",
    value=st.session_state.get("prefill", ""),
    placeholder="e.g. Which customers are over-utilizing their plans?"
)

# ── ANSWER ────────────────────────────────────────────────────────────────────
if question and domains:
    domain_name = classify_domain(question, domains)

    if not domain_name or domain_name not in domains:
        st.markdown('<p style="color:#f87171;font-size:13px">Could not identify domain — try rephrasing.</p>', unsafe_allow_html=True)
    else:
        domain_data = domains[domain_name]
        cfg = DOMAIN_UI.get(domain_name, {"icon": "📊", "label": domain_name.replace("_", " ").title()})

        st.markdown(f"""
        <div style="display:inline-flex;align-items:center;gap:8px;
        background:rgba(99,102,241,0.12);border:1px solid rgba(99,102,241,0.28);
        color:#a5b4fc;font-size:0.8rem;font-weight:600;padding:7px 14px;
        border-radius:8px;margin:8px 0 12px">
            {cfg['icon']} Routed to
            <strong style="color:white;margin-left:2px">{cfg['label'].upper()}</strong>
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("Analysing..."):
            context = build_context(domain_data)
            answer = ask_groq(question, context)

        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);
        border-left:3px solid #7c3aed;border-radius:12px;padding:18px 22px;margin-bottom:12px">
            <div style="color:#a78bfa;font-weight:700;font-size:0.68rem;letter-spacing:0.1em;
            text-transform:uppercase;margin-bottom:10px">Answer</div>
            <div style="color:#e2e8f0;font-size:0.92rem;line-height:1.8">{answer}</div>
        </div>
        """, unsafe_allow_html=True)

        col_kb, _ = st.columns([1, 3])
        if col_kb.button("🔍 View knowledge base", key="toggle_kb"):
            st.session_state["show_kb"] = not st.session_state.get("show_kb", False)

        if st.session_state.get("show_kb", False):
            st.markdown('<p style="color:#475569;font-size:0.73rem;margin:6px 0 4px">Structured domain knowledge retrieved and passed to the AI.</p>', unsafe_allow_html=True)
            st.code(context, language="yaml")

elif question and not domains:
    st.markdown('<p style="color:#f87171;font-size:13px">No domain files found. Check your repository.</p>', unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="border-top:1px solid rgba(255,255,255,0.06);margin-top:1.25rem;
padding:0.6rem 0 0.25rem;text-align:center;color:#2d3748;font-size:0.7rem">
    Built by <span style="color:#818cf8;font-weight:600">Ajantika Paul</span> &nbsp;·&nbsp;
    <a href="https://github.com/ajantika/analytics-ai-skill-system"
    style="color:#818cf8;text-decoration:none">GitHub</a>
</div>
""", unsafe_allow_html=True)
