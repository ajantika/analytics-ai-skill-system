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
    background: #1a1535 !important;
    background-image: radial-gradient(ellipse at top, #2d2060 0%, #0f0c29 70%) !important;
}
[data-testid="stHeader"] { background: transparent !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 1rem !important;
    padding-bottom: 0.5rem !important;
    max-width: 740px !important;
    background: transparent !important;
}
[data-testid="stVerticalBlock"], [data-testid="element-container"],
div[class*="stMarkdown"], div[class*="stButton"],
[data-testid="stHorizontalBlock"] { background: transparent !important; }

div[data-testid="column"] .stButton > button {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
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
    background: rgba(99,102,241,0.3) !important;
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
}
.stButton > button:hover {
    background: rgba(99,102,241,0.12) !important;
    border-color: rgba(99,102,241,0.3) !important;
    color: white !important;
}
.stTextInput > div > div {
    background: rgba(15, 12, 41, 0.95) !important;
    border-radius: 12px !important;
}
.stTextInput > div > div > input {
    background: rgba(15, 12, 41, 0.95) !important;
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
    color: #4a5568 !important;
    -webkit-text-fill-color: #4a5568 !important;
}
.stTextInput label {
    color: #64748b !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
}
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary { color: #475569 !important; font-size: 0.78rem !important; }
[data-testid="stSpinner"] > div { border-top-color: #818cf8 !important; }
</style>
""", unsafe_allow_html=True)


# ── Domain UI config ──────────────────────────────────────────────────────────
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
    }
}

# Preferred display order
PREFERRED_ORDER = ["product_usage", "marketing", "sales", "hr"]


# ── Load domains from YAML files ──────────────────────────────────────────────
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
                        data["domain"] = key  # normalize
                        domains[key] = data
            except Exception:
                pass
    return domains


def classify_domain(question, domains):
    if not domains:
        return None
    q = question.lower()
    scores = {}
    for domain_name, data in domains.items():
        score = 0
        keywords = data.get("keywords", [])
        for kw in keywords:
            kw_clean = str(kw).lower().replace("-", " ").replace("_", " ")
            if kw_clean in q:
                score += 2
            else:
                for word in kw_clean.split():
                    if len(word) > 3 and word in q:
                        score += 1
        scores[domain_name] = score
    best = max(scores, key=scores.get)
    # Return best if it has any score, else return first available domain
    if scores[best] > 0:
        return best
    # Fallback: match active domain if available
    active = st.session_state.get("active_domain", "")
    if active in domains:
        return active
    return list(domains.keys())[0]


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
        prompt = f"""You are a senior analytics expert at a B2B SaaS company.
Answer the question directly and specifically using the domain knowledge below.
Be concise — 3 to 5 sentences or a short list. Use specific metrics and thresholds from the knowledge base.
Do NOT give generic step-by-step instructions.

DOMAIN KNOWLEDGE:
{context}

QUESTION: {question}

Direct expert answer:"""
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"


# ── Session state ─────────────────────────────────────────────────────────────
if "active_domain" not in st.session_state:
    st.session_state["active_domain"] = "product_usage"
if "prefill" not in st.session_state:
    st.session_state["prefill"] = ""

# Load domains early so we can debug
domains = load_domains()

# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:1.25rem 0 0.75rem">
  <div style="display:inline-flex;align-items:center;gap:8px;background:rgba(99,102,241,0.15);
  border:1px solid rgba(99,102,241,0.35);color:#a5b4fc;font-size:11px;font-weight:600;
  padding:5px 16px;border-radius:20px;margin-bottom:14px;letter-spacing:0.05em">
    🤖 AI-POWERED &nbsp;·&nbsp; NOT A DASHBOARD &nbsp;·&nbsp; GROQ + LLAMA 3.1
  </div>
  <h1 style="color:white;font-size:2rem;font-weight:700;margin:0 0 8px;line-height:1.2">
    Analytics AI <span style="color:#a78bfa">Skill System</span>
  </h1>
  <p style="color:#94a3b8;font-size:0.88rem;margin:0">
    Ask a question in plain English &nbsp;·&nbsp; AI routes to the right domain &nbsp;·&nbsp; Instant answer
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background:rgba(234,179,8,0.07);border:1px solid rgba(234,179,8,0.22);
border-radius:8px;padding:7px 16px;margin-bottom:1rem;text-align:center;font-size:11.5px;color:#a16207">
  ⚠️ Demo environment &nbsp;·&nbsp; Knowledge bases contain illustrative data, not production data
</div>
""", unsafe_allow_html=True)

# ── DOMAIN ROW ────────────────────────────────────────────────────────────────
st.markdown('<p style="color:#64748b;font-size:0.72rem;font-weight:600;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:0.4rem">Select a domain</p>', unsafe_allow_html=True)

# Build display order — preferred order first, then any extra loaded domains
display_order = [d for d in PREFERRED_ORDER if d in domains]
extra = [d for d in domains if d not in PREFERRED_ORDER]
display_order += extra

dcols = st.columns(max(len(display_order), 1))
for i, dk in enumerate(display_order):
    cfg = DOMAIN_UI.get(dk, {"icon": "📊", "label": dk.replace("_", " ").title()})
    if dcols[i].button(f"{cfg['icon']} {cfg['label']}", key=f"dom_{dk}", use_container_width=True):
        st.session_state["active_domain"] = dk
        questions = cfg.get("questions", [])
        st.session_state["prefill"] = questions[0] if questions else ""
        st.rerun()

active = st.session_state.get("active_domain", display_order[0] if display_order else "")
# Make sure active domain is valid
if active not in domains and display_order:
    active = display_order[0]
    st.session_state["active_domain"] = active

active_cfg = DOMAIN_UI.get(active, {"icon": "📊", "label": active.replace("_", " ").title(), "questions": []})

st.markdown('<hr style="border:none;border-top:1px solid rgba(255,255,255,0.07);margin:0.75rem 0"/>', unsafe_allow_html=True)

# ── DYNAMIC EXAMPLE QUESTIONS ─────────────────────────────────────────────────
st.markdown(f'<p style="color:#64748b;font-size:0.72rem;font-weight:600;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:0.4rem">{active_cfg["icon"]} {active_cfg["label"]} — example questions</p>', unsafe_allow_html=True)

qcols = st.columns(2)
for i, q in enumerate(active_cfg.get("questions", [])):
    if qcols[i % 2].button(q, key=f"q_{active}_{i}", use_container_width=True):
        st.session_state["prefill"] = q
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
        st.error("Could not identify domain. Please try rephrasing your question.")
    else:
        domain_data = domains[domain_name]
        cfg = DOMAIN_UI.get(domain_name, {"icon": "📊", "label": domain_name.replace("_"," ").title()})

        st.markdown(f"""
        <div style="display:inline-flex;align-items:center;gap:8px;
        background:rgba(99,102,241,0.12);border:1px solid rgba(99,102,241,0.28);
        color:#a5b4fc;font-size:0.8rem;font-weight:600;padding:7px 14px;
        border-radius:8px;margin:8px 0 12px">
            {cfg['icon']} Routed to
            <strong style="color:white;margin-left:2px">{cfg['label'].upper()}</strong> domain
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("Analysing..."):
            context = build_context(domain_data)
            answer = ask_groq(question, context)

        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);
        border-left:3px solid #7c3aed;border-radius:12px;padding:18px 22px;margin-bottom:8px">
            <div style="color:#a78bfa;font-weight:700;font-size:0.7rem;letter-spacing:0.1em;
            text-transform:uppercase;margin-bottom:10px">Analytics Insight</div>
            <div style="color:#e2e8f0;font-size:0.92rem;line-height:1.75">{answer}</div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("🔍 View knowledge base used"):
            st.markdown('<p style="color:#64748b;font-size:0.75rem;margin-bottom:8px">Structured domain knowledge retrieved and passed to the AI to generate the answer above.</p>', unsafe_allow_html=True)
            st.code(context, language="yaml")

elif question and not domains:
    st.error("No domain YAML files found. Please check your repository files.")

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="border-top:1px solid rgba(255,255,255,0.06);margin-top:1.5rem;
padding:0.75rem 0 0.5rem;text-align:center;color:#334155;font-size:0.72rem">
    Built by <span style="color:#818cf8;font-weight:600">Ajantika Paul</span> &nbsp;·&nbsp;
    <a href="https://github.com/ajantika/analytics-ai-skill-system"
    style="color:#818cf8;text-decoration:none">View on GitHub</a>
</div>
""", unsafe_allow_html=True)
