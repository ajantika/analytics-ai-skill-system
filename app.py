import os
import yaml
import streamlit as st
from groq import Groq

st.set_page_config(page_title="Analytics AI Skill System", page_icon="🤖", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
*, html, body, [class*="css"], [data-testid="stAppViewContainer"], [data-testid="stHeader"],
[data-testid="stToolbar"], [data-testid="stDecoration"], section[data-testid="stSidebar"],
.main, .block-container {
    font-family: 'Inter', sans-serif !important;
    background: transparent !important;
}
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e) !important;
}
[data-testid="stHeader"] { background: transparent !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem !important; max-width: 720px !important; }

/* Remove ALL white backgrounds */
[data-testid="stVerticalBlock"], [data-testid="element-container"],
div[class*="stMarkdown"], div[class*="stButton"] {
    background: transparent !important;
}

/* Domain buttons styled as cards */
div[data-testid="column"] .stButton > button {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 16px !important;
    color: white !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    padding: 20px 12px !important;
    width: 100% !important;
    height: 100px !important;
    transition: all 0.2s !important;
    line-height: 1.6 !important;
}
div[data-testid="column"] .stButton > button:hover {
    background: rgba(99,102,241,0.25) !important;
    border-color: rgba(99,102,241,0.6) !important;
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 25px rgba(99,102,241,0.2) !important;
}
div[data-testid="column"] .stButton > button:active,
div[data-testid="column"] .stButton > button:focus {
    background: rgba(99,102,241,0.35) !important;
    border-color: #818cf8 !important;
}

/* Example question buttons */
.example-btn .stButton > button {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #cbd5e1 !important;
    font-size: 0.8rem !important;
    font-weight: 400 !important;
    padding: 9px 12px !important;
    height: auto !important;
    text-align: left !important;
}
.example-btn .stButton > button:hover {
    background: rgba(99,102,241,0.15) !important;
    border-color: rgba(99,102,241,0.4) !important;
    color: white !important;
}

/* Text input */
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 12px !important;
    color: white !important;
    font-size: 0.95rem !important;
    padding: 13px 18px !important;
}
.stTextInput > div > div > input:focus {
    border-color: #818cf8 !important;
    box-shadow: 0 0 0 3px rgba(129,140,248,0.15) !important;
}
.stTextInput > div > div > input::placeholder { color: #475569 !important; }
.stTextInput label { color: #64748b !important; font-size: 0.78rem !important; font-weight: 600 !important; text-transform: uppercase !important; letter-spacing: 0.07em !important; }

/* Spinner */
[data-testid="stSpinner"] > div { border-top-color: #818cf8 !important; }

/* Expander */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary { color: #64748b !important; font-size: 0.82rem !important; }
</style>
""", unsafe_allow_html=True)


# ── Load domains ──────────────────────────────────────────────────────────────
@st.cache_data
def load_domains():
    domains = {}
    for file in os.listdir("."):
        if file.endswith(".yaml"):
            with open(file) as f:
                data = yaml.safe_load(f)
                domains[data["domain"]] = data
    return domains

def classify_domain(question, domains):
    q = question.lower()
    scores = {d: sum(1 for kw in data["keywords"] if kw in q) for d, data in domains.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else list(domains.keys())[0]

def build_context(domain_data):
    ctx = f"Domain: {domain_data['domain'].upper()}\nDescription: {domain_data['description']}\n\nKey Metrics:\n"
    for m in domain_data["metrics"]:
        ctx += f"  - {m['name']}: {m['definition']}\n"
    ctx += "\nSample Q&A:\n"
    for qa in domain_data["sample_qa"]:
        ctx += f"  Q: {qa['q']}\n  A: {qa['a']}\n\n"
    return ctx

def ask_groq(question, context):
    try:
        api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
        if not api_key:
            return "Add GROQ_API_KEY to Streamlit secrets."
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": f"""You are an expert analytics assistant.
Use ONLY the domain knowledge below to answer. Be concise and actionable.

{context}

Question: {question}
Answer:"""}],
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"


# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:2rem 0 1rem">
  <div style="display:inline-block;background:rgba(99,102,241,0.2);border:1px solid rgba(99,102,241,0.4);color:#a5b4fc;font-size:11px;font-weight:600;padding:4px 16px;border-radius:20px;margin-bottom:14px;letter-spacing:0.06em">
    ⚡ POWERED BY GROQ + LLAMA 3.1
  </div>
  <h1 style="color:white;font-size:2.2rem;font-weight:700;margin:0 0 8px;line-height:1.2">
    Analytics AI <span style="background:linear-gradient(90deg,#818cf8,#c084fc);-webkit-background-clip:text;-webkit-text-fill-color:transparent">Skill System</span>
  </h1>
  <p style="color:#94a3b8;font-size:0.95rem;margin:0">
    Ask any analytics question · AI routes to the right domain · Get instant answers
  </p>
</div>
""", unsafe_allow_html=True)

# ── DOMAIN CARDS (real clickable buttons) ────────────────────────────────────
st.markdown('<p style="color:#64748b;font-size:0.75rem;font-weight:600;text-transform:uppercase;letter-spacing:0.07em;margin:0.5rem 0 0.5rem">Select a domain</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
if col1.button("💰\n\nSales\nMRR · Churn · Pipeline", key="domain_sales", use_container_width=True):
    st.session_state["prefill"] = "Why did our churn increase last quarter?"
if col2.button("📢\n\nMarketing\nCAC · CPL · ROAS", key="domain_marketing", use_container_width=True):
    st.session_state["prefill"] = "Which campaign has the lowest CAC?"
if col3.button("👥\n\nHR\nHeadcount · Attrition", key="domain_hr", use_container_width=True):
    st.session_state["prefill"] = "What is our attrition rate this year?"

st.markdown('<hr style="border:none;border-top:1px solid rgba(255,255,255,0.07);margin:1.25rem 0"/>', unsafe_allow_html=True)

# ── EXAMPLE QUESTIONS ─────────────────────────────────────────────────────────
st.markdown('<p style="color:#64748b;font-size:0.75rem;font-weight:600;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:0.5rem">Or try an example</p>', unsafe_allow_html=True)

examples = [
    "Why did our churn increase last quarter?",
    "Which campaign has the lowest CAC?",
    "What is our attrition rate this year?",
    "How is our pipeline trending?"
]
st.markdown('<div class="example-btn">', unsafe_allow_html=True)
ecols = st.columns(2)
for i, ex in enumerate(examples):
    if ecols[i % 2].button(ex, key=f"ex_{i}", use_container_width=True):
        st.session_state["prefill"] = ex
st.markdown('</div>', unsafe_allow_html=True)

# ── INPUT ─────────────────────────────────────────────────────────────────────
st.markdown('<div style="margin-top:1rem"></div>', unsafe_allow_html=True)
question = st.text_input(
    "Ask your own question",
    value=st.session_state.get("prefill", ""),
    placeholder="e.g. Which team has the highest attrition this quarter?"
)

# ── ANSWER ────────────────────────────────────────────────────────────────────
domains = load_domains()

if question:
    domain_name = classify_domain(question, domains)
    domain_data = domains[domain_name]
    icons = {"sales": "💰", "hr": "👥", "marketing": "📢"}
    icon = icons.get(domain_name, "📊")

    st.markdown(f"""
    <div style="display:inline-flex;align-items:center;gap:8px;background:rgba(99,102,241,0.15);
    border:1px solid rgba(99,102,241,0.35);color:#a5b4fc;font-size:0.83rem;font-weight:600;
    padding:8px 16px;border-radius:8px;margin:10px 0 14px">
        {icon} Routed to <strong style="color:white;margin-left:4px">{domain_name.upper()}</strong>&nbsp;domain
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Generating answer..."):
        context = build_context(domain_data)
        answer = ask_groq(question, context)

    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.1);
    border-left:3px solid #818cf8;border-radius:14px;padding:20px 24px;margin-bottom:12px">
        <div style="color:#818cf8;font-weight:700;font-size:0.72rem;letter-spacing:0.09em;
        text-transform:uppercase;margin-bottom:10px">Answer</div>
        <div style="color:#e2e8f0;font-size:0.93rem;line-height:1.75">{answer}</div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("🔍 View domain knowledge used"):
        st.code(context, language="yaml")

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style="border:none;border-top:1px solid rgba(255,255,255,0.07);margin:2rem 0 1rem"/>
<div style="text-align:center;color:#475569;font-size:0.75rem;padding-bottom:1rem">
    Built by <span style="color:#818cf8;font-weight:600">Ajantika Paul</span> · Ex-Cloudflare · 
    <a href="https://github.com/ajantika" style="color:#818cf8;text-decoration:none">github.com/ajantika</a>
</div>
""", unsafe_allow_html=True)
