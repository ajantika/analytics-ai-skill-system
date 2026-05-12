import os
import yaml
import streamlit as st
from groq import Groq

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Analytics AI Skill System",
    page_icon="🤖",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    #MainMenu, footer, header {visibility: hidden;}
    .stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); min-height: 100vh; }
    .hero { text-align: center; padding: 2.5rem 1rem 1.5rem; }
    .hero-badge { display: inline-block; background: rgba(99,102,241,0.2); border: 1px solid rgba(99,102,241,0.5); color: #a5b4fc; font-size: 12px; font-weight: 600; padding: 4px 14px; border-radius: 20px; margin-bottom: 16px; letter-spacing: 0.05em; }
    .hero-title { font-size: 2.4rem; font-weight: 700; color: white; margin: 0 0 10px; line-height: 1.2; }
    .hero-title span { background: linear-gradient(90deg, #818cf8, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .hero-sub { color: #94a3b8; font-size: 1rem; margin-bottom: 2rem; }
    .domain-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 1.5rem 0; }
    .domain-card { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); border-radius: 14px; padding: 16px; text-align: center; }
    .domain-icon { font-size: 1.8rem; margin-bottom: 6px; }
    .domain-name { color: white; font-weight: 600; font-size: 0.95rem; }
    .domain-tags { color: #94a3b8; font-size: 0.75rem; margin-top: 4px; }
    .stTextInput > div > div > input { background: rgba(255,255,255,0.08) !important; border: 1px solid rgba(255,255,255,0.15) !important; border-radius: 12px !important; color: white !important; font-size: 1rem !important; padding: 14px 18px !important; }
    .stTextInput > div > div > input:focus { border-color: #818cf8 !important; box-shadow: 0 0 0 3px rgba(129,140,248,0.2) !important; }
    .stTextInput > div > div > input::placeholder { color: #64748b !important; }
    .stButton > button { background: rgba(255,255,255,0.06) !important; border: 1px solid rgba(255,255,255,0.12) !important; border-radius: 10px !important; color: #cbd5e1 !important; font-size: 0.82rem !important; font-weight: 500 !important; padding: 8px 12px !important; transition: all 0.2s !important; width: 100% !important; }
    .stButton > button:hover { background: rgba(99,102,241,0.2) !important; border-color: rgba(99,102,241,0.5) !important; color: white !important; }
    .route-badge { display: inline-flex; align-items: center; gap: 8px; background: rgba(99,102,241,0.15); border: 1px solid rgba(99,102,241,0.3); color: #a5b4fc; font-size: 0.85rem; font-weight: 600; padding: 8px 16px; border-radius: 8px; margin: 12px 0; }
    .answer-card { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.1); border-left: 3px solid #818cf8; border-radius: 14px; padding: 20px 24px; color: #e2e8f0; font-size: 0.95rem; line-height: 1.7; margin: 12px 0; }
    .answer-label { color: #818cf8; font-weight: 700; font-size: 0.8rem; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 10px; }
    .section-label { color: #64748b; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.07em; margin: 1.5rem 0 0.75rem; }
    .divider { border: none; border-top: 1px solid rgba(255,255,255,0.07); margin: 1.5rem 0; }
    .footer { text-align: center; color: #475569; font-size: 0.78rem; padding: 2rem 0 1rem; }
    .streamlit-expanderHeader { background: rgba(255,255,255,0.04) !important; border: 1px solid rgba(255,255,255,0.08) !important; border-radius: 10px !important; color: #94a3b8 !important; font-size: 0.85rem !important; }
</style>
""", unsafe_allow_html=True)


# ── Load domains ──────────────────────────────────────────────────────────────
@st.cache_data
def load_domains():
    domains = {}
    domain_dir = "."
    for file in os.listdir(domain_dir):
        if file.endswith(".yaml"):
            with open(os.path.join(domain_dir, file)) as f:
                data = yaml.safe_load(f)
                domains[data["domain"]] = data
    return domains

def classify_domain(question, domains):
    question_lower = question.lower()
    scores = {}
    for domain_name, data in domains.items():
        score = sum(1 for kw in data["keywords"] if kw in question_lower)
        scores[domain_name] = score
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else list(domains.keys())[0]

def build_context(domain_data):
    ctx = f"Domain: {domain_data['domain'].upper()}\n"
    ctx += f"Description: {domain_data['description']}\n\n"
    ctx += "Key Metrics:\n"
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
        prompt = f"""You are an expert analytics assistant helping a data analytics team.
Use ONLY the domain knowledge provided below to answer the question.
Be concise, practical and specific. Give actionable guidance.

DOMAIN KNOWLEDGE:
{context}

QUESTION: {question}

Answer:"""
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"


# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">POWERED BY GROQ + LLAMA 3.1</div>
    <div class="hero-title">Analytics AI <span>Skill System</span></div>
    <div class="hero-sub">Ask any analytics question · AI routes to the right domain · Get instant answers</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="domain-grid">
    <div class="domain-card">
        <div class="domain-icon">💰</div>
        <div class="domain-name">Sales</div>
        <div class="domain-tags">MRR · ARR · Churn · Pipeline</div>
    </div>
    <div class="domain-card">
        <div class="domain-icon">📢</div>
        <div class="domain-name">Marketing</div>
        <div class="domain-tags">CAC · CPL · MQL · ROAS</div>
    </div>
    <div class="domain-card">
        <div class="domain-icon">👥</div>
        <div class="domain-name">HR</div>
        <div class="domain-tags">Headcount · Attrition · Hiring</div>
    </div>
</div>
<hr class="divider">
""", unsafe_allow_html=True)

st.markdown('<div class="section-label">Try an example</div>', unsafe_allow_html=True)
examples = [
    "Why did our churn increase last quarter?",
    "Which campaign has the lowest CAC?",
    "What is our attrition rate this year?",
    "How is our pipeline trending?"
]
cols = st.columns(2)
for i, ex in enumerate(examples):
    if cols[i % 2].button(ex, key=f"ex_{i}", use_container_width=True):
        st.session_state["prefill"] = ex

st.markdown('<div class="section-label" style="margin-top:1.5rem">Ask your own question</div>', unsafe_allow_html=True)
question = st.text_input(
    label="",
    value=st.session_state.get("prefill", ""),
    placeholder="e.g. Which team has the highest attrition this quarter?"
)

domains = load_domains()

if question:
    domain_name = classify_domain(question, domains)
    domain_data = domains[domain_name]
    domain_icons = {"sales": "💰", "hr": "👥", "marketing": "📢"}
    icon = domain_icons.get(domain_name, "📊")

    st.markdown(f'<div class="route-badge">{icon} Routed to <strong style="color:white;margin-left:4px">{domain_name.upper()}</strong> domain</div>', unsafe_allow_html=True)

    with st.spinner("Generating answer..."):
        context = build_context(domain_data)
        answer = ask_groq(question, context)

    st.markdown(f'<div class="answer-card"><div class="answer-label">Answer</div>{answer}</div>', unsafe_allow_html=True)

    with st.expander("View domain knowledge used"):
        st.code(context, language="yaml")

st.markdown("""
<hr class="divider">
<div class="footer">
    Built by <strong style="color:#818cf8">Ajantika Paul</strong> · Ex-Cloudflare · 
    <a href="https://github.com/ajantika" style="color:#818cf8;text-decoration:none">github.com/ajantika</a>
</div>
""", unsafe_allow_html=True)
