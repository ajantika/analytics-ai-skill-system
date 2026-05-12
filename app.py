import os
import yaml
import streamlit as st
import google.generativeai as genai

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Analytics AI Skill System",
    page_icon="🤖",
    layout="centered"
)

# ── Load all domain YAML files ────────────────────────────────────────────────
@st.cache_data
def load_domains():
    domains = {}
    domain_dir = "domains"
    for file in os.listdir(domain_dir):
        if file.endswith(".yaml"):
            with open(os.path.join(domain_dir, file)) as f:
                data = yaml.safe_load(f)
                domains[data["domain"]] = data
    return domains

# ── Classify which domain the question belongs to ────────────────────────────
def classify_domain(question, domains):
    question_lower = question.lower()
    scores = {}
    for domain_name, data in domains.items():
        score = sum(1 for kw in data["keywords"] if kw in question_lower)
        scores[domain_name] = score
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else list(domains.keys())[0]

# ── Build context string from YAML ───────────────────────────────────────────
def build_context(domain_data):
    ctx = f"Domain: {domain_data['domain'].upper()}\n"
    ctx += f"Description: {domain_data['description']}\n\n"
    ctx += "Key Metrics:\n"
    for m in domain_data["metrics"]:
        ctx += f"  - {m['name']}: {m['definition']}\n"
    ctx += "\nSample Q&A from this domain:\n"
    for qa in domain_data["sample_qa"]:
        ctx += f"  Q: {qa['q']}\n  A: {qa['a']}\n\n"
    return ctx

# ── Ask Gemini (FREE) ─────────────────────────────────────────────────────────
def ask_gemini(question, context):
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
        if not api_key:
            return "⚠️ API key not found. Add GEMINI_API_KEY to Streamlit secrets."
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"""You are an expert analytics assistant helping a data analytics team.
Use ONLY the domain knowledge provided below to answer the question.
Be concise, practical and specific. Give actionable guidance.

DOMAIN KNOWLEDGE:
{context}

QUESTION: {question}

Answer:"""
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🤖 Analytics AI Skill System")
st.caption("Multi-domain analytics Q&A · Built by Ajantika Paul · Ex-Cloudflare")
st.markdown("---")

# Domain cards
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("💰 **Sales**")
    st.caption("MRR, Churn, Pipeline")
with col2:
    st.markdown("📢 **Marketing**")
    st.caption("CAC, CPL, MQL, ROAS")
with col3:
    st.markdown("👥 **HR**")
    st.caption("Headcount, Attrition, Hiring")

st.markdown("---")

# Load domains
domains = load_domains()

# Example questions
st.markdown("**Try these example questions:**")
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

# Question input
question = st.text_input(
    "Or type your own question:",
    value=st.session_state.get("prefill", ""),
    placeholder="e.g. Which team has the highest attrition?"
)

if question:
    # Classify domain
    domain_name = classify_domain(question, domains)
    domain_data = domains[domain_name]
    domain_icons = {"sales": "💰", "hr": "👥", "marketing": "📢"}
    icon = domain_icons.get(domain_name, "📊")

    st.info(f"{icon} Routed to **{domain_name.upper()}** domain")

    with st.spinner("Generating answer..."):
        context = build_context(domain_data)
        answer = ask_gemini(question, context)

    st.success("**Answer:**")
    st.write(answer)

    with st.expander("🔍 View domain knowledge used"):
        st.code(context, language="yaml")

st.markdown("---")
st.caption("This project mirrors the multi-domain AI analytics skill system built at Cloudflare · github.com/ajantika")
