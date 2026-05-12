# 🤖 Analytics AI Skill System

A multi-domain AI analytics Q&A system with domain-aware query routing — built to mirror the production system deployed.

🔗 **Live demo: [ajantika-analytics-ai.streamlit.app](https://ajantika-analytics-ai.streamlit.app)**

---

## What it does

```
User question → Domain classifier → YAML knowledge retrieval → Llama 3.1 via Groq → Answer
```

- Ask any analytics question in plain English
- System identifies the domain (Sales, Marketing, or HR)
- Retrieves the relevant YAML knowledge base for that domain
- Llama 3.1 generates a precise, contextual answer instantly

---

## Why this matters

This project recreates the architecture of a real production system built at Cloudflare that:

- Eliminated 3–4 hours/week of manual ad hoc queries for an 11-person analytics team
- Generated **~$216K in annual productivity value** across 4 business domains
- Enabled non-technical stakeholders to self-serve analytics insights without analyst intervention

---

## Tech stack

| Tool | Purpose |
|---|---|
| **Python** | Core application logic |
| **Groq API + Llama 3.1** | Free, fast LLM for answer generation |
| **YAML** | Structured domain knowledge bases |
| **Streamlit** | Web interface and cloud deployment |
| **Domain-aware routing** | Keyword classifier routes questions to the correct domain |

---

## Project structure

```
analytics-ai-skill-system/
├── sales.yaml          # Sales domain: MRR, churn, pipeline, conversion
├── marketing.yaml      # Marketing domain: CAC, CPL, MQL, ROAS
├── hr.yaml             # HR domain: headcount, attrition, hiring
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md
```

---

## Run locally

```bash
# Install dependencies
pip install -r requirements.txt

# Add your free Groq API key
export GROQ_API_KEY=your_key_here

# Run the app
streamlit run app.py
```

Get your **FREE** Groq API key at: [console.groq.com](https://console.groq.com) — no credit card needed.

---

## Deploy on Streamlit Cloud (free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo → set main file to `app.py`
4. Advanced settings → Secrets → add: `GROQ_API_KEY = "your_key"`
5. Deploy → get a live URL in 2 minutes

---

## Add a new domain

Create a new `.yaml` file following the same structure as `sales.yaml` — the system picks it up automatically with no code changes needed.

---

## Built by

**Ajantika Paul** — Analytics & AI Systems Lead 

[![LinkedIn](https://img.shields.io/badge/LinkedIn-ajantika--paul-blue?style=flat&logo=linkedin)](https://linkedin.com/in/ajantika-paul)
[![GitHub](https://img.shields.io/badge/GitHub-ajantika-black?style=flat&logo=github)](https://github.com/ajantika)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=flat&logo=streamlit)](https://ajantika-analytics-ai.streamlit.app)
