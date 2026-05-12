# 🤖 Analytics AI Skill System

A multi-domain AI analytics Q&A system with domain-aware query routing.
Built to mirror the production system deployed at Cloudflare by Ajantika Paul.

## What it does

```
User question → Domain classifier → YAML knowledge retrieval → Gemini AI → Answer
```

- Ask a question in plain English
- System identifies the domain (Sales, Marketing, or HR)
- Retrieves the relevant knowledge base
- Gemini AI generates a precise, contextual answer

## Why this matters

This project recreates the architecture of a real production system that:
- Eliminated 3–4 hours/week of manual ad hoc queries for an 11-person analytics team
- Generated ~$216K in annual productivity value across 4 business domains
- Enabled non-technical stakeholders to self-serve analytics insights without analyst help

## Tech stack

- **Python** — core logic
- **Google Gemini API** — free LLM for answer generation
- **YAML** — structured domain knowledge bases
- **Streamlit** — web interface and deployment
- **Domain-aware routing** — keyword classifier routes questions to correct domain

## Run locally

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=your_free_key_here
streamlit run app.py
```

Get your FREE Gemini API key at: aistudio.google.com

## Deploy on Streamlit Cloud (free)

1. Push this repo to GitHub
2. Go to share.streamlit.io
3. Connect your GitHub repo
4. Add secret: GEMINI_API_KEY = your_key
5. Deploy and get a live URL

## Add a new domain

Create a new YAML file in the domains/ folder — system picks it up automatically.

## Built by

**Ajantika Paul** — Analytics & AI Systems Lead · Ex-Cloudflare  
[LinkedIn](https://linkedin.com/in/ajantika-paul) · [GitHub](https://github.com/ajantika)
