# 🤖 Analytics AI Skill System

A multi-domain AI analytics Q&A system with domain-aware query routing — built to mirror the production system deployed at Cloudflare.

## What it does

Users ask natural language analytics questions → the system identifies the domain (Sales, Marketing, or HR) → retrieves the relevant knowledge base → Claude generates a precise, contextual answer.

```
User question → Domain classifier → YAML knowledge retrieval → Claude API → Answer
```

## Why this matters

This project recreates the architecture of a real production system that:
- Eliminated 3–4 hours/week of manual ad hoc query handling for an 11-person analytics team
- Generated ~$216K in annual productivity value across 4 business domains
- Enabled non-technical stakeholders to self-serve analytics insights

## Tech stack

- **Python** — core logic
- **Claude API (Anthropic)** — LLM for answer generation
- **YAML** — structured domain knowledge bases
- **Streamlit** — web interface and deployment
- **Domain-aware routing** — keyword-based classifier

## Project structure

```
analytics-ai-skill-system/
├── domains/
│   ├── sales.yaml        # Sales, MRR, churn, pipeline
│   ├── marketing.yaml    # CAC, CPL, MQL, attribution
│   └── hr.yaml           # Headcount, attrition, hiring
├── app.py                # Main Streamlit application
├── requirements.txt      # Python dependencies
└── README.md
```

## Run locally

```bash
# Install dependencies
pip install -r requirements.txt

# Add your Anthropic API key
export ANTHROPIC_API_KEY=your_key_here

# Run the app
streamlit run app.py
```

## Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to share.streamlit.io
3. Connect your GitHub repo
4. Add `ANTHROPIC_API_KEY` in the Secrets section
5. Deploy

## How to extend

Add a new domain by creating a new YAML file in the `domains/` folder following the same structure. The system automatically picks it up — no code changes needed.

## Built by

**Ajantika Paul** — Analytics & AI Systems Lead  
[LinkedIn](https://linkedin.com/in/ajantika-paul) · [GitHub](https://github.com/ajantika)
