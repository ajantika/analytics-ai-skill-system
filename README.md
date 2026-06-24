# Multi-Domain AI Analytics Skill System

A conversational analytics product that routes plain-English questions across business domains and returns precise, data-backed answers instantly.

🔗 **[Live Demo](https://ajantika-analytics-ai.streamlit.app/)**

## The problem
Analysts were spending 3-4 hours per week answering the same ad hoc questions from across the business. Non-technical teams couldn't self-serve insights — every question required an analyst in the loop.

## What it does
A user types any question in plain English. The system:
1. Classifies the domain automatically (Product, Marketing, Sales, HR)
2. Retrieves the structured YAML knowledge base for that domain
3. Returns a direct answer with real figures

No SQL, no wait, no analyst needed. Each domain encodes its own KPI definitions and business logic, so answers stay accurate to how each team measures success.

## Architecture
Owned end-to-end: canonical KPI taxonomy → per-domain YAML knowledge schema → retrieval and routing layer → evaluation harness keeping answers trustworthy as new domains are added.

🧩 **Framework: Domain-Routing + Eval Harness** — add a new domain as a skill, not a rebuild

## Impact
- 💡 **~$216K in annual productivity value** (≈3-4 analyst hrs/week × loaded cost)
- 🚀 4 business domains live, self-serve from day one
- 🧠 Analytics team redirected to strategic work

## Stack
Claude · MCP · Python · Streamlit · YAML

---

Built by [Ajantika Paul](https://ajantika.github.io) · Lead Product Data Analyst @ Cloudflare
