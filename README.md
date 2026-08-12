# Analytics AI Skill System

A multi-domain Generative AI Analytics prototype that routes natural-language business questions
to governed domain skills and generates explainable, evaluated answers grounded in defined business metrics.

**Live demo:** https://ajantika-analytics-ai.streamlit.app

---

## Problem

Analytics teams repeatedly answer the same business questions — MRR recovery opportunities,
utilization health, pipeline coverage, attrition risk — whose logic already exists across
dashboards, SQL, documentation, and analyst knowledge.

Generic LLM chatbots don't solve this. They confidently invent metrics, ignore governance,
and produce answers nobody should trust.

This prototype demonstrates a different approach: **LLM reasoning constrained by a governed
semantic layer**, with visible routing, evaluation checks, and honest labelling of what
is and isn't verified.

---

## Architecture

```
User Question (plain English)
        ↓
Domain Router         ←  keyword scoring (deterministic, not semantic)
        ↓
Analytics Skill       ←  YAML file for the detected domain
        ↓
Governed Metric Layer ←  metric definitions, formulas, owners, validation dates
        ↓
LLM                   ←  Groq / Llama 3.1, constrained to supplied context
        ↓
Evaluation            ←  deterministic + heuristic checks, labelled by method
        ↓
Business Answer       ←  Insight / Why it matters / Recommended action
```

### Module structure

| File | Responsibility |
|---|---|
| `app.py` | Streamlit UI and orchestration |
| `router.py` | Keyword-based domain classification with confidence scoring |
| `skills.py` | YAML skill loading, metric layer, context building, follow-up questions |
| `llm.py` | Groq/Llama API interaction, structured output parsing |
| `evals.py` | Evaluation framework, eval dataset, routing accuracy runner |
| `tests/` | 17 unit tests — routing (10) and evaluation (7) |
| `*.yaml` | Domain skill files — one per domain |
| `.streamlit/config.toml` | Dark theme configuration |

---

## Governed Metric Layer

Business metric definitions live in YAML skill files, not inside the LLM prompt template.

Each metric contains:

```yaml
- name: Plan Fit Score
  business_definition: Measures actual utilization relative to contracted capacity
  formula: actual_usage / contracted_capacity
  interpretation: "<0.5 = churn risk; 0.5–1.0 = healthy; >1.0 = expansion opportunity"
  dimensions: [customer_id, plan_tier, region]
  owner: Product Analytics
  source: Demo Product Usage Dataset
  last_validated: "2026-Q1"
```

This means:
- The LLM cannot invent KPI definitions — it is constrained to governed definitions
- Adding a domain requires only a new YAML file, no application code changes
- Metric ownership and validation dates are explicit

---

## Domain Routing

Routing is **deterministic keyword scoring** — explicitly labelled throughout the codebase and UI.

- Each YAML skill contains a `keywords` list
- Questions are scored by keyword overlap (exact phrase: +2, partial word: +1)
- Confidence is derived from score separation between top and runner-up domains
- Questions below the confidence threshold trigger a disambiguation UI
- `router.py` is designed to be replaced with embedding or LLM-based routing without
  changing the rest of the application

**This is never misrepresented as semantic, embedding, or AI routing.**

---

## Evaluation

Every generated response is evaluated by `evals.py` across four dimensions:

| Dimension | Method | What it checks |
|---|---|---|
| Numeric grounding | **Deterministic** | Do figures in the answer appear in the governed context? |
| Metric definition | **Deterministic** | Are referenced metrics defined in the semantic layer? |
| Answer relevance | **Heuristic** | Term overlap + length + non-answer phrase detection |
| Unsupported claims | **Heuristic** | Generalisation phrase matching |

All methods are labelled in the UI. No evaluation scores are fabricated.

The System Evaluation tab runs routing accuracy against 13 curated questions (11 with expected
domain labels, 2 intentionally ambiguous). Failures are shown and explained.

---

## Demo — recommended sequence

1. *"Which customers are over-utilizing their plans?"* — strong product routing
2. *"What is the MRR recovery opportunity from right-sizing?"* — revenue metric grounding
3. *"Which regions have the highest over-utilization?"* — regional dimension
4. Click a suggested follow-up question
5. Try *"Why is revenue declining?"* — triggers multi-domain disambiguation

---

## Prototype vs Production

### This demo uses
- Streamlit (UI)
- YAML skill files (governed metric layer)
- Keyword-based routing (deterministic)
- Groq / Llama 3.1 (LLM)
- Illustrative pre-built data — no live database
- Deterministic + heuristic evaluation

### Production would require
- Snowflake / warehouse integration (live query execution)
- dbt semantic layer or Cube (governed metrics at scale)
- Embedding-based or LLM-based semantic routing
- Role-based access controls and PII masking
- Query auditing and observability (latency, cost, tokens)
- Evaluation datasets and automated regression testing
- Human feedback loop
- Caching, fallback models, graceful degradation
- Prompt injection testing and security review

**None of the production capabilities above exist in this demo.**

---

## Limitations

- **No live data.** All figures are pre-embedded in YAML skill files.
- **Keyword routing is brittle** on ambiguous or multi-domain questions.
  The confidence threshold and disambiguation UI mitigate this but do not eliminate it.
- **Evaluation is heuristic** — it catches common failure modes but is not
  a substitute for ground-truth evaluation datasets.
- **LLM output is non-deterministic.** The same question may produce slightly different answers.
- **No conversation memory.** Each question is independent.
- **No RAG, no agents, no Snowflake, no semantic routing** — these are not implemented.

Technical honesty is a feature of this portfolio.

---

## Running locally

```bash
pip install -r requirements.txt

# Add your Groq API key to .streamlit/secrets.toml:
# GROQ_API_KEY = "gsk_..."

streamlit run app.py
```

### Running tests

```bash
python tests/test_router.py
python tests/test_evals.py
```

---

## Built by

**Ajantika Paul** — Lead Product Data Analyst  
[ajantika.github.io](https://ajantika.github.io) · [github.com/ajantika](https://github.com/ajantika)
