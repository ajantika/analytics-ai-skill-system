# Analytics AI Skill System

A multi-domain analytics assistant that routes natural-language business questions to governed
domain skills and returns explainable, evaluated answers grounded in defined business metrics.

**Live demo:** https://ajantika-analytics-ai.streamlit.app

---

## The Problem

Analytics teams repeatedly answer the same business questions — MRR recovery opportunities,
utilization health, pipeline coverage, attrition risk — whose logic already exists across
dashboards, SQL, documentation, and analyst knowledge.

Generic LLM chatbots do not solve this. They confidently invent metrics, ignore governance,
and produce answers that cannot be audited or trusted.

This prototype demonstrates a different approach: **LLM reasoning constrained by a governed
semantic layer**, with visible routing, deterministic evaluation checks, and honest labelling
of what is and is not verified.

---

## The Solution

A question-answering system where:

- Business metric definitions live in structured **YAML skill files**, not inside LLM prompts
- Questions are **routed to the correct domain** before any LLM call is made
- The LLM is **constrained to the supplied context** — it cannot invent KPI definitions
- Every answer is **evaluated** across four dimensions with clearly labelled methods
- Failures and limitations are **transparent**, not hidden

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
| `skills.py` | YAML skill loading, governed metric layer, context building, follow-up questions |
| `llm.py` | Groq/Llama API interaction, structured output parsing |
| `evals.py` | Evaluation framework, eval dataset, routing accuracy runner |
| `tests/` | 17 unit tests — routing (10) and evaluation (7) |
| `*.yaml` | Domain skill files — one per domain |
| `.streamlit/config.toml` | Dark theme configuration |

---

## Governed Metric Layer

Business metric definitions live in YAML skill files, not inside the LLM prompt template.

Each metric definition can include:

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
- Adding a new domain requires only a new YAML file, with no application code changes
- Metric ownership and validation dates are explicit and auditable

---

## Domain Routing

Routing is **deterministic keyword scoring** — explicitly labelled throughout the codebase and UI.

- Each YAML skill file contains a `keywords` list
- Questions are scored by keyword overlap (exact phrase match: +2, partial word: +1)
- Confidence is derived from **score separation** between top and runner-up domains — a routing heuristic, not a calibrated probability
- The UI displays routing confidence as **High / Medium / Low**, not a raw percentage
- Questions below the confidence threshold trigger a disambiguation UI rather than a confident wrong answer
- `router.py` is designed to be swapped out for embedding-based or LLM-based routing without changing any other module

**This is never misrepresented as semantic, embedding, or AI routing.**

---

## Evaluation

### Routing evaluation

The system is evaluated against 13 curated questions: 11 with expected domain labels
and 2 intentionally ambiguous questions excluded from accuracy calculation.

**Current results: 81% baseline routing accuracy (9 of 11)**

| Question | Expected | Predicted | Result |
|---|---|---|---|
| Which customers are over-utilizing their plans? | Product | Product | ✓ |
| What is the MRR recovery opportunity from right-sizing? | Product | Product | ✓ |
| Which regions have the highest over-utilization? | Product | Product | ✓ |
| Which campaign brought the highest number of customers? | Marketing | **Sales** | ✗ |
| How are our MQL to SQL conversion rates trending? | Marketing | Marketing | ✓ |
| Which sales rep gives the highest discounts? | Sales | Sales | ✓ |
| What is our pipeline coverage ratio? | Sales | **Product** | ✗ |
| Which teams have the highest attrition? | HR | HR | ✓ |
| Are we on track with our hiring plan? | HR | HR | ✓ |
| What is our CSAT score? | Support | Support | ✓ |
| Who are the top performing support agents? | Support | Support | ✓ |

**Failure analysis:**

- *"Which campaign brought the highest number of customers?"* → mis-routed to Sales.
  `customers` scores +2 for Sales; `new customer` keyword adds a partial +1.
  Marketing only gets +2 from `campaign`. Shared customer terminology gives Sales the edge.
  Whole-word or TF-IDF weighting would likely correct this.

- *"What is our pipeline coverage ratio?"* → mis-routed to Product.
  `coverage` contains `overage` as a substring — a Product keyword — triggering a +2 false-positive match.
  Substring matching cannot distinguish word boundaries.
  Whole-word matching would prevent this false positive.

Failures are not hidden. They identify exactly where keyword scoring breaks down and what routing improvements would close the gap.

### Answer quality checks

Every generated response is evaluated across four dimensions:

| Dimension | Method | What it checks |
|---|---|---|
| Numeric grounding | **Deterministic** | Do figures in the answer appear in the governed context? |
| Metric recognition | **Deterministic** | Are referenced metrics defined in the domain skill? |
| Answer relevance | **Heuristic** | Term overlap + length + non-answer phrase detection |
| Unsupported claims | **Heuristic** | Generalisation phrase matching |

All methods are labelled in the UI. No evaluation scores are fabricated.

**What these checks do not claim:** They do not verify factual correctness, eliminate hallucinations,
or confirm the LLM's reasoning. Numeric grounding confirms a figure appeared in the supplied context —
not that the figure was used correctly.

### Evaluation limitations

- Small, curated evaluation dataset — not a statistically meaningful benchmark
- Not a held-out test set — keyword lists and eval questions were developed together; overfitting is possible
- Deterministic keyword baseline — a semantic router would likely outperform on ambiguous questions
- No independent human evaluation of answer correctness or usefulness
- No calibrated routing probability — High/Medium/Low labels reflect score separation, not a trained classifier
- All figures are pre-embedded in YAML skill files, not live business data

---

## Example questions — recommended demo sequence

1. *"Which customers are over-utilizing their plans?"* — strong Product routing, grounded in specific figures
2. *"What is the MRR recovery opportunity from right-sizing?"* — tests $1.4M numeric grounding
3. *"Which regions have the highest over-utilization?"* — regional dimension retrieval
4. Click a suggested follow-up question
5. *"Why is revenue declining?"* — triggers multi-domain disambiguation (by design)

---

## Prototype vs Production

### This demo uses

- Streamlit (UI)
- YAML domain skill files (governed metric layer)
- Keyword-based domain routing (deterministic, not semantic)
- Groq / Llama 3.1 (LLM)
- Illustrative pre-built data — no live database connection
- Deterministic + heuristic evaluation checks
- 17 unit tests covering routing and evaluation logic

### Production would require

| Capability | Production approach |
|---|---|
| Data layer | Snowflake / data warehouse with live query execution |
| Semantic layer | dbt semantic layer or Cube (governed metrics at scale) |
| Routing | Embedding-based or LLM-based semantic routing |
| Access control | RBAC, row-level security, PII masking |
| Observability | Latency, cost, and token monitoring |
| Evaluation | Held-out eval datasets, automated regression testing |
| Human feedback | Analyst review loop for correctness and usefulness |
| Reliability | Caching, fallback models, graceful degradation |
| Security | Prompt injection testing, audit logging |

**None of the production capabilities above exist in this demo.**

---

## Limitations

- **No live data.** All figures are pre-embedded in YAML skill files.
- **Keyword routing is brittle** on ambiguous or multi-domain questions. The confidence threshold and disambiguation UI mitigate this but do not eliminate it.
- **Substring matching causes false positives** — e.g. `overage` matches inside `coverage`. Whole-word matching is a known improvement.
- **Evaluation is heuristic** — it catches common failure modes but is not a substitute for ground-truth evaluation datasets or human review.
- **LLM output is non-deterministic.** The same question may produce slightly different answers.
- **No conversation memory.** Each question is independent.
- **No RAG, no agents, no Snowflake, no semantic routing** — these are not implemented.

Technical honesty is a feature of this portfolio, not a limitation of it.

---

## Running locally

```bash
git clone https://github.com/ajantika/analytics-ai-skill-system.git
cd analytics-ai-skill-system
pip install -r requirements.txt
```

Add your Groq API key to `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "gsk_..."
```

```bash
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
[LinkedIn](https://linkedin.com/in/ajantika-paul) · [GitHub](https://github.com/ajantika)
