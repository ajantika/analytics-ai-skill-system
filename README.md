# Analytics AI Skill System

A governed analytics assistant with an end-to-end LLM product-quality and evaluation
framework: deterministic evaluators, a written human rubric, an LLM-as-a-Judge, a
60-case adversarial golden dataset, a failure taxonomy, judge calibration against human
raters, and versioned regression testing.

**Live demo:** https://ajantika-analytics-ai.streamlit.app

---

## The claim

> This system does not just generate AI answers. It defines what a high-quality answer
> looks like, evaluates responses using both humans and an automated judge, measures
> whether the automated judge agrees with human judgement, classifies failures into a
> structured taxonomy, and tracks whether a prompt, router or model change actually
> improved quality.

The assistant itself is the smaller half of the project. The larger half is the
apparatus for deciding whether its answers are any good — and for knowing when the
thing measuring quality is itself wrong.

---

## What the evaluation found

Two results, both of which contradict what the numbers look like at first glance.

### The automated judge agreed 89% of the time and was measuring nothing

Across the 18 responses scored by hand and by the LLM judge:

| | |
|---|---|
| Agreement within ±1 point | **89%** |
| Pass/fail agreement | **100%** |
| Cohen's κ, quadratic-weighted | **0.00** |
| Pearson r | −0.12 |
| Human mean / judge mean | 4.45 / 4.95 |

Raw agreement looks strong and means almost nothing. Both raters cluster at the top of
the scale, so nearly all of that agreement is what chance produces anyway. Correct for
it and there is none left: the judge was rating almost everything a 5 and being credited
wherever the human happened to land in the same place.

Pass/fail κ is **undefined**, not zero — both raters passed all 18 cases, so there is no
variance to correct for. The application renders it as an em-dash with that reason rather
than as a number.

The bias is directional on every dimension and largest exactly where the rubric is most
subjective:

| Dimension | Judge − human |
|---|---|
| Helpfulness | +0.72 |
| Clarity | +0.67 |
| Instruction following | +0.56 |
| Correctness | +0.44 |
| Relevance | +0.39 |
| **Groundedness** | **+0.22** |

That shape is the practical finding. The judge tracks humans best on the most objective
dimension and worst on the most subjective ones, which argues for automating groundedness
and correctness first and keeping human review on helpfulness and clarity.

**A dashboard reporting only the 89% would have shipped this judge.**

### A prompt change reasoned from real failures made things worse

`sysprompt-v2` was written to fix observed instruction-following and missing-context
failures. Measured against a baseline run under the original prompt, it was a 16-point
regression on required-fact coverage across all 60 cases. It was not shipped. Details in
[How evaluation feeds product improvement](#how-evaluation-feeds-product-improvement).

Both findings are the same argument: an evaluation number you have not checked against a
reference is not a measurement.

---

## Architecture

```
                          Question
                              │
                    ┌─────────▼─────────┐
                    │   Domain Router   │  keyword scoring, IDF-weighted
                    │  v3_idf_weighted  │  no-match and ties flagged, not guessed
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │ Governed Semantic │  YAML skill files
                    │      Context      │  metric definitions, formulas, owners
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │   LLM Response    │  constrained to the supplied context
                    │   sysprompt-v2    │
                    └─────────┬─────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────▼───────┐   ┌─────────▼─────────┐   ┌───────▼────────┐
│ Deterministic │   │  Human Evaluation │   │ LLM-as-a-Judge │
│     Evals     │   │   6-dim rubric    │   │  same rubric   │
│               │   │                   │   │                │
│ grounding     │   │ THE REFERENCE     │   │ full coverage  │
│ required facts│   │ STANDARD          │   │ cheap, fast    │
│ forbidden     │   │                   │   │ unvalidated    │
│ routing       │   │ slow, low coverage│   │ until measured │
└───────┬───────┘   └─────────┬─────────┘   └───────┬────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │  Human ↔ AI       │  agreement, κ, correlation
                    │  Alignment        │  judge bias by dimension
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │ Failure Analysis  │  taxonomy, severity, root cause
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │ Prompt / Router / │
                    │ Rubric Change     │
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │    Regression     │  same golden set, versioned run
                    │    Evaluation     │  accepted or rejected
                    └───────────────────┘
```

---

## Why three evaluators

### Why deterministic checks are used, and not replaced by the LLM

An LLM is strictly worse than `in` at answering *does this exact string appear in this
text*. It is slower, costs money, varies between calls, and can be wrong. Anything
expressible as a rule stays a rule:

- **Numeric grounding** — do the figures in the answer appear in the governed context?
- **Metric recognition** — is the referenced metric defined in the semantic layer?
- **Required facts** — does the answer contain each fact the reference case specifies?
- **Forbidden claims** — does it contain a phrase the case forbids?
- **Expected-domain routing** — did the router select the declared domain?

Cheap, instant, perfectly repeatable, no drift. They also have a hard limit: they verify
**presence, not correctness**. A real figure attached to the wrong metric passes numeric
grounding. That boundary is where the subjective evaluators begin, and drawing it
explicitly is what keeps each tier honest about its own scope.

### Why humans remain the reference

Relevance, helpfulness, and whether an inference was adequately marked are judgement
calls. There is no rule that decides them, which is why the rubric specifies **anchors**
rather than thresholds.

The automated judge is a model predicting what a human would say. Its scores mean
something only to the extent that prediction holds — an empirical question with an
answer. When human and judge disagree, the human is right *by definition*, because the
human is what the judge approximates.

The cost is coverage: human evaluation is always a sample. The whole architecture exists
to get the judge's coverage with the authority of the human sample, which only works if
the gap between them is measured rather than assumed.

### Why the judge must be calibrated

An uncalibrated judge is not a cheap human rater. It is a different instrument that
happens to emit numbers on the same scale, and treating its scores as a target optimises
toward its idiosyncrasies instead of toward quality.

Failure modes this system measures rather than assumes away:

| Bias | How it is handled |
|---|---|
| **Verbosity bias** — long, fluent answers score higher | Rubric separates helpfulness from clarity; judge prompt states a short correct answer beats a long plausible one; per-dimension agreement shows whether it worked |
| **Self-preference** — a model favours its own output | Judge model chain puts a different, larger model first; the AI Judge page states plainly when judge and generator are the same model |
| **Leniency drift** — scores cluster at 4 | Mean signed difference reported per dimension |
| **Self-inconsistency** — arithmetic and pass-rule errors | Both recomputed from the judge's own dimension scores; discrepancy rates reported; stored records use the recomputed values, never the judge's claim |

### How disagreements are analysed

Aggregate agreement says whether there is a problem; only individual cases say what it
is. Every material disagreement is attributed to a structural cause by rule — *judge
leniency, judge over-severity, unsupported inference accepted, ambiguous rubric,
ambiguous reference answer, human annotation inconsistency, model verbosity bias,
missing-context handling, failure-mode mismatch*.

The classifier is deliberately conservative and returns *uncategorised* rather than
guessing: a plausible-sounding wrong diagnosis is worse than an honest blank, because it
gets acted on.

**Human-versus-human disagreement is checked first.** If two raters applying the same
rubric disagree with each other, the judge is being compared against an unstable
reference and no amount of judge tuning fixes it. Those cases are attributed to the
rubric, and each produces a specific proposed clarification.

### How evaluation feeds product improvement

An evaluation programme that produces dashboards and no changes has failed. The loop:

**Failure detected → root cause identified → change made → golden set re-run → quality
compared → accepted or rejected.**

Four worked examples ship on the Regression Testing page. Three recompute live on every
page load, so the before-and-after is evidence rather than narration:

| # | Finding | Change | Measured result |
|---|---|---|---|
| 1 | `"pipeline coverage"` routed to Product — `"coverage"` contains the keyword `"overage"` | `v2_token_aware`: word-boundary matching | Substring false positives eliminated; accuracy barely moved — the bug was real but not dominant |
| 2 | Most remaining misroutes were **ties** broken by dict iteration order | `v3_idf_weighted`: inverse-domain-frequency weighting; no-match and ties flagged explicitly | Routing accuracy **76.3% → 88.1%**; silent misroutes **10 → 1** |
| 3 | Five cases matched *no* keyword — skill files listed `reopened` but not `reopen`, defined NRR but never listed `nrr` | Added the missing terms to three skill files | Three of five resolved; the other two ask about named individuals and are left failing and visible |
| 4 | Every `instruction_following` case failed — the prompt hard-coded *"use these exact headers"* | `sysprompt-v2`: template becomes a default that explicit user instructions override, plus premise-checking and an explicit "not in the governed layer" outcome | **Rejected.** Required-fact pass rate **89% → 73%**, grounding **65% → 57%**. Not shipped |

Case 4 is the one worth reading twice. The change was reasoned from real failures and
predicted to help. Measured against a baseline run under the original prompt across the
full 60 cases, it was a 16-point regression on the metric that matters most: whether the
answer actually contains the figures the reference requires.

The mechanism is visible in the responses. Telling the model to decline when the governed
layer lacks an answer, and to mark inference explicitly, made it hedge on questions it
could answer. The two metrics that improved — unsupported claims 3% → 0%, forbidden-claim
violations 7% → 4% — are the ones that reward saying less. That is a real effect and the
wrong trade for an assistant whose job is reporting governed figures.

Both prompt versions remain callable and the default is unchanged. One deterministic
comparison is not sufficient grounds to switch, and the judge and human dimensions that
would confirm the trade-off are not yet measured on the baseline.

*(An earlier version of this comparison rested on 55 v1 cases: the baseline lost its final
five to rate limiting on a free-tier daily budget. Those have since been generated and the
figures above cover all 60 on both sides. Generation is now paced to prevent a recurrence,
and an empty completion — which arrives as a 200 with no content — is recorded as a failure
rather than stored as a zero-length answer.)*

The second column of case 2 matters more than the first: a flagged uncertain route lets
the user pick a domain; a silent one produces a fluent, well-formatted answer built on
the wrong governed context.

---

## The golden dataset

60 cases in `data/golden_eval_set.json`, authored against the five governed YAML skill
files. Every `required_fact` is a figure that exists verbatim in a skill file — enforced
by a test that loads each file and checks.

| Test type | n | What it probes |
|---|---|---|
| `standard` | 15 | Baseline competence |
| `grounding` | 11 | Figure fidelity under adjacent-metric pressure |
| `adversarial` | 7 | False premises, policy boundaries |
| `cross_domain` | 7 | Questions spanning two skills |
| `missing_context` | 6 | Answers genuinely absent from the governed layer |
| `ambiguous` | 6 | Underspecified scope |
| `instruction_following` | 5 | Explicit format instructions conflicting with the template |
| `unsupported_inference` | 3 | Causal claims the data cannot support |

**Three genuine inconsistencies in the governed layer are included deliberately** rather
than corrected:

- Total MRR differs between the Product and Sales skills ($4.7M vs $4.2M)
- New-customer counts coincide between Sales and Marketing (both 312) without the layer
  stating whether they describe the same population
- One figure (90-day new-hire retention) exists only in a skill's sample Q&A, not in its
  governed metric definitions

A governance layer with conflicts is the normal condition. An evaluation suite that
quietly avoids them is not testing governance.

---

## The rubric

Six dimensions, 1–5, with written anchors so two evaluators would apply comparable
standards: **relevance, groundedness, correctness, instruction following, helpfulness,
clarity**. Each dimension states both what to score on and what to exclude.

```
PASS requires all of: groundedness ≥ 3, correctness ≥ 3, relevance ≥ 3,
and no critical failure mode. Any single dimension at 1 is an automatic FAIL.
```

The same rule is applied by the human UI default, the demo profiles, and the LLM judge —
which is what makes their pass rates comparable.

---

## Running it

```bash
pip install -r requirements.txt

# The app. Reads stored evaluation records; works without an API key
# (answer generation and the live judge need one).
streamlit run app.py

# Generate the evaluation artifacts the dashboards read.
export GROQ_API_KEY="..."          # read from env, falling back to Streamlit secrets
python eval_runner.py --all --label "current"

# A baseline under the earlier prompt, for regression comparison
python eval_runner.py --all --prompt-version sysprompt-v1 --label "baseline prompt"

python eval_runner.py --list
python eval_runner.py --compare <baseline-run-id> <current-run-id>

# Full test suite — no dependencies beyond the app's own
python tests/run_tests.py
```

The key is never read from, or written to, any file in this repository.

---

## Project structure

```
app.py                 shell: page config, theme, navigation, state assembly
router.py              3 keyword routers, all callable, so improvements stay measurable
skills.py              YAML skill loading, governed context construction
llm.py                 Groq calls + versioned system prompts
evals.py               deterministic + heuristic evaluators, golden-set loading
human_evals.py         rubric, annotation records, storage, rater calibration
llm_judge.py           judge prompt, strict JSON parsing, self-consistency checks
demo_raters.py         scripted rubric profiles — NOT human annotators
alignment.py           human ↔ judge agreement, bias, disagreement attribution
failure_taxonomy.py    12 failure modes with severity, cause, remediation
stats_utils.py         κ, Pearson, Spearman, agreement — no SciPy dependency
eval_runner.py         run orchestration, artifact freezing, regression comparison

ui/theme.py            styling and presentation primitives
ui/data.py             cached loaders, assembled evaluation state
ui/page_*.py           10 pages

data/                  golden set (committed) + generated artifacts (see data/README.md)
tests/                 297 tests, runnable with pytest or tests/run_tests.py
```

---

## Technical honesty

### Implemented

Golden dataset · six-dimension human rubric · deterministic evaluators · LLM-as-a-Judge
with structured-output validation · human annotation workflow with provenance tracking ·
human↔AI agreement (exact, ±1, pass agreement, weighted Cohen's κ, Pearson, Spearman,
per dimension) · human↔human calibration with disagreement diagnosis · 12-category
failure taxonomy · versioned runs with regression comparison · three router versions kept
callable · 297 unit tests.

### Demo or simulated — labelled as such in the UI

- **Demo rater profiles** (`demo_raters.py`) — two scripted rubric profiles, strict and
  lenient, that read the real generated responses and apply explicit anchor rules. Every
  record carries `rater_type: "demo_profile"`. They exist so the calibration workflow is
  demonstrable before human ratings exist. **They are not human annotators and no figure
  derived from them is presented as human agreement anywhere.**
- **Governed data** — all figures in the YAML skill files are illustrative, written for
  this project. They resemble SaaS analytics but describe no real company.
- **Scale** — 60 cases. Enough to expose failure modes and compare configurations; not a
  statistically powered benchmark.

### Not claimed

Production traffic · real customer, employee or revenue data · a managed annotation
programme or contractor workforce · production monitoring or an automated quality gate in
a deployment pipeline · a held-out benchmark · semantic or embedding-based routing · that
deterministic checks verify factual correctness · that the AI judge is validated.

---

## Known limitations

**The golden set is not held out.** It was authored by the person who built the system,
against the same YAML files the system answers from. Overfitting is possible and should
be assumed. An independently authored set is the single highest-value addition this
project could receive.

**Sample sizes are small.** Chance-corrected statistics such as Cohen's κ are unstable
below roughly 15–20 paired observations, and per-slice figures often rest on five or six
cases. Where a statistic is undefined it renders as an em-dash, never as zero — zero
means "no agreement beyond chance", which is a different claim from "not enough data".

**Routing is keyword scoring.** Labelled `keyword_idf_weighted` everywhere and never
described as semantic. Embedding-based routing would almost certainly do better on the
ambiguous cases; it is not implemented, and no claim is made about how much better.

**Confidence labels are not calibrated probabilities.** Routing confidence reflects
keyword score separation, and is shown as High/Medium/Low rather than a percentage for
that reason.

**Generation is non-deterministic.** Responses are generated at temperature 0.2; the
judge runs at temperature 0. Run-to-run variance is not quantified, so small deltas on
the regression page should not be over-read. Comparing two identically configured runs is
the cheapest way to bound it.

---

Built by [Ajantika Paul](https://ajantika.github.io) · portfolio project · illustrative
data only
