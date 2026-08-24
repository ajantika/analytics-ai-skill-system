"""
ui/page_methodology.py — Methodology and honest scope.

Explanation lives here so the other pages can lead with numbers. This page also
carries the implemented / simulated / not-claimed breakdown, which is the part a
reader should be able to find without hunting.
"""

from __future__ import annotations

import streamlit as st

from human_evals import CONFIDENCE_RULE, CRITICAL_RULE, PASS_RULE, rubric_markdown
from llm import PROMPT_VERSION_NOTES
from router import ROUTER_VERSION_NOTES
from ui import data as D
from ui import theme as T

ARCHITECTURE = """
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
"""


def _scope_columns(state: D.EvalState) -> None:
    cov = state.coverage

    implemented = [
        f"Golden evaluation dataset — {len(state.cases)} adversarial cases authored against the "
        f"governed skill files",
        "Six-dimension human rubric with written anchors, a pass rule and a critical-failure rule",
        "Deterministic evaluators — numeric grounding, metric recognition, required facts, "
        "forbidden claims, expected-domain routing",
        "LLM-as-a-Judge with strict structured output, one repair retry, and recomputation of its "
        "own arithmetic and pass rule",
        "Human annotation workflow with provenance tracking and persistence",
        "Human ↔ AI agreement — exact, within-±1, pass agreement, weighted Cohen's κ, Pearson, "
        "Spearman, per dimension and overall",
        "Human ↔ human calibration with disagreement diagnosis and rubric clarifications",
        "Failure taxonomy — 12 categories with severity, cause and remediation",
        "Versioned evaluation runs with regression comparison across every metric",
        "Three router implementations kept callable so improvements are measurable",
        "296 unit tests covering the statistics, parsing, taxonomy and dataset integrity",
    ]

    demo = [
        ("Demo rater profiles",
         "Two scripted rubric profiles (strict and lenient) that read the real generated responses "
         "and apply explicit anchor rules. They are labelled <code>demo_profile</code> everywhere "
         "and are never counted as human annotations. They exist so the calibration workflow is "
         "demonstrable before human ratings exist."),
        ("Governed data",
         "All figures in the YAML skill files are illustrative, written for this project. They "
         "resemble SaaS analytics but describe no real company."),
        ("Evaluation scale",
         f"{cov['total_cases']} cases, {cov['judge_evaluated']} judged, "
         f"{cov['human_annotated']} rated by a person. Enough to expose failure modes and compare "
         f"configurations; not a statistically powered benchmark."),
    ]

    not_claimed = [
        "Production traffic — this system has never served a user question outside this demo",
        "Real customer, employee or revenue data of any kind",
        "A managed annotation programme, contractor workforce, or annotation vendor",
        "Production monitoring, alerting, or an automated quality gate in a deployment pipeline",
        "A held-out benchmark — the golden set was authored by the same person who built the system",
        "Semantic or embedding-based routing — routing is keyword scoring and is labelled as such",
        "That the deterministic checks verify factual correctness — they verify presence",
        "That the AI judge is validated — its agreement with human raters is measured and reported, "
        "and at current coverage that measurement is directional",
    ]

    col1, col2 = st.columns(2, gap="medium")

    with col1:
        st.markdown(
            T.panel(
                T.label_text("Implemented", T.GOOD)
                + "".join(
                    f'<div style="color:{T.TEXT};font-size:0.78rem;line-height:1.65;padding:3px 0;'
                    f'border-bottom:1px solid rgba(255,255,255,0.03)">✓ {item}</div>'
                    for item in implemented
                ),
                accent=T.GOOD,
            ),
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            T.panel(
                T.label_text("Demo or simulated — clearly labelled in the UI", T.WARN)
                + "".join(
                    f'<div style="padding:5px 0;border-bottom:1px solid rgba(255,255,255,0.03)">'
                    f'<div style="color:{T.INK};font-size:0.78rem;font-weight:600">{title}</div>'
                    f'<div style="color:{T.FAINT};font-size:0.74rem;line-height:1.6">{desc}</div>'
                    f'</div>'
                    for title, desc in demo
                ),
                accent=T.WARN,
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            T.panel(
                T.label_text("Not claimed", T.BAD)
                + "".join(
                    f'<div style="color:{T.TEXT};font-size:0.78rem;line-height:1.65;padding:3px 0;'
                    f'border-bottom:1px solid rgba(255,255,255,0.03)">✗ {item}</div>'
                    for item in not_claimed
                ),
                accent=T.BAD,
            ),
            unsafe_allow_html=True,
        )


def render(state: D.EvalState) -> None:
    T.page_header(
        "Methodology",
        "How quality is defined, measured and acted on in this system — and an explicit statement "
        "of what is real, what is demonstration, and what is not being claimed.",
        eyebrow="Reference",
    )

    T.section("The claim this system makes", top_rule=False)
    st.markdown(
        T.panel(
            T.body_text(
                "This system does not just generate AI answers. It defines what a high-quality "
                "answer looks like, evaluates responses using both humans and an automated judge, "
                "measures whether the automated judge agrees with human judgement, classifies "
                "failures into a structured taxonomy, and tracks whether a prompt, router or model "
                "change actually improved quality.",
                T.INK, "0.92rem",
            ),
            accent=T.ACCENT,
        ),
        unsafe_allow_html=True,
    )

    T.section("Architecture")
    st.code(ARCHITECTURE, language="text")

    T.section("The evaluator hierarchy", "Three evaluators, different jobs. Confusing them is the "
                                          "most common way an evaluation programme goes wrong.")
    T.evaluator_tier_legend()

    with st.expander("Why deterministic checks are not replaced by the LLM", expanded=False):
        st.markdown("""
An LLM is strictly worse than `in` at answering *does this exact string appear in this
text*. It is slower, costs money, varies between calls, and can be wrong. Every check that
can be expressed as a rule stays a rule:

- **Numeric grounding** — do the figures in the answer appear in the governed context?
- **Metric recognition** — is the referenced metric defined in the semantic layer?
- **Required facts** — does the answer contain each fact the reference specifies?
- **Forbidden claims** — does the answer contain a phrase the case forbids?
- **Expected-domain routing** — did the router select the declared domain?

These are cheap, instant, perfectly repeatable, and they never drift. They also have a hard
limit: they verify *presence*, not correctness. A real figure attached to the wrong metric
passes numeric grounding. That is exactly the boundary where the subjective evaluators
start, and drawing the boundary explicitly is what keeps each tier honest about its scope.
""")

    with st.expander("Why humans remain the reference standard"):
        st.markdown("""
Relevance, helpfulness and whether an inference was adequately marked are judgement calls.
There is no rule that decides them, which is why the rubric specifies anchors rather than
thresholds.

The automated judge is a model trained to predict what a human would say. Its scores are
only meaningful to the extent that prediction holds — and that is an empirical question
with an answer, measured on the Alignment page. When human and judge disagree, the human
is right by definition, because the human is what the judge is trying to approximate.

The cost of this is coverage. Humans are slow and expensive, so human evaluation is always
a sample. The judge covers everything cheaply. The whole architecture exists to get the
coverage of the judge with the authority of the human sample — which only works if the gap
between them is measured rather than assumed.
""")

    with st.expander("Why the automated judge must be calibrated"):
        st.markdown("""
An uncalibrated judge is not a cheap human rater. It is a different instrument that happens
to output numbers on the same scale, and using its scores as a quality target optimises the
system toward the judge's idiosyncrasies rather than toward quality.

Known failure modes this application measures rather than assumes away:

**Verbosity bias.** Judges reward long, fluent, confident answers. The rubric explicitly
separates helpfulness from clarity, and the judge prompt states that a short correct answer
beats a long plausible one — but whether that instruction worked is visible only in the
per-dimension agreement figures.

**Self-preference.** A model rates its own output more favourably than a third party would.
The judge model chain puts a different model first, and the AI Judge page states plainly
when the judge and generator ended up being the same model.

**Leniency drift.** Judges tend to cluster scores at 4 and reserve 1 and 2 for obvious
failures, which compresses exactly the range where a quality gate operates. Mean signed
difference per dimension is reported for this reason.

**Self-inconsistency.** The judge's own arithmetic and its own application of the pass rule
are both checked against its dimension scores, and the discrepancy rates are reported. The
stored record uses the recomputed values, never the judge's claim.
""")

    with st.expander("How disagreements are analysed"):
        st.markdown("""
Aggregate agreement says whether there is a problem. Only individual cases say what it is.
Every material disagreement is attributed to a structural cause by rule:

*judge leniency · judge over-severity · unsupported inference accepted · ambiguous rubric ·
ambiguous reference answer · human annotation inconsistency · model verbosity bias ·
missing-context handling · failure-mode mismatch*

The classifier is deliberately conservative and returns *uncategorised* rather than guessing.
A plausible-sounding wrong diagnosis is worse than an honest blank, because it gets acted on.

One ordering rule matters more than the rest: **human-versus-human disagreement is checked
first**. If two raters applying the same rubric disagree with each other, the judge is being
compared against an unstable reference, and no amount of judge tuning fixes that. Those
cases are attributed to the rubric, not to the judge, and each one produces a specific
proposed clarification.
""")

    with st.expander("How evaluation results feed product improvement"):
        st.markdown("""
An evaluation programme that produces dashboards and no changes has failed. The loop:

1. **Failure detected** — a golden case fails, or evaluators disagree about it.
2. **Root cause identified** — the taxonomy supplies the category; the case record supplies
   the specifics.
3. **Change made** — to the router, the system prompt, the rubric, or the skill files.
4. **Golden set re-run** — same cases, new run, versioned configuration.
5. **Quality compared** — every metric against its own good direction.
6. **Accepted or rejected** — by a person, on the evidence.

Four worked examples are on the Regression Testing page, three of which recompute live on
every page load. One of the four is reported as *not yet measured*, because it changes
generated text and needs a model run under each prompt version. Leaving it unmeasured
rather than asserting a plausible improvement is the same discipline the rest of the system
applies.
""")

    T.section("The rubric", "The written standard both humans and the judge apply.")
    with st.expander("Full rubric with anchors", expanded=False):
        st.markdown(rubric_markdown())

    st.markdown(
        T.panel(
            T.label_text("Pass rule", T.GOOD) + T.body_text(PASS_RULE, T.TEXT, "0.8rem")
            + '<div style="height:8px"></div>'
            + T.label_text("Critical failure", T.BAD) + T.body_text(CRITICAL_RULE, T.TEXT, "0.8rem")
            + '<div style="height:8px"></div>'
            + T.label_text("Evaluator confidence", T.INFO)
            + T.body_text(CONFIDENCE_RULE, T.TEXT, "0.8rem"),
            accent=T.ACCENT,
        ),
        unsafe_allow_html=True,
    )

    T.section("Versioned components", "Every run records which version of each produced it.")
    for title, notes in (("Router", ROUTER_VERSION_NOTES), ("System prompt", PROMPT_VERSION_NOTES)):
        st.markdown(
            f'<div style="color:{T.DIM};font-size:0.7rem;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:0.08em;margin:10px 0 5px">{title}</div>',
            unsafe_allow_html=True,
        )
        for version, note in notes.items():
            st.markdown(
                f'<div style="padding:5px 0;border-bottom:1px solid rgba(255,255,255,0.04)">'
                f'<code style="font-size:0.72rem">{version}</code>'
                f'<div style="color:{T.FAINT};font-size:0.75rem;line-height:1.65;margin-top:2px">'
                f'{note}</div></div>',
                unsafe_allow_html=True,
            )

    T.section("Technical honesty", "What is real, what is demonstration, what is not claimed.")
    _scope_columns(state)

    T.section("Known limitations", "Stated here rather than discovered by a reader.")
    st.markdown(f"""
**The golden set is not held out.** It was authored by the person who built the system,
against the same YAML skill files the system answers from. Overfitting is possible and
should be assumed. An independently authored evaluation set is the single highest-value
addition this project could receive.

**Sample sizes are small.** {len(state.cases)} cases total; human coverage is
{T.pct(state.coverage.get('human_coverage'))}. Chance-corrected statistics such as Cohen's κ
are unstable below roughly 15–20 paired observations, and per-slice figures often rest on
five or six cases. Where a statistic is undefined it renders as an em-dash, never as zero.

**Routing is keyword scoring.** It is labelled `keyword_idf_weighted` everywhere and is
never described as semantic. Embedding-based or LLM-based routing would almost certainly
outperform it on the ambiguous cases; it is not implemented here, and no claim is made
about how much better it would be.

**Confidence labels are not calibrated probabilities.** Routing confidence reflects keyword
score separation. It is a heuristic, displayed as High / Medium / Low rather than as a
percentage for that reason.

**Generation is non-deterministic.** Responses are generated at temperature 0.2, so two
runs with identical configuration produce different text and slightly different scores.
The judge runs at temperature 0. Run-to-run variance is not currently quantified, which
means small deltas on the regression page should not be over-read.

**Demo rater profiles are not people.** They read real responses and apply real anchor
rules, which makes them useful for demonstrating the calibration workflow. They cannot
substitute for human judgement, and no figure derived from them is presented as human
agreement anywhere in this application.
""")

    T.section("Running the evaluation pipeline")
    st.code("""# Generate responses, run deterministic checks, and run the LLM judge
export GROQ_API_KEY="..."
python eval_runner.py --all --label "current"

# A baseline for comparison, under the earlier prompt
python eval_runner.py --all --prompt-version sysprompt-v1 --label "baseline prompt"

# Inspect and compare
python eval_runner.py --list
python eval_runner.py --compare <baseline-run-id> <current-run-id>

# The full test suite — no dependencies beyond the app's own
python tests/run_tests.py""", language="bash")
