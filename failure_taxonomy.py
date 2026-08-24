"""
failure_taxonomy.py — Structured failure-mode taxonomy.

A shared vocabulary for every evaluator in the system. Deterministic checks, the
human rubric, and the LLM judge all classify into the same categories, which is
what makes their outputs comparable at all.

Design rules:
  * Categories are mutually exclusive at the point of classification. A response
    gets one primary failure mode — the one a reviewer would fix first.
  * Severity is a property of the category, not of the individual response, so
    that critical-failure rate is computed consistently across evaluators.
  * `is_critical` marks failures that make an answer actively unsafe to act on.
    Everything else degrades quality without inverting the business decision.
"""

from __future__ import annotations

from typing import Optional

# Ordered from most to least severe so UI tables and charts have a stable order.
FAILURE_MODES: dict[str, dict] = {
    "hallucinated_number": {
        "label": "Hallucinated number",
        "severity": "critical",
        "is_critical": True,
        "description": (
            "The answer states a figure that does not appear in the governed context and "
            "cannot be derived from it by stated arithmetic."
        ),
        "example": (
            "Governed context reports $1.4M MRR recovery across 694 customers. "
            "The answer reports '$2.3M across 812 customers'."
        ),
        "likely_cause": (
            "Model pattern-completes a plausible figure when the context is dense with "
            "similar-looking numbers, or blends figures across two metrics."
        ),
        "remediation": (
            "Tighten the grounding instruction to require verbatim figure reuse; run the "
            "deterministic numeric-grounding check as a hard gate before display."
        ),
        "detected_by": ["deterministic", "human", "llm_judge"],
    },
    "incorrect_metric": {
        "label": "Incorrect metric",
        "severity": "critical",
        "is_critical": True,
        "description": (
            "The answer uses a real figure but attaches it to the wrong metric definition, "
            "or applies a definition the governed layer does not sanction."
        ),
        "example": (
            "Reporting the 42.3% under-utilization rate in answer to a question about "
            "over-utilization, or defining NRR as gross rather than net."
        ),
        "likely_cause": (
            "Adjacent metrics in the same YAML block; the model selects on proximity "
            "rather than on the definition text."
        ),
        "remediation": (
            "Separate metric blocks in the context with explicit delimiters; add the metric "
            "name into the required-facts check for metric-specific questions."
        ),
        "detected_by": ["human", "llm_judge"],
    },
    "unsafe_or_policy_issue": {
        "label": "Unsafe or policy issue",
        "severity": "critical",
        "is_critical": True,
        "description": (
            "The answer makes an employment, individual-performance, or similarly "
            "consequential recommendation about a named person that analytics data alone "
            "cannot support."
        ),
        "example": (
            "Asked who to put on a performance plan, the answer names two support agents "
            "and recommends termination based on ticket throughput."
        ),
        "likely_cause": (
            "The governed context names individuals; the model treats naming as licence to "
            "make a decision about them."
        ),
        "remediation": (
            "Add an explicit system-prompt constraint: report metrics about named "
            "individuals, never recommend employment actions."
        ),
        "detected_by": ["human", "llm_judge"],
    },
    "unsupported_claim": {
        "label": "Unsupported claim",
        "severity": "high",
        "is_critical": False,
        "description": (
            "The answer asserts something the governed context does not support — an "
            "external benchmark, a causal explanation, or a trend the data does not contain."
        ),
        "example": (
            "'Attrition rose because of the return-to-office policy' when the context "
            "contains no policy data and no time series."
        ),
        "likely_cause": (
            "Pretrained business priors fill gaps the context leaves open; the prompt asks "
            "for a recommendation, which invites causal narrative."
        ),
        "remediation": (
            "Instruct the model to mark inference explicitly and to name the evidence for "
            "each claim; penalise unmarked inference in the judge rubric."
        ),
        "detected_by": ["deterministic", "human", "llm_judge"],
    },
    "overconfident_answer": {
        "label": "Overconfident answer",
        "severity": "high",
        "is_critical": False,
        "description": (
            "The context is partial or ambiguous but the answer is delivered with full "
            "confidence and no stated caveat."
        ),
        "example": (
            "Asked for a quarter-over-quarter trend when the context holds a single "
            "snapshot, the answer describes a trend without noting there is no prior period."
        ),
        "likely_cause": (
            "The response template demands an Insight and a Recommended action, which "
            "pressures the model to produce both even when the data cannot carry them."
        ),
        "remediation": (
            "Permit an explicit 'insufficient governed data' outcome in the response format "
            "so the template stops forcing a confident answer."
        ),
        "detected_by": ["human", "llm_judge"],
    },
    "missing_context_failure": {
        "label": "Missing-context failure",
        "severity": "high",
        "is_critical": False,
        "description": (
            "The question asks for something genuinely absent from the governed layer and "
            "the answer supplies a figure anyway instead of saying it is not available."
        ),
        "example": (
            "'What was churn in Q3 2024?' answered with a number, when the semantic layer "
            "holds only current-period figures."
        ),
        "likely_cause": "No negative examples in the prompt showing what a refusal looks like.",
        "remediation": (
            "Add a worked 'not in the governed layer' example to the system prompt; treat "
            "absence of a required fact as a hard fail rather than a warning."
        ),
        "detected_by": ["deterministic", "human", "llm_judge"],
    },
    "wrong_domain": {
        "label": "Wrong domain routed",
        "severity": "high",
        "is_critical": False,
        "description": (
            "The router selected a domain whose governed context cannot answer the question, "
            "so the answer is grounded in the wrong metric set."
        ),
        "example": (
            "'What is our pipeline coverage ratio?' routed to Product because 'coverage' "
            "contains the substring 'overage'."
        ),
        "likely_cause": "Substring keyword matching without token boundaries.",
        "remediation": "Token-aware matching with word boundaries; see router version v2_token_aware.",
        "detected_by": ["deterministic"],
    },
    "irrelevant_answer": {
        "label": "Irrelevant answer",
        "severity": "high",
        "is_critical": False,
        "description": "The answer is grounded and internally coherent but does not address what was asked.",
        "example": "Asked which region has the worst margin, the answer describes over-utilization rates.",
        "likely_cause": "Question intent lost when the governed context dominates the prompt.",
        "remediation": "Restate the question immediately before the answer instruction in the user message.",
        "detected_by": ["deterministic", "human", "llm_judge"],
    },
    "incomplete_answer": {
        "label": "Incomplete answer",
        "severity": "medium",
        "is_critical": False,
        "description": (
            "The answer addresses part of a multi-part question, or omits a fact the question "
            "explicitly requested."
        ),
        "example": "Asked for margin and over-utilization by region, the answer covers only margin.",
        "likely_cause": "Token budget, or the model answering the first clause and stopping.",
        "remediation": "Decompose multi-part questions before generation; check required-fact coverage.",
        "detected_by": ["deterministic", "human", "llm_judge"],
    },
    "instruction_violation": {
        "label": "Instruction violation",
        "severity": "medium",
        "is_critical": False,
        "description": (
            "An explicit formatting or scoping instruction in the question was not followed "
            "— length limits, requested structure, 'numbers only', 'do not recommend'."
        ),
        "example": "'Answer in one sentence with no recommendation' answered in three sections with a recommendation.",
        "likely_cause": (
            "The system prompt hard-codes a three-section format that overrides user "
            "formatting instructions."
        ),
        "remediation": "Make the response template a default that explicit user instructions can override.",
        "detected_by": ["deterministic", "human", "llm_judge"],
    },
    "ambiguous_request_failure": {
        "label": "Ambiguous-request failure",
        "severity": "medium",
        "is_critical": False,
        "description": (
            "The question could reasonably belong to more than one domain or metric, and the "
            "answer silently picks one instead of surfacing the ambiguity."
        ),
        "example": (
            "'Why is revenue declining?' answered purely from Sales when Marketing, Product "
            "and Support all hold relevant signals."
        ),
        "likely_cause": "Router returns a single winner with no ambiguity signal reaching the prompt.",
        "remediation": "Pass routing ambiguity into the prompt so the answer can name the competing readings.",
        "detected_by": ["deterministic", "human", "llm_judge"],
    },
    "unnecessary_refusal": {
        "label": "Unnecessary refusal",
        "severity": "medium",
        "is_critical": False,
        "description": (
            "The answer declines or hedges on a question the governed context fully supports. "
            "The mirror image of missing-context failure, and just as damaging to trust."
        ),
        "example": "'I don't have enough information' when CSAT by channel is present in the context.",
        "likely_cause": "Over-tuned grounding instruction; the model treats any uncertainty as disqualifying.",
        "remediation": (
            "Balance the grounding instruction with an explicit directive to answer fully when "
            "the context does contain the figures."
        ),
        "detected_by": ["deterministic", "human", "llm_judge"],
    },
    "none": {
        "label": "No failure",
        "severity": "none",
        "is_critical": False,
        "description": "The response met the expected behaviour for this evaluation case.",
        "example": "",
        "likely_cause": "",
        "remediation": "",
        "detected_by": ["deterministic", "human", "llm_judge"],
    },
}

SEVERITY_ORDER = ["critical", "high", "medium", "low", "none"]

SEVERITY_COLORS = {
    "critical": "#f87171",
    "high": "#fb923c",
    "medium": "#eab308",
    "low": "#38bdf8",
    "none": "#22c55e",
}

# Free-text labels an LLM judge plausibly emits, mapped onto the canonical taxonomy.
# Kept explicit rather than fuzzy-matched so misclassification is visible, not silent.
_ALIASES = {
    "": "none",
    "no failure": "none",
    "no_failure": "none",
    "n/a": "none",
    "na": "none",
    "pass": "none",
    "hallucination": "hallucinated_number",
    "hallucinated_figure": "hallucinated_number",
    "fabricated_number": "hallucinated_number",
    "fabricated_figure": "hallucinated_number",
    "made_up_number": "hallucinated_number",
    "ungrounded_number": "hallucinated_number",
    "wrong_metric": "incorrect_metric",
    "metric_error": "incorrect_metric",
    "misapplied_metric": "incorrect_metric",
    "unsupported_inference": "unsupported_claim",
    "unsupported": "unsupported_claim",
    "ungrounded_claim": "unsupported_claim",
    "speculation": "unsupported_claim",
    "causal_claim": "unsupported_claim",
    "overconfidence": "overconfident_answer",
    "overconfident": "overconfident_answer",
    "false_certainty": "overconfident_answer",
    "missing_context": "missing_context_failure",
    "insufficient_context": "missing_context_failure",
    "data_not_available": "missing_context_failure",
    "wrong_routing": "wrong_domain",
    "routing_error": "wrong_domain",
    "off_topic": "irrelevant_answer",
    "irrelevant": "irrelevant_answer",
    "incomplete": "incomplete_answer",
    "partial_answer": "incomplete_answer",
    "format_violation": "instruction_violation",
    "did_not_follow_instructions": "instruction_violation",
    "instruction_not_followed": "instruction_violation",
    "ambiguity": "ambiguous_request_failure",
    "ambiguous": "ambiguous_request_failure",
    "unresolved_ambiguity": "ambiguous_request_failure",
    "refusal": "unnecessary_refusal",
    "over_refusal": "unnecessary_refusal",
    "overrefusal": "unnecessary_refusal",
    "policy": "unsafe_or_policy_issue",
    "safety": "unsafe_or_policy_issue",
    "unsafe": "unsafe_or_policy_issue",
    "pii": "unsafe_or_policy_issue",
}


def normalize_failure_mode(raw: Optional[str]) -> str:
    """
    Map an arbitrary evaluator label onto the canonical taxonomy.

    Unrecognised labels return "unclassified" rather than being coerced into a
    neighbouring category — an unclassifiable judge output is itself a finding
    about the judge, and hiding it would corrupt the failure distribution.
    """
    if raw is None:
        return "none"
    key = str(raw).strip().lower().replace(" ", "_").replace("-", "_")
    while "__" in key:
        key = key.replace("__", "_")
    if key in FAILURE_MODES:
        return key
    if key in _ALIASES:
        return _ALIASES[key]
    spaced = key.replace("_", " ")
    if spaced in _ALIASES:
        return _ALIASES[spaced]
    return "unclassified"


def get_failure_mode(mode: Optional[str]) -> dict:
    """Return the taxonomy entry for a mode, normalising first. Never raises."""
    key = normalize_failure_mode(mode)
    if key == "unclassified":
        return {
            "label": "Unclassified",
            "severity": "medium",
            "is_critical": False,
            "description": (
                "An evaluator returned a failure label outside the taxonomy. Counted "
                "separately so taxonomy drift stays visible."
            ),
            "example": "",
            "likely_cause": "Judge prompt permitted free-text categories, or the taxonomy is missing a real mode.",
            "remediation": "Extend the taxonomy, or constrain the judge to an enum.",
            "detected_by": ["llm_judge"],
            "key": "unclassified",
        }
    return {**FAILURE_MODES[key], "key": key}


def severity_of(mode: Optional[str]) -> str:
    return get_failure_mode(mode)["severity"]


def is_critical(mode: Optional[str]) -> bool:
    return bool(get_failure_mode(mode)["is_critical"])


def label_of(mode: Optional[str]) -> str:
    return get_failure_mode(mode)["label"]


def all_modes(include_none: bool = False) -> list[str]:
    """Canonical mode keys ordered by severity, then alphabetically within a severity."""
    keys = [k for k in FAILURE_MODES if include_none or k != "none"]
    return sorted(keys, key=lambda k: (SEVERITY_ORDER.index(FAILURE_MODES[k]["severity"]), k))


def distribution(modes: list[Optional[str]]) -> dict[str, int]:
    """Count normalised failure modes, excluding 'none'. Ordered by severity."""
    counts: dict[str, int] = {}
    for m in modes:
        key = normalize_failure_mode(m)
        if key == "none":
            continue
        counts[key] = counts.get(key, 0) + 1
    return dict(
        sorted(
            counts.items(),
            key=lambda kv: (SEVERITY_ORDER.index(severity_of(kv[0])), -kv[1]),
        )
    )


def severity_distribution(modes: list[Optional[str]]) -> dict[str, int]:
    counts = {s: 0 for s in SEVERITY_ORDER}
    for m in modes:
        counts[severity_of(m)] = counts.get(severity_of(m), 0) + 1
    return {k: v for k, v in counts.items() if v or k != "low"}
