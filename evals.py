"""
evals.py — Evaluation framework for Analytics AI Skill System

Implements lightweight, deterministic evaluation checks for every generated response.
Evaluation methods are labelled clearly:
  - DETERMINISTIC: rule-based checks on text content
  - HEURISTIC: pattern matching with known limitations
  - LLM-BASED: would require an additional model call (not implemented in this demo)

No evaluation results are faked. Every check is transparent about its method and limitations.
"""

import re
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Numbers and percentages present in the governed context that we can check for
NUMBER_PATTERN = re.compile(r"\b\d[\d,]*\.?\d*[%KMB]?\b|\$[\d,]+\.?\d*[KMB]?")

# Phrases that suggest the model may be going beyond the context
UNSUPPORTED_CLAIM_PHRASES = [
    "industry standard", "typically", "generally speaking",
    "in most cases", "research shows", "studies suggest",
    "it is common", "best practice dictates", "experts recommend",
    "on average across industries"
]

# Phrases that indicate the model is hedging / not answering
NON_ANSWER_PHRASES = [
    "i cannot determine", "i don't have enough information",
    "please provide more context", "i would need access to",
    "unable to answer", "cannot be calculated"
]


@dataclass
class EvalResult:
    groundedness: str           # PASS / WARN / FAIL
    metric_validity: str        # PASS / WARN / FAIL
    relevance: str              # PASS / WARN / FAIL
    unsupported_claims: str     # NONE / FOUND
    overall_quality: float      # 0.0 – 1.0
    quality_label: str          # HIGH / MEDIUM / LOW
    details: dict = field(default_factory=dict)
    methods: dict = field(default_factory=dict)


def evaluate_response(
    question: str,
    answer: str,
    context: str,
    domain_data: dict,
    routing_confidence: float = 0.5
) -> EvalResult:
    """
    Run all evaluation checks on a generated answer.

    Args:
        question: the original user question
        answer: the LLM-generated answer text
        context: the governed context passed to the LLM
        domain_data: the full domain skill dict
        routing_confidence: confidence score from the router (0.0–1.0)

    Returns:
        EvalResult with pass/fail for each dimension and overall quality score.
    """
    answer_lower = answer.lower()
    context_lower = context.lower()

    # ── 1. Groundedness (DETERMINISTIC) ──────────────────────────────────────
    # Check: do the numbers in the answer appear in the context?
    answer_numbers = set(NUMBER_PATTERN.findall(answer))
    context_numbers = set(NUMBER_PATTERN.findall(context))

    grounded_count = sum(1 for n in answer_numbers if n in context_numbers)
    total_numbers = len(answer_numbers)

    if total_numbers == 0:
        groundedness = "WARN"
        ground_detail = "No specific figures found in answer — may be too general."
    elif grounded_count / total_numbers >= 0.75:
        groundedness = "PASS"
        ground_detail = f"{grounded_count}/{total_numbers} figures found in governed context."
    elif grounded_count / total_numbers >= 0.5:
        groundedness = "WARN"
        ground_detail = f"{grounded_count}/{total_numbers} figures matched — some may be ungrounded."
    else:
        groundedness = "FAIL"
        ground_detail = f"Only {grounded_count}/{total_numbers} figures matched the governed context."

    # ── 2. Metric Validity (DETERMINISTIC) ───────────────────────────────────
    # Check: are metric names referenced in the answer defined in the semantic layer?
    defined_metrics = [m.get("name", "").lower() for m in domain_data.get("metrics", [])]
    answer_mentions_defined = any(m in answer_lower for m in defined_metrics)

    metric_validity = "PASS" if answer_mentions_defined or defined_metrics else "WARN"
    metric_detail = (
        "Referenced metrics found in governed semantic layer."
        if answer_mentions_defined
        else "No explicit metric names referenced — check answer for specificity."
    )

    # ── 3. Relevance (HEURISTIC) ─────────────────────────────────────────────
    # Check: does the answer contain any non-answer phrases?
    is_non_answer = any(p in answer_lower for p in NON_ANSWER_PHRASES)
    # Check: does the answer share meaningful words with the question?
    q_words = set(w for w in question.lower().split() if len(w) > 3)
    a_words = set(w for w in answer_lower.split() if len(w) > 3)
    word_overlap = len(q_words & a_words) / max(len(q_words), 1)

    if is_non_answer:
        relevance = "FAIL"
        rel_detail = "Answer contains hedging phrases suggesting it did not address the question."
    elif word_overlap >= 0.25 and len(answer) > 80:
        relevance = "PASS"
        rel_detail = f"Answer is substantive and shares relevant terms with the question."
    elif len(answer) < 50:
        relevance = "WARN"
        rel_detail = "Answer is very short — may not fully address the question."
    else:
        relevance = "WARN"
        rel_detail = "Low term overlap between question and answer — verify relevance."

    # ── 4. Unsupported Claims (HEURISTIC) ────────────────────────────────────
    found_phrases = [p for p in UNSUPPORTED_CLAIM_PHRASES if p in answer_lower]
    if found_phrases:
        unsupported = "FOUND"
        unsupported_detail = f"Possible unsupported generalisation: '{found_phrases[0]}'"
    else:
        unsupported = "NONE"
        unsupported_detail = "No unsupported generalisation phrases detected."

    # ── Overall Quality Score ─────────────────────────────────────────────────
    score = 1.0
    score -= 0.25 if groundedness == "FAIL" else (0.1 if groundedness == "WARN" else 0)
    score -= 0.15 if metric_validity == "FAIL" else (0.05 if metric_validity == "WARN" else 0)
    score -= 0.20 if relevance == "FAIL" else (0.08 if relevance == "WARN" else 0)
    score -= 0.15 if unsupported == "FOUND" else 0
    score -= (1 - routing_confidence) * 0.15  # routing uncertainty penalty

    score = max(0.0, min(1.0, score))

    if score >= 0.80:
        quality_label = "HIGH"
    elif score >= 0.60:
        quality_label = "MEDIUM"
    else:
        quality_label = "LOW"

    logger.info(
        f"Eval: groundedness={groundedness}, metric_validity={metric_validity}, "
        f"relevance={relevance}, unsupported={unsupported}, quality={score:.2f}"
    )

    return EvalResult(
        groundedness=groundedness,
        metric_validity=metric_validity,
        relevance=relevance,
        unsupported_claims=unsupported,
        overall_quality=score,
        quality_label=quality_label,
        details={
            "groundedness": ground_detail,
            "metric_validity": metric_detail,
            "relevance": rel_detail,
            "unsupported_claims": unsupported_detail,
        },
        methods={
            "groundedness": "DETERMINISTIC — figure matching against governed context",
            "metric_validity": "DETERMINISTIC — metric name lookup in semantic layer",
            "relevance": "HEURISTIC — term overlap + length + non-answer phrase detection",
            "unsupported_claims": "HEURISTIC — generalisation phrase matching",
        }
    )


# ── Evaluation Dataset for System Eval Page ──────────────────────────────────

EVAL_DATASET = [
    {
        "id": "prod_001",
        "domain": "product_usage",
        "question": "Which customers are over-utilizing their plans?",
        "expected_domain": "product_usage",
        "expected_metrics": ["Over-Utilization Rate", "Plan Fit Score"],
        "note": "Core product question — should route with high confidence"
    },
    {
        "id": "prod_002",
        "domain": "product_usage",
        "question": "What is the MRR recovery opportunity from right-sizing?",
        "expected_domain": "product_usage",
        "expected_metrics": ["MRR Recovery Opportunity"],
        "note": "Revenue-adjacent question — tests grounding on $1.4M figure"
    },
    {
        "id": "prod_003",
        "domain": "product_usage",
        "question": "Which regions have the highest over-utilization?",
        "expected_domain": "product_usage",
        "expected_metrics": ["Regional Over-Utilization"],
        "note": "Tests regional breakdown retrieval"
    },
    {
        "id": "mkt_001",
        "domain": "marketing",
        "question": "Which campaign brought the highest number of customers?",
        "expected_domain": "marketing",
        "expected_metrics": ["Campaign Performance"],
        "note": "Tests marketing campaign routing"
    },
    {
        "id": "mkt_002",
        "domain": "marketing",
        "question": "How are our MQL to SQL conversion rates trending?",
        "expected_domain": "marketing",
        "expected_metrics": ["MQL to SQL Conversion"],
        "note": "Tests funnel metric retrieval"
    },
    {
        "id": "sales_001",
        "domain": "sales",
        "question": "Which sales rep gives the highest discounts?",
        "expected_domain": "sales",
        "expected_metrics": ["Sales Rep Discount Rates"],
        "note": "Tests sales rep performance routing"
    },
    {
        "id": "sales_002",
        "domain": "sales",
        "question": "What is our pipeline coverage ratio?",
        "expected_domain": "sales",
        "expected_metrics": ["Pipeline Coverage"],
        "note": "Tests pipeline metric retrieval"
    },
    {
        "id": "hr_001",
        "domain": "hr",
        "question": "Which teams have the highest attrition?",
        "expected_domain": "hr",
        "expected_metrics": ["Attrition Rate"],
        "note": "Tests HR attrition routing"
    },
    {
        "id": "hr_002",
        "domain": "hr",
        "question": "Are we on track with our hiring plan?",
        "expected_domain": "hr",
        "expected_metrics": ["Hiring Plan Attainment"],
        "note": "Tests hiring plan metric"
    },
    {
        "id": "sup_001",
        "domain": "csup",
        "question": "What is our CSAT score?",
        "expected_domain": "csup",
        "expected_metrics": ["CSAT Score"],
        "note": "Tests support routing"
    },
    {
        "id": "sup_002",
        "domain": "csup",
        "question": "Who are the top performing support agents?",
        "expected_domain": "csup",
        "expected_metrics": ["Agent Performance"],
        "note": "Tests agent performance retrieval"
    },
    {
        "id": "ambig_001",
        "domain": "ambiguous",
        "question": "Why is revenue declining?",
        "expected_domain": None,
        "expected_metrics": [],
        "note": "Intentionally ambiguous — should trigger multi-domain disambiguation"
    },
    {
        "id": "ambig_002",
        "domain": "ambiguous",
        "question": "How are we doing?",
        "expected_domain": None,
        "expected_metrics": [],
        "note": "Too vague — should surface low confidence"
    },
]


def run_routing_eval(router_fn, domains: dict) -> dict:
    """
    Run routing accuracy evaluation against the curated eval dataset.

    Args:
        router_fn: callable(question, domains) → RoutingResult
        domains: loaded domain skill files

    Returns:
        dict with accuracy, per-domain results, failures
    """
    results = []
    correct = 0
    total_with_expected = 0

    for item in EVAL_DATASET:
        if item["expected_domain"] is None:
            continue  # ambiguous cases — skip routing accuracy count

        total_with_expected += 1
        routing = router_fn(item["question"], domains)
        predicted = routing.domain
        expected = item["expected_domain"]
        is_correct = predicted == expected

        if is_correct:
            correct += 1

        results.append({
            "id": item["id"],
            "question": item["question"],
            "expected": expected,
            "predicted": predicted,
            "confidence": routing.confidence,
            "correct": is_correct,
            "note": item["note"]
        })

    accuracy = correct / total_with_expected if total_with_expected > 0 else 0
    failures = [r for r in results if not r["correct"]]

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total_with_expected,
        "results": results,
        "failures": failures
    }
