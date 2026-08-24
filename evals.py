"""
evals.py — Deterministic and heuristic evaluation layer.

The bottom tier of a three-tier evaluator hierarchy:

  DETERMINISTIC  Objective, cheap, perfectly repeatable. Numeric grounding,
                 metric recognition, required-fact coverage, forbidden-claim
                 detection, expected-domain routing. These are not replaced by an
                 LLM — an LLM is strictly worse at "does this exact string appear".
  HEURISTIC      Pattern matching with known false positives and false negatives.
                 Labelled as such wherever it is shown.
  (see human_evals.py and llm_judge.py for the subjective tiers)

No evaluation result in this module is faked. Every check reports its method and
its limitations alongside its verdict.
"""

import json
import logging
import pathlib
import re
from dataclasses import dataclass, field
from typing import Optional, Union

logger = logging.getLogger(__name__)

DATA_DIR = pathlib.Path(__file__).parent / "data"
GOLDEN_SET_PATH = DATA_DIR / "golden_eval_set.json"

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


# ══════════════════════════════════════════════════════════════════════════════
# GOLDEN EVALUATION SET
# ══════════════════════════════════════════════════════════════════════════════


def load_golden_set(path: Optional[str] = None) -> dict:
    """
    Load the golden evaluation dataset.

    Returns {"cases": [], ...} when the file is missing rather than raising, so the
    application renders an honest empty state instead of a stack trace.
    """
    p = pathlib.Path(path) if path else GOLDEN_SET_PATH
    if not p.exists():
        logger.warning(f"Golden set not found at {p}")
        return {"cases": [], "dataset_version": None}
    try:
        with open(p, "r") as f:
            data = json.load(f)
        cases = data.get("cases", [])
        logger.info(f"Loaded golden set v{data.get('dataset_version')} — {len(cases)} cases")
        return data
    except Exception as e:
        logger.error(f"Failed to load golden set from {p}: {e}")
        return {"cases": [], "dataset_version": None, "load_error": str(e)}


def golden_cases(path: Optional[str] = None) -> list[dict]:
    return load_golden_set(path).get("cases", [])


def case_by_id(eval_id: str, cases: Optional[list[dict]] = None) -> Optional[dict]:
    for c in cases if cases is not None else golden_cases():
        if c.get("eval_id") == eval_id:
            return c
    return None


def validate_golden_set(data: dict) -> list[str]:
    """Structural validation. Returns a list of problems; empty means the set is well formed."""
    problems = []
    cases = data.get("cases", [])
    if not cases:
        return ["dataset contains no cases"]

    required_fields = [
        "eval_id", "question", "domain", "expected_domain", "test_type", "difficulty",
        "governed_context", "expected_answer_summary", "required_facts",
        "forbidden_claims", "expected_behavior", "expected_failure_mode", "severity",
    ]
    seen = set()
    for c in cases:
        cid = c.get("eval_id", "<missing id>")
        for f in required_fields:
            if f not in c:
                problems.append(f"{cid}: missing field '{f}'")
        if cid in seen:
            problems.append(f"{cid}: duplicate eval_id")
        seen.add(cid)
        if not isinstance(c.get("required_facts", []), list):
            problems.append(f"{cid}: required_facts must be a list")
        if not isinstance(c.get("forbidden_claims", []), list):
            problems.append(f"{cid}: forbidden_claims must be a list")
    return problems


# ══════════════════════════════════════════════════════════════════════════════
# DETERMINISTIC CHECKS
# ══════════════════════════════════════════════════════════════════════════════


def _normalize(text: str) -> str:
    """
    Normalise for literal matching: lowercase, collapse whitespace, drop thousands
    separators, and unify unicode punctuation the model may emit.

    Commas are stripped so that '35,109' in the dataset matches '35109' in an answer
    and vice versa. This is the only lossy step and it is deliberate.
    """
    t = str(text).lower()
    t = t.replace(" ", " ").replace(" ", " ").replace(" ", " ")
    t = t.replace("–", "-").replace("—", "-").replace("−", "-")
    t = t.replace("‘", "'").replace("’", "'")
    t = t.replace("“", '"').replace("”", '"')
    t = t.replace("×", "x")
    t = t.replace(",", "")
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def _as_alternatives(fact: Union[str, list]) -> list[str]:
    """A required fact is either a string or a list of acceptable surface forms."""
    if isinstance(fact, (list, tuple)):
        return [str(f) for f in fact]
    return [str(fact)]


def _contains(haystack_norm: str, needle: str) -> bool:
    return _normalize(needle) in haystack_norm


def check_required_facts(answer: str, required_facts: list) -> dict:
    """
    DETERMINISTIC — does the answer contain every fact the case requires?

    Each entry may be a string or a list of acceptable surface forms; any one form
    satisfies the requirement. A case with no required facts returns status "N/A"
    rather than a vacuous PASS, so empty requirements never inflate a pass rate.
    """
    if not required_facts:
        return {
            "status": "N/A",
            "found": [],
            "missing": [],
            "coverage": None,
            "detail": "This case specifies no required facts — the expected behaviour is qualitative.",
            "method": "DETERMINISTIC — literal string matching with normalised surface forms",
        }

    hay = _normalize(answer)
    found, missing = [], []
    for fact in required_facts:
        alts = _as_alternatives(fact)
        hit = next((a for a in alts if _contains(hay, a)), None)
        if hit:
            found.append({"required": alts[0], "matched": hit})
        else:
            missing.append(alts[0])

    coverage = len(found) / len(required_facts)
    if coverage == 1.0:
        status, detail = "PASS", f"All {len(required_facts)} required facts present."
    elif coverage >= 0.5:
        status, detail = "WARN", f"{len(found)}/{len(required_facts)} required facts present. Missing: {', '.join(missing)}."
    else:
        status, detail = "FAIL", f"Only {len(found)}/{len(required_facts)} required facts present. Missing: {', '.join(missing)}."

    return {
        "status": status,
        "found": found,
        "missing": missing,
        "coverage": coverage,
        "detail": detail,
        "method": "DETERMINISTIC — literal string matching with normalised surface forms",
    }


def check_forbidden_claims(answer: str, forbidden_claims: list) -> dict:
    """
    DETERMINISTIC — does the answer contain a phrase the case forbids?

    Precision-oriented and recall-limited by design: it catches the obvious surface
    form of a forbidden claim. A model that expresses the same forbidden idea in
    different words will not be caught here — that is what the human rubric and the
    LLM judge are for. The UI states this limitation wherever the check is shown.
    """
    if not forbidden_claims:
        return {
            "status": "N/A",
            "violations": [],
            "detail": "This case specifies no forbidden claims.",
            "method": "DETERMINISTIC — literal phrase matching (precision-oriented, recall-limited)",
        }

    hay = _normalize(answer)
    violations = [c for c in forbidden_claims if _contains(hay, c)]
    return {
        "status": "FAIL" if violations else "PASS",
        "violations": violations,
        "detail": (
            f"Forbidden phrase present: {'; '.join(repr(v) for v in violations)}"
            if violations
            else f"None of the {len(forbidden_claims)} forbidden phrases detected. "
                 "Semantic violations are not caught by literal matching."
        ),
        "method": "DETERMINISTIC — literal phrase matching (precision-oriented, recall-limited)",
    }


def check_expected_domain(routed_domain: Optional[str], expected_domain: Optional[str]) -> dict:
    """DETERMINISTIC — did the router select the domain the case expects?"""
    if expected_domain is None:
        return {
            "status": "N/A",
            "routed": routed_domain,
            "expected": None,
            "detail": "Case has no single correct domain — it tests ambiguity handling, not routing accuracy.",
            "method": "DETERMINISTIC — exact domain key comparison",
        }
    ok = routed_domain == expected_domain
    return {
        "status": "PASS" if ok else "FAIL",
        "routed": routed_domain,
        "expected": expected_domain,
        "detail": (
            f"Routed to {routed_domain} as expected."
            if ok
            else f"Routed to {routed_domain}; case expects {expected_domain}."
        ),
        "method": "DETERMINISTIC — exact domain key comparison",
    }


def run_deterministic_suite(
    case: dict,
    answer: str,
    context: str,
    domain_data: dict,
    routed_domain: Optional[str] = None,
    routing_confidence: float = 0.5,
) -> dict:
    """
    Every deterministic and heuristic check for one golden case, in one record.

    The returned `verdict` is a hard gate, not a quality score: it fails only on
    objectively checkable violations. Subjective quality is deliberately out of
    scope here — that is what the human rubric and the LLM judge measure. Keeping
    the tiers separate is what makes human-vs-judge agreement meaningful.
    """
    base = evaluate_response(
        question=case.get("question", ""),
        answer=answer,
        context=context,
        domain_data=domain_data or {},
        routing_confidence=routing_confidence,
    )

    facts = check_required_facts(answer, case.get("required_facts", []))
    forbidden = check_forbidden_claims(answer, case.get("forbidden_claims", []))
    domain = check_expected_domain(routed_domain, case.get("expected_domain"))

    hard_failures = []
    if facts["status"] == "FAIL":
        hard_failures.append("required_facts")
    if forbidden["status"] == "FAIL":
        hard_failures.append("forbidden_claims")
    if domain["status"] == "FAIL":
        hard_failures.append("expected_domain")
    if base.groundedness == "FAIL":
        hard_failures.append("numeric_grounding")

    return {
        "eval_id": case.get("eval_id"),
        "verdict": "FAIL" if hard_failures else "PASS",
        "hard_failures": hard_failures,
        "checks": {
            "numeric_grounding": {
                "status": base.groundedness,
                "detail": base.details["groundedness"],
                "method": base.methods["groundedness"],
            },
            "metric_recognition": {
                "status": base.metric_validity,
                "detail": base.details["metric_validity"],
                "method": base.methods["metric_validity"],
            },
            "required_facts": facts,
            "forbidden_claims": forbidden,
            "expected_domain": domain,
            "relevance_heuristic": {
                "status": base.relevance,
                "detail": base.details["relevance"],
                "method": base.methods["relevance"],
            },
            "unsupported_claim_heuristic": {
                "status": base.unsupported_claims,
                "detail": base.details["unsupported_claims"],
                "method": base.methods["unsupported_claims"],
            },
        },
        "legacy_quality_score": base.overall_quality,
        "legacy_quality_label": base.quality_label,
    }


def deterministic_summary(results: list[dict]) -> dict:
    """Aggregate deterministic verdicts across a run. All rates computed, none asserted."""
    if not results:
        return {"n": 0}

    n = len(results)

    def rate_of(check: str, status: str) -> Optional[float]:
        applicable = [r for r in results if r["checks"][check]["status"] != "N/A"]
        if not applicable:
            return None
        return sum(1 for r in applicable if r["checks"][check]["status"] == status) / len(applicable)

    def applicable_n(check: str) -> int:
        return sum(1 for r in results if r["checks"][check]["status"] != "N/A")

    return {
        "n": n,
        "verdict_pass_rate": sum(1 for r in results if r["verdict"] == "PASS") / n,
        "numeric_grounding_pass_rate": rate_of("numeric_grounding", "PASS"),
        "metric_recognition_pass_rate": rate_of("metric_recognition", "PASS"),
        "required_facts_pass_rate": rate_of("required_facts", "PASS"),
        "required_facts_n": applicable_n("required_facts"),
        "forbidden_claims_violation_rate": rate_of("forbidden_claims", "FAIL"),
        "forbidden_claims_n": applicable_n("forbidden_claims"),
        "routing_accuracy": rate_of("expected_domain", "PASS"),
        "routing_n": applicable_n("expected_domain"),
        "relevance_pass_rate": rate_of("relevance_heuristic", "PASS"),
        "unsupported_claim_rate": rate_of("unsupported_claim_heuristic", "FOUND"),
        "mean_required_fact_coverage": (
            sum(r["checks"]["required_facts"]["coverage"] for r in results
                if r["checks"]["required_facts"]["coverage"] is not None)
            / max(applicable_n("required_facts"), 1)
        ) if applicable_n("required_facts") else None,
    }


def run_golden_routing_eval(router_fn, domains: dict, cases: Optional[list[dict]] = None) -> dict:
    """
    Routing accuracy over the golden set.

    Separate from run_routing_eval() (which uses the original 13-item EVAL_DATASET
    and is retained so the legacy routing view keeps working). Cases with
    expected_domain of null test ambiguity handling and are excluded from accuracy,
    but are reported so the exclusion is visible rather than silent.
    """
    cases = cases if cases is not None else golden_cases()
    results, excluded = [], []

    for case in cases:
        routing = router_fn(case["question"], domains)
        expected = case.get("expected_domain")
        record = {
            "eval_id": case["eval_id"],
            "question": case["question"],
            "expected": expected,
            "predicted": routing.domain,
            "confidence": routing.confidence,
            "is_ambiguous": routing.is_ambiguous,
            "test_type": case.get("test_type"),
            "difficulty": case.get("difficulty"),
        }
        if expected is None:
            record["correct"] = None
            excluded.append(record)
        else:
            record["correct"] = routing.domain == expected
            results.append(record)

    correct = sum(1 for r in results if r["correct"])
    return {
        "accuracy": correct / len(results) if results else None,
        "correct": correct,
        "total": len(results),
        "results": results,
        "failures": [r for r in results if not r["correct"]],
        "excluded_ambiguous": excluded,
        "n_excluded": len(excluded),
    }
