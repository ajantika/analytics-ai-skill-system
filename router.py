"""
router.py — Domain routing for Analytics AI Skill System

Routing method: deterministic keyword scoring.
Labelled clearly so it is never misrepresented as semantic or embedding-based routing.
Architecture note: replace classify_domain() with an embedding-based or LLM-based
  router by swapping this module; the rest of the application is unaffected.
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Confidence threshold below which a question is treated as ambiguous
AMBIGUITY_THRESHOLD = 0.35
# Minimum raw score for any match (questions with score 0 everywhere are ambiguous)
MIN_SCORE_FOR_ROUTING = 1


@dataclass
class RoutingResult:
    domain: Optional[str]
    confidence: float          # 0.0 – 1.0
    method: str                # always "keyword" in this implementation
    top_domains: list          # [(domain, score), ...] sorted descending
    is_ambiguous: bool
    reasoning: str


def classify_domain(question: str, domains: dict, fallback_domain: str = "") -> RoutingResult:
    """
    Keyword-based domain classifier.

    Returns a RoutingResult with confidence score and ambiguity flag.
    Routing method is always labelled as 'keyword' — not 'semantic' or 'LLM-based'.
    """
    if not domains:
        return RoutingResult(
            domain=None, confidence=0.0, method="keyword",
            top_domains=[], is_ambiguous=True,
            reasoning="No domain skill files loaded."
        )

    q = question.lower()
    raw_scores: dict[str, float] = {}

    for domain_name, data in domains.items():
        score = 0.0
        for kw in data.get("keywords", []):
            kw_clean = str(kw).lower().replace("-", " ").replace("_", " ")
            if kw_clean in q:
                score += 2.0
            else:
                for word in kw_clean.split():
                    if len(word) > 3 and word in q:
                        score += 1.0
        raw_scores[domain_name] = score

    sorted_domains = sorted(raw_scores.items(), key=lambda x: x[1], reverse=True)
    top_score = sorted_domains[0][1] if sorted_domains else 0
    second_score = sorted_domains[1][1] if len(sorted_domains) > 1 else 0

    # Normalise confidence: ratio of top score to total score, with separation bonus
    total_score = sum(raw_scores.values())

    if total_score == 0 or top_score < MIN_SCORE_FOR_ROUTING:
        # No keyword match — ambiguous, fall back if a domain is active
        domain = fallback_domain if fallback_domain in domains else (list(domains.keys())[0])
        logger.info(f"Routing: no keyword match for '{question[:60]}' — fallback to '{domain}'")
        return RoutingResult(
            domain=domain, confidence=0.30, method="keyword",
            top_domains=sorted_domains[:3], is_ambiguous=True,
            reasoning="No strong keyword match found. Showing closest domain."
        )

    # Separation score: how much better is top vs second?
    separation = (top_score - second_score) / max(top_score, 1)
    confidence = min(0.98, (top_score / max(total_score, 1)) * 0.6 + separation * 0.4)

    is_ambiguous = confidence < AMBIGUITY_THRESHOLD or (
        second_score > 0 and (top_score / max(second_score, 0.01)) < 1.4
    )

    best_domain = sorted_domains[0][0]

    logger.info(
        f"Routing: '{question[:60]}' → '{best_domain}' "
        f"(confidence={confidence:.2f}, ambiguous={is_ambiguous}, method=keyword)"
    )

    return RoutingResult(
        domain=best_domain,
        confidence=confidence,
        method="keyword",
        top_domains=sorted_domains[:3],
        is_ambiguous=is_ambiguous,
        reasoning=_build_reasoning(best_domain, confidence, sorted_domains)
    )


def _build_reasoning(domain: str, confidence: float, sorted_domains: list) -> str:
    label = domain.replace("_", " ").title()
    pct = int(confidence * 100)
    if len(sorted_domains) > 1 and sorted_domains[1][1] > 0:
        runner = sorted_domains[1][0].replace("_", " ").title()
        return f"Primary signal → {label} ({pct}% confidence). Runner-up: {runner}."
    return f"Strong signal → {label} ({pct}% confidence)."


def get_ambiguous_domains(routing: RoutingResult, domains: dict) -> list[str]:
    """Return domain names that have meaningful scores (for multi-domain disambiguation)."""
    if not routing.top_domains:
        return list(domains.keys())[:3]
    threshold = routing.top_domains[0][1] * 0.5
    return [d for d, s in routing.top_domains if s >= threshold]
