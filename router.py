"""
router.py — Domain routing for Analytics AI Skill System

Routing method: deterministic keyword scoring. Labelled clearly so it is never
misrepresented as semantic or embedding-based routing.

THREE VERSIONS ARE KEPT DELIBERATELY.

  v1_substring     The original implementation. Scores a keyword when it appears
                   anywhere in the question as a substring. This has a real defect:
                   "pipeline coverage" scores for the Product domain because
                   "coverage" contains "overage".

  v2_token_aware   Matches on word boundaries with light singular/plural handling
                   and rewards contiguous multi-word phrase matches. Fixes the
                   substring class of error, but leaves keyword collisions between
                   domains resolved by arbitrary tie-breaking.

  v3_idf_weighted  Weights each keyword token by the inverse of how many domains
                   claim it, so a token shared across skills carries less signal than
                   one unique to a domain. Flags no-match and tied outcomes instead of
                   resolving them silently. Current default.

Earlier versions are retained, unchanged and still callable, so the regression page
can recompute routing accuracy for all three against the same golden set. Deleting a
superseded version would have made the improvement unmeasurable — a claimed fix with
no before is not evidence of anything.

Architecture note: replace these with an embedding-based or LLM-based router by
adding an entry to ROUTER_VERSIONS; the rest of the application is unaffected.
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Confidence threshold below which a question is treated as ambiguous
AMBIGUITY_THRESHOLD = 0.35
# Minimum raw score for any match (questions with score 0 everywhere are ambiguous)
MIN_SCORE_FOR_ROUTING = 1

DEFAULT_ROUTER_VERSION = "v3_idf_weighted"


@dataclass
class RoutingResult:
    domain: Optional[str]
    confidence: float          # 0.0 – 1.0
    method: str                # scoring method label
    top_domains: list          # [(domain, score), ...] sorted descending
    is_ambiguous: bool
    reasoning: str
    version: str = DEFAULT_ROUTER_VERSION
    no_match: bool = False     # nothing in the question matched any domain's keywords
    is_tie: bool = False       # top two domains scored within TIE_MARGIN of each other

    @property
    def is_confident(self) -> bool:
        return not (self.is_ambiguous or self.no_match or self.is_tie)


def classify_domain(
    question: str,
    domains: dict,
    fallback_domain: str = "",
    version: Optional[str] = None,
) -> RoutingResult:
    """
    Route a question to a domain.

    version selects the scoring implementation and defaults to DEFAULT_ROUTER_VERSION.
    The signature is backward compatible: existing three-argument callers get the
    current default without change.
    """
    impl = ROUTER_VERSIONS.get(version or DEFAULT_ROUTER_VERSION)
    if impl is None:
        raise ValueError(
            f"unknown router version {version!r}; available: {sorted(ROUTER_VERSIONS)}"
        )
    return impl(question, domains, fallback_domain)


def classify_domain_v1(question: str, domains: dict, fallback_domain: str = "") -> RoutingResult:
    """
    v1_substring — the original keyword classifier, preserved verbatim.

    Known defect: substring matching produces false positives when a keyword is
    contained inside an unrelated word. Retained as the regression baseline.
    """
    if not domains:
        return RoutingResult(
            domain=None, confidence=0.0, method="keyword",
            top_domains=[], is_ambiguous=True,
            reasoning="No domain skill files loaded.", version="v1_substring"
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

    return _assemble(raw_scores, question, domains, fallback_domain,
                     method="keyword", version="v1_substring")


# ── v2: token-aware ───────────────────────────────────────────────────────────

_TOKEN_SPLIT = re.compile(r"[^a-z0-9%$&+]+")


def _tokens(text: str) -> list[str]:
    """Lowercase word tokens. Hyphens and underscores split, so 'over-utilized' is two tokens."""
    return [t for t in _TOKEN_SPLIT.split(str(text).lower()) if t]


def _stem(word: str) -> str:
    """
    Minimal plural handling only. Not a real stemmer, and deliberately not one —
    aggressive stemming would reintroduce the false-positive class this version fixes.
    """
    if len(word) > 3 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _phrase_present(needle: list[str], haystack: list[str]) -> bool:
    """Contiguous subsequence match on stemmed tokens."""
    if not needle or len(needle) > len(haystack):
        return False
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i:i + len(needle)] == needle:
            return True
    return False


def classify_domain_v2(question: str, domains: dict, fallback_domain: str = "") -> RoutingResult:
    """
    v2_token_aware — word-boundary keyword matching.

    Scoring:
      3.0  contiguous multi-word phrase match ("pipeline coverage", "hiring plan")
      2.0  single-word keyword matching a question token exactly (after plural stemming)
      0.75 per constituent token of a multi-word keyword that appears independently

    The change from v1 is that a keyword must align to a token boundary. "coverage"
    no longer scores the Product keyword "overage"; "planning" no longer scores "plan".
    Multi-word phrases are weighted above single words because they carry far more
    domain signal.
    """
    if not domains:
        return RoutingResult(
            domain=None, confidence=0.0, method="keyword_token_aware",
            top_domains=[], is_ambiguous=True,
            reasoning="No domain skill files loaded.", version="v2_token_aware"
        )

    q_tokens = [_stem(t) for t in _tokens(question)]
    q_set = set(q_tokens)
    raw_scores: dict[str, float] = {}

    for domain_name, data in domains.items():
        score = 0.0
        for kw in data.get("keywords", []):
            kw_tokens = [_stem(t) for t in _tokens(kw)]
            if not kw_tokens:
                continue
            if len(kw_tokens) == 1:
                if kw_tokens[0] in q_set:
                    score += 2.0
            elif _phrase_present(kw_tokens, q_tokens):
                score += 3.0
            else:
                hits = sum(1 for t in kw_tokens if len(t) > 3 and t in q_set)
                score += hits * 0.75
        raw_scores[domain_name] = score

    return _assemble(raw_scores, question, domains, fallback_domain,
                     method="keyword_token_aware", version="v2_token_aware")


# ── v3: token-aware + inverse domain frequency ────────────────────────────────


def domain_frequency(domains: dict) -> dict[str, int]:
    """
    How many domains claim each keyword token.

    'csat' appears in one skill file; 'customers' appears in two; 'pipeline' in two.
    A token claimed by every domain carries no routing signal at all, and treating
    it as equal to a unique token is what produced the arbitrary tie-breaking that
    dominated v2's errors.
    """
    df: dict[str, int] = {}
    for data in domains.values():
        claimed = {_stem(t) for kw in data.get("keywords", []) for t in _tokens(kw)}
        for token in claimed:
            df[token] = df.get(token, 0) + 1
    return df


def classify_domain_v3(question: str, domains: dict, fallback_domain: str = "") -> RoutingResult:
    """
    v3_idf_weighted — v2 plus inverse-domain-frequency weighting.

    Each keyword token is weighted by 1/(number of domains that claim it), scaled by
    the domain count so absolute scores stay comparable to earlier versions. A
    multi-word keyword uses the mean weight of its tokens.

    Two further changes, neither of which raises accuracy but both of which convert
    silently-wrong routes into honestly-uncertain ones:
      * A question matching no keyword at all sets no_match, instead of quietly
        returning whichever domain happened to load first.
      * A top-two score gap inside TIE_MARGIN sets is_tie, because picking between
        two equal scores is a coin flip, not a routing decision.
    """
    if not domains:
        return RoutingResult(
            domain=None, confidence=0.0, method="keyword_idf_weighted",
            top_domains=[], is_ambiguous=True, no_match=True,
            reasoning="No domain skill files loaded.", version="v3_idf_weighted"
        )

    df = domain_frequency(domains)
    n_domains = len(domains)
    q_tokens = [_stem(t) for t in _tokens(question)]
    q_set = set(q_tokens)
    raw_scores: dict[str, float] = {}

    for domain_name, data in domains.items():
        score = 0.0
        partial_counted: set[str] = set()
        for kw in data.get("keywords", []):
            kw_tokens = [_stem(t) for t in _tokens(kw)]
            if not kw_tokens:
                continue
            weight = sum(1.0 / df.get(t, 1) for t in kw_tokens) / len(kw_tokens)

            if len(kw_tokens) == 1:
                if kw_tokens[0] in q_set:
                    score += 2.0 * weight * n_domains
            elif _phrase_present(kw_tokens, q_tokens):
                score += 3.0 * weight * n_domains
            else:
                # Partial credit at most once per question token per domain, so a
                # common token appearing across several multi-word keywords cannot
                # accumulate spurious score.
                for t in kw_tokens:
                    if len(t) > 3 and t in q_set and t not in partial_counted:
                        partial_counted.add(t)
                        score += 0.75 * (1.0 / df.get(t, 1)) * n_domains
        raw_scores[domain_name] = score

    return _assemble(raw_scores, question, domains, fallback_domain,
                     method="keyword_idf_weighted", version="v3_idf_weighted")


# ── Shared scoring assembly ───────────────────────────────────────────────────

# Top-two scores within this relative margin are treated as a tie rather than a win.
TIE_MARGIN = 0.02


def _assemble(
    raw_scores: dict,
    question: str,
    domains: dict,
    fallback_domain: str,
    method: str,
    version: str,
) -> RoutingResult:
    """
    Turn raw keyword scores into a RoutingResult.

    Shared by both versions so a routing-accuracy difference between them reflects
    the scoring change alone, not a difference in how confidence is derived.
    """
    sorted_domains = sorted(raw_scores.items(), key=lambda x: x[1], reverse=True)
    top_score = sorted_domains[0][1] if sorted_domains else 0
    second_score = sorted_domains[1][1] if len(sorted_domains) > 1 else 0
    total_score = sum(raw_scores.values())

    if total_score == 0 or top_score < MIN_SCORE_FOR_ROUTING:
        # domain is still populated with a best guess so existing callers keep working,
        # but no_match records that nothing actually matched. Consumers that care about
        # routing honesty check no_match, not domain.
        domain = fallback_domain if fallback_domain in domains else (list(domains.keys())[0])
        logger.debug(f"Routing[{version}]: no keyword match for '{question[:60]}' — fallback to '{domain}'")
        return RoutingResult(
            domain=domain, confidence=0.30, method=method,
            top_domains=sorted_domains[:3], is_ambiguous=True, no_match=True,
            reasoning="No keyword in this question matched any domain skill file.",
            version=version,
        )

    separation = (top_score - second_score) / max(top_score, 1)
    confidence = min(0.98, (top_score / max(total_score, 1)) * 0.6 + separation * 0.4)

    is_tie = second_score > 0 and separation <= TIE_MARGIN
    is_ambiguous = is_tie or confidence < AMBIGUITY_THRESHOLD or (
        second_score > 0 and (top_score / max(second_score, 0.01)) < 1.4
    )

    best_domain = sorted_domains[0][0]
    logger.debug(
        f"Routing[{version}]: '{question[:60]}' → '{best_domain}' "
        f"(confidence={confidence:.2f}, ambiguous={is_ambiguous}, tie={is_tie})"
    )

    reasoning = _build_reasoning(best_domain, confidence, sorted_domains)
    if is_tie:
        runner = sorted_domains[1][0].replace("_", " ").title()
        reasoning = (
            f"Tied signal between {best_domain.replace('_', ' ').title()} and {runner} — "
            "the keyword scores are effectively equal, so this selection is arbitrary."
        )

    return RoutingResult(
        domain=best_domain,
        confidence=confidence,
        method=method,
        top_domains=sorted_domains[:3],
        is_ambiguous=is_ambiguous,
        reasoning=reasoning,
        version=version,
        is_tie=is_tie,
    )


ROUTER_VERSIONS = {
    "v1_substring": classify_domain_v1,
    "v2_token_aware": classify_domain_v2,
    "v3_idf_weighted": classify_domain_v3,
}

ROUTER_VERSION_NOTES = {
    "v1_substring": (
        "Original implementation. A keyword scores when it appears anywhere in the "
        "question string, including inside an unrelated word. Kept as the regression "
        "baseline so later improvements are measurable rather than asserted."
    ),
    "v2_token_aware": (
        "Word-boundary matching with plural stemming and a bonus for contiguous "
        "multi-word phrases. Removes the substring false-positive class, but leaves "
        "keyword collisions between domains unresolved."
    ),
    "v3_idf_weighted": (
        "Adds inverse-domain-frequency weighting so a keyword claimed by several "
        "domains carries less signal than a keyword unique to one, caps partial credit "
        "at once per question token, and flags no-match and tied outcomes explicitly "
        "instead of resolving them arbitrarily."
    ),
}


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
