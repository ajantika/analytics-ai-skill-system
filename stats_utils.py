"""
stats_utils.py — Agreement and correlation primitives.

Pure-Python/NumPy implementations so the deployed app does not need SciPy and so
every statistic used on the dashboards is inspectable and unit-tested rather than
delegated to an opaque call.

All functions are total: they return None (not an exception, not a fabricated
number) when the input is too small or degenerate for the statistic to be defined.
A None result must be rendered as "not defined" in the UI — never as 0.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

# ── Pairing ───────────────────────────────────────────────────────────────────


def paired(a: dict, b: dict) -> tuple[list, list, list]:
    """
    Align two {key: value} mappings on their shared keys.

    Returns (keys, a_values, b_values) with None-valued entries dropped, so that
    a rater who skipped a dimension does not silently contribute a zero.
    """
    keys = [k for k in a if k in b and a[k] is not None and b[k] is not None]
    keys.sort()
    return keys, [a[k] for k in keys], [b[k] for k in keys]


# ── Agreement ─────────────────────────────────────────────────────────────────


def exact_agreement(x: Sequence, y: Sequence) -> Optional[float]:
    """Proportion of paired observations that are identical."""
    if not x or len(x) != len(y):
        return None
    return sum(1 for a, b in zip(x, y) if a == b) / len(x)


def within_n_agreement(x: Sequence[float], y: Sequence[float], n: float = 1.0) -> Optional[float]:
    """Proportion of paired observations within +/- n of each other."""
    if not x or len(x) != len(y):
        return None
    return sum(1 for a, b in zip(x, y) if abs(a - b) <= n) / len(x)


def mean_signed_difference(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    """
    Mean of (y - x). Used for judge-bias direction: positive means y (the judge)
    scores higher than x (the human) on average.
    """
    if not x or len(x) != len(y):
        return None
    return sum(b - a for a, b in zip(x, y)) / len(x)


# ── Cohen's kappa ─────────────────────────────────────────────────────────────


def cohens_kappa(
    x: Sequence,
    y: Sequence,
    categories: Optional[Sequence] = None,
    weights: Optional[str] = None,
) -> Optional[float]:
    """
    Cohen's kappa: agreement corrected for chance.

    weights=None       nominal kappa. Correct for unordered labels such as
                       pass/fail or failure-mode category.
    weights="linear"   ordinal kappa, disagreement penalty grows linearly.
    weights="quadratic" ordinal kappa, penalty grows with the square of the gap.
                       Appropriate for 1-5 rubric scores, where 4-vs-5 is a far
                       smaller disagreement than 1-vs-5.

    Returns None when fewer than 2 paired observations exist, or when expected
    agreement is 1.0 (both raters used exactly one category, so chance-corrected
    agreement is undefined rather than perfect).
    """
    if not x or len(x) != len(y) or len(x) < 2:
        return None

    cats = list(categories) if categories is not None else sorted(set(list(x) + list(y)))
    if len(cats) < 2:
        return None
    idx = {c: i for i, c in enumerate(cats)}
    k = len(cats)

    if any(v not in idx for v in x) or any(v not in idx for v in y):
        return None

    n = len(x)
    obs = [[0.0] * k for _ in range(k)]
    for a, b in zip(x, y):
        obs[idx[a]][idx[b]] += 1.0

    row = [sum(obs[i]) / n for i in range(k)]
    col = [sum(obs[i][j] for i in range(k)) / n for j in range(k)]

    if weights is None:
        w = [[0.0 if i == j else 1.0 for j in range(k)] for i in range(k)]
    elif weights == "linear":
        w = [[abs(i - j) / (k - 1) for j in range(k)] for i in range(k)]
    elif weights == "quadratic":
        w = [[((i - j) / (k - 1)) ** 2 for j in range(k)] for i in range(k)]
    else:
        raise ValueError(f"unknown weights: {weights!r}")

    d_obs = sum(w[i][j] * obs[i][j] / n for i in range(k) for j in range(k))
    d_exp = sum(w[i][j] * row[i] * col[j] for i in range(k) for j in range(k))

    if d_exp == 0:
        # Zero expected disagreement: every cell chance-expects perfect agreement.
        return None
    return 1.0 - (d_obs / d_exp)


def kappa_interpretation(k: Optional[float]) -> str:
    """Landis & Koch descriptive bands. Descriptive only — not a quality target."""
    if k is None:
        return "not defined"
    if k < -0.01:
        return "worse than chance"
    if k < 0.01:
        return "no better than chance"
    if k < 0.20:
        return "slight"
    if k < 0.40:
        return "fair"
    if k < 0.60:
        return "moderate"
    if k < 0.80:
        return "substantial"
    return "almost perfect"


# ── Correlation ───────────────────────────────────────────────────────────────


def pearson(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    """Pearson product-moment correlation. None if n < 3 or either series is constant."""
    if not x or len(x) != len(y) or len(x) < 3:
        return None
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    dx = [a - mx for a in x]
    dy = [b - my for b in y]
    num = sum(a * b for a, b in zip(dx, dy))
    den = math.sqrt(sum(a * a for a in dx) * sum(b * b for b in dy))
    if den == 0:
        return None
    return num / den


def _ranks(values: Sequence[float]) -> list[float]:
    """Fractional ranks with ties averaged (required for correct Spearman on 1-5 scores)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for t in range(i, j + 1):
            ranks[order[t]] = avg
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    """Spearman rank correlation, tie-corrected. Preferred over Pearson for ordinal rubric scores."""
    if not x or len(x) != len(y) or len(x) < 3:
        return None
    return pearson(_ranks(x), _ranks(y))


# ── Confusion matrix ──────────────────────────────────────────────────────────


def confusion_matrix(x: Sequence, y: Sequence, categories: Sequence) -> list[list[int]]:
    """Counts indexed [x_category][y_category]. Rows = first rater, columns = second."""
    idx = {c: i for i, c in enumerate(categories)}
    k = len(categories)
    m = [[0] * k for _ in range(k)]
    for a, b in zip(x, y):
        if a in idx and b in idx:
            m[idx[a]][idx[b]] += 1
    return m


def binary_rates(x: Sequence[bool], y: Sequence[bool]) -> dict:
    """
    Treating x as reference and y as the system under test, return the four cells
    plus precision/recall of y against x. Used for judge-vs-human pass/fail.
    """
    tp = sum(1 for a, b in zip(x, y) if a and b)
    tn = sum(1 for a, b in zip(x, y) if not a and not b)
    fp = sum(1 for a, b in zip(x, y) if not a and b)
    fn = sum(1 for a, b in zip(x, y) if a and not b)
    n = tp + tn + fp + fn
    return {
        "true_pass": tp,
        "true_fail": tn,
        "false_pass": fp,   # judge passed a response the human failed — the costly error
        "false_fail": fn,
        "n": n,
        "accuracy": (tp + tn) / n if n else None,
        "precision": tp / (tp + fp) if (tp + fp) else None,
        "recall": tp / (tp + fn) if (tp + fn) else None,
    }


# ── Small helpers ─────────────────────────────────────────────────────────────


def mean(values: Iterable[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def rate(numerator: int, denominator: int) -> Optional[float]:
    return numerator / denominator if denominator else None
