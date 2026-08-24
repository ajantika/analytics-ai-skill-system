"""
tests/test_stats_utils.py — Agreement and correlation primitives.

Kappa and correlation values are checked against hand-computed expectations rather
than against another library, so a silent change in behaviour is caught.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stats_utils import (
    binary_rates,
    cohens_kappa,
    confusion_matrix,
    exact_agreement,
    kappa_interpretation,
    mean,
    mean_signed_difference,
    paired,
    pearson,
    rate,
    spearman,
    within_n_agreement,
)


# ── pairing ───────────────────────────────────────────────────────────────────

def test_paired_uses_shared_keys_only():
    keys, a, b = paired({"x": 1, "y": 2, "z": 3}, {"x": 5, "y": 6, "w": 9})
    assert keys == ["x", "y"]
    assert a == [1, 2] and b == [5, 6]


def test_paired_drops_none_values():
    """A rater who skipped a dimension must not contribute a zero."""
    keys, a, b = paired({"x": 1, "y": None}, {"x": 2, "y": 4})
    assert keys == ["x"], "None-valued entries must be dropped, not coerced"
    assert a == [1] and b == [2]


# ── agreement ─────────────────────────────────────────────────────────────────

def test_exact_agreement():
    assert exact_agreement([1, 2, 3, 4], [1, 2, 3, 5]) == 0.75
    assert exact_agreement([1, 1], [1, 1]) == 1.0
    assert exact_agreement([1, 1], [2, 2]) == 0.0


def test_exact_agreement_empty_returns_none():
    assert exact_agreement([], []) is None
    assert exact_agreement([1, 2], [1]) is None


def test_within_one_agreement_is_looser_than_exact():
    x, y = [1, 2, 3, 4], [2, 3, 3, 5]
    assert exact_agreement(x, y) == 0.25
    assert within_n_agreement(x, y, 1) == 1.0


def test_mean_signed_difference_direction():
    """Positive means the second series scores higher — used for judge leniency."""
    assert mean_signed_difference([3, 3, 3], [4, 4, 4]) == 1.0
    assert mean_signed_difference([4, 4], [3, 3]) == -1.0


# ── Cohen's kappa ─────────────────────────────────────────────────────────────

def test_kappa_perfect_agreement_is_one():
    x = [1, 2, 3, 4, 5, 1, 2]
    assert abs(cohens_kappa(x, x, categories=[1, 2, 3, 4, 5]) - 1.0) < 1e-9


def test_kappa_chance_level_is_near_zero():
    """Independent raters with matched marginals should land near 0, not near accuracy."""
    x = [1, 1, 2, 2, 1, 1, 2, 2]
    y = [1, 2, 1, 2, 1, 2, 1, 2]
    k = cohens_kappa(x, y, categories=[1, 2])
    assert k is not None and abs(k) < 0.3, f"expected near-chance kappa, got {k}"


def test_kappa_below_zero_when_worse_than_chance():
    x = [1, 1, 1, 2, 2, 2]
    y = [2, 2, 2, 1, 1, 1]
    k = cohens_kappa(x, y, categories=[1, 2])
    assert k is not None and k < 0


def test_kappa_known_value():
    """
    Hand-computed 2x2, n=100, cells 60/10/10/20.
      observed agreement = (60+20)/100                  = 0.80
      marginals          = 0.7/0.3 on both raters
      expected agreement = 0.7*0.7 + 0.3*0.3            = 0.58
      kappa              = (0.80-0.58)/(1-0.58) = 0.22/0.42 = 0.523809...
    """
    x = [1] * 60 + [1] * 10 + [2] * 10 + [2] * 20
    y = [1] * 60 + [2] * 10 + [1] * 10 + [2] * 20
    k = cohens_kappa(x, y, categories=[1, 2])
    assert abs(k - (0.22 / 0.42)) < 1e-9, f"expected {0.22/0.42}, got {k}"


def test_quadratic_weighting_rewards_near_misses():
    """4-vs-5 disagreement should be penalised far less than 1-vs-5."""
    near_x, near_y = [4, 4, 5, 5, 3, 3], [5, 4, 5, 4, 3, 4]
    far_x, far_y = [4, 4, 5, 5, 3, 3], [1, 4, 1, 4, 3, 1]
    k_near = cohens_kappa(near_x, near_y, categories=[1, 2, 3, 4, 5], weights="quadratic")
    k_far = cohens_kappa(far_x, far_y, categories=[1, 2, 3, 4, 5], weights="quadratic")
    assert k_near > k_far, f"near-miss kappa {k_near} should exceed far-miss {k_far}"


def test_kappa_returns_none_when_undefined():
    assert cohens_kappa([], []) is None
    assert cohens_kappa([3], [3]) is None, "n=1 is not enough for kappa"
    assert cohens_kappa([3, 3, 3], [3, 3, 3]) is None, \
        "single category used by both raters -> expected agreement 1.0 -> undefined"


def test_kappa_rejects_unknown_weights():
    try:
        cohens_kappa([1, 2], [1, 2], categories=[1, 2], weights="cubic")
        assert False, "should raise on unknown weighting scheme"
    except ValueError:
        pass


def test_kappa_interpretation_bands():
    assert kappa_interpretation(None) == "not defined"
    assert kappa_interpretation(-0.1) == "worse than chance"
    assert kappa_interpretation(0.5) == "moderate"
    assert kappa_interpretation(0.9) == "almost perfect"


# ── correlation ───────────────────────────────────────────────────────────────

def test_pearson_perfect_positive_and_negative():
    assert abs(pearson([1, 2, 3, 4], [2, 4, 6, 8]) - 1.0) < 1e-9
    assert abs(pearson([1, 2, 3, 4], [8, 6, 4, 2]) + 1.0) < 1e-9


def test_pearson_none_on_constant_series():
    assert pearson([3, 3, 3, 3], [1, 2, 3, 4]) is None, \
        "zero variance must return None, not 0.0"


def test_pearson_none_when_too_few_points():
    assert pearson([1, 2], [1, 2]) is None


def test_spearman_captures_monotonic_nonlinear():
    """Spearman should see a perfect monotonic relationship Pearson understates."""
    x = [1, 2, 3, 4, 5]
    y = [1, 4, 9, 16, 25]
    assert abs(spearman(x, y) - 1.0) < 1e-9
    assert pearson(x, y) < 1.0


def test_spearman_handles_ties():
    """Tied ranks must be averaged; 1-5 rubric scores are full of ties."""
    x = [3, 3, 4, 5, 5]
    y = [3, 4, 4, 5, 5]
    s = spearman(x, y)
    assert s is not None and 0 < s <= 1.0


# ── confusion matrix ──────────────────────────────────────────────────────────

def test_confusion_matrix_orientation():
    """Rows index the first series, columns the second."""
    m = confusion_matrix([True, True, False], [True, False, False], [False, True])
    # categories = [False, True] -> index 0 = False, 1 = True
    assert m[1][1] == 1, "human pass / judge pass"
    assert m[1][0] == 1, "human pass / judge fail"
    assert m[0][0] == 1, "human fail / judge fail"
    assert m[0][1] == 0


def test_binary_rates_labels_false_pass():
    human = [True, False, False, True]
    judge = [True, True, False, False]
    r = binary_rates(human, judge)
    assert r["true_pass"] == 1
    assert r["false_pass"] == 1, "judge passed what the human failed"
    assert r["false_fail"] == 1
    assert r["true_fail"] == 1
    assert r["accuracy"] == 0.5


def test_binary_rates_none_precision_when_no_positives():
    r = binary_rates([False, False], [False, False])
    assert r["precision"] is None and r["recall"] is None


# ── helpers ───────────────────────────────────────────────────────────────────

def test_mean_ignores_none():
    assert mean([1, None, 3]) == 2.0
    assert mean([None, None]) is None
    assert mean([]) is None


def test_rate_guards_zero_denominator():
    assert rate(3, 4) == 0.75
    assert rate(0, 0) is None
