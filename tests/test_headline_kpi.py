"""
tests/test_headline_kpi.py — The large figure shown above an answer.

Setting a number in 2.5rem type asserts "this is the answer". The extractor was
doing that for a figure it picked by pattern order, which put a region's MRR on
screen as the headline answer to a question about margins — and truncated it from
$1.4 M to $1.4 on the way.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ui.page_ask import _headline_kpi

MARGIN_BREAKDOWN = (
    "Product margin by region: EMEA 71% margin ($1.4 M MRR), North America 68% "
    "margin ($2.1 M MRR), APAC 52% margin ($890 K MRR), LATAM 44% margin ($310 K MRR)."
)


def test_breakdown_gets_no_headline():
    """The regression. Eight figures, none of which is 'the' answer."""
    assert _headline_kpi(MARGIN_BREAKDOWN) == ""


def test_magnitude_suffix_survives_a_space():
    """'$1.4 M' must not become '$1.4' — a factor of a million."""
    assert _headline_kpi("Right-sizing represents $1.4 M in additional annual MRR.") == "$1.4M"


def test_magnitude_suffix_without_space():
    assert _headline_kpi("The recovery opportunity is $1.4M.") == "$1.4M"


def test_thousands_suffix():
    assert _headline_kpi("Churned MRR was $89 K last quarter.") == "$89K"


def test_plain_currency_with_separators():
    assert _headline_kpi("Blended average contract value is $28,600.") == "$28,600"


def test_single_percentage():
    assert _headline_kpi("Our CSAT score is 87.3% overall.") == "87.3%"


def test_small_percentage_suppressed():
    """A 4.2% set in poster type reads as noise, not emphasis."""
    assert _headline_kpi("Regrettable attrition is 4.2% this quarter.") == ""


def test_two_figures_suppressed():
    assert _headline_kpi("Coverage sits at 3.8x against the 3x healthy threshold.") == ""


def test_bare_count_is_not_a_kpi():
    """694 is a count, not a headline metric; without a unit it reads as a mystery number."""
    assert _headline_kpi("694 customers are over-utilizing their plans.") == ""


def test_repeated_figure_counts_once():
    assert _headline_kpi("Total MRR is $4.2 M. That $4.2 M is flat on last quarter.") == "$4.2M"


def test_empty_and_figureless_text():
    assert _headline_kpi("") == ""
    assert _headline_kpi("The governed layer does not contain this metric.") == ""


def test_multiple_x_ratios_suppressed():
    assert _headline_kpi("SMB is at 2.1x while Enterprise runs 4.5x.") == ""
