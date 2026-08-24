"""
ui/theme.py — Shared styling and presentation primitives.

Two rules govern everything here:

  1. A statistic that is not defined renders as "—" with a reason, never as 0%.
     Padding an undefined metric with a zero is the single easiest way for a quality
     dashboard to lie.
  2. Every number on screen is traceable to the records that produced it. Helpers
     that render a metric take an optional denominator or note so the reader can see
     what the figure rests on.
"""

from __future__ import annotations

from typing import Optional

import streamlit as st

# ── Palette ───────────────────────────────────────────────────────────────────

INK = "#f1f5f9"
TEXT = "#cbd5e1"
MUTED = "#94a3b8"
DIM = "#64748b"
FAINT = "#475569"
ACCENT = "#818cf8"
ACCENT_DEEP = "#6366f1"

GOOD = "#22c55e"
WARN = "#eab308"
BAD = "#f87171"
INFO = "#38bdf8"

SURFACE = "rgba(255,255,255,0.028)"
BORDER = "rgba(255,255,255,0.075)"

DIMENSION_COLORS = {
    "relevance": "#818cf8",
    "groundedness": "#34d399",
    "correctness": "#fbbf24",
    "instruction_following": "#f472b6",
    "helpfulness": "#38bdf8",
    "clarity": "#a78bfa",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, -apple-system, sans-serif", color=MUTED, size=12),
    margin=dict(l=8, r=8, t=28, b=8),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.08)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.08)"),
    hoverlabel=dict(bgcolor="#161428", bordercolor=BORDER, font=dict(color=INK)),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED, size=11)),
)


def style_fig(fig, height: int = 260, showlegend: bool = False):
    """Apply the house layout to a Plotly figure."""
    fig.update_layout(**PLOTLY_LAYOUT, height=height, showlegend=showlegend)
    return fig


def count_tick(max_value: float, target_ticks: int = 8) -> int:
    """
    Tick spacing for an integer count axis.

    Hard-coding dtick=1 renders one label per unit, which is unreadable the moment a
    count exceeds about ten. This keeps ticks integral and roughly `target_ticks` apart.
    """
    import math

    if not max_value or max_value <= target_ticks:
        return 1
    return max(1, int(math.ceil(max_value / target_ticks)))


# ── Global CSS ────────────────────────────────────────────────────────────────

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}
.stApp { background: linear-gradient(160deg, #0d0b1e 0%, #111827 55%, #0d0b1e 100%) !important; }
section[data-testid="stMain"] > div { background: transparent !important; }
#MainMenu, footer { visibility: hidden !important; }
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { display: none !important; }

.block-container { max-width: 1180px !important; padding: 1.6rem 1.6rem 4rem !important; }
.narrow .block-container { max-width: 820px !important; }

[data-testid="stVerticalBlock"], [data-testid="element-container"],
[data-testid="stHorizontalBlock"], div[class*="stMarkdown"] { background: transparent !important; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(8,7,20,0.92) !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] { padding-top: 0.4rem; }

/* Buttons */
div[data-testid="stButton"] > button {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 8px !important;
    color: #94a3b8 !important;
    font-size: 0.8rem !important; font-weight: 500 !important;
    padding: 9px 14px !important; transition: all 0.15s ease !important;
}
div[data-testid="stButton"] > button:hover {
    background: rgba(99,102,241,0.15) !important;
    border-color: rgba(99,102,241,0.42) !important; color: #e2e8f0 !important;
}
div[data-testid="stButton"] > button[kind="primary"] {
    background: rgba(99,102,241,0.85) !important; color: #fff !important;
    border-color: rgba(99,102,241,0.9) !important;
}

/* Inputs */
div[data-testid="stTextInput"] > div > div, div[data-testid="stTextArea"] textarea {
    background: rgba(15,12,40,0.95) !important;
    border: 1px solid rgba(99,102,241,0.32) !important; border-radius: 10px !important;
}
div[data-testid="stTextInput"] input, div[data-testid="stTextArea"] textarea {
    background: transparent !important; color: #f1f5f9 !important;
}
div[data-testid="stTextInput"] input { padding: 13px 16px !important; }

/* Expanders */
div[data-testid="stExpander"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important; margin-bottom: 8px !important;
}
div[data-testid="stExpander"] summary {
    color: #94a3b8 !important; font-size: 0.8rem !important; font-weight: 600 !important;
}

/* Tabs */
button[data-baseweb="tab"] { color: #64748b !important; font-size: 0.82rem !important; }
button[data-baseweb="tab"][aria-selected="true"] { color: #a5b4fc !important; }
div[data-baseweb="tab-highlight"] { background-color: #6366f1 !important; }

/* Sliders / radios */
div[data-testid="stSlider"] label, div[data-testid="stRadio"] label,
div[data-testid="stSelectbox"] label, div[data-testid="stMultiSelect"] label {
    color: #94a3b8 !important; font-size: 0.78rem !important; font-weight: 600 !important;
}

/* Dataframes */
div[data-testid="stDataFrame"] { border: 1px solid rgba(255,255,255,0.07) !important; border-radius: 10px !important; }

/* Typography */
p, li { color: #cbd5e1 !important; }
h1 { color: #f1f5f9 !important; }
h2, h3, h4, h5 { color: #e2e8f0 !important; }
code { background: rgba(99,102,241,0.14) !important; color: #a5b4fc !important; border-radius: 4px !important; }
pre { background: rgba(0,0,0,0.32) !important; border: 1px solid rgba(255,255,255,0.07) !important; border-radius: 8px !important; }
strong { color: #e2e8f0 !important; }
hr { border-color: rgba(255,255,255,0.07) !important; }
table { color: #cbd5e1 !important; }
th { color: #94a3b8 !important; font-size: 0.75rem !important; }
a { color: #818cf8 !important; }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


# ── Formatting ────────────────────────────────────────────────────────────────

DASH = "—"


def pct(value: Optional[float], places: int = 0) -> str:
    """Percentage, or an em-dash when undefined. Never renders None as 0%."""
    return DASH if value is None else f"{value * 100:.{places}f}%"


def score(value: Optional[float], places: int = 2) -> str:
    return DASH if value is None else f"{value:.{places}f}"


def num(value: Optional[float]) -> str:
    return DASH if value is None else f"{value:,g}"


def pp(value: Optional[float], places: int = 1) -> str:
    """Signed percentage-point delta."""
    return DASH if value is None else f"{value * 100:+.{places}f}pp"


def signed(value: Optional[float], places: int = 2) -> str:
    return DASH if value is None else f"{value:+.{places}f}"


def kappa_label(value: Optional[float]) -> str:
    from stats_utils import kappa_interpretation
    return DASH if value is None else f"{value:.2f} ({kappa_interpretation(value)})"


def dim_label(dimension: str) -> str:
    from human_evals import DIMENSION_LABELS
    return DIMENSION_LABELS.get(dimension, dimension.replace("_", " ").title())


def score_color(value: Optional[float], good: float = 4.0, warn: float = 3.0) -> str:
    if value is None:
        return DIM
    return GOOD if value >= good else WARN if value >= warn else BAD


def rate_color(value: Optional[float], good: float = 0.8, warn: float = 0.6,
               lower_is_better: bool = False) -> str:
    if value is None:
        return DIM
    if lower_is_better:
        return GOOD if value <= (1 - good) else WARN if value <= (1 - warn) else BAD
    return GOOD if value >= good else WARN if value >= warn else BAD


# ── Building blocks ───────────────────────────────────────────────────────────


def page_header(title: str, subtitle: str = "", eyebrow: str = "") -> None:
    eyebrow_html = (
        f'<div style="color:{ACCENT_DEEP};font-size:0.66rem;font-weight:700;'
        f'text-transform:uppercase;letter-spacing:0.12em;margin-bottom:6px">{eyebrow}</div>'
        if eyebrow else ""
    )
    sub_html = (
        f'<p style="color:{DIM};font-size:0.88rem;margin:6px 0 0;line-height:1.65;'
        f'max-width:760px">{subtitle}</p>' if subtitle else ""
    )
    st.markdown(
        f'<div style="padding:0.4rem 0 1.1rem">{eyebrow_html}'
        f'<h2 style="color:{INK};font-size:1.55rem;font-weight:800;margin:0;'
        f'letter-spacing:-0.025em">{title}</h2>{sub_html}</div>',
        unsafe_allow_html=True,
    )


def section(title: str, subtitle: str = "", top_rule: bool = True) -> None:
    rule = f"border-top:1px solid {BORDER};padding-top:18px;" if top_rule else ""
    sub = (f'<p style="color:{DIM};font-size:0.79rem;margin:3px 0 0;line-height:1.6;'
           f'max-width:780px">{subtitle}</p>') if subtitle else ""
    st.markdown(
        f'<div style="{rule}margin:14px 0 12px">'
        f'<h3 style="color:{INK};font-size:1.0rem;font-weight:700;margin:0;'
        f'letter-spacing:-0.01em">{title}</h3>{sub}</div>',
        unsafe_allow_html=True,
    )


def metric_card(
    label: str,
    value: str,
    delta: Optional[str] = None,
    delta_good: Optional[bool] = None,
    footnote: str = "",
    target: str = "",
    accent: str = ACCENT,
    value_color: Optional[str] = None,
) -> str:
    """
    One executive metric tile.

    `footnote` is where the denominator goes. A rate without its denominator is not
    a measurement, so the tile is built to always have somewhere to put it.
    """
    delta_html = ""
    if delta:
        if delta_good is None:
            bg, fg = "rgba(148,163,184,0.12)", MUTED
        elif delta_good:
            bg, fg = "rgba(34,197,94,0.13)", GOOD
        else:
            bg, fg = "rgba(248,113,113,0.13)", BAD
        delta_html = (
            f'<span style="background:{bg};color:{fg};font-size:0.66rem;font-weight:700;'
            f'padding:2px 8px;border-radius:5px;margin-left:8px;white-space:nowrap">{delta}</span>'
        )

    target_html = (
        f'<div style="color:{FAINT};font-size:0.63rem;margin-top:2px">{target}</div>'
        if target else ""
    )
    foot_html = (
        f'<div style="color:{DIM};font-size:0.65rem;margin-top:7px;line-height:1.45">{footnote}</div>'
        if footnote else ""
    )

    return (
        f'<div style="background:{SURFACE};border:1px solid {BORDER};'
        f'border-top:2px solid {accent};border-radius:11px;padding:13px 15px 12px;height:100%">'
        f'<div style="color:{DIM};font-size:0.63rem;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:0.08em;line-height:1.35;min-height:2.0em">{label}</div>'
        f'<div style="display:flex;align-items:baseline;flex-wrap:wrap;margin-top:5px">'
        f'<span style="color:{value_color or INK};font-size:1.62rem;font-weight:800;'
        f'letter-spacing:-0.03em;line-height:1.1">{value}</span>{delta_html}</div>'
        f'{target_html}{foot_html}</div>'
    )


def metric_row(cards: list[str], columns: Optional[int] = None) -> None:
    """Render metric cards in an evenly spaced grid."""
    cols = st.columns(columns or len(cards), gap="small")
    for col, card in zip(cols, cards):
        col.markdown(card, unsafe_allow_html=True)


def panel(content_html: str, accent: str = ACCENT_DEEP, tint: str = "") -> str:
    bg = tint or SURFACE
    return (
        f'<div style="background:{bg};border:1px solid {BORDER};border-left:3px solid {accent};'
        f'border-radius:10px;padding:14px 17px;margin-bottom:11px">{content_html}</div>'
    )


def label_text(text: str, color: str = ACCENT_DEEP) -> str:
    return (f'<div style="color:{color};font-size:0.62rem;font-weight:700;'
            f'text-transform:uppercase;letter-spacing:0.1em;margin-bottom:5px">{text}</div>')


def body_text(text: str, color: str = TEXT, size: str = "0.84rem") -> str:
    return f'<div style="color:{color};font-size:{size};line-height:1.7">{text}</div>'


_MD_BOLD = None


def response_html(text: str) -> str:
    """
    Render a model response inside an HTML panel.

    Model output uses **bold** section headers. Streamlit's markdown parser does not
    run on raw HTML we inject, so those would otherwise display as literal asterisks.
    This converts the small subset the response template actually uses and escapes
    everything else, since the text is model output and must not be able to inject markup.
    """
    import html
    import re

    global _MD_BOLD
    if _MD_BOLD is None:
        _MD_BOLD = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)

    escaped = html.escape(str(text).strip())
    bolded = _MD_BOLD.sub(
        rf'<strong style="color:{INK};font-size:0.72rem;text-transform:uppercase;'
        rf'letter-spacing:0.07em">\1</strong>',
        escaped,
    )
    return bolded.replace("\n", "<br>")


def chip(text: str, color: str = MUTED, bg: Optional[str] = None) -> str:
    background = bg or "rgba(148,163,184,0.10)"
    return (
        f'<span style="background:{background};color:{color};font-size:0.66rem;font-weight:600;'
        f'padding:2px 9px;border-radius:11px;margin-right:5px;white-space:nowrap;'
        f'display:inline-block;margin-bottom:3px">{text}</span>'
    )


def status_chip(status: str) -> str:
    """PASS / WARN / FAIL / FOUND / NONE / N/A rendered consistently everywhere."""
    mapping = {
        "PASS": (GOOD, "rgba(34,197,94,0.13)", "Pass"),
        "NONE": (GOOD, "rgba(34,197,94,0.13)", "None"),
        "WARN": (WARN, "rgba(234,179,8,0.13)", "Warn"),
        "FAIL": (BAD, "rgba(248,113,113,0.13)", "Fail"),
        "FOUND": (BAD, "rgba(248,113,113,0.13)", "Found"),
        "N/A": (FAINT, "rgba(148,163,184,0.08)", "n/a"),
    }
    color, bg, text = mapping.get(status, (DIM, "rgba(148,163,184,0.10)", status))
    return chip(text, color, bg)


def pass_chip(passed: Optional[bool]) -> str:
    if passed is None:
        return chip("not rated", FAINT, "rgba(148,163,184,0.08)")
    return (chip("PASS", GOOD, "rgba(34,197,94,0.13)") if passed
            else chip("FAIL", BAD, "rgba(248,113,113,0.13)"))


def severity_chip(severity: str) -> str:
    from failure_taxonomy import SEVERITY_COLORS
    color = SEVERITY_COLORS.get(severity, DIM)
    tints = {
        "critical": "rgba(248,113,113,0.14)", "high": "rgba(251,146,60,0.14)",
        "medium": "rgba(234,179,8,0.13)", "low": "rgba(56,189,248,0.13)",
        "none": "rgba(34,197,94,0.12)",
    }
    return chip(severity.title(), color, tints.get(severity, "rgba(148,163,184,0.10)"))


def provenance_chip(rater_type: str) -> str:
    """
    Provenance is shown wherever an annotation appears. There is no view in this
    application where a demo profile can be mistaken for a person.
    """
    if rater_type == "human":
        return chip("Human annotation", GOOD, "rgba(34,197,94,0.13)")
    if rater_type == "demo_profile":
        return chip("Demo profile · not human", WARN, "rgba(234,179,8,0.13)")
    return chip(rater_type, DIM)


def score_bar(value: Optional[float], maximum: float = 5.0, width: int = 68) -> str:
    """Inline 1-5 score bar for dense tables."""
    if value is None:
        return f'<span style="color:{FAINT}">{DASH}</span>'
    filled = max(0.0, min(1.0, value / maximum))
    color = score_color(value)
    return (
        f'<span style="display:inline-flex;align-items:center;gap:7px">'
        f'<span style="display:inline-block;width:{width}px;height:5px;border-radius:3px;'
        f'background:rgba(255,255,255,0.08);overflow:hidden">'
        f'<span style="display:block;width:{filled * 100:.0f}%;height:100%;background:{color}"></span>'
        f'</span><span style="color:{color};font-size:0.76rem;font-weight:600">{value:.1f}</span></span>'
    )


def note(text: str, kind: str = "info") -> None:
    """A small qualifying note. Used for denominators, caveats and provenance."""
    palette = {
        "info": (DIM, "rgba(148,163,184,0.06)", "rgba(148,163,184,0.16)"),
        "warn": (WARN, "rgba(234,179,8,0.05)", "rgba(234,179,8,0.20)"),
        "good": (GOOD, "rgba(34,197,94,0.05)", "rgba(34,197,94,0.20)"),
    }
    color, bg, border = palette.get(kind, palette["info"])
    st.markdown(
        f'<div style="background:{bg};border:1px solid {border};border-radius:8px;'
        f'padding:9px 13px;margin:6px 0 12px;color:{color};font-size:0.74rem;'
        f'line-height:1.65">{text}</div>',
        unsafe_allow_html=True,
    )


def empty_state(title: str, explanation: str, command: str = "") -> None:
    """
    Shown when an artifact has not been generated yet.

    Says what is missing and exactly how to produce it, rather than rendering an
    empty chart that reads as "zero" instead of "not measured".
    """
    cmd_html = (
        f'<div style="background:rgba(0,0,0,0.3);border:1px solid {BORDER};border-radius:7px;'
        f'padding:9px 13px;margin-top:11px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;'
        f'font-size:0.76rem;color:{ACCENT};white-space:pre-line;line-height:1.7">{command}</div>'
        if command else ""
    )
    st.markdown(
        f'<div style="background:{SURFACE};border:1px dashed rgba(255,255,255,0.13);'
        f'border-radius:12px;padding:26px 28px;margin:10px 0 16px;text-align:left">'
        f'<div style="color:{MUTED};font-size:0.95rem;font-weight:700;margin-bottom:7px">{title}</div>'
        f'<div style="color:{DIM};font-size:0.82rem;line-height:1.75;max-width:660px">{explanation}</div>'
        f'{cmd_html}</div>',
        unsafe_allow_html=True,
    )


def evaluator_tier_legend() -> None:
    """
    The evaluator hierarchy, shown wherever the three tiers appear together.
    Making the division of labour explicit is the point of the architecture.
    """
    tiers = [
        ("Deterministic", GOOD,
         "Objective, repeatable checks: does this figure appear in the governed context, "
         "is this metric defined, did the router pick the declared domain.",
         "Best for objective checks. Cannot judge whether an answer is useful."),
        ("Human", ACCENT,
         "A person applies the six-dimension rubric. The reference standard for "
         "subjective quality in this system.",
         "Authoritative but slow. Coverage is the binding constraint."),
        ("LLM-as-a-Judge", WARN,
         "A model applies the same rubric to the same responses, at full coverage.",
         "Scalable approximation of human judgement — only trustworthy to the extent "
         "it agrees with humans, which is measured on the Alignment page."),
    ]
    cols = st.columns(3, gap="small")
    for col, (name, color, what, caveat) in zip(cols, tiers):
        col.markdown(
            f'<div style="background:{SURFACE};border:1px solid {BORDER};'
            f'border-top:2px solid {color};border-radius:10px;padding:12px 14px;height:100%">'
            f'<div style="color:{color};font-size:0.8rem;font-weight:700;margin-bottom:6px">{name}</div>'
            f'<div style="color:{TEXT};font-size:0.74rem;line-height:1.6;margin-bottom:7px">{what}</div>'
            f'<div style="color:{FAINT};font-size:0.68rem;line-height:1.55;font-style:italic">{caveat}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


def footer() -> None:
    st.markdown(
        f'<div style="border-top:1px solid rgba(255,255,255,0.05);margin-top:2.5rem;'
        f'padding:1rem 0 0.5rem;text-align:center;font-size:0.7rem;color:{FAINT}">'
        f'Built by <a href="https://ajantika.github.io" style="color:{ACCENT_DEEP};'
        f'text-decoration:none;font-weight:600">Ajantika Paul</a> · '
        f'<a href="https://github.com/ajantika/analytics-ai-skill-system" '
        f'style="color:{ACCENT_DEEP};text-decoration:none">GitHub</a> · '
        f'Portfolio project — illustrative data, no production traffic'
        f'</div>',
        unsafe_allow_html=True,
    )
