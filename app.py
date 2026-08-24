"""
app.py — Analytics AI Skill System.

Application shell only: page configuration, theme, navigation and state assembly.
Every page lives in ui/page_*.py, all evaluation logic lives in the top-level modules
(router, skills, llm, evals, human_evals, llm_judge, alignment, failure_taxonomy,
eval_runner), and nothing in this file computes a metric.
"""

import logging
import pathlib

import streamlit as st

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Analytics AI Skill System",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

from ui import data as D           # noqa: E402
from ui import theme as T          # noqa: E402
from ui import (                   # noqa: E402
    page_alignment,
    page_ask,
    page_dashboard,
    page_disagreements,
    page_failures,
    page_golden,
    page_human_eval,
    page_judge,
    page_methodology,
    page_regression,
)

T.inject_css()

GA_ID = "G-BEKZJV5CJJ"


def _inject_ga4() -> None:
    """Best-effort analytics injection. Silently skipped where static assets are read-only."""
    script = (
        f'<script async src="https://www.googletagmanager.com/gtag/js?id={GA_ID}"></script>'
        f'<script>window.dataLayer=window.dataLayer||[];'
        f'function gtag(){{dataLayer.push(arguments);}}gtag("js",new Date());'
        f'gtag("config","{GA_ID}");</script>'
    )
    try:
        index_path = pathlib.Path(st.__file__).parent / "static" / "index.html"
        html = index_path.read_text()
        if GA_ID not in html:
            index_path.write_text(html.replace("</head>", script + "</head>"))
    except Exception:
        pass


_inject_ga4()

STATE = D.load_state()


def _sidebar_status() -> None:
    """
    A standing statement of what data exists, so no page can be read as reporting on
    something that was never measured.
    """
    st.sidebar.markdown(
        f'<div style="padding:2px 0 10px">'
        f'<div style="color:{T.INK};font-size:0.92rem;font-weight:800;letter-spacing:-0.02em">'
        f'◈ Analytics AI</div>'
        f'<div style="color:{T.FAINT};font-size:0.66rem;margin-top:1px">'
        f'Quality &amp; evaluation platform</div></div>',
        unsafe_allow_html=True,
    )

    cov = STATE.coverage
    rows = [
        ("Golden cases", str(len(STATE.cases)), T.MUTED),
        ("Responses", str(len(STATE.responses)) if STATE.has_responses else "none",
         T.MUTED if STATE.has_responses else T.BAD),
        ("Judge evaluations", str(cov["judge_evaluated"]) if STATE.has_judge else "none",
         T.MUTED if STATE.has_judge else T.BAD),
        ("Human coverage", f"{cov['human_annotated']} cases" if STATE.has_human else "none",
         T.GOOD if STATE.has_human else T.WARN),
        ("Demo-profile coverage", f"{cov['demo_annotated']} cases" if STATE.has_demo else "none", T.FAINT),
        ("Evaluation runs", str(len(STATE.runs)) if STATE.runs else "none",
         T.MUTED if STATE.runs else T.BAD),
    ]
    body = "".join(
        f'<div style="display:flex;justify-content:space-between;padding:3px 0;font-size:0.7rem">'
        f'<span style="color:{T.FAINT}">{label}</span>'
        f'<span style="color:{color};font-weight:600">{value}</span></div>'
        for label, value, color in rows
    )
    st.sidebar.markdown(
        f'<div style="background:rgba(255,255,255,0.025);border:1px solid {T.BORDER};'
        f'border-radius:9px;padding:10px 12px;margin-bottom:10px">'
        f'<div style="color:{T.DIM};font-size:0.6rem;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:0.09em;margin-bottom:6px">Data in this build</div>{body}</div>',
        unsafe_allow_html=True,
    )

    if not STATE.has_responses:
        st.sidebar.markdown(
            f'<div style="background:rgba(248,113,113,0.06);border:1px solid rgba(248,113,113,0.2);'
            f'border-radius:8px;padding:9px 11px;font-size:0.68rem;color:{T.BAD};line-height:1.6">'
            f'No evaluation run yet. Pages show empty states rather than placeholder numbers.'
            f'</div>',
            unsafe_allow_html=True,
        )
    elif not STATE.has_human:
        st.sidebar.markdown(
            f'<div style="background:rgba(234,179,8,0.06);border:1px solid rgba(234,179,8,0.2);'
            f'border-radius:8px;padding:9px 11px;font-size:0.68rem;color:{T.WARN};line-height:1.6">'
            f'No human annotations yet. The AI judge is unvalidated until some responses are '
            f'rated by a person.</div>',
            unsafe_allow_html=True,
        )

    if STATE.latest_run:
        cfg = STATE.latest_run["config"]
        st.sidebar.markdown(
            f'<div style="margin-top:10px;color:{T.FAINT};font-size:0.63rem;line-height:1.7">'
            f'<strong style="color:{T.DIM}">Latest run</strong><br>'
            f'{cfg["run_id"]}<br>{cfg["model_version"]}<br>'
            f'{cfg["system_prompt_version"]} · {cfg["router_version"]}</div>',
            unsafe_allow_html=True,
        )


_sidebar_status()


def _page(fn, slug: str, narrow: bool = False):
    """
    Wrap a page renderer so it receives the assembled state and gets a footer.

    Streamlit derives a page's URL path from the callable's name, so every wrapper
    is renamed to its slug — otherwise all ten closures would collide on `_render`.
    """
    def _render():
        if narrow:
            st.markdown('<style>.block-container{max-width:880px !important}</style>',
                        unsafe_allow_html=True)
        fn(STATE)
        T.footer()

    _render.__name__ = slug
    return _render


PAGES = [
    # The default page is served at "/" — giving it a url_path as well makes that
    # path 404, so it deliberately has none.
    st.Page(_page(page_ask.render, "ask", narrow=True),
            title="Ask AI", icon="💬", default=True),
    st.Page(_page(page_dashboard.render, "dashboard"),
            title="AI Quality Dashboard", icon="📊", url_path="dashboard"),
    st.Page(_page(page_human_eval.render, "human_evaluation"),
            title="Human Evaluation", icon="✍️", url_path="human-evaluation"),
    st.Page(_page(page_judge.render, "ai_judge"),
            title="AI Judge", icon="⚖️", url_path="ai-judge"),
    st.Page(_page(page_alignment.render, "alignment"),
            title="Human ↔ AI Alignment", icon="🎯", url_path="alignment"),
    st.Page(_page(page_disagreements.render, "disagreements"),
            title="Human ↔ AI Disagreements", icon="⚡", url_path="disagreements"),
    st.Page(_page(page_failures.render, "failure_analysis"),
            title="Failure Analysis", icon="🔍", url_path="failure-analysis"),
    st.Page(_page(page_golden.render, "golden_dataset"),
            title="Golden Dataset", icon="📚", url_path="golden-dataset"),
    st.Page(_page(page_regression.render, "regression"),
            title="Regression Testing", icon="📈", url_path="regression"),
    st.Page(_page(page_methodology.render, "methodology"),
            title="Methodology", icon="📖", url_path="methodology"),
]

st.navigation(PAGES).run()
