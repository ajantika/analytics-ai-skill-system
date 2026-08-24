"""
ui/data.py — Cached loaders and derived evaluation state.

One place assembles every artifact the pages read, so no page invents its own view
of the data and no two pages can disagree about a headline number.

Human annotations are deliberately NOT cached: they are written by the user during
a session and must be re-read on every rerun.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Optional

import streamlit as st

from alignment import alignment_by_provenance, build_pairs, coverage
from eval_runner import load_judge_results, load_responses, load_runs
from evals import golden_cases, load_golden_set
from human_evals import calibration_summary, load_annotations
from llm_judge import judge_summary
from skills import load_domains

ROOT = pathlib.Path(__file__).parent.parent

DOMAIN_LABELS = {
    "product_usage": "Product Analytics",
    "marketing": "Marketing Analytics",
    "sales": "Sales Analytics",
    "hr": "People Analytics",
    "csup": "Support Analytics",
    "cross_domain": "Cross-domain",
}

DOMAIN_ICONS = {
    "product_usage": "📊", "marketing": "📣", "sales": "💰",
    "hr": "👥", "csup": "🎧", "cross_domain": "🔀",
}

TEST_TYPE_LABELS = {
    "standard": "Standard",
    "grounding": "Grounding",
    "adversarial": "Adversarial",
    "missing_context": "Missing context",
    "cross_domain": "Cross-domain",
    "ambiguous": "Ambiguous",
    "instruction_following": "Instruction following",
    "unsupported_inference": "Unsupported inference",
}


def domain_label(key: Optional[str]) -> str:
    if not key:
        return "Unrouted"
    return DOMAIN_LABELS.get(key, key.replace("_", " ").title())


# ── Cached loaders ────────────────────────────────────────────────────────────


DATA_DIR = ROOT / "data"


def _stamp(*names: str) -> float:
    """
    Modification-time fingerprint for a set of data files.

    Passed into the cached loaders as a cache key so that artifacts written by
    `python eval_runner.py` while the app is running are picked up on the next
    rerun. Without this, a freshly generated run stays invisible until the server
    restarts — which is exactly when a reader would conclude the pipeline is broken.

    The receiving parameter must NOT be named with a leading underscore: Streamlit
    excludes underscore-prefixed arguments from the cache key, which would silently
    disable the invalidation this function exists to provide.
    """
    total = 0.0
    for name in names:
        path = DATA_DIR / name
        if path.exists():
            total += path.stat().st_mtime
    return round(total, 3)


@st.cache_data(show_spinner=False)
def _load_domains(stamp_key: float) -> dict:
    return load_domains(str(ROOT))


def get_domains() -> dict:
    yaml_stamp = sum(p.stat().st_mtime for p in ROOT.glob("*.yaml"))
    return _load_domains(round(yaml_stamp, 3))


@st.cache_data(show_spinner=False)
def _load_golden(stamp_key: float) -> dict:
    return load_golden_set()


def get_golden() -> dict:
    return _load_golden(_stamp("golden_eval_set.json"))


def get_cases() -> list[dict]:
    return get_golden().get("cases", [])


@st.cache_data(show_spinner=False)
def _load_responses(run_id: Optional[str], stamp_key: float) -> list[dict]:
    return load_responses(run_id)


def get_responses(run_id: Optional[str] = None) -> list[dict]:
    return _load_responses(run_id, _stamp("model_responses.json"))


@st.cache_data(show_spinner=False)
def _load_judge(run_id: Optional[str], stamp_key: float) -> list[dict]:
    return load_judge_results(run_id)


def get_judge_results(run_id: Optional[str] = None) -> list[dict]:
    return _load_judge(run_id, _stamp("judge_results.json"))


@st.cache_data(show_spinner=False)
def _load_runs(stamp_key: float) -> list[dict]:
    return load_runs()


def get_runs() -> list[dict]:
    return _load_runs(_stamp("evaluation_runs.json"))


def get_annotations() -> list[dict]:
    """Not cached — the user writes these during the session."""
    return load_annotations()


# ── Assembled state ───────────────────────────────────────────────────────────


@dataclass
class EvalState:
    """Everything the pages read, assembled once per rerun."""
    cases: list[dict]
    cases_by_id: dict
    domains: dict
    responses: list[dict]
    responses_by_id: dict
    judge_results: list[dict]
    judge_by_id: dict
    annotations: list[dict]
    runs: list[dict]
    latest_run: Optional[dict]
    alignment: dict
    coverage: dict
    judge_stats: dict
    calibration: dict
    deterministic: list[dict] = field(default_factory=list)
    deterministic_by_id: dict = field(default_factory=dict)

    # ── availability ──
    @property
    def has_responses(self) -> bool:
        return bool(self.responses)

    @property
    def has_judge(self) -> bool:
        return any(r.get("parse_ok") for r in self.judge_results)

    @property
    def has_human(self) -> bool:
        return any(a.get("rater_type") == "human" for a in self.annotations)

    @property
    def has_demo(self) -> bool:
        return any(a.get("rater_type") == "demo_profile" for a in self.annotations)

    @property
    def has_any_annotations(self) -> bool:
        return bool(self.annotations)

    @property
    def human_annotations(self) -> list[dict]:
        return [a for a in self.annotations if a.get("rater_type") == "human"]

    @property
    def demo_annotations(self) -> list[dict]:
        return [a for a in self.annotations if a.get("rater_type") == "demo_profile"]

    @property
    def human_rater_ids(self) -> list[str]:
        return sorted({a["evaluator_id"] for a in self.human_annotations})

    def response_for(self, eval_id: str) -> Optional[dict]:
        return self.responses_by_id.get(eval_id)

    def judge_for(self, eval_id: str) -> Optional[dict]:
        return self.judge_by_id.get(eval_id)

    def case_for(self, eval_id: str) -> Optional[dict]:
        return self.cases_by_id.get(eval_id)

    def annotations_for(self, eval_id: str) -> list[dict]:
        return [a for a in self.annotations if a.get("eval_id") == eval_id]

    def deterministic_for(self, eval_id: str) -> Optional[dict]:
        return self.deterministic_by_id.get(eval_id)

    def best_alignment(self) -> dict:
        """
        The alignment view to headline.

        Human annotations are the reference standard, so they are used whenever any
        exist. The demo-profile view is only the headline when no human has rated
        anything yet, and every caller renders the returned `label` alongside the
        numbers so the reader always knows which population produced them.
        """
        human = self.alignment["human_only"]
        return human if human.get("n") else self.alignment["demo_only"]


def load_state(run_id: Optional[str] = None) -> EvalState:
    """Assemble the full evaluation state for one page render."""
    cases = get_cases()
    responses = get_responses(run_id)
    judge_results = get_judge_results(run_id)
    annotations = get_annotations()
    runs = get_runs()

    latest = runs[-1] if runs else None
    deterministic = (latest or {}).get("deterministic_records", []) if latest else []

    return EvalState(
        cases=cases,
        cases_by_id={c["eval_id"]: c for c in cases},
        domains=get_domains(),
        responses=responses,
        responses_by_id={r["eval_id"]: r for r in responses},
        judge_results=judge_results,
        judge_by_id={r["eval_id"]: r for r in judge_results if r.get("eval_id")},
        annotations=annotations,
        runs=runs,
        latest_run=latest,
        alignment=alignment_by_provenance(annotations, judge_results),
        coverage=coverage(annotations, judge_results, len(cases)),
        judge_stats=judge_summary(judge_results),
        calibration=calibration_summary(annotations),
        deterministic=deterministic,
        deterministic_by_id={d["eval_id"]: d for d in deterministic},
    )


def clear_caches() -> None:
    """Force a reload of every cached artifact."""
    for fn in (_load_domains, _load_golden, _load_responses, _load_judge, _load_runs):
        fn.clear()


# ── Shared empty states ───────────────────────────────────────────────────────

GENERATE_COMMAND = "export GROQ_API_KEY='...'\npython eval_runner.py --all"

NO_RUN_EXPLANATION = (
    "No evaluation run has been executed yet, so there is nothing to report. This page shows "
    "figures computed from stored evaluation records only — it will not display placeholder "
    "numbers. Run the pipeline against the golden set to populate it."
)

NO_HUMAN_EXPLANATION = (
    "No human annotations exist yet. Human judgement is the reference standard this system "
    "measures the automated judge against, so the human-versus-AI figures cannot be computed "
    "until some responses have been rated. Open the Human Evaluation page and rate a few cases."
)
