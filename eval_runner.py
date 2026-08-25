"""
eval_runner.py — Orchestrates an evaluation run and freezes the result as an artifact.

An evaluation run is a fully specified experiment. Every run records the router
version, system-prompt version, generator model, judge model, judge version and
dataset version that produced it, because a quality number without that metadata
cannot be compared to anything.

    python eval_runner.py --all                    full run: generate, evaluate, judge
    python eval_runner.py --all --label "baseline" --prompt-version sysprompt-v1
    python eval_runner.py --all --router-version v1_substring
    python eval_runner.py --responses               generation + deterministic only, no judge
    python eval_runner.py --judge --run-id <id>     judge an existing run's responses
    python eval_runner.py --list                    list stored runs
    python eval_runner.py --compare <base> <curr>   print a regression comparison

The Groq key is read from GROQ_API_KEY in the environment, falling back to Streamlit
secrets. It is never read from, or written to, any file in this repository.

Artifacts are written to data/:
    runs/<run_id>/model_responses.json   frozen generation output for that run
    runs/<run_id>/judge_results.json     frozen judge output for that run
    model_responses.json                 copy of the most recent run, for the app
    judge_results.json                   copy of the most recent run, for the app
    human_annotations.json               demo-profile + human annotations (merged, never overwritten)
    evaluation_runs.json                 run metadata and computed summaries
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional

from alignment import alignment_by_provenance, coverage
from demo_raters import DEMO_RATERS, annotate_all
from evals import (
    deterministic_summary,
    golden_cases,
    load_golden_set,
    run_deterministic_suite,
    run_golden_routing_eval,
)
from failure_taxonomy import distribution as failure_distribution
from human_evals import (
    DIMENSIONS,
    load_annotations,
    save_annotations,
)
from llm import DEFAULT_MODEL, DEFAULT_PROMPT_VERSION, ask_groq, get_api_key
from llm_judge import JUDGE_VERSION, judge_response, judge_summary
from router import DEFAULT_ROUTER_VERSION, ROUTER_VERSIONS
from skills import build_context, load_domains

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("eval_runner")

ROOT = pathlib.Path(__file__).parent
DATA_DIR = ROOT / "data"
RUNS_DIR = DATA_DIR / "runs"
RUNS_INDEX = DATA_DIR / "evaluation_runs.json"
CURRENT_RESPONSES = DATA_DIR / "model_responses.json"
CURRENT_JUDGE = DATA_DIR / "judge_results.json"


# ── Run configuration ─────────────────────────────────────────────────────────


@dataclass
class RunConfig:
    run_id: str
    timestamp: str
    label: str = ""
    model_version: str = DEFAULT_MODEL
    system_prompt_version: str = DEFAULT_PROMPT_VERSION
    router_version: str = DEFAULT_ROUTER_VERSION
    judge_version: str = JUDGE_VERSION
    judge_model: Optional[str] = None
    dataset_version: Optional[str] = None
    n_cases: int = 0
    temperature: float = 0.2
    notes: str = ""
    artifact_kind: str = "real_model_run"   # never "simulated" unless it actually is
    demo_rater_ids: list = field(default_factory=lambda: list(DEMO_RATERS))


def new_run_config(**overrides) -> RunConfig:
    now = datetime.now(timezone.utc)
    cfg = RunConfig(
        run_id=f"run-{now:%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}",
        timestamp=now.isoformat(timespec="seconds"),
        dataset_version=load_golden_set().get("dataset_version"),
    )
    for k, v in overrides.items():
        if v is not None and hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


# ── IO ────────────────────────────────────────────────────────────────────────


def _write_json(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)
    logger.info(f"wrote {path.relative_to(ROOT)}")


def _read_json(path: pathlib.Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"failed to read {path}: {e}")
        return None


def load_responses(run_id: Optional[str] = None) -> list[dict]:
    path = (RUNS_DIR / run_id / "model_responses.json") if run_id else CURRENT_RESPONSES
    data = _read_json(path)
    return data.get("responses", []) if data else []


def load_judge_results(run_id: Optional[str] = None) -> list[dict]:
    path = (RUNS_DIR / run_id / "judge_results.json") if run_id else CURRENT_JUDGE
    data = _read_json(path)
    return data.get("results", []) if data else []


def load_runs() -> list[dict]:
    data = _read_json(RUNS_INDEX)
    return data.get("runs", []) if data else []


def get_run(run_id: str) -> Optional[dict]:
    return next((r for r in load_runs() if r["config"]["run_id"] == run_id), None)


# ── Generation ────────────────────────────────────────────────────────────────


def generate_responses(
    cases: list[dict],
    domains: dict,
    config: RunConfig,
    api_key: Optional[str] = None,
    progress=None,
) -> tuple[list[dict], list[dict]]:
    """
    Route each case, build its governed context, call the model, and run the
    deterministic suite over the result.

    Returns (responses, deterministic_records). A model error is recorded on the
    response rather than raising, so one bad call does not discard the whole run.
    """
    router_fn = ROUTER_VERSIONS[config.router_version]
    responses, deterministic = [], []

    for i, case in enumerate(cases, 1):
        routing = router_fn(case["question"], domains)

        # The case declares which skill supplies the governed context. Where it
        # declares none, the router's choice is used and recorded as such.
        declared = case.get("governed_context", {}).get("skill")
        context_skill = declared if declared in domains else routing.domain
        domain_data = domains.get(context_skill, {})
        context = build_context(domain_data) if domain_data else ""

        result = ask_groq(
            question=case["question"],
            context=context,
            api_key=api_key,
            model=config.model_version,
            prompt_version=config.system_prompt_version,
            temperature=config.temperature,
        )

        record = {
            "eval_id": case["eval_id"],
            "question": case["question"],
            "answer": result["answer"],
            "error": result["error"],
            "model": result["model"],
            "prompt_version": result["prompt_version"],
            "latency_seconds": result.get("latency_seconds"),
            "routed_domain": routing.domain,
            "routing_confidence": round(routing.confidence, 4),
            "routing_ambiguous": routing.is_ambiguous,
            "routing_no_match": routing.no_match,
            "routing_tie": routing.is_tie,
            "router_version": routing.version,
            "context_skill": context_skill,
            "context_source": "case_declared" if declared in domains else "router_selected",
            "run_id": config.run_id,
        }
        responses.append(record)

        if not result["error"]:
            deterministic.append(run_deterministic_suite(
                case=case,
                answer=result["answer"],
                context=context,
                domain_data=domain_data,
                routed_domain=routing.domain,
                routing_confidence=routing.confidence,
            ))

        msg = f"[{i}/{len(cases)}] {case['eval_id']} → {routing.domain}"
        logger.info(msg + (" (MODEL ERROR)" if result["error"] else ""))
        if progress:
            progress(i, len(cases), case["eval_id"])

    return responses, deterministic


# ── Judging ───────────────────────────────────────────────────────────────────


# Groq's free tier caps tokens per minute, and a judge call costs roughly 4,800 of
# them. Firing calls back to back means every one after the first is rejected, and
# reactive backoff then wastes more time than it saves. Pacing proactively to just
# under the sustainable rate turns a 75-minute run of retries into a predictable one.
JUDGE_PACE_SECONDS = 36.0


def judge_all(
    cases_by_id: dict,
    responses: list[dict],
    domains: dict,
    config: RunConfig,
    api_key: Optional[str] = None,
    progress=None,
    pace_seconds: float = JUDGE_PACE_SECONDS,
    keep: Optional[list[dict]] = None,
) -> list[dict]:
    """
    Run the LLM judge over every successfully generated response.

    `keep` carries forward judgements from a previous pass; any case already judged
    successfully is skipped. A full 60-case pass costs more tokens than a free-tier
    daily budget allows, so re-judging work that already succeeded is not merely
    wasteful — it makes completing the set impossible.
    """
    # Only successful judgements carry forward. Keeping the failures too would leave a
    # stale failed record beside each newly successful one for the same case, inflating
    # the denominator and reporting a parse success rate that never happened.
    results = [r for r in (keep or []) if r.get("parse_ok")]
    already = {r["eval_id"] for r in results}
    usable = [r for r in responses if not r.get("error") and r["eval_id"] not in already]
    if already:
        logger.info(f"reusing {len(already)} existing judgements, judging {len(usable)} remaining")
    last_call_at = 0.0

    for i, resp in enumerate(usable, 1):
        if pace_seconds and last_call_at:
            elapsed = time.monotonic() - last_call_at
            if elapsed < pace_seconds:
                time.sleep(pace_seconds - elapsed)
        last_call_at = time.monotonic()

        case = cases_by_id.get(resp["eval_id"])
        if not case:
            continue
        domain_data = domains.get(resp.get("context_skill"), {})
        context = build_context(domain_data) if domain_data else ""

        record = judge_response(
            case=case,
            response_text=resp["answer"],
            governed_context=context,
            api_key=api_key,
            model=config.judge_model,
        )
        record["run_id"] = config.run_id
        results.append(record)

        logger.info(
            f"[{i}/{len(usable)}] judged {resp['eval_id']}: "
            + (f"overall {record['overall_score']}" if record["parse_ok"]
               else f"FAILED ({record['parse_error']})")
        )
        if progress:
            progress(i, len(usable), resp["eval_id"])

    return results


# ── Summary ───────────────────────────────────────────────────────────────────


def summarise_run(
    config: RunConfig,
    cases: list[dict],
    responses: list[dict],
    deterministic: list[dict],
    judge_results: list[dict],
    annotations: list[dict],
    domains: dict,
) -> dict:
    """
    Compute every headline metric for a run from its records.

    Nothing here is hard-coded. Where a statistic cannot be computed from the data
    available, it is None and must render as "not defined".
    """
    router_fn = ROUTER_VERSIONS[config.router_version]
    routing = run_golden_routing_eval(lambda q, d: router_fn(q, d), domains, cases)
    det = deterministic_summary(deterministic)
    judge = judge_summary(judge_results)
    align = alignment_by_provenance(annotations, judge_results)
    cov = coverage(annotations, judge_results, len(cases))

    latencies = [r["latency_seconds"] for r in responses
                 if r.get("latency_seconds") is not None and not r.get("error")]

    human_pairs = align["human_only"]
    demo_pairs = align["demo_only"]

    return {
        "generation": {
            "n_cases": len(cases),
            "n_responses": len(responses),
            "n_errors": sum(1 for r in responses if r.get("error")),
            "mean_latency_seconds": sum(latencies) / len(latencies) if latencies else None,
            "mean_answer_chars": (
                sum(len(r["answer"]) for r in responses if not r.get("error"))
                / max(sum(1 for r in responses if not r.get("error")), 1)
            ) if responses else None,
        },
        "routing": {
            "accuracy": routing["accuracy"],
            "correct": routing["correct"],
            "total": routing["total"],
            "n_excluded_ambiguous": routing["n_excluded"],
            "no_match_count": sum(1 for r in responses if r.get("routing_no_match")),
            "tie_count": sum(1 for r in responses if r.get("routing_tie")),
            "silent_misroutes": sum(
                1 for r in routing["results"]
                if not r["correct"] and not r["is_ambiguous"]
            ),
        },
        "deterministic": det,
        "judge": judge,
        "alignment": {
            "human_only": _slim_alignment(human_pairs),
            "demo_only": _slim_alignment(demo_pairs),
            "combined": _slim_alignment(align["combined"]),
        },
        "coverage": cov,
        "failure_distribution": {
            "judge": failure_distribution([r.get("failure_mode") for r in judge_results if r.get("parse_ok")]),
            "human": failure_distribution([
                a.get("failure_mode") for a in annotations if a.get("rater_type") == "human"
            ]),
            "demo_profiles": failure_distribution([
                a.get("failure_mode") for a in annotations if a.get("rater_type") == "demo_profile"
            ]),
        },
    }


def _slim_alignment(a: dict) -> dict:
    """The alignment fields worth storing in a run summary — full detail is recomputed live."""
    if not a or a.get("n", 0) == 0:
        return {"n": 0}
    return {
        "n": a["n"],
        "human_mean_score": a["human_mean_score"],
        "judge_mean_score": a["judge_mean_score"],
        "score_gap": a["score_gap"],
        "human_pass_rate": a["human_pass_rate"],
        "judge_pass_rate": a["judge_pass_rate"],
        "exact_agreement": a["exact_agreement"],
        "within_1_agreement": a["within_1_agreement"],
        "pass_agreement": a["pass_agreement"],
        "kappa_pass": a["kappa_pass"],
        "pearson": a["pearson"],
        "spearman": a["spearman"],
        "disagreement_rate": a["disagreement_rate"],
        "bias_direction": a["bias"]["direction"],
        "by_dimension": {
            d: {"within_1": v["within_1"], "kappa_quadratic": v["kappa_quadratic"],
                "human_mean": v["human_mean"], "judge_mean": v["judge_mean"], "n": v["n"]}
            for d, v in a["by_dimension"].items()
        },
    }


def save_run(config: RunConfig, summary: dict, deterministic: list[dict]) -> None:
    """Append this run to the run index, keeping every prior run."""
    runs = load_runs()
    runs = [r for r in runs if r["config"]["run_id"] != config.run_id]
    runs.append({
        "config": asdict(config),
        "summary": summary,
        "deterministic_records": deterministic,
    })
    runs.sort(key=lambda r: r["config"]["timestamp"])
    _write_json(RUNS_INDEX, {
        "schema_version": 1,
        "note": (
            "Each run is a real execution of the pipeline against the golden set. "
            "artifact_kind states whether a run used live model calls."
        ),
        "runs": runs,
    })


# ── Regression comparison ─────────────────────────────────────────────────────

COMPARISON_METRICS = [
    ("Overall human score",        ["alignment", "human_only", "human_mean_score"],  "score",   "up"),
    ("Overall AI judge score",     ["judge", "mean_overall_score"],                  "score",   "up"),
    ("Human pass rate",            ["alignment", "human_only", "human_pass_rate"],   "percent", "up"),
    ("AI judge pass rate",         ["judge", "pass_rate"],                           "percent", "up"),
    ("Human ↔ AI agreement (±1)",  ["alignment", "human_only", "within_1_agreement"], "percent", "up"),
    ("Human ↔ AI pass agreement",  ["alignment", "human_only", "pass_agreement"],    "percent", "up"),
    ("Critical failure rate",      ["judge", "critical_failure_rate"],               "percent", "down"),
    ("Groundedness (judge mean)",  ["judge", "by_dimension", "groundedness"],        "score",   "up"),
    ("Relevance (judge mean)",     ["judge", "by_dimension", "relevance"],           "score",   "up"),
    ("Correctness (judge mean)",   ["judge", "by_dimension", "correctness"],         "score",   "up"),
    ("Instruction following (judge mean)", ["judge", "by_dimension", "instruction_following"], "score", "up"),
    ("Routing accuracy",           ["routing", "accuracy"],                          "percent", "up"),
    ("Silent misroutes",           ["routing", "silent_misroutes"],                  "count",   "down"),
    ("Required-fact pass rate",    ["deterministic", "required_facts_pass_rate"],    "percent", "up"),
    ("Numeric grounding pass rate", ["deterministic", "numeric_grounding_pass_rate"], "percent", "up"),
    ("Unsupported claim rate",     ["deterministic", "unsupported_claim_rate"],      "percent", "down"),
    ("Forbidden claim violation rate", ["deterministic", "forbidden_claims_violation_rate"], "percent", "down"),
    ("Deterministic gate pass rate", ["deterministic", "verdict_pass_rate"],         "percent", "up"),
    ("Judge parse success rate",   ["judge", "parse_success_rate"],                  "percent", "up"),
]


def _dig(obj: dict, path: list[str]):
    for key in path:
        if not isinstance(obj, dict) or key not in obj:
            return None
        obj = obj[key]
    return obj if isinstance(obj, (int, float)) else None


def compare_runs(baseline: dict, current: dict) -> dict:
    """
    Compare two stored runs metric by metric.

    `direction` records whether up is good for that metric, so the UI can mark an
    improvement or a regression without re-deriving the semantics. A metric missing
    from either run is reported as not comparable rather than treated as zero.
    """
    rows = []
    for label, path, fmt, better in COMPARISON_METRICS:
        b = _dig(baseline["summary"], path)
        c = _dig(current["summary"], path)
        if b is None or c is None:
            rows.append({
                "metric": label, "format": fmt, "better": better,
                "baseline": b, "current": c, "delta": None,
                "verdict": "not comparable",
                "reason": "metric absent from one of the two runs",
            })
            continue
        delta = c - b
        if abs(delta) < 1e-9:
            verdict = "unchanged"
        elif (delta > 0) == (better == "up"):
            verdict = "improved"
        else:
            verdict = "regressed"
        rows.append({
            "metric": label, "format": fmt, "better": better,
            "baseline": b, "current": c, "delta": delta, "verdict": verdict, "reason": "",
        })

    counted = [r for r in rows if r["verdict"] in ("improved", "regressed", "unchanged")]
    return {
        "baseline_config": baseline["config"],
        "current_config": current["config"],
        "config_diff": _config_diff(baseline["config"], current["config"]),
        "rows": rows,
        "n_improved": sum(1 for r in counted if r["verdict"] == "improved"),
        "n_regressed": sum(1 for r in counted if r["verdict"] == "regressed"),
        "n_unchanged": sum(1 for r in counted if r["verdict"] == "unchanged"),
        "n_not_comparable": sum(1 for r in rows if r["verdict"] == "not comparable"),
    }


def _config_diff(a: dict, b: dict) -> list[dict]:
    """What actually changed between two runs — the independent variable of the experiment."""
    tracked = ["model_version", "system_prompt_version", "router_version",
               "judge_version", "judge_model", "dataset_version", "temperature"]
    return [
        {"field": f, "baseline": a.get(f), "current": b.get(f)}
        for f in tracked if a.get(f) != b.get(f)
    ]


# ── Orchestration ─────────────────────────────────────────────────────────────


def run_full(
    config: Optional[RunConfig] = None,
    limit: Optional[int] = None,
    do_judge: bool = True,
    api_key: Optional[str] = None,
    progress=None,
) -> dict:
    """Execute a complete evaluation run and persist every artifact."""
    domains = load_domains(str(ROOT))
    cases = golden_cases()
    if limit:
        cases = cases[:limit]

    config = config or new_run_config()
    config.n_cases = len(cases)
    cases_by_id = {c["eval_id"]: c for c in cases}

    key = get_api_key(api_key)
    if not key:
        raise RuntimeError(
            "No GROQ_API_KEY found. Set it in your shell:\n"
            "    export GROQ_API_KEY='...'\n"
            "or add it to .streamlit/secrets.toml (which is gitignored)."
        )

    logger.info(f"=== run {config.run_id} — {len(cases)} cases ===")
    logger.info(f"    model={config.model_version} prompt={config.system_prompt_version} "
                f"router={config.router_version}")

    responses, deterministic = generate_responses(cases, domains, config, key, progress)
    det_by_id = {d["eval_id"]: d for d in deterministic}

    run_dir = RUNS_DIR / config.run_id
    responses_payload = {"config": asdict(config), "responses": responses}
    _write_json(run_dir / "model_responses.json", responses_payload)
    _write_json(CURRENT_RESPONSES, responses_payload)

    judge_results = []
    if do_judge:
        judge_results = judge_all(cases_by_id, responses, domains, config, key, progress)
        judge_payload = {"config": asdict(config), "results": judge_results}
        _write_json(run_dir / "judge_results.json", judge_payload)
        _write_json(CURRENT_JUDGE, judge_payload)

    # Demo-profile annotations are regenerated for this run's responses; any human
    # annotations already on disk are preserved untouched.
    existing = load_annotations()
    human_annotations = [a for a in existing if a.get("rater_type") == "human"]
    demo_annotations = annotate_all(cases_by_id, responses, det_by_id, config.timestamp)
    annotations = human_annotations + demo_annotations
    save_annotations(annotations)
    logger.info(f"annotations: {len(human_annotations)} human preserved, "
                f"{len(demo_annotations)} demo-profile regenerated")

    summary = summarise_run(config, cases, responses, deterministic, judge_results,
                            annotations, domains)
    save_run(config, summary, deterministic)

    return {"config": config, "summary": summary, "responses": responses,
            "deterministic": deterministic, "judge_results": judge_results}


# ── CLI ───────────────────────────────────────────────────────────────────────


def _fmt(value, fmt: str) -> str:
    if value is None:
        return "n/a"
    if fmt == "percent":
        return f"{value * 100:.1f}%"
    if fmt == "score":
        return f"{value:.2f}"
    return f"{value:g}"


def _print_summary(summary: dict) -> None:
    g, r, d, j = summary["generation"], summary["routing"], summary["deterministic"], summary["judge"]
    print(f"\n  Generation   {g['n_responses']} responses, {g['n_errors']} errors, "
          f"mean latency {_fmt(g['mean_latency_seconds'], 'score')}s")
    print(f"  Routing      {_fmt(r['accuracy'], 'percent')} accuracy "
          f"({r['correct']}/{r['total']}), {r['silent_misroutes']} silent misroutes, "
          f"{r['no_match_count']} no-match, {r['tie_count']} ties")
    print(f"  Determ.      gate pass {_fmt(d.get('verdict_pass_rate'), 'percent')}, "
          f"required facts {_fmt(d.get('required_facts_pass_rate'), 'percent')}, "
          f"grounding {_fmt(d.get('numeric_grounding_pass_rate'), 'percent')}")
    if j.get("n_parsed"):
        print(f"  Judge        mean {_fmt(j['mean_overall_score'], 'score')}/5, "
              f"pass {_fmt(j['pass_rate'], 'percent')}, "
              f"critical {_fmt(j['critical_failure_rate'], 'percent')}, "
              f"parsed {j['n_parsed']}/{j['n']}")
        print(f"  Judge self   arithmetic errors {_fmt(j['arithmetic_error_rate'], 'percent')}, "
              f"pass-rule errors {_fmt(j['pass_rule_error_rate'], 'percent')}, "
              f"off-taxonomy {_fmt(j['off_taxonomy_rate'], 'percent')}")
    cov = summary["coverage"]
    print(f"  Coverage     human {cov['human_annotated']}/{cov['total_cases']}, "
          f"demo {cov['demo_annotated']}/{cov['total_cases']}, "
          f"judge {cov['judge_evaluated']}/{cov['total_cases']}")
    ha = summary["alignment"]["human_only"]
    if ha.get("n"):
        print(f"  Human↔AI     n={ha['n']}, ±1 agreement {_fmt(ha['within_1_agreement'], 'percent')}, "
              f"pass agreement {_fmt(ha['pass_agreement'], 'percent')}, "
              f"bias {ha['bias_direction']}")
    else:
        print("  Human↔AI     no human annotations yet — rate cases in the Human Evaluation page")


def _print_comparison(cmp: dict) -> None:
    print(f"\n  {cmp['baseline_config']['run_id']}  →  {cmp['current_config']['run_id']}")
    if cmp["config_diff"]:
        print("\n  What changed:")
        for c in cmp["config_diff"]:
            print(f"    {c['field']}: {c['baseline']} → {c['current']}")
    else:
        print("\n  Configuration identical between runs.")
    print(f"\n  {'Metric':38s} {'Baseline':>10s} {'Current':>10s} {'Delta':>10s}  Verdict")
    print("  " + "─" * 84)
    for row in cmp["rows"]:
        if row["verdict"] == "not comparable":
            print(f"  {row['metric']:38s} {'n/a':>10s} {'n/a':>10s} {'':>10s}  not comparable")
            continue
        delta = row["delta"]
        delta_str = (f"{delta * 100:+.1f}pp" if row["format"] == "percent"
                     else f"{delta:+.2f}" if row["format"] == "score" else f"{delta:+g}")
        mark = {"improved": "▲", "regressed": "▼", "unchanged": "="}[row["verdict"]]
        print(f"  {row['metric']:38s} {_fmt(row['baseline'], row['format']):>10s} "
              f"{_fmt(row['current'], row['format']):>10s} {delta_str:>10s}  {mark} {row['verdict']}")
    print(f"\n  {cmp['n_improved']} improved, {cmp['n_regressed']} regressed, "
          f"{cmp['n_unchanged']} unchanged, {cmp['n_not_comparable']} not comparable")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Run the golden-set evaluation pipeline and freeze the results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--all", action="store_true", help="full run: generate, evaluate, judge")
    p.add_argument("--responses", action="store_true", help="generate and evaluate, skip the judge")
    p.add_argument("--judge", action="store_true", help="judge an existing run's responses")
    p.add_argument("--list", action="store_true", help="list stored runs")
    p.add_argument("--compare", nargs=2, metavar=("BASELINE", "CURRENT"), help="compare two run ids")
    p.add_argument("--run-id", help="target run id for --judge")
    p.add_argument("--only-missing", action="store_true",
                   help="with --judge, only judge cases that have no successful judgement yet")
    p.add_argument("--label", default="", help="human-readable label for this run")
    p.add_argument("--notes", default="", help="what this run is testing")
    p.add_argument("--model", help=f"generator model (default {DEFAULT_MODEL})")
    p.add_argument("--judge-model", help="judge model (default: first available in the judge chain)")
    p.add_argument("--prompt-version", choices=["sysprompt-v1", "sysprompt-v2"],
                   help=f"system prompt version (default {DEFAULT_PROMPT_VERSION})")
    p.add_argument("--router-version", choices=sorted(ROUTER_VERSIONS),
                   help=f"router version (default {DEFAULT_ROUTER_VERSION})")
    p.add_argument("--limit", type=int, help="only run the first N cases (for a smoke test)")
    args = p.parse_args(argv)

    if args.list:
        runs = load_runs()
        if not runs:
            print("No runs stored. Run:  python eval_runner.py --all")
            return 0
        print(f"\n{len(runs)} stored run(s):\n")
        for r in runs:
            c, s = r["config"], r["summary"]
            j = s.get("judge", {})
            print(f"  {c['run_id']}")
            print(f"    {c['timestamp']}  {c.get('label') or '(no label)'}")
            print(f"    model={c['model_version']}  prompt={c['system_prompt_version']}  "
                  f"router={c['router_version']}")
            print(f"    cases={c['n_cases']}  routing={_fmt(s['routing']['accuracy'], 'percent')}  "
                  f"judge_mean={_fmt(j.get('mean_overall_score'), 'score')}\n")
        return 0

    if args.compare:
        base, curr = get_run(args.compare[0]), get_run(args.compare[1])
        if not base or not curr:
            missing = args.compare[0] if not base else args.compare[1]
            print(f"Run not found: {missing}\nUse --list to see stored run ids.")
            return 1
        _print_comparison(compare_runs(base, curr))
        return 0

    if args.judge:
        run_id = args.run_id
        if not run_id:
            runs = load_runs()
            if not runs:
                print("No stored runs to judge. Run --all or --responses first.")
                return 1
            run_id = runs[-1]["config"]["run_id"]
            print(f"No --run-id given; judging most recent run {run_id}")
        stored = get_run(run_id)
        if not stored:
            print(f"Run not found: {run_id}")
            return 1
        responses = load_responses(run_id)
        if not responses:
            print(f"No stored responses for {run_id}")
            return 1

        domains = load_domains(str(ROOT))
        cases = golden_cases()
        cases_by_id = {c["eval_id"]: c for c in cases}
        config = RunConfig(**stored["config"])
        if args.judge_model:
            config.judge_model = args.judge_model

        key = get_api_key()
        if not key:
            print("No GROQ_API_KEY found. export GROQ_API_KEY='...' and retry.")
            return 1

        existing = load_judge_results(run_id) if args.only_missing else None
        results = judge_all(cases_by_id, responses, domains, config, key, keep=existing)
        payload = {"config": asdict(config), "results": results}
        _write_json(RUNS_DIR / run_id / "judge_results.json", payload)
        _write_json(CURRENT_JUDGE, payload)

        annotations = load_annotations()
        summary = summarise_run(config, cases, responses,
                                stored.get("deterministic_records", []), results,
                                annotations, domains)
        save_run(config, summary, stored.get("deterministic_records", []))
        _print_summary(summary)
        return 0

    if not (args.all or args.responses):
        p.print_help()
        print("\nNothing to do. Start with:  python eval_runner.py --all")
        return 0

    config = new_run_config(
        label=args.label,
        notes=args.notes,
        model_version=args.model,
        judge_model=args.judge_model,
        system_prompt_version=args.prompt_version,
        router_version=args.router_version,
    )

    try:
        result = run_full(config=config, limit=args.limit, do_judge=args.all)
    except RuntimeError as e:
        print(f"\n{e}")
        return 1

    print(f"\n=== run {result['config'].run_id} complete ===")
    _print_summary(result["summary"])

    runs = load_runs()
    if len(runs) > 1:
        print(f"\n  Compare against the previous run:")
        print(f"    python eval_runner.py --compare {runs[-2]['config']['run_id']} "
              f"{result['config'].run_id}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
