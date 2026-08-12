"""
skills.py — Skill loading and governed semantic layer

Loads YAML domain skill files and exposes metric definitions as a governed semantic layer.
The semantic layer ensures the LLM uses pre-defined, validated business metric definitions
rather than inventing KPI definitions at inference time.
"""

import os
import yaml
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def load_domains(directory: str = ".") -> dict:
    """Load all YAML domain skill files from the given directory."""
    domains = {}
    for filename in os.listdir(directory):
        if not filename.endswith(".yaml"):
            continue
        path = os.path.join(directory, filename)
        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict) and "domain" in data:
                key = str(data["domain"]).strip().lower().replace(" ", "_")
                data["domain"] = key
                domains[key] = data
                logger.debug(f"Loaded domain skill: {key} ({filename})")
        except Exception as e:
            logger.warning(f"Failed to load {filename}: {e}")
    return domains


def get_metric(domain_data: dict, metric_name: str) -> Optional[dict]:
    """Return the full metric metadata dict for a given metric name (case-insensitive)."""
    for m in domain_data.get("metrics", []):
        if m.get("name", "").lower() == metric_name.lower():
            return m
    return None


def get_all_metric_names(domain_data: dict) -> list[str]:
    """Return all metric names defined in a domain skill."""
    return [m.get("name", "") for m in domain_data.get("metrics", [])]


def build_context(domain_data: dict) -> str:
    """
    Build the context string injected into the LLM prompt.
    Includes domain description, full metric definitions, and sample Q&A.
    This is the governed semantic layer — the LLM is grounded in these definitions.
    """
    ctx = f"Domain: {domain_data.get('domain', '').upper()}\n"
    ctx += f"Description: {domain_data.get('description', '')}\n\n"
    ctx += "Governed Metric Definitions:\n"

    for m in domain_data.get("metrics", []):
        ctx += f"\n  Metric: {m.get('name', '')}\n"
        if m.get("business_definition"):
            ctx += f"  Business Definition: {m['business_definition']}\n"
        elif m.get("definition"):
            ctx += f"  Definition: {m['definition']}\n"
        if m.get("formula"):
            ctx += f"  Formula: {m['formula']}\n"
        if m.get("interpretation"):
            ctx += f"  Interpretation: {m['interpretation']}\n"
        if m.get("dimensions"):
            dims = ", ".join(m["dimensions"]) if isinstance(m["dimensions"], list) else m["dimensions"]
            ctx += f"  Dimensions: {dims}\n"
        if m.get("owner"):
            ctx += f"  Owner: {m['owner']}\n"
        if m.get("source"):
            ctx += f"  Source: {m['source']}\n"
        if m.get("last_validated"):
            ctx += f"  Last Validated: {m['last_validated']}\n"

    ctx += "\nSample Q&A (grounded examples):\n"
    for qa in domain_data.get("sample_qa", []):
        ctx += f"  Q: {qa.get('q', '')}\n  A: {qa.get('a', '')}\n\n"

    return ctx


def get_follow_up_questions(domain_data: dict, question: str) -> list[str]:
    """
    Return contextual follow-up questions from the domain skill's follow_ups list,
    filtered by relevance to the original question. Falls back to first 3 domain follow-ups.
    Follow-ups only reference analyses the demo can actually perform.
    """
    all_followups = domain_data.get("follow_up_questions", [])
    if not all_followups:
        return []

    q_lower = question.lower()
    scored = []
    for fq in all_followups:
        fq_words = set(fq.lower().split())
        q_words = set(q_lower.split())
        overlap = len(fq_words & q_words)
        scored.append((overlap, fq))

    scored.sort(key=lambda x: x[0], reverse=True)
    # Return top 3, avoiding exact duplicates of the original question
    result = [fq for _, fq in scored if fq.lower().strip() != question.lower().strip()]
    return result[:3]


def metric_is_defined(metric_name: str, domain_data: dict) -> bool:
    """Check if a metric name is explicitly defined in the governed semantic layer."""
    defined = [m.get("name", "").lower() for m in domain_data.get("metrics", [])]
    return metric_name.lower() in defined


def get_metric_display(metric: dict) -> str:
    """Build a compact human-readable metric definition for display."""
    lines = []
    name = metric.get("name", "")
    if name:
        lines.append(f"**{name}**")
    defn = metric.get("business_definition") or metric.get("definition", "")
    if defn:
        lines.append(defn)
    if metric.get("formula"):
        lines.append(f"Formula: `{metric['formula']}`")
    if metric.get("interpretation"):
        lines.append(f"Interpretation: {metric['interpretation']}")
    if metric.get("owner"):
        lines.append(f"Owner: {metric['owner']}")
    if metric.get("source"):
        lines.append(f"Source: {metric['source']}")
    if metric.get("last_validated"):
        lines.append(f"Last validated: {metric['last_validated']}")
    return "\n\n".join(lines)
