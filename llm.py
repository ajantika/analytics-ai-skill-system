"""
llm.py — LLM interaction for Analytics AI Skill System

Handles all Groq API calls and owns the versioned system prompts.

PROMPTS ARE VERSIONED, AND OLD VERSIONS ARE KEPT.
v1 is the original prompt. v2 is a revision driven by specific findings from the
golden-set evaluation, not by intuition:

  Finding: every instruction_following case failed. The v1 prompt hard-codes a
    three-section format with "use these exact headers", which overrides an explicit
    user instruction such as "answer in one sentence with no recommendation".
  Fix in v2: the template becomes the default for open-ended questions and yields to
    explicit user formatting instructions.

  Finding: missing_context cases produced fabricated figures. The v1 prompt demands
    an Insight and a Recommended action for every question, which pressures the model
    to manufacture both when the governed layer holds neither.
  Fix in v2: an explicit "not in the governed layer" outcome is permitted and shown.

  Finding: adversarial cases with false premises were accepted and elaborated.
  Fix in v2: an explicit instruction to check the premise against the context first.

Both versions stay callable so the regression page can measure whether v2 actually
helped rather than assuming it did.
"""

import logging
import os
import re
import time
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "openai/gpt-oss-20b"
DEFAULT_PROMPT_VERSION = "sysprompt-v2"


SYSTEM_PROMPT_V1 = """You are a senior analytics expert advising a business leadership team.

You have access to a governed knowledge base with REAL, SPECIFIC numbers.
You MUST answer using ONLY the exact numbers, percentages, and figures in the knowledge base.
Do NOT invent numbers, benchmarks, or metrics that are not in the knowledge base.
Do NOT generate SQL. Do NOT say "calculate this". Do NOT give generic steps.

Structure every response in exactly this format — use these exact headers:

**Insight**
[1–2 sentences: the concise analytical finding with relevant numbers from the knowledge base]

**Why it matters**
[1–2 sentences: the business implication of this finding]

**Recommended action**
[1–2 sentences: the next analytical or business action, grounded in the available data. If the data does not support a specific recommendation, say so explicitly.]

Be direct and confident. Use the actual figures provided."""


SYSTEM_PROMPT_V2 = """You are a senior analytics expert advising a business leadership team.

You have a governed knowledge base containing specific, validated figures. It is your
only source of truth.

GROUNDING
- Use only figures that appear in the governed knowledge base. Reuse them verbatim.
- Never invent a number, an industry benchmark, or a metric definition. If a benchmark
  is not stated in the knowledge base, you do not have one.
- If you calculate something from the supplied figures, say so and show the inputs.
- Do not generate SQL and do not describe how one might calculate the answer.

CHECK THE PREMISE FIRST
If the question asserts a figure or a trend, verify it against the knowledge base before
answering. If the assertion is wrong, correct it plainly and then answer. Never accept a
stated figure that contradicts the governed knowledge base.

WHEN THE ANSWER IS NOT IN THE KNOWLEDGE BASE
If the knowledge base does not contain what was asked — a historical period, a metric
that is not defined, an individual customer or employee record, a breakdown that does
not exist — say so directly and name what is available instead. This is a complete and
correct answer. Do not estimate, do not substitute a related figure, and do not fill the
gap with general business knowledge.

Equally: if the knowledge base DOES contain what was asked, answer it fully. Hedging or
declining when the figures are present is a failure, not caution.

SEPARATE EVIDENCE FROM INFERENCE
State findings that come from the knowledge base as findings. Anything you conclude
beyond it — a cause, a driver, an explanation — must be marked as inference or
hypothesis, in those words. The knowledge base contains outcomes, not causes.

INDIVIDUALS
The knowledge base names some employees and sales reps. You may report their governed
metrics. You must not recommend or justify employment actions about them; the data does
not support that and it is not an analytics output.

FORMAT
If the question specifies a format — a length, a table, "numbers only", "no
recommendation", "one sentence" — follow that instruction exactly and ignore the default
structure below.

Otherwise, use this default structure:

**Insight**
[1–2 sentences: the finding, with the relevant figures from the knowledge base]

**Why it matters**
[1–2 sentences: the business implication]

**Recommended action**
[1–2 sentences: the next action the data supports. If the data does not support a
specific recommendation, say that instead of inventing one.]

Be direct. Be concise. Confidence should match the evidence."""


PROMPT_VERSIONS = {
    "sysprompt-v1": SYSTEM_PROMPT_V1,
    "sysprompt-v2": SYSTEM_PROMPT_V2,
}

PROMPT_VERSION_NOTES = {
    "sysprompt-v1": (
        "Original prompt. Hard-codes a three-section response format with 'use these exact "
        "headers', and requires an Insight and a Recommended action for every question."
    ),
    "sysprompt-v2": (
        "Revision driven by golden-set findings. The response template becomes a default that "
        "explicit user formatting instructions override; an explicit 'not in the governed layer' "
        "outcome is permitted; premise-checking, evidence-versus-inference marking, and a "
        "boundary on recommendations about named individuals are added."
    ),
}

# Retained for backward compatibility with any existing import of SYSTEM_PROMPT.
SYSTEM_PROMPT = SYSTEM_PROMPT_V1


def get_api_key(explicit: Optional[str] = None) -> str:
    """Resolve the Groq key from an explicit argument, the environment, then Streamlit secrets."""
    if explicit:
        return explicit
    env = os.environ.get("GROQ_API_KEY", "")
    if env:
        return env
    try:
        import streamlit as st
        return st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        return ""


def ask_groq(
    question: str,
    context: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    prompt_version: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 500,
) -> dict:
    """
    Send a question plus governed context to the model.

    Returns a dict with: answer, model, prompt_version, error, latency_seconds.
    The version and model are returned rather than assumed so that every stored
    evaluation record states exactly which configuration produced it.
    """
    version = prompt_version or DEFAULT_PROMPT_VERSION
    system_prompt = PROMPT_VERSIONS.get(version)
    if system_prompt is None:
        raise ValueError(f"unknown prompt version {version!r}; available: {sorted(PROMPT_VERSIONS)}")

    model_name = model or DEFAULT_MODEL

    try:
        from groq import Groq

        key = get_api_key(api_key)
        if not key:
            return {
                "answer": "No GROQ_API_KEY configured. Add it to the environment or Streamlit secrets.",
                "model": "none", "prompt_version": version, "error": True,
                "latency_seconds": None,
            }

        client = Groq(api_key=key)
        user_message = (
            f"GOVERNED KNOWLEDGE BASE:\n{context}\n\n"
            f"BUSINESS QUESTION: {question}\n\n"
            "Answer using only the figures from the knowledge base above."
        )

        started = time.perf_counter()
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        latency = time.perf_counter() - started
        raw = (response.choices[0].message.content or "").strip()
        logger.info(f"LLM response generated ({len(raw)} chars, {latency:.2f}s, {version})")
        return {
            "answer": raw,
            "model": model_name,
            "prompt_version": version,
            "error": False,
            "latency_seconds": round(latency, 3),
        }

    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        kind, message = _classify_error(e)
        return {
            "answer": message,
            "model": model_name, "prompt_version": version, "error": True,
            "error_kind": kind, "error_detail": str(e),
            "latency_seconds": None,
        }


_RETRY_AFTER = re.compile(r"try again in ([\dhms.]+)", re.I)
_USAGE = re.compile(r"Limit (\d+), Used (\d+)", re.I)


def _classify_error(e: Exception) -> tuple[str, str]:
    """
    Turn a provider exception into something a reader can act on.

    Rate-limit errors arrive as a wall of JSON naming an org id, a service tier and
    a token count. Rendering that verbatim tells the reader nothing about what went
    wrong or what to do, so the three cases worth distinguishing are separated here
    and everything else keeps its original text.
    """
    text = str(e)
    lowered = text.lower()

    if "rate_limit" in lowered or "429" in text:
        retry = _RETRY_AFTER.search(text)
        usage = _USAGE.search(text)
        detail = ""
        if usage:
            limit, used = int(usage.group(1)), int(usage.group(2))
            detail = f" {used:,} of {limit:,} tokens used."
        wait = f" Capacity returns in about {retry.group(1)}." if retry else ""
        if "tpd" in lowered or "per day" in lowered:
            return "quota_exhausted", (
                "The daily token budget for this Groq account is used up, so no new answers "
                f"can be generated right now.{detail}{wait} "
                "Everything else in this application reads stored evaluation records and "
                "works normally."
            )
        return "rate_limited", (
            "Too many requests to the model in a short window."
            f"{wait} Try that question again in a moment."
        )

    if "model_not_found" in lowered or "does not exist" in lowered:
        return "model_unavailable", (
            "The configured model is not available on this account. Check the model id in "
            "llm.py against the models your Groq key can reach."
        )

    if "authentication" in lowered or "invalid api key" in lowered or "401" in text:
        return "auth_failed", (
            "The Groq API key was rejected. Check GROQ_API_KEY in your environment or in "
            ".streamlit/secrets.toml."
        )

    return "unknown", f"The model call failed: {text}"


def parse_structured_answer(raw_answer: str) -> dict:
    """
    Parse the structured LLM output into sections.

    Returns dict with keys: insight, why_it_matters, recommended_action, raw
    Falls back gracefully if the model doesn't follow the format exactly.
    """
    sections = {"insight": "", "why_it_matters": "", "recommended_action": "", "raw": raw_answer}

    current = None
    lines = []

    for line in raw_answer.split("\n"):
        stripped = line.strip()
        low = stripped.lower().lstrip("*# ").rstrip("*: ")

        if low in ("insight", "answer", "summary", "finding"):
            if current and lines:
                sections[current] = " ".join(lines).strip()
            current = "insight"
            lines = []
        elif low in ("why it matters", "why", "business impact", "impact", "implication"):
            if current and lines:
                sections[current] = " ".join(lines).strip()
            current = "why_it_matters"
            lines = []
        elif low in ("recommended action", "recommendation", "action", "next steps", "next step"):
            if current and lines:
                sections[current] = " ".join(lines).strip()
            current = "recommended_action"
            lines = []
        elif stripped and current:
            clean = stripped.lstrip("*#").strip()
            if clean:
                lines.append(clean)

    if current and lines:
        sections[current] = " ".join(lines).strip()

    # If parsing failed, put everything in insight
    if not sections["insight"] and not sections["why_it_matters"]:
        sections["insight"] = raw_answer

    return sections
