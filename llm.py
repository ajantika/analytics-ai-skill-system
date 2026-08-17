"""
llm.py — LLM interaction for Analytics AI Skill System

Handles all Groq/Llama API calls.
Prompt engineering ensures structured output (Insight / Why it matters / Recommended action).
The LLM is instructed to stay grounded in the supplied context and not invent metrics.
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a senior analytics expert advising a business leadership team.

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


def ask_groq(question: str, context: str, api_key: Optional[str] = None) -> dict:
    """
    Send a question + governed context to the Groq/Llama model.

    Returns:
        dict with keys: answer (str), model (str), error (bool)
    """
    try:
        from groq import Groq

        key = api_key or os.environ.get("GROQ_API_KEY", "")
        if not key:
            return {
                "answer": "No GROQ_API_KEY configured. Add it to Streamlit secrets.",
                "model": "none",
                "error": True
            }

        client = Groq(api_key=key)
        user_message = f"""GOVERNED KNOWLEDGE BASE:
{context}

BUSINESS QUESTION: {question}

Answer using ONLY the figures from the knowledge base above. Follow the exact format specified."""

        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            max_tokens=500,
            temperature=0.2
        )
        raw = response.choices[0].message.content.strip()
        logger.info(f"LLM response generated ({len(raw)} chars)")
        return {"answer": raw, "model": "llama3-70b-8192", "error": False}

    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return {
            "answer": f"Model call failed: {str(e)}",
            "model": "none",
            "error": True
        }


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
        low = stripped.lower()

        if "**insight**" in low or stripped.startswith("Insight"):
            if current and lines:
                sections[current] = " ".join(lines).strip()
            current = "insight"
            lines = []
        elif "**why it matters**" in low or stripped.startswith("Why it matters"):
            if current and lines:
                sections[current] = " ".join(lines).strip()
            current = "why_it_matters"
            lines = []
        elif "**recommended action**" in low or stripped.startswith("Recommended action"):
            if current and lines:
                sections[current] = " ".join(lines).strip()
            current = "recommended_action"
            lines = []
        elif stripped and current:
            # Strip leading ** from content lines
            clean = stripped.lstrip("*").strip()
            if clean:
                lines.append(clean)

    if current and lines:
        sections[current] = " ".join(lines).strip()

    # If parsing failed, put everything in insight
    if not sections["insight"] and not sections["why_it_matters"]:
        sections["insight"] = raw_answer

    return sections
