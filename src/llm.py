"""Groq LLM helpers for the symptom chatbot.

Two responsibilities, kept separate so the model prediction sits between them:
1. ``extract_symptoms`` — map free-text to canonical symptom tokens (JSON mode,
   constrained to the known vocabulary).
2. ``narrate_results`` — turn the symptom-checker's ranked predictions into a
   friendly, safety-aware explanation.

The API key is read from the environment (loaded from a git-ignored .env).
"""
from __future__ import annotations

import json
import os
from functools import lru_cache

from dotenv import load_dotenv

from .config import PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


class LLMNotConfigured(RuntimeError):
    """Raised when no Groq API key is available."""


def is_configured() -> bool:
    return bool(os.getenv("GROQ_API_KEY"))


@lru_cache(maxsize=1)
def _client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise LLMNotConfigured(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    from groq import Groq

    return Groq(api_key=api_key)


def _chat(messages, *, model=None, temperature=0.2, json_mode=False, max_tokens=1024) -> str:
    kwargs = {
        "model": model or DEFAULT_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = _client().chat.completions.create(**kwargs)
    return resp.choices[0].message.content


def extract_symptoms(user_text: str, vocab: list[str], current: list[str] | None = None) -> list[str]:
    """Return canonical symptom tokens (a subset of ``vocab``) implied by the text.

    Merges with any ``current`` symptoms so the conversation accumulates context.
    """
    current = current or []
    system = (
        "You are a clinical intake assistant. Map the patient's free-text description to "
        "symptom identifiers from the ALLOWED list only. Never invent identifiers. "
        "Match synonyms and lay terms (e.g. 'throwing up' -> 'vomiting', 'temperature' -> "
        "'high_fever'). Merge newly mentioned symptoms with the ones already selected. "
        'Respond ONLY as JSON: {"symptoms": ["id1", "id2", ...]}.'
    )
    user = (
        f"ALLOWED symptom identifiers:\n{', '.join(vocab)}\n\n"
        f"Already selected: {', '.join(current) if current else '(none)'}\n\n"
        f"Patient says: \"{user_text}\"\n\n"
        "Return the full merged symptom list as JSON."
    )
    raw = _chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        json_mode=True,
        temperature=0.0,
    )
    try:
        data = json.loads(raw)
        found = data.get("symptoms", [])
    except (json.JSONDecodeError, AttributeError):
        found = []

    allowed = set(vocab)
    merged: list[str] = list(current)
    for s in found:
        token = str(s).strip()
        if token in allowed and token not in merged:
            merged.append(token)
    return merged


def explain_risk(disease_label: str, probability: float, factors: list[dict]) -> str:
    """Explain a binary risk prediction in plain language.

    ``factors`` is a list of {"feature": str, "contribution": float}, where a positive
    contribution pushed the prediction toward the disease and negative pushed away.
    """
    risk = "higher" if probability >= 0.5 else "lower"
    factor_lines = []
    for f in factors:
        direction = "raises" if f["contribution"] > 0 else "lowers"
        factor_lines.append(f"- {f['feature']} ({direction} the estimated risk)")
    factor_text = "\n".join(factor_lines) or "- (no dominant factors identified)"

    system = (
        "You are a careful, warm health-information assistant. You are NOT a doctor and must "
        "not diagnose. Explain a machine-learning risk estimate in plain language: what the "
        "probability means, which patient factors drove it (using the provided directions), "
        "and practical, non-alarming next steps. Note these are statistical estimates from a "
        "public dataset. Be concise (up to ~140 words) and end with a one-line disclaimer. "
        "Do not invent factors or numbers beyond those provided."
    )
    user = (
        f"Condition: {disease_label}\n"
        f"Model-estimated probability of {disease_label}: {probability:.0%} "
        f"(this indicates {risk} risk).\n\n"
        f"Most influential factors for this patient:\n{factor_text}\n\n"
        "Write the explanation now."
    )
    return _chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.4,
        max_tokens=400,
    )


def narrate_results(symptoms: list[str], predictions: list[dict], disease_info: dict) -> str:
    """Explain the ranked predictions conversationally, with a safety disclaimer.

    ``predictions`` is a list of {"disease": str, "probability": float}, best first.
    """
    pretty_symptoms = ", ".join(s.replace("_", " ") for s in symptoms) or "none provided"
    lines = []
    for p in predictions:
        info = disease_info.get(p["disease"], {})
        desc = info.get("description", "")
        precs = "; ".join(info.get("precautions", []))
        lines.append(
            f"- {p['disease']} ({p['probability']:.0%} model confidence). "
            f"Description: {desc or 'n/a'}. Suggested precautions: {precs or 'n/a'}."
        )
    context = "\n".join(lines)

    system = (
        "You are a careful, warm health-information assistant. You are NOT a doctor and must "
        "not diagnose. Explain the model's output in plain language, note that these are "
        "statistical estimates from a small dataset, and encourage seeing a clinician. "
        "Be concise (up to ~150 words). Do not add symptoms or conditions beyond those given."
    )
    user = (
        f"The patient reported these symptoms: {pretty_symptoms}.\n\n"
        f"A machine-learning symptom checker produced these ranked possibilities:\n{context}\n\n"
        "Write a brief, reassuring explanation of what this might suggest, mention the most "
        "likely possibility and its precautions, and remind them to consult a healthcare "
        "professional. End with a one-line disclaimer."
    )
    return _chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.4,
        max_tokens=512,
    )
