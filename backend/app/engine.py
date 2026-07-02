"""Bridge between the FastAPI layer and the existing ``src/`` ML engine.

All heavy objects (metadata, preprocessors, models) are cached so repeated API
requests don't re-read artifacts from disk.
"""
from __future__ import annotations

from functools import lru_cache

from src import artifacts, llm
from src.config import DATASETS, FIGURES_DIR
from src.explain import local_explanation
from src.inference import _densify, _standardise_proba, build_row

TOP_CONTRIBUTIONS = 8
TOP_K_MULTICLASS = 3


# --- cached artifact loaders ---------------------------------------------
@lru_cache(maxsize=None)
def meta(slug: str) -> dict:
    return artifacts.load_metadata(slug)


@lru_cache(maxsize=None)
def metrics(slug: str) -> dict:
    return artifacts.load_metrics(slug)


@lru_cache(maxsize=None)
def _preprocessor(slug: str):
    return artifacts.load_preprocessor(slug)


@lru_cache(maxsize=None)
def _nn(slug: str):
    return artifacts.load_nn(slug)


@lru_cache(maxsize=None)
def _classical(slug: str, name: str):
    return artifacts.load_classical(slug, name)


# --- helpers --------------------------------------------------------------
def available_slugs() -> list[str]:
    return [s for s in DATASETS if artifacts.is_trained(s)]


def _figure_urls(slug: str) -> dict[str, str]:
    kinds = {
        "confusion_matrix": f"{slug}_confusion_matrix.png",
        "importance": f"{slug}_importance.png",
        "eda": f"{slug}_eda.png",
    }
    return {k: f"/figures/{name}" for k, name in kinds.items() if (FIGURES_DIR / name).exists()}


def list_models() -> list[dict]:
    out = []
    for slug in available_slugs():
        m = meta(slug)
        out.append(
            {
                "slug": slug,
                "label": m["label"],
                "task": m["task"],
                "input_mode": m["input_mode"],
                "best_model": m["best_model"],
                "n_samples": m["n_samples"],
            }
        )
    return out


def model_detail(slug: str) -> dict:
    m = meta(slug)
    return {
        "slug": slug,
        "label": m["label"],
        "task": m["task"],
        "input_mode": m["input_mode"],
        "target_names": m["target_names"],
        "positive_class": m.get("positive_class"),
        "models": m["models"],
        "best_model": m["best_model"],
        "notes": m["notes"],
        "n_samples": m["n_samples"],
        "n_features_transformed": m["n_features_transformed"],
        "feature_specs": m["feature_specs"],
        "symptom_vocab": m.get("symptom_vocab", []),
        "metrics": metrics(slug),
        "figures": _figure_urls(slug),
    }


def _proba(slug: str, model_name: str, values: dict):
    m = meta(slug)
    model_name = model_name or m["best_model"]
    row = build_row(m, values)
    xt = _densify(_preprocessor(slug).transform(row))
    if model_name == artifacts.NN_NAME:
        raw = _nn(slug).predict(xt, verbose=0)
    else:
        raw = _classical(slug, model_name).predict_proba(xt)
    return m, model_name, row, _standardise_proba(raw, m["task"])[0]


def predict(slug: str, model_name: str | None, values: dict, with_contributions: bool = True) -> dict:
    m, used_model, row, proba = _proba(slug, model_name, values)
    names = m["target_names"]
    order = sorted(range(len(proba)), key=lambda i: proba[i], reverse=True)
    k = TOP_K_MULTICLASS if m["task"] != "binary" else len(names)
    top = [{"label": names[i], "probability": float(proba[i])} for i in order[:k]]

    contributions = []
    if with_contributions:
        try:
            expl = local_explanation(slug, row, top_n=TOP_CONTRIBUTIONS)
            contributions = [
                {"feature": r["raw_feature"], "contribution": float(r["contribution"])}
                for _, r in expl.iterrows()
            ]
        except Exception:  # pragma: no cover - defensive
            contributions = []

    return {
        "slug": slug,
        "task": m["task"],
        "model": used_model,
        "target_names": names,
        "positive_class": m.get("positive_class"),
        "probabilities": [float(p) for p in proba],
        "top_predictions": top,
        "contributions": contributions,
    }


def ai_explanation(slug: str, model_name: str | None, values: dict) -> dict:
    if not llm.is_configured():
        return {"available": False, "text": "", "reason": "GROQ_API_KEY not configured on the server."}
    result = predict(slug, model_name, values, with_contributions=True)
    if result["task"] != "binary":
        return {"available": False, "text": "", "reason": "AI risk explanation applies to binary models only."}
    disease = result["positive_class"] or meta(slug)["label"]
    prob = result["probabilities"][1]
    factors = result["contributions"][:5]
    text = llm.explain_risk(disease, prob, factors)
    return {"available": True, "text": text, "reason": None}


def chat(message: str, symptoms: list[str]) -> dict:
    m = meta("symptoms")
    vocab = m["symptom_vocab"]
    if not llm.is_configured():
        return {
            "symptoms": symptoms,
            "predictions": [],
            "reply": "The chatbot needs a GROQ_API_KEY configured on the server.",
        }
    merged = llm.extract_symptoms(message, vocab, current=symptoms or [])
    if not merged:
        return {
            "symptoms": [],
            "predictions": [],
            "reply": ("I couldn't map that to any known symptoms yet. Could you describe what "
                      "you're feeling in more detail (e.g. fever, cough, rash)?"),
        }
    result = predict("symptoms", None, {"symptoms": merged}, with_contributions=False)
    preds = [{"disease": p["label"], "probability": p["probability"]} for p in result["top_predictions"]]
    reply = llm.narrate_results(merged, preds, m.get("disease_info", {}))
    return {"symptoms": merged, "predictions": preds, "reply": reply}
