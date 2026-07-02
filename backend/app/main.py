"""FastAPI application exposing the diagnostic models over a REST API."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from src.config import FIGURES_DIR  # noqa: E402

from . import engine  # noqa: E402
from .schemas import ChatRequest, PredictRequest  # noqa: E402

app = FastAPI(
    title="AI Healthcare Diagnosis API",
    description="Serves multi-disease ML models (symptom checker + diabetes, heart, "
    "cardiovascular, stroke) with predictions, explanations, and an LLM chatbot.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev: relax; tighten for production
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/figures", StaticFiles(directory=str(FIGURES_DIR)), name="figures")


def _require_slug(slug: str) -> None:
    if slug not in engine.available_slugs():
        raise HTTPException(status_code=404, detail=f"Model '{slug}' not found or not trained.")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "models": engine.available_slugs()}


@app.get("/api/models")
def get_models() -> list[dict]:
    return engine.list_models()


@app.get("/api/models/{slug}")
def get_model(slug: str) -> dict:
    _require_slug(slug)
    return engine.model_detail(slug)


@app.post("/api/models/{slug}/predict")
def post_predict(slug: str, req: PredictRequest) -> dict:
    _require_slug(slug)
    try:
        return engine.predict(slug, req.model_name, req.values)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc


@app.post("/api/models/{slug}/ai-explanation")
def post_ai_explanation(slug: str, req: PredictRequest) -> dict:
    _require_slug(slug)
    try:
        return engine.ai_explanation(slug, req.model_name, req.values)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Explanation failed: {exc}") from exc


@app.post("/api/chat")
def post_chat(req: ChatRequest) -> dict:
    if "symptoms" not in engine.available_slugs():
        raise HTTPException(status_code=404, detail="Symptom model not trained.")
    try:
        return engine.chat(req.message, req.symptoms)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Chat failed: {exc}") from exc
