"""Request models for the API (responses are returned as plain dicts)."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    model_name: str | None = Field(default=None, description="Which trained model to use; defaults to best.")
    values: dict[str, Any] = Field(default_factory=dict, description="Raw feature values, or {'symptoms': [...]}.")


class ChatRequest(BaseModel):
    message: str
    symptoms: list[str] = Field(default_factory=list, description="Symptoms accumulated so far in the chat.")
