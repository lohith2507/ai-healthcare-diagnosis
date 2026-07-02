"""Save/load helpers so training and the app agree on artifact layout.

Per dataset slug we persist:
    preprocessor.joblib        fitted sklearn ColumnTransformer
    clf_<name>.joblib          fitted classical estimators
    nn.keras                   fitted Keras model
    metadata.json             schema the app needs to build forms & interpret output
    metrics.json              evaluation metrics per model
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import joblib
from tensorflow import keras

from .config import model_dir

PREPROCESSOR = "preprocessor.joblib"
NN = "nn.keras"
METADATA = "metadata.json"
METRICS = "metrics.json"
NN_NAME = "neural_network"


def clf_filename(name: str) -> str:
    return f"clf_{name}.joblib"


def save_preprocessor(slug: str, preprocessor) -> None:
    joblib.dump(preprocessor, model_dir(slug) / PREPROCESSOR)


def load_preprocessor(slug: str):
    return joblib.load(model_dir(slug) / PREPROCESSOR)


def save_classical(slug: str, name: str, model) -> None:
    joblib.dump(model, model_dir(slug) / clf_filename(name))


def load_classical(slug: str, name: str):
    return joblib.load(model_dir(slug) / clf_filename(name))


def save_nn(slug: str, model) -> None:
    model.save(model_dir(slug) / NN)


def load_nn(slug: str):
    return keras.models.load_model(model_dir(slug) / NN)


def save_metadata(slug: str, metadata: dict) -> None:
    (model_dir(slug) / METADATA).write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_metadata(slug: str) -> dict:
    return json.loads((model_dir(slug) / METADATA).read_text(encoding="utf-8"))


def save_metrics(slug: str, metrics: dict) -> None:
    (model_dir(slug) / METRICS).write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def load_metrics(slug: str) -> dict:
    return json.loads((model_dir(slug) / METRICS).read_text(encoding="utf-8"))


def feature_specs_to_dicts(feature_specs) -> list[dict]:
    return [asdict(fs) for fs in feature_specs]


def is_trained(slug: str) -> bool:
    return (model_dir(slug) / METADATA).exists()
