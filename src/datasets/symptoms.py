"""Multi-disease symptom checker (dataset.csv + companions).

Each raw row is a disease plus up to 17 free-text symptom tokens. We normalise
the tokens, build a fixed vocabulary, and turn every row into a severity-weighted
multi-hot vector (weight from Symptom-severity.csv, or 1.0 if unknown).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..config import DATA_RAW
from .base import FeatureSpec, PreparedData

SLUG = "symptoms"


def _normalise(token: str) -> str:
    """Canonicalise a symptom token: strip, lowercase, collapse spaces to `_`."""
    token = str(token).strip().lower()
    token = re.sub(r"\s+", "_", token)
    token = re.sub(r"_+", "_", token)
    return token.strip("_")


def _severity_map() -> dict[str, float]:
    sev = pd.read_csv(DATA_RAW / "Symptom-severity.csv")
    return {_normalise(s): float(w) for s, w in zip(sev["Symptom"], sev["weight"])}


def _disease_info() -> dict[str, dict]:
    desc = pd.read_csv(DATA_RAW / "symptom_Description.csv")
    prec = pd.read_csv(DATA_RAW / "symptom_precaution.csv")
    info: dict[str, dict] = {}
    for _, row in desc.iterrows():
        info[str(row["Disease"]).strip()] = {"description": str(row["Description"]).strip(), "precautions": []}
    for _, row in prec.iterrows():
        name = str(row["Disease"]).strip()
        precs = [str(row[c]).strip() for c in prec.columns[1:] if pd.notna(row[c]) and str(row[c]).strip()]
        info.setdefault(name, {"description": "", "precautions": []})["precautions"] = precs
    return info


def load() -> PreparedData:
    df = pd.read_csv(DATA_RAW / "dataset.csv")
    symptom_cols = [c for c in df.columns if c.lower().startswith("symptom")]

    severity = _severity_map()

    # Collect every symptom token that appears, in a stable sorted order.
    vocab: set[str] = set()
    rows_tokens: list[set[str]] = []
    for _, row in df.iterrows():
        tokens = {_normalise(row[c]) for c in symptom_cols if pd.notna(row[c]) and str(row[c]).strip()}
        tokens.discard("")
        rows_tokens.append(tokens)
        vocab.update(tokens)
    vocab_list = sorted(vocab)

    # Severity-weighted multi-hot matrix.
    weights = np.array([severity.get(s, 1.0) for s in vocab_list], dtype=float)
    matrix = np.zeros((len(rows_tokens), len(vocab_list)), dtype=float)
    index = {s: i for i, s in enumerate(vocab_list)}
    for r, tokens in enumerate(rows_tokens):
        for t in tokens:
            matrix[r, index[t]] = weights[index[t]]

    X = pd.DataFrame(matrix, columns=vocab_list)
    diseases = df["Disease"].astype(str).str.strip()
    target_names = sorted(diseases.unique())
    code = {name: i for i, name in enumerate(target_names)}
    y = diseases.map(code).to_numpy()

    # For a symptom checker the "features" are the vocabulary; the app renders a
    # single multiselect rather than 130+ individual controls.
    feature_specs = [
        FeatureSpec(name=s, kind="numeric", label=s.replace("_", " ").title(), minimum=0, maximum=1, default=0)
        for s in vocab_list
    ]

    return PreparedData(
        slug=SLUG,
        label="Multi-Disease Symptom Checker",
        task="multiclass",
        X=X,
        y=y,
        target_names=target_names,
        numeric_cols=vocab_list,
        categorical_cols=[],
        feature_specs=feature_specs,
        input_mode="symptoms",
        notes=(
            "41-class disease prediction from a severity-weighted symptom vector. "
            "The dataset is small and highly separable, so accuracy is near-perfect; "
            "treat this as a pattern-recognition demo, not a clinical tool."
        ),
        symptom_vocab=vocab_list,
        symptom_weights={s: float(weights[index[s]]) for s in vocab_list},
        disease_info=_disease_info(),
    )
