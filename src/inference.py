"""Single-row inference used by the Streamlit app and the tests.

Given raw feature values (matching a dataset's ``feature_specs``), build the
input frame, run it through the saved preprocessor, and return class probabilities
with any chosen trained model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import artifacts


def _densify(matrix) -> np.ndarray:
    if sp.issparse(matrix):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def _standardise_proba(raw, task: str) -> np.ndarray:
    raw = np.asarray(raw)
    if task == "binary" and raw.ndim == 2 and raw.shape[1] == 1:
        p = raw.ravel()
        return np.column_stack([1.0 - p, p])
    if raw.ndim == 1:
        return np.column_stack([1.0 - raw, raw])
    return raw


def build_row(meta: dict, values: dict) -> pd.DataFrame:
    """Assemble a one-row DataFrame in the raw feature schema the preprocessor expects."""
    if meta["input_mode"] == "symptoms":
        weights = meta["symptom_weights"]
        selected = set(values.get("symptoms", []))
        row = {s: (weights.get(s, 1.0) if s in selected else 0.0) for s in meta["symptom_vocab"]}
        return pd.DataFrame([row])

    cols = meta["numeric_cols"] + meta["categorical_cols"]
    return pd.DataFrame([{c: values.get(c) for c in cols}])


def predict_proba(slug: str, model_name: str, row: pd.DataFrame) -> np.ndarray:
    meta = artifacts.load_metadata(slug)
    pre = artifacts.load_preprocessor(slug)
    xt = _densify(pre.transform(row))
    if model_name == artifacts.NN_NAME:
        model = artifacts.load_nn(slug)
        raw = model.predict(xt, verbose=0)
    else:
        model = artifacts.load_classical(slug, model_name)
        raw = model.predict_proba(xt)
    return _standardise_proba(raw, meta["task"])[0]
