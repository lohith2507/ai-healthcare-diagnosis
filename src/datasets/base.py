"""Shared types and helpers for dataset preparation.

Every dataset module returns a :class:`PreparedData`. This gives the training
pipeline and the Streamlit app one uniform contract regardless of whether the
underlying data is a symptom checklist or a table of clinical measurements.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

Task = Literal["binary", "multiclass"]
InputMode = Literal["form", "symptoms"]


@dataclass
class FeatureSpec:
    """Describes one *raw* input feature so the app can render a control for it."""

    name: str
    kind: Literal["numeric", "categorical"]
    label: str
    minimum: float | None = None
    maximum: float | None = None
    default: float | str | None = None
    options: list[str] | None = None
    help: str | None = None


@dataclass
class PreparedData:
    """Standardised bundle describing a modelling task."""

    slug: str
    label: str
    task: Task
    X: pd.DataFrame
    y: np.ndarray
    target_names: list[str]
    numeric_cols: list[str]
    categorical_cols: list[str]
    feature_specs: list[FeatureSpec]
    positive_class: str | None = None
    input_mode: InputMode = "form"
    notes: str = ""
    # Symptom-checker extras (unused by tabular datasets)
    symptom_vocab: list[str] = field(default_factory=list)
    symptom_weights: dict[str, float] = field(default_factory=dict)
    disease_info: dict[str, dict] = field(default_factory=dict)

    def build_preprocessor(self) -> ColumnTransformer:
        """Create an *unfitted* preprocessor matching this dataset's columns."""
        numeric_pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]
        )
        categorical_pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers = []
        if self.numeric_cols:
            transformers.append(("num", numeric_pipe, self.numeric_cols))
        if self.categorical_cols:
            transformers.append(("cat", categorical_pipe, self.categorical_cols))
        return ColumnTransformer(transformers, remainder="drop")


def coerce_categoricals(X: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Force categorical columns to plain ``str`` objects with ``np.nan`` for missing.

    This avoids two pandas-3 / scikit-learn pitfalls: the ``string`` dtype's
    ``pd.NA`` crashes SimpleImputer, and boolean categories (True/False) would not
    match the string inputs the app sends at prediction time.
    """
    X = X.copy()
    for c in cols:
        col = X[c]
        missing = col.isna().to_numpy()
        col = col.astype(object).astype(str)
        col[missing] = np.nan
        X[c] = col
    return X


def feature_names_from_preprocessor(preprocessor: ColumnTransformer) -> list[str]:
    """Best-effort readable names for the columns produced by a fitted preprocessor."""
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:  # pragma: no cover - defensive
        return [f"f{i}" for i in range(preprocessor.transform_output_shape_[1])]
