"""UCI Heart Disease (heart_disease_uci.csv).

The raw target `num` is 0-4 (severity); we binarise to presence/absence of
heart disease. `id` and `dataset` (study site) are dropped as non-clinical.
"""
from __future__ import annotations

import pandas as pd

from ..config import DATA_RAW
from .base import FeatureSpec, PreparedData, coerce_categoricals

SLUG = "heart"

_NUMERIC = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]
_CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]


def load() -> PreparedData:
    df = pd.read_csv(DATA_RAW / "heart_disease_uci.csv")
    df = df.drop(columns=[c for c in ["id", "dataset"] if c in df.columns])

    y = (df["num"].astype(float) > 0).astype(int).to_numpy()
    X = coerce_categoricals(df.drop(columns=["num"]), _CATEGORICAL)

    specs = [
        FeatureSpec("age", "numeric", "Age (years)", 20, 100, 54),
        FeatureSpec("sex", "categorical", "Sex", options=["Male", "Female"], default="Male"),
        FeatureSpec("cp", "categorical", "Chest pain type",
                    options=["typical angina", "atypical angina", "non-anginal", "asymptomatic"],
                    default="asymptomatic"),
        FeatureSpec("trestbps", "numeric", "Resting blood pressure (mm Hg)", 80, 220, 130),
        FeatureSpec("chol", "numeric", "Serum cholesterol (mg/dl)", 100, 600, 240),
        FeatureSpec("fbs", "categorical", "Fasting blood sugar > 120 mg/dl",
                    options=["True", "False"], default="False"),
        FeatureSpec("restecg", "categorical", "Resting ECG",
                    options=["normal", "lv hypertrophy", "st-t abnormality"], default="normal"),
        FeatureSpec("thalch", "numeric", "Max heart rate achieved", 60, 220, 140),
        FeatureSpec("exang", "categorical", "Exercise-induced angina",
                    options=["True", "False"], default="False"),
        FeatureSpec("oldpeak", "numeric", "ST depression (oldpeak)", -3.0, 7.0, 0.8),
        FeatureSpec("slope", "categorical", "Slope of peak exercise ST",
                    options=["upsloping", "flat", "downsloping"], default="flat"),
        FeatureSpec("ca", "numeric", "Major vessels colored (0-3)", 0, 3, 0),
        FeatureSpec("thal", "categorical", "Thalassemia",
                    options=["normal", "fixed defect", "reversable defect"], default="normal"),
    ]

    return PreparedData(
        slug=SLUG,
        label="Heart Disease",
        task="binary",
        X=X,
        y=y,
        target_names=["No Heart Disease", "Heart Disease"],
        numeric_cols=_NUMERIC,
        categorical_cols=_CATEGORICAL,
        feature_specs=specs,
        positive_class="Heart Disease",
        notes="Binary heart-disease classification from 13 clinical features (~920 patients, multi-site).",
    )
