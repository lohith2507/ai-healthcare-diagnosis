"""Stroke Prediction (healthcare-dataset-stroke-data.csv).

BMI contains the literal string "N/A"; the target is heavily imbalanced (~5%
positive), which the training pipeline addresses with balanced class weights.
"""
from __future__ import annotations

import pandas as pd

from ..config import DATA_RAW
from .base import FeatureSpec, PreparedData, coerce_categoricals

SLUG = "stroke"

_NUMERIC = ["age", "avg_glucose_level", "bmi"]
_CATEGORICAL = [
    "gender", "hypertension", "heart_disease", "ever_married",
    "work_type", "Residence_type", "smoking_status",
]
_YESNO = {0: "No", 1: "Yes"}


def load() -> PreparedData:
    df = pd.read_csv(DATA_RAW / "healthcare-dataset-stroke-data.csv")
    df = df.drop(columns=[c for c in ["id"] if c in df.columns])
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
    df["hypertension"] = df["hypertension"].map(_YESNO)
    df["heart_disease"] = df["heart_disease"].map(_YESNO)

    y = df["stroke"].astype(int).to_numpy()
    X = coerce_categoricals(df[_NUMERIC + _CATEGORICAL].copy(), _CATEGORICAL)

    specs = [
        FeatureSpec("age", "numeric", "Age (years)", 1, 100, 45),
        FeatureSpec("avg_glucose_level", "numeric", "Average glucose level", 50.0, 300.0, 106.0),
        FeatureSpec("bmi", "numeric", "Body mass index", 12.0, 60.0, 28.0),
        FeatureSpec("gender", "categorical", "Gender", options=["Male", "Female"], default="Female"),
        FeatureSpec("hypertension", "categorical", "Hypertension", options=["No", "Yes"], default="No"),
        FeatureSpec("heart_disease", "categorical", "Heart disease", options=["No", "Yes"], default="No"),
        FeatureSpec("ever_married", "categorical", "Ever married", options=["Yes", "No"], default="Yes"),
        FeatureSpec("work_type", "categorical", "Work type",
                    options=["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
                    default="Private"),
        FeatureSpec("Residence_type", "categorical", "Residence", options=["Urban", "Rural"], default="Urban"),
        FeatureSpec("smoking_status", "categorical", "Smoking status",
                    options=["never smoked", "formerly smoked", "smokes", "Unknown"],
                    default="never smoked"),
    ]

    return PreparedData(
        slug=SLUG,
        label="Stroke Risk",
        task="binary",
        X=X,
        y=y,
        target_names=["No Stroke", "Stroke"],
        numeric_cols=_NUMERIC,
        categorical_cols=_CATEGORICAL,
        feature_specs=specs,
        positive_class="Stroke",
        notes="Binary stroke-risk classification from 10 features (~5k patients, ~5% positive).",
    )
