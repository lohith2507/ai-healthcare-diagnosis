"""Cardiovascular Disease (cardio_train.csv).

Semicolon-delimited, ~70k rows. Age is stored in days; blood-pressure columns
contain data-entry outliers. We convert age to years, map coded categoricals to
readable labels, and drop physiologically implausible rows.
"""
from __future__ import annotations

import pandas as pd

from ..config import DATA_RAW
from .base import FeatureSpec, PreparedData, coerce_categoricals

SLUG = "cardio"

_NUMERIC = ["age_years", "height", "weight", "ap_hi", "ap_lo"]
_CATEGORICAL = ["gender", "cholesterol", "gluc", "smoke", "alco", "active"]

_LEVEL = {1: "Normal", 2: "Above normal", 3: "Well above normal"}
_YESNO = {0: "No", 1: "Yes"}


def load() -> PreparedData:
    df = pd.read_csv(DATA_RAW / "cardio_train.csv", sep=";")

    df["age_years"] = (df["age"] / 365.25).round(1)
    df["gender"] = df["gender"].map({1: "Women", 2: "Men"})
    df["cholesterol"] = df["cholesterol"].map(_LEVEL)
    df["gluc"] = df["gluc"].map(_LEVEL)
    for col in ["smoke", "alco", "active"]:
        df[col] = df[col].map(_YESNO)

    # Drop implausible measurements (blood pressure typos, extreme body metrics).
    before = len(df)
    df = df[
        df["ap_hi"].between(80, 250)
        & df["ap_lo"].between(40, 200)
        & (df["ap_hi"] >= df["ap_lo"])
        & df["height"].between(120, 220)
        & df["weight"].between(30, 200)
    ].reset_index(drop=True)
    dropped = before - len(df)

    y = df["cardio"].astype(int).to_numpy()
    X = coerce_categoricals(df[_NUMERIC + _CATEGORICAL].copy(), _CATEGORICAL)

    specs = [
        FeatureSpec("age_years", "numeric", "Age (years)", 20, 90, 52),
        FeatureSpec("gender", "categorical", "Gender", options=["Women", "Men"], default="Women"),
        FeatureSpec("height", "numeric", "Height (cm)", 120, 220, 165),
        FeatureSpec("weight", "numeric", "Weight (kg)", 30, 200, 74),
        FeatureSpec("ap_hi", "numeric", "Systolic blood pressure", 80, 250, 120),
        FeatureSpec("ap_lo", "numeric", "Diastolic blood pressure", 40, 200, 80),
        FeatureSpec("cholesterol", "categorical", "Cholesterol",
                    options=list(_LEVEL.values()), default="Normal"),
        FeatureSpec("gluc", "categorical", "Glucose", options=list(_LEVEL.values()), default="Normal"),
        FeatureSpec("smoke", "categorical", "Smoker", options=["No", "Yes"], default="No"),
        FeatureSpec("alco", "categorical", "Alcohol intake", options=["No", "Yes"], default="No"),
        FeatureSpec("active", "categorical", "Physically active", options=["No", "Yes"], default="Yes"),
    ]

    return PreparedData(
        slug=SLUG,
        label="Cardiovascular Disease",
        task="binary",
        X=X,
        y=y,
        target_names=["No Cardiovascular Disease", "Cardiovascular Disease"],
        numeric_cols=_NUMERIC,
        categorical_cols=_CATEGORICAL,
        feature_specs=specs,
        positive_class="Cardiovascular Disease",
        notes=(
            f"Binary cardiovascular-disease classification from 11 features "
            f"(~70k patients; {dropped} implausible rows removed during cleaning)."
        ),
    )
