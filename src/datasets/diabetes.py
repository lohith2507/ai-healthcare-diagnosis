"""Pima Indians Diabetes (diabetes.csv).

Several columns use 0 to mean "not recorded"; we convert those to NaN so the
median imputer in the preprocessor handles them honestly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import DATA_RAW
from .base import FeatureSpec, PreparedData

SLUG = "diabetes"
_ZERO_IS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def load() -> PreparedData:
    df = pd.read_csv(DATA_RAW / "diabetes.csv")
    for col in _ZERO_IS_MISSING:
        df[col] = df[col].replace(0, np.nan)

    y = df["Outcome"].astype(int).to_numpy()
    X = df.drop(columns=["Outcome"])
    numeric_cols = list(X.columns)

    specs = [
        FeatureSpec("Pregnancies", "numeric", "Number of pregnancies", 0, 17, 1),
        FeatureSpec("Glucose", "numeric", "Plasma glucose (mg/dL)", 40, 200, 117),
        FeatureSpec("BloodPressure", "numeric", "Diastolic blood pressure (mm Hg)", 40, 130, 72),
        FeatureSpec("SkinThickness", "numeric", "Triceps skinfold thickness (mm)", 0, 100, 23),
        FeatureSpec("Insulin", "numeric", "2-hour serum insulin (mu U/ml)", 0, 850, 30),
        FeatureSpec("BMI", "numeric", "Body mass index", 15.0, 60.0, 32.0),
        FeatureSpec("DiabetesPedigreeFunction", "numeric", "Diabetes pedigree function", 0.05, 2.5, 0.37),
        FeatureSpec("Age", "numeric", "Age (years)", 18, 100, 29),
    ]

    return PreparedData(
        slug=SLUG,
        label="Diabetes Risk",
        task="binary",
        X=X,
        y=y,
        target_names=["No Diabetes", "Diabetes"],
        numeric_cols=numeric_cols,
        categorical_cols=[],
        feature_specs=specs,
        positive_class="Diabetes",
        notes="Binary diabetes classification from 8 clinical measurements (768 patients).",
    )
