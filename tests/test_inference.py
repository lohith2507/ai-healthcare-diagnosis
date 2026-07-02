"""End-to-end inference smoke tests against the trained artifacts.

Skipped automatically if a model has not been trained yet (run `python -m src.train`).
"""
import numpy as np
import pytest

from src import artifacts
from src.config import DATASETS
from src.inference import build_row, predict_proba

TRAINED = [s for s in DATASETS if artifacts.is_trained(s)]

pytestmark = pytest.mark.skipif(not TRAINED, reason="no trained models found")


def _default_values(meta: dict) -> dict:
    if meta["input_mode"] == "symptoms":
        return {"symptoms": meta["symptom_vocab"][:3]}
    return {spec["name"]: spec["default"] for spec in meta["feature_specs"]}


@pytest.mark.parametrize("slug", TRAINED)
def test_predict_returns_valid_distribution(slug):
    meta = artifacts.load_metadata(slug)
    values = _default_values(meta)
    row = build_row(meta, values)

    for model_name in meta["models"]:
        proba = predict_proba(slug, model_name, row)
        assert proba.shape == (len(meta["target_names"]),)
        assert np.all(np.isfinite(proba))
        assert proba.min() >= -1e-6
        assert abs(float(proba.sum()) - 1.0) < 1e-3, f"{model_name} proba must sum to 1"


@pytest.mark.parametrize("slug", TRAINED)
def test_build_row_matches_schema(slug):
    meta = artifacts.load_metadata(slug)
    row = build_row(meta, _default_values(meta))
    assert len(row) == 1
    if meta["input_mode"] != "symptoms":
        expected = set(meta["numeric_cols"] + meta["categorical_cols"])
        assert set(row.columns) == expected
