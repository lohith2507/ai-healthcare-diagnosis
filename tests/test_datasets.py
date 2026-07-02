"""Data-preparation contract tests: every loader returns consistent, clean data."""
import numpy as np
import pytest

from src.config import DATASETS
from src.datasets import load_dataset


@pytest.mark.parametrize("slug", list(DATASETS))
def test_shapes_and_targets(slug):
    data = load_dataset(slug)
    assert len(data.X) == len(data.y), "X/y row mismatch"
    assert len(data.X) > 0

    n_classes = len(data.target_names)
    assert n_classes >= 2
    assert set(np.unique(data.y)).issubset(set(range(n_classes)))

    # Declared columns must exist in the feature frame.
    for col in data.numeric_cols + data.categorical_cols:
        assert col in data.X.columns, f"{col} missing from X"

    # Task label is consistent with the number of classes.
    if data.task == "binary":
        assert n_classes == 2
    else:
        assert n_classes > 2


@pytest.mark.parametrize("slug", list(DATASETS))
def test_preprocessor_fits(slug):
    data = load_dataset(slug)
    pre = data.build_preprocessor()
    Xt = pre.fit_transform(data.X.head(200))
    assert Xt.shape[0] == min(200, len(data.X))
    assert Xt.shape[1] > 0


def test_symptom_encoding_is_weighted():
    data = load_dataset("symptoms")
    assert data.input_mode == "symptoms"
    assert len(data.symptom_vocab) > 100
    # Every vocabulary token has a positive severity weight.
    assert all(w > 0 for w in data.symptom_weights.values())
