"""Training pipeline: fit classical + neural models for one or all datasets.

Usage:
    python -m src.train                # train every dataset in the registry
    python -m src.train --dataset heart
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from . import artifacts
from .config import DATASETS, FIGURES_DIR, RANDOM_STATE, TEST_SIZE
from .datasets import load_dataset
from .evaluate import compute_metrics, save_confusion_matrix, selection_score
from .models.classical import build_classical_models
from .models.neural import build_mlp, train_mlp


def _densify(matrix) -> np.ndarray:
    if sp.issparse(matrix):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def _standardise_proba(raw, task: str) -> np.ndarray:
    """Coerce any model's probability output to shape (n_samples, n_classes)."""
    raw = np.asarray(raw)
    if task == "binary" and raw.ndim == 2 and raw.shape[1] == 1:
        p = raw.ravel()
        return np.column_stack([1.0 - p, p])
    if raw.ndim == 1:
        return np.column_stack([1.0 - raw, raw])
    return raw


def train_dataset(slug: str) -> dict:
    print(f"\n=== Training '{slug}' ({DATASETS[slug]}) ===")
    data = load_dataset(slug)
    n_classes = len(data.target_names)

    X_train, X_test, y_train, y_test = train_test_split(
        data.X, data.y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=data.y
    )

    preprocessor = data.build_preprocessor()
    Xtr = _densify(preprocessor.fit_transform(X_train))
    Xte = _densify(preprocessor.transform(X_test))
    print(f"  samples: train={len(Xtr)} test={len(Xte)} | features after transform: {Xtr.shape[1]}")

    metrics: dict[str, dict] = {}

    # --- Classical baselines ---------------------------------------------
    for name, model in build_classical_models(data.task, y_train).items():
        t0 = time.perf_counter()
        model.fit(Xtr, y_train)
        train_time = time.perf_counter() - t0
        proba = _standardise_proba(model.predict_proba(Xte), data.task)
        y_pred = proba.argmax(axis=1)
        m = compute_metrics(y_test, y_pred, proba, data.task, n_classes)
        m["train_time_s"] = round(train_time, 2)
        metrics[name] = m
        artifacts.save_classical(slug, name, model)
        print(f"  {name:>20}: f1={m['f1_macro']:.3f} acc={m['accuracy']:.3f} "
              f"({train_time:.1f}s)")

    # --- Neural network --------------------------------------------------
    class_weight = None
    if data.task == "binary":
        classes = np.unique(y_train)
        weights = compute_class_weight("balanced", classes=classes, y=y_train)
        class_weight = {int(c): float(w) for c, w in zip(classes, weights)}

    nn = build_mlp(Xtr.shape[1], n_classes, data.task)
    t0 = time.perf_counter()
    train_mlp(nn, Xtr, y_train, class_weight=class_weight)
    train_time = time.perf_counter() - t0
    proba = _standardise_proba(nn.predict(Xte, verbose=0), data.task)
    y_pred = proba.argmax(axis=1)
    m = compute_metrics(y_test, y_pred, proba, data.task, n_classes)
    m["train_time_s"] = round(train_time, 2)
    metrics[artifacts.NN_NAME] = m
    print(f"  {artifacts.NN_NAME:>20}: f1={m['f1_macro']:.3f} acc={m['accuracy']:.3f} "
          f"({train_time:.1f}s)")

    # --- Persist ----------------------------------------------------------
    best_model = max(metrics, key=lambda k: selection_score(metrics[k], data.task))
    print(f"  -> best model: {best_model}")

    artifacts.save_preprocessor(slug, preprocessor)
    artifacts.save_nn(slug, nn)

    try:
        feature_names_out = list(preprocessor.get_feature_names_out())
    except Exception:
        feature_names_out = [f"f{i}" for i in range(Xtr.shape[1])]

    metadata = {
        "slug": slug,
        "label": data.label,
        "task": data.task,
        "input_mode": data.input_mode,
        "target_names": data.target_names,
        "positive_class": data.positive_class,
        "numeric_cols": data.numeric_cols,
        "categorical_cols": data.categorical_cols,
        "feature_specs": artifacts.feature_specs_to_dicts(data.feature_specs),
        "feature_names_out": feature_names_out,
        "models": list(metrics.keys()),
        "best_model": best_model,
        "notes": data.notes,
        "n_samples": int(len(data.X)),
        "n_features_transformed": int(Xtr.shape[1]),
        "symptom_vocab": data.symptom_vocab,
        "symptom_weights": data.symptom_weights,
        "disease_info": data.disease_info,
    }
    artifacts.save_metadata(slug, metadata)
    artifacts.save_metrics(slug, metrics)

    # Confusion matrix for the best model on the held-out test set.
    best_est_pred = y_pred if best_model == artifacts.NN_NAME else None
    if best_est_pred is None:
        best_model_obj = artifacts.load_classical(slug, best_model)
        best_est_pred = _standardise_proba(
            best_model_obj.predict_proba(Xte), data.task
        ).argmax(axis=1)
    save_confusion_matrix(
        y_test, best_est_pred, data.target_names,
        FIGURES_DIR / f"{slug}_confusion_matrix.png",
        f"{data.label} — {best_model} (confusion matrix)",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train diagnostic models.")
    parser.add_argument("--dataset", choices=list(DATASETS), help="Train a single dataset.")
    args = parser.parse_args()

    slugs = [args.dataset] if args.dataset else list(DATASETS)
    for slug in slugs:
        train_dataset(slug)
    print("\nAll requested datasets trained. Artifacts in models/, figures in reports/figures/.")


if __name__ == "__main__":
    main()
