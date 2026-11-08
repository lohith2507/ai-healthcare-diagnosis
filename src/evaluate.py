"""Evaluation metrics and plots shared by the training pipeline."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    top_k_accuracy_score,
)


def compute_metrics(y_true, y_pred, y_proba, task: str, n_classes: int) -> dict:
    """Return a JSON-serialisable metrics dict for one model."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if task == "binary":
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
    else:
        labels = list(range(n_classes))
        metrics["top3_accuracy"] = float(
            top_k_accuracy_score(y_true, y_proba, k=3, labels=labels)
        )
    return metrics


def selection_score(metrics: dict, task: str) -> float:
    """Single number used to pick the best model.

    Binary tasks use ROC-AUC; multiclass tasks use macro-F1 (accuracy is
    misleading on imbalanced labels such as stroke).
    """
    return metrics["roc_auc"] if task == "binary" else metrics["f1_macro"]


def save_confusion_matrix(y_true, y_pred, target_names, path: Path, title: str) -> None:
    """Write a confusion-matrix PNG to ``path``.

    Figure size scales with the class count and caps at 16 inches so the
    41-class symptom checker stays readable. Cell counts are shown as
    integers when there are 12 or fewer classes; larger matrices omit the
    numbers to avoid overlapping labels.
    """
    labels = list(range(len(target_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    n = len(target_names)
    fig, ax = plt.subplots(figsize=(min(2 + n * 0.5, 16), min(2 + n * 0.5, 16)))
    disp = ConfusionMatrixDisplay(cm, display_labels=target_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation="vertical",
              values_format="d" if n <= 12 else "")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
