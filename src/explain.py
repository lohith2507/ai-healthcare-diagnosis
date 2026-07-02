"""Model explainability: global feature importance + per-prediction attributions.

Tree models use SHAP for fast, signed local explanations. Linear and neural
models fall back to coefficient- or permutation-based importance. Everything maps
transformed feature contributions back to the original human-readable feature.
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import artifacts
from .config import DATASETS, FIGURES_DIR
from .datasets import load_dataset


def _densify(matrix) -> np.ndarray:
    if sp.issparse(matrix):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def _raw_label_for(transformed_name: str, numeric_cols, categorical_cols) -> str:
    """Map a transformed feature name (e.g. 'cat__sex_Male') to its raw feature."""
    name = transformed_name
    for prefix in ("num__", "cat__", "remainder__"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    if name in numeric_cols:
        return name
    for col in categorical_cols:
        if name.startswith(f"{col}_"):
            return f"{col}={name[len(col) + 1:]}"
    return name


def _load_model(slug: str, model_name: str):
    if model_name == artifacts.NN_NAME:
        return artifacts.load_nn(slug)
    return artifacts.load_classical(slug, model_name)


def compute_global_importance(slug: str, top_n: int = 20) -> pd.DataFrame:
    """Return a DataFrame(feature, importance) for the dataset's best model."""
    meta = artifacts.load_metadata(slug)
    data = load_dataset(slug)
    pre = artifacts.load_preprocessor(slug)
    feat_names = meta["feature_names_out"]
    model_name = meta["best_model"]
    model = _load_model(slug, model_name)

    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        importances = np.abs(np.asarray(model.coef_, dtype=float)).mean(axis=0)
    else:  # neural net -> permutation importance on a sample
        importances = _permutation_importance(model, pre, data)

    df = pd.DataFrame({"feature": feat_names, "importance": importances})
    df["raw_feature"] = df["feature"].map(
        lambda n: _raw_label_for(n, data.numeric_cols, data.categorical_cols)
    )
    grouped = (
        df.groupby("raw_feature", as_index=False)["importance"].sum()
        .sort_values("importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return grouped


def _permutation_importance(model, pre, data, sample: int = 500) -> np.ndarray:
    from sklearn.metrics import accuracy_score

    X = data.X.sample(min(sample, len(data.X)), random_state=0)
    y = data.y[X.index]
    Xt = _densify(pre.transform(X))
    base = accuracy_score(y, model.predict(Xt, verbose=0).argmax(axis=1)
                          if data.task != "binary"
                          else (model.predict(Xt, verbose=0).ravel() >= 0.5).astype(int))
    rng = np.random.default_rng(0)
    scores = np.zeros(Xt.shape[1])
    for j in range(Xt.shape[1]):
        Xp = Xt.copy()
        rng.shuffle(Xp[:, j])
        pred = (model.predict(Xp, verbose=0).argmax(axis=1) if data.task != "binary"
                else (model.predict(Xp, verbose=0).ravel() >= 0.5).astype(int))
        scores[j] = base - accuracy_score(y, pred)
    return np.clip(scores, 0, None)


def local_explanation(slug: str, x_row: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    """Signed per-feature contributions for a single prediction (best-effort)."""
    meta = artifacts.load_metadata(slug)
    data = load_dataset(slug)
    pre = artifacts.load_preprocessor(slug)
    feat_names = meta["feature_names_out"]
    model_name = meta["best_model"]
    model = _load_model(slug, model_name)
    xt = _densify(pre.transform(x_row))

    contributions = None
    if hasattr(model, "feature_importances_"):  # tree models -> SHAP
        try:
            import shap

            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(xt)
            contributions = _pick_shap_row(sv)
        except Exception:
            contributions = None
    if contributions is None and hasattr(model, "coef_"):  # linear
        coef = np.asarray(model.coef_, dtype=float)
        coef = coef[0] if coef.ndim == 2 else coef
        contributions = coef * xt[0]
    if contributions is None:  # fallback: unsigned global importance * presence
        gi = compute_global_importance(slug, top_n=len(feat_names))
        imp_map = dict(zip(gi["raw_feature"], gi["importance"]))
        raw = [_raw_label_for(n, data.numeric_cols, data.categorical_cols) for n in feat_names]
        contributions = np.array([imp_map.get(r, 0.0) for r in raw]) * xt[0]

    df = pd.DataFrame({"feature": feat_names, "contribution": np.asarray(contributions).ravel()})
    df["raw_feature"] = df["feature"].map(
        lambda n: _raw_label_for(n, data.numeric_cols, data.categorical_cols)
    )
    grouped = df.groupby("raw_feature", as_index=False)["contribution"].sum()
    grouped["abs"] = grouped["contribution"].abs()
    return grouped.sort_values("abs", ascending=False).head(top_n).drop(columns="abs").reset_index(drop=True)


def _pick_shap_row(shap_values) -> np.ndarray:
    """Normalise the many SHAP output shapes to a 1-D contribution vector."""
    sv = shap_values
    if isinstance(sv, list):  # list per class -> use positive/last class
        sv = sv[-1]
    sv = np.asarray(sv)
    if sv.ndim == 3:  # (samples, features, classes)
        sv = sv[0, :, -1]
    elif sv.ndim == 2:  # (samples, features)
        sv = sv[0]
    return sv


def save_importance_figure(slug: str) -> None:
    df = compute_global_importance(slug)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(df))))
    ax.barh(df["raw_feature"][::-1], df["importance"][::-1], color="#2b8cbe")
    ax.set_title(f"{DATASETS[slug]} — top feature importances")
    ax.set_xlabel("importance")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{slug}_importance.png", dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate explainability figures.")
    parser.add_argument("--dataset", choices=list(DATASETS))
    args = parser.parse_args()
    slugs = [args.dataset] if args.dataset else list(DATASETS)
    for slug in slugs:
        if artifacts.is_trained(slug):
            save_importance_figure(slug)
            print(f"  saved importance figure for {slug}")
        else:
            print(f"  skip {slug} (not trained)")


if __name__ == "__main__":
    main()
