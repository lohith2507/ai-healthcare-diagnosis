"""Exploratory data analysis figures, saved to reports/figures/.

Run:  python -m src.eda            (all datasets)
      python -m src.eda --dataset stroke
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .config import DATASETS, FIGURES_DIR
from .datasets import load_dataset


def _target_distribution(data, ax) -> None:
    counts = np.bincount(data.y, minlength=len(data.target_names))
    order = np.argsort(counts)[::-1]
    names = [data.target_names[i] for i in order]
    vals = counts[order]
    if len(names) > 15:
        names, vals = names[:15], vals[:15]
    sns.barplot(x=vals, y=names, ax=ax, color="#3182bd")
    ax.set_title(f"{data.label} — class distribution")
    ax.set_xlabel("count")


def _numeric_correlation(data, ax) -> None:
    if not data.numeric_cols:
        ax.axis("off")
        return
    corr = data.X[data.numeric_cols].corr(numeric_only=True)
    sns.heatmap(corr, annot=len(data.numeric_cols) <= 10, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, cbar=False)
    ax.set_title(f"{data.label} — numeric feature correlation")


def generate(slug: str) -> None:
    data = load_dataset(slug)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    _target_distribution(data, axes[0])
    _numeric_correlation(data, axes[1])
    fig.tight_layout()
    out = FIGURES_DIR / f"{slug}_eda.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  saved {out.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate EDA figures.")
    parser.add_argument("--dataset", choices=list(DATASETS))
    args = parser.parse_args()
    slugs = [args.dataset] if args.dataset else list(DATASETS)
    for slug in slugs:
        generate(slug)


if __name__ == "__main__":
    main()
