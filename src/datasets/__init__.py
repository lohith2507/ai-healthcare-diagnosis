"""Dataset loaders keyed by slug.

Import :func:`load_dataset` and pass a slug from ``src.config.DATASETS``.
"""
from __future__ import annotations

from . import cardio, diabetes, heart, stroke, symptoms
from .base import FeatureSpec, PreparedData

_LOADERS = {
    symptoms.SLUG: symptoms.load,
    diabetes.SLUG: diabetes.load,
    heart.SLUG: heart.load,
    cardio.SLUG: cardio.load,
    stroke.SLUG: stroke.load,
}


def load_dataset(slug: str) -> PreparedData:
    if slug not in _LOADERS:
        raise KeyError(f"Unknown dataset '{slug}'. Available: {sorted(_LOADERS)}")
    return _LOADERS[slug]()


__all__ = ["load_dataset", "PreparedData", "FeatureSpec"]
