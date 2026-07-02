"""Central configuration: paths and the dataset registry.

Everything that needs to know *where* things live imports from here, so there is a
single source of truth for the project layout.
"""
from __future__ import annotations

from pathlib import Path

# --- Project layout -------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"

for _d in (MODELS_DIR, FIGURES_DIR, METRICS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Reproducibility ------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2

# --- Dataset registry -----------------------------------------------------
# Each key maps a short slug (used for artifact filenames and app routing) to a
# human-friendly label. The actual loading/prep logic lives in src/datasets/.
DATASETS: dict[str, str] = {
    "symptoms": "Multi-Disease Symptom Checker",
    "diabetes": "Diabetes Risk",
    "heart": "Heart Disease",
    "cardio": "Cardiovascular Disease",
    "stroke": "Stroke Risk",
}


def model_dir(slug: str) -> Path:
    """Directory holding all trained artifacts for a dataset slug."""
    d = MODELS_DIR / slug
    d.mkdir(parents=True, exist_ok=True)
    return d
