"""Pytest setup: put the repo root on sys.path.

This lets tests import the project packages (``src``, ``backend``) directly
when pytest is run from the repo root without installing the project.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
