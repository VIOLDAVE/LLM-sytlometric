"""
Command-line script for running classification modeling.

This script evaluates whether stylometric features can predict the LLM
model family that generated a text.

Run from the project root:

    python scripts/run_classification_modeling.py
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.classification import run_classification_modeling


def main() -> None:
    """
    Run the classification modeling pipeline.
    """
    run_classification_modeling(
        n_splits=5,
    )


if __name__ == "__main__":
    main()