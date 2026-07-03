"""
Command-line script for running statistical tests.

This script runs the full statistical testing pipeline for the
LLM stylometric fingerprinting project.

It produces:
- Kruskal-Wallis model-family tests;
- genre-specific Kruskal-Wallis robustness tests;
- pairwise Mann-Whitney U post-hoc tests for the strongest features.

Run from the project root:

    python scripts/run_statistical_tests.py
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.statistical_tests import run_statistical_tests


def main() -> None:
    """
    Run the full statistical testing pipeline.

    The pipeline saves outputs to:

        outputs/statistical_tests/kruskal_model_family_tests.csv
        outputs/statistical_tests/kruskal_model_family_tests_by_genre.csv
        outputs/statistical_tests/pairwise_posthoc_mannwhitney_tests.csv
    """
    run_statistical_tests(
        n_posthoc_features=10,
    )


if __name__ == "__main__":
    main()