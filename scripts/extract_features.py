"""
Command-line script for extracting stylometric features.

Run from the project root:

    python scripts/extract_features.py
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.stylometry import run_feature_extraction


def main() -> None:
    """
    Run stylometric feature extraction.
    """
    run_feature_extraction()


if __name__ == "__main__":
    main()