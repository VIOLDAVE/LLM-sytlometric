"""
Command-line script for preprocessing raw model outputs.

Run from the project root:

    python scripts/preprocess_outputs.py
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import preprocess_raw_outputs


def main() -> None:
    """
    Run the preprocessing pipeline.
    """
    preprocess_raw_outputs()


if __name__ == "__main__":
    main()