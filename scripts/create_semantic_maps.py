"""
Command-line script for creating semantic maps from cleaned model outputs.

Run from the project root:

    python scripts/create_semantic_maps.py
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings import run_semantic_mapping


def main() -> None:
    """
    Create embeddings, PCA coordinates, t-SNE coordinates, and silhouette scores.
    """
    run_semantic_mapping()


if __name__ == "__main__":
    main()