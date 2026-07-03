"""
Command-line script for generating LLM outputs.

This script should be run from the project root.

Example usage:
    python scripts/generate_outputs.py --model OpenAI --limit 1
    python scripts/generate_outputs.py --model Claude --limit 1
    python scripts/generate_outputs.py --model Gemini --limit 1
    python scripts/generate_outputs.py --model DeepSeek --limit 1
    python scripts/generate_outputs.py --model Mistral --limit 1
"""

from __future__ import annotations
from pathlib import Path
import sys
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    PROMPTS_PATH,
    RAW_OUTPUTS_PATH,
)
from src.model_runner import generate_outputs_from_prompts


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate LLM responses for stylometric fingerprinting."
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=["OpenAI", "Claude", "Gemini", "DeepSeek", "Mistral"],
        help="Model provider to call.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of prompts to process.",
    )

    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Prompt row index to start from.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature.",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum output tokens.",
    )

    return parser.parse_args()


def main() -> None:
    """
    Generate model outputs and save them to the raw outputs CSV file.
    """
    args = parse_args()

    print("Starting model output generation...")
    print(f"Model: {args.model}")
    print(f"Prompt file: {PROMPTS_PATH}")
    print(f"Output file: {RAW_OUTPUTS_PATH}")

    generated_df = generate_outputs_from_prompts(
        prompts_path=PROMPTS_PATH,
        model_key=args.model,
        output_path=RAW_OUTPUTS_PATH,
        limit=args.limit,
        start_index=args.start_index,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    print("\nGeneration completed.")
    print(f"Rows generated: {len(generated_df)}")
    print(f"Saved to: {RAW_OUTPUTS_PATH}")


if __name__ == "__main__":
    main()