"""
Prompt loading utilities.

This module loads and validates the prompt dataset used for API generation.
"""

from pathlib import Path
import pandas as pd


REQUIRED_PROMPT_COLUMNS = [
    "prompt_id",
    "genre",
    "theme",
    "word_limit",
    "prompt_text",
]


def load_prompts(path: str | Path) -> pd.DataFrame:
    """
    Load prompts from a CSV file.

    Parameters
    ----------
    path : str | Path
        Path to the prompt CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded prompt dataset.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    return pd.read_csv(path)


def validate_prompts(df: pd.DataFrame) -> None:
    """
    Validate that the prompt dataset contains required columns and non-empty prompts.

    Parameters
    ----------
    df : pd.DataFrame
        Prompt dataset.

    Raises
    ------
    ValueError
        If required columns are missing or prompt text is empty.
    """
    missing_columns = [col for col in REQUIRED_PROMPT_COLUMNS if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required prompt columns: {missing_columns}")

    missing_prompt_count = df["prompt_text"].isna().sum()

    if missing_prompt_count > 0:
        raise ValueError(f"Found {missing_prompt_count} missing prompt texts.")


def summarize_prompts(df: pd.DataFrame) -> None:
    """
    Print a simple prompt summary by genre and theme.

    Parameters
    ----------
    df : pd.DataFrame
        Prompt dataset.
    """
    print("Prompt dataset summary")
    print("=" * 40)
    print(f"Total prompts: {len(df)}")

    if "genre" in df.columns:
        print("\\nPrompts by genre:")
        print(df["genre"].value_counts())

    if "theme" in df.columns:
        print("\\nPrompts by theme:")
        print(df["theme"].value_counts())
