"""
Preprocessing module for LLM stylometric fingerprinting.

This module prepares raw model outputs for downstream stylometric analysis.
The goal is to clean technical artifacts while preserving each model's
writing style as much as possible.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.config import RAW_OUTPUTS_PATH, PROCESSED_DIR
from src.text_utils import count_words


CLEAN_OUTPUTS_PATH = PROCESSED_DIR / "model_outputs_clean.csv"


REQUIRED_RAW_COLUMNS = [
    "text_id",
    "prompt_id",
    "genre",
    "theme",
    "model_family",
    "model_name",
    "provider",
    "original_prompt_text",
    "controlled_prompt_text",
    "raw_output_text",
    "length_controlled_output_text",
    "output_text",
    "raw_word_count",
    "final_word_count",
    "length_status",
    "within_10_percent_word_limit",
    "api_status",
]


def load_raw_outputs(path: str | Path = RAW_OUTPUTS_PATH) -> pd.DataFrame:
    """
    Load raw model outputs from CSV.

    Parameters
    ----------
    path : str | Path
        Path to model_outputs_raw.csv.

    Returns
    -------
    pd.DataFrame
        Raw model output dataframe.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Raw output file not found: {path}")

    return pd.read_csv(path)


def validate_raw_outputs(df: pd.DataFrame) -> None:
    """
    Validate that the raw model output dataframe has required columns.

    Parameters
    ----------
    df : pd.DataFrame
        Raw model output dataframe.

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    missing_columns = [col for col in REQUIRED_RAW_COLUMNS if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required raw output columns: {missing_columns}")


def filter_successful_valid_outputs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only successful and length-valid model outputs.

    Parameters
    ----------
    df : pd.DataFrame
        Raw model output dataframe.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe with usable rows only.
    """
    filtered = df[
        (df["api_status"] == "success")
        & (df["within_10_percent_word_limit"] == True)
        & (df["length_status"] == "valid")
    ].copy()

    return filtered


def normalize_text_spacing(text: str) -> str:
    """
    Apply light technical text normalization.

    This function avoids aggressive cleaning because punctuation, casing,
    sentence structure, and formatting may be meaningful stylometric signals.

    Cleaning performed:
    - normalize repeated whitespace;
    - remove spaces before punctuation;
    - remove spaces around apostrophes;
    - repair some missing spaces after punctuation;
    - strip leading/trailing whitespace.

    Parameters
    ----------
    text : str
        Input model output text.

    Returns
    -------
    str
        Lightly cleaned text.
    """
    if not isinstance(text, str):
        return ""

    cleaned = text.strip()

    # Convert escaped newline sequences if they appear as literal text.
    cleaned = cleaned.replace("\\n", "\n")

    # Normalize excessive spaces and tabs, but preserve paragraph breaks first.
    cleaned = re.sub(r"[ \t]+", " ", cleaned)

    # Remove spaces before punctuation.
    cleaned = re.sub(r"\s+([.,!?;:])", r"\1", cleaned)

    # Fix spaces around apostrophes/curly apostrophes.
    cleaned = re.sub(r"\s+(['’])\s*", r"\1", cleaned)

    # Add a missing space after punctuation when two words are joined.
    # Example: "circulation,each" -> "circulation, each"
    cleaned = re.sub(r"([.,!?;:])([A-Za-z])", r"\1 \2", cleaned)

    # Add a space when a lowercase letter is directly followed by uppercase.
    # Example: "datasetWas" -> "dataset Was"
    cleaned = re.sub(r"([a-z])([A-Z])", r"\1 \2", cleaned)

    # Normalize paragraph spacing.
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    # Normalize remaining repeated spaces.
    cleaned = re.sub(r" {2,}", " ", cleaned)

    return cleaned.strip()


def add_clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cleaned text and basic clean-text diagnostics.

    Parameters
    ----------
    df : pd.DataFrame
        Filtered model output dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with clean_output_text and clean_word_count.
    """
    df = df.copy()

    df["clean_output_text"] = df["output_text"].apply(normalize_text_spacing)
    df["clean_word_count"] = df["clean_output_text"].apply(count_words)

    df["clean_length_status"] = df["clean_word_count"].apply(
        lambda count: "valid" if 180 <= count <= 220 else (
            "too_short" if count < 180 else "too_long"
        )
    )

    df["clean_within_10_percent_word_limit"] = df["clean_word_count"].between(
        180,
        220,
        inclusive="both",
    )

    return df


def remove_duplicate_outputs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate model-prompt rows.

    If duplicates exist, keep the latest occurrence. This is useful after
    reruns where a model response was regenerated for the same prompt.

    Parameters
    ----------
    df : pd.DataFrame
        Model output dataframe.

    Returns
    -------
    pd.DataFrame
        Deduplicated dataframe.
    """
    deduplicated = df.drop_duplicates(
        subset=["model_family", "prompt_id"],
        keep="last",
    ).copy()

    return deduplicated


def summarize_clean_outputs(df: pd.DataFrame) -> None:
    """
    Print a concise summary of the cleaned dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned model output dataframe.
    """
    print("Cleaned model output summary")
    print("=" * 50)

    print(f"Total rows: {len(df)}")

    print("\nRows by model family:")
    print(df["model_family"].value_counts().sort_index())

    print("\nRows by genre:")
    print(df["genre"].value_counts().sort_index())

    print("\nRows by model family and genre:")
    print(pd.crosstab(df["model_family"], df["genre"]))

    print("\nClean length status:")
    print(df["clean_length_status"].value_counts())

    print("\nClean word count summary:")
    print(df["clean_word_count"].describe())


def preprocess_raw_outputs(
    raw_path: str | Path = RAW_OUTPUTS_PATH,
    output_path: str | Path = CLEAN_OUTPUTS_PATH,
) -> pd.DataFrame:
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    raw_path : str | Path
        Path to raw model outputs.
    output_path : str | Path
        Path where cleaned outputs should be saved.

    Returns
    -------
    pd.DataFrame
        Cleaned model output dataframe.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_raw_outputs(raw_path)
    validate_raw_outputs(df)

    df = filter_successful_valid_outputs(df)
    df = remove_duplicate_outputs(df)
    df = add_clean_columns(df)

    df.to_csv(output_path, index=False)

    summarize_clean_outputs(df)

    print(f"\nSaved cleaned outputs to: {output_path}")

    return df