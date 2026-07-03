"""
Feature summary script for stylometric feature dataset.

This script performs quick quality checks before exploratory analysis:
- dataset shape;
- class balance;
- genre balance;
- missing values;
- constant features;
- feature means by model family;
- features with strongest between-model variation.

Run from project root:

    python scripts/feature_summary.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

FEATURE_PATH = PROJECT_ROOT / "data" / "features" / "final" / "stylometric_features.csv"
SUMMARY_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tables"


def load_features(path: Path = FEATURE_PATH) -> pd.DataFrame:
    """
    Load stylometric feature dataset.
    """
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")

    return pd.read_csv(path)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Return stylometric feature columns.
    """
    return [column for column in df.columns if column.startswith("stylo_")]


def summarize_dataset(df: pd.DataFrame, feature_columns: list[str]) -> None:
    """
    Print core dataset summary.
    """
    print("Stylometric Feature Dataset Summary")
    print("=" * 60)

    print(f"Dataset shape: {df.shape}")
    print(f"Number of stylometric features: {len(feature_columns)}")

    print("\nRows by model family:")
    print(df["model_family"].value_counts().sort_index())

    print("\nRows by genre:")
    print(df["genre"].value_counts().sort_index())

    print("\nModel family × genre balance:")
    print(pd.crosstab(df["model_family"], df["genre"]))


def summarize_missing_and_constant_features(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    """
    Print missing-value and constant-feature diagnostics.
    """
    print("\nMissing values in feature columns:")
    missing_counts = df[feature_columns].isna().sum()
    missing_counts = missing_counts[missing_counts > 0]

    if missing_counts.empty:
        print("No missing values found.")
    else:
        print(missing_counts.sort_values(ascending=False))

    constant_features = [
        column for column in feature_columns
        if df[column].nunique(dropna=False) <= 1
    ]

    print("\nConstant features:")
    if constant_features:
        for feature in constant_features:
            print(f"- {feature}")
    else:
        print("No constant features found.")


def compute_model_means(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Compute mean stylometric features by model family.
    """
    return (
        df.groupby("model_family")[feature_columns]
        .mean()
        .round(4)
        .reset_index()
    )


def compute_between_model_variation(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Rank features by between-model variation.

    This uses the variance of model-family means. It is not a formal
    statistical test, but it helps identify features that vary strongly
    between models before formal testing.
    """
    model_means = df.groupby("model_family")[feature_columns].mean()

    variation = model_means.var(axis=0).sort_values(ascending=False)

    variation_df = variation.reset_index()
    variation_df.columns = ["feature", "between_model_variance"]

    return variation_df


def save_summary_tables(
    model_means: pd.DataFrame,
    variation_df: pd.DataFrame,
    output_dir: Path = SUMMARY_OUTPUT_DIR,
) -> None:
    """
    Save summary tables to outputs/tables.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    model_means_path = output_dir / "feature_means_by_model_family.csv"
    variation_path = output_dir / "feature_between_model_variation.csv"

    model_means.to_csv(model_means_path, index=False)
    variation_df.to_csv(variation_path, index=False)

    print(f"\nSaved model-family feature means to: {model_means_path}")
    print(f"Saved between-model variation table to: {variation_path}")


def main() -> None:
    """
    Run feature summary diagnostics.
    """
    df = load_features()
    feature_columns = get_feature_columns(df)

    summarize_dataset(df, feature_columns)
    summarize_missing_and_constant_features(df, feature_columns)

    model_means = compute_model_means(df, feature_columns)
    variation_df = compute_between_model_variation(df, feature_columns)

    print("\nTop 15 features by between-model variation:")
    print(variation_df.head(15).to_string(index=False))

    print("\nFeature means by model family, first 10 features:")
    preview_columns = ["model_family"] + feature_columns[:10]
    print(model_means[preview_columns].to_string(index=False))

    save_summary_tables(model_means, variation_df)


if __name__ == "__main__":
    main()