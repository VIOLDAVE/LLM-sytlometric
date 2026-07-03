"""
Statistical testing module for LLM stylometric fingerprinting.

This module tests whether stylometric features differ significantly across
model families. It uses non-parametric tests because stylometric features
are often skewed and may not satisfy normality assumptions.

Main tests:
- Kruskal-Wallis omnibus tests across model families;
- Benjamini-Hochberg FDR correction;
- epsilon-squared effect size;
- genre-specific robustness tests;
- pairwise Mann-Whitney U post-hoc tests for the strongest features.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu

# Attempt dynamic import of statsmodels' multipletests. Some environments
# (e.g., linting or minimal CI) may not have statsmodels installed; use
# a graceful fallback in that case.
try:
    import importlib

    _smm = importlib.import_module("statsmodels.stats.multitest")
    multipletests = getattr(_smm, "multipletests")
    STATSMODELS_AVAILABLE = True
except Exception:
    multipletests = None
    STATSMODELS_AVAILABLE = False

from src.config import FEATURE_DIR, OUTPUT_DIR


FEATURE_PATH = FEATURE_DIR / "final" / "stylometric_features.csv"
STAT_OUTPUT_DIR = OUTPUT_DIR / "statistical_tests"


def load_feature_data(path: str | Path = FEATURE_PATH) -> pd.DataFrame:
    """
    Load stylometric feature dataset.

    Parameters
    ----------
    path : str | Path
        Path to stylometric_features.csv.

    Returns
    -------
    pd.DataFrame
        Feature dataframe.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")

    return pd.read_csv(path)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Identify stylometric feature columns.

    Parameters
    ----------
    df : pd.DataFrame
        Feature dataframe.

    Returns
    -------
    list[str]
        Columns beginning with stylo_.
    """
    return [column for column in df.columns if column.startswith("stylo_")]


def epsilon_squared_kruskal(
    h_statistic: float,
    n_observations: int,
    n_groups: int,
) -> float:
    """
    Compute epsilon-squared effect size for Kruskal-Wallis test.

    Formula
    -------
    epsilon_squared = (H - k + 1) / (n - k)

    where:
    - H is the Kruskal-Wallis test statistic;
    - k is the number of groups;
    - n is the total number of observations.

    Parameters
    ----------
    h_statistic : float
        Kruskal-Wallis H statistic.
    n_observations : int
        Total number of observations.
    n_groups : int
        Number of groups.

    Returns
    -------
    float
        Epsilon-squared effect size.
    """
    if pd.isna(h_statistic):
        return np.nan

    denominator = n_observations - n_groups

    if denominator <= 0:
        return 0.0

    effect = (h_statistic - n_groups + 1) / denominator

    return max(0.0, float(effect))


def interpret_effect_size(epsilon_squared: float) -> str:
    """
    Interpret epsilon-squared effect size using practical thresholds.

    These thresholds are approximate and used for descriptive interpretation.

    Parameters
    ----------
    epsilon_squared : float
        Effect size.

    Returns
    -------
    str
        Effect size category.
    """
    if pd.isna(epsilon_squared):
        return "unknown"

    if epsilon_squared >= 0.14:
        return "large"

    if epsilon_squared >= 0.06:
        return "medium"

    if epsilon_squared >= 0.01:
        return "small"

    return "negligible"


def run_kruskal_tests_by_model(
    df: pd.DataFrame,
    feature_columns: list[str],
    group_column: str = "model_family",
) -> pd.DataFrame:
    """
    Run Kruskal-Wallis tests for each stylometric feature across model families.

    Kruskal-Wallis is an omnibus test. It tests whether at least one model
    family has a different distribution for a given feature.

    Parameters
    ----------
    df : pd.DataFrame
        Feature dataframe.
    feature_columns : list[str]
        Stylometric feature columns.
    group_column : str
        Grouping column, usually model_family.

    Returns
    -------
    pd.DataFrame
        Test results for each feature.
    """
    results = []

    groups = sorted(df[group_column].dropna().unique())
    n_groups = len(groups)
    n_observations = len(df)

    for feature in feature_columns:
        grouped_values = []

        for group in groups:
            group_mask = df[group_column] == group
            group_series = pd.Series(df.loc[group_mask, feature])
            group_values = group_series.dropna().to_numpy(dtype=float)

            if len(group_values) > 0:
                grouped_values.append(group_values)

        if len(grouped_values) < 2:
            continue

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                h_statistic, p_value = kruskal(*grouped_values)
        except ValueError:
            h_statistic, p_value = np.nan, np.nan

        epsilon_sq = epsilon_squared_kruskal(
            h_statistic=h_statistic,
            n_observations=n_observations,
            n_groups=n_groups,
        )

        model_means = (
            df.groupby(group_column)[feature]
            .mean()
            .to_dict()
        )

        max_mean_model = max(model_means, key=lambda key: model_means[key])
        min_mean_model = min(model_means, key=lambda key: model_means[key])

        results.append(
            {
                "feature": feature,
                "kruskal_h": h_statistic,
                "p_value": p_value,
                "epsilon_squared": epsilon_sq,
                "effect_size_interpretation": interpret_effect_size(epsilon_sq),
                "max_mean_model": max_mean_model,
                "min_mean_model": min_mean_model,
                "max_mean": model_means[max_mean_model],
                "min_mean": model_means[min_mean_model],
                "mean_difference": model_means[max_mean_model] - model_means[min_mean_model],
                "n_observations": n_observations,
                "n_groups": n_groups,
            }
        )

    return pd.DataFrame(results)


def apply_fdr_correction(
    results_df: pd.DataFrame,
    p_column: str = "p_value",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Apply Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    results_df : pd.DataFrame
        Statistical test results.
    p_column : str
        Name of p-value column.
    alpha : float
        Significance level.

    Returns
    -------
    pd.DataFrame
        Results with corrected p-values.
    """
    results_df = results_df.copy()

    if results_df.empty:
        return results_df

    valid_p = results_df[p_column].notna()

    results_df["p_value_fdr"] = np.nan
    results_df["significant_fdr_0_05"] = False

    if valid_p.sum() == 0:
        results_df["neg_log10_p_fdr"] = np.nan
        return results_df

    if STATSMODELS_AVAILABLE and multipletests is not None:
        rejected, corrected_p, _, _ = multipletests(
            results_df.loc[valid_p, p_column],
            alpha=alpha,
            method="fdr_bh",
        )

        results_df.loc[valid_p, "p_value_fdr"] = corrected_p
        results_df.loc[valid_p, "significant_fdr_0_05"] = rejected
    else:
        ranked = results_df.loc[valid_p, p_column].rank(method="first")
        m_tests = valid_p.sum()

        corrected_p = (
            results_df.loc[valid_p, p_column] * m_tests / ranked
        ).clip(upper=1.0)

        results_df.loc[valid_p, "p_value_fdr"] = corrected_p
        results_df.loc[valid_p, "significant_fdr_0_05"] = corrected_p < alpha

    results_df["neg_log10_p_fdr"] = -np.log10(
        results_df["p_value_fdr"].replace(0, np.nextafter(0, 1))
    )

    return results_df


def run_genre_specific_tests(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Run model-family Kruskal-Wallis tests separately within each genre.

    This checks whether model-family differences persist when genre is held
    constant.

    Parameters
    ----------
    df : pd.DataFrame
        Feature dataframe.
    feature_columns : list[str]
        Stylometric feature columns.

    Returns
    -------
    pd.DataFrame
        Genre-specific test results.
    """
    genre_results = []

    for genre, genre_df in df.groupby("genre"):
        result = run_kruskal_tests_by_model(
            genre_df,
            feature_columns,
            group_column="model_family",
        )

        if not result.empty:
            result["genre"] = genre
            genre_results.append(result)

    if not genre_results:
        return pd.DataFrame()

    genre_results_df = pd.concat(genre_results, ignore_index=True)
    genre_results_df = apply_fdr_correction(genre_results_df)

    genre_results_df = genre_results_df.sort_values(
        ["genre", "significant_fdr_0_05", "p_value_fdr", "epsilon_squared"],
        ascending=[True, False, True, False],
    ).reset_index(drop=True)

    return genre_results_df


def run_pairwise_posthoc_tests(
    df: pd.DataFrame,
    features: list[str],
    group_column: str = "model_family",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Run pairwise Mann-Whitney U tests between model families.

    Kruskal-Wallis is an omnibus test. It tells us whether at least one
    model family differs from the others, but it does not identify which
    model pairs are different. This function performs pairwise post-hoc
    tests for selected significant features.

    Parameters
    ----------
    df : pd.DataFrame
        Stylometric feature dataframe.
    features : list[str]
        Selected stylometric features to test.
    group_column : str
        Grouping column, usually model_family.
    alpha : float
        Significance level for FDR correction.

    Returns
    -------
    pd.DataFrame
        Pairwise post-hoc test results.
    """
    results = []

    groups = sorted(df[group_column].dropna().unique())

    for feature in features:
        for group_a, group_b in combinations(groups, 2):
            values_a = (
                pd.Series(df.loc[df[group_column] == group_a, feature])
                .dropna()
                .to_numpy(dtype=float)
            )

            values_b = (
                pd.Series(df.loc[df[group_column] == group_b, feature])
                .dropna()
                .to_numpy(dtype=float)
            )

            if len(values_a) == 0 or len(values_b) == 0:
                continue

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    statistic, p_value = mannwhitneyu(
                        values_a,
                        values_b,
                        alternative="two-sided",
                    )
            except ValueError:
                statistic, p_value = np.nan, np.nan

            mean_a = float(np.mean(values_a))
            mean_b = float(np.mean(values_b))

            results.append(
                {
                    "feature": feature,
                    "group_a": group_a,
                    "group_b": group_b,
                    "mannwhitney_u": statistic,
                    "p_value": p_value,
                    "mean_group_a": mean_a,
                    "mean_group_b": mean_b,
                    "mean_difference_a_minus_b": mean_a - mean_b,
                    "absolute_mean_difference": abs(mean_a - mean_b),
                    "higher_mean_group": group_a if mean_a > mean_b else group_b,
                    "lower_mean_group": group_b if mean_a > mean_b else group_a,
                }
            )

    posthoc_df = pd.DataFrame(results)

    if posthoc_df.empty:
        return posthoc_df

    posthoc_df = apply_fdr_correction(
        posthoc_df,
        p_column="p_value",
        alpha=alpha,
    )

    posthoc_df = posthoc_df.sort_values(
        ["feature", "p_value_fdr", "absolute_mean_difference"],
        ascending=[True, True, False],
    ).reset_index(drop=True)

    return posthoc_df


def summarize_statistical_results(
    overall_results: pd.DataFrame,
    genre_results: pd.DataFrame | None = None,
    posthoc_results: pd.DataFrame | None = None,
) -> None:
    """
    Print concise summary of statistical test results.

    Parameters
    ----------
    overall_results : pd.DataFrame
        FDR-corrected Kruskal-Wallis results.
    genre_results : pd.DataFrame | None
        Genre-specific results.
    posthoc_results : pd.DataFrame | None
        Pairwise post-hoc results.
    """
    print("Statistical testing summary")
    print("=" * 60)

    print(f"Total features tested: {len(overall_results)}")
    print(
        "Significant features after FDR correction:",
        int(overall_results["significant_fdr_0_05"].sum()),
    )

    print("\nEffect size counts:")
    print(overall_results["effect_size_interpretation"].value_counts())

    print("\nTop 15 significant features by FDR-corrected p-value:")
    top = overall_results.sort_values(
        ["significant_fdr_0_05", "p_value_fdr", "epsilon_squared"],
        ascending=[False, True, False],
    ).head(15)

    display_columns = [
        "feature",
        "kruskal_h",
        "p_value_fdr",
        "epsilon_squared",
        "effect_size_interpretation",
        "max_mean_model",
        "min_mean_model",
    ]

    top_display = top[display_columns].copy()
    top_display["kruskal_h"] = top_display["kruskal_h"].round(3)
    top_display["p_value_fdr"] = top_display["p_value_fdr"].map(lambda x: f"{x:.3e}")
    top_display["epsilon_squared"] = top_display["epsilon_squared"].round(3)

    print(top_display.to_string(index=False))

    if genre_results is not None and not genre_results.empty:
        genre_summary = (
            genre_results.groupby("genre")
            .agg(
                total_features=("feature", "count"),
                significant_features=("significant_fdr_0_05", "sum"),
                mean_epsilon_squared=("epsilon_squared", "mean"),
                median_epsilon_squared=("epsilon_squared", "median"),
            )
            .reset_index()
        )

        genre_summary["share_significant"] = (
            genre_summary["significant_features"] / genre_summary["total_features"]
        )

        print("\nGenre-specific summary:")
        print(genre_summary.to_string(index=False))

    if posthoc_results is not None and not posthoc_results.empty:
        print("\nPairwise post-hoc summary:")
        print(f"Total pairwise tests: {len(posthoc_results)}")
        print(
            "Significant pairwise tests after FDR correction:",
            int(posthoc_results["significant_fdr_0_05"].sum()),
        )

        posthoc_display_columns = [
            "feature",
            "group_a",
            "group_b",
            "p_value_fdr",
            "mean_group_a",
            "mean_group_b",
            "higher_mean_group",
            "lower_mean_group",
        ]

        posthoc_top = posthoc_results.sort_values(
            ["p_value_fdr", "absolute_mean_difference"],
            ascending=[True, False],
        ).head(15)

        posthoc_display = posthoc_top[posthoc_display_columns].copy()
        posthoc_display["p_value_fdr"] = posthoc_display["p_value_fdr"].map(
            lambda x: f"{x:.3e}"
        )
        posthoc_display["mean_group_a"] = posthoc_display["mean_group_a"].round(3)
        posthoc_display["mean_group_b"] = posthoc_display["mean_group_b"].round(3)

        print("\nTop 15 pairwise post-hoc comparisons:")
        print(posthoc_display.to_string(index=False))


def run_statistical_tests(
    feature_path: str | Path = FEATURE_PATH,
    output_dir: str | Path = STAT_OUTPUT_DIR,
    n_posthoc_features: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run the full statistical testing pipeline.

    Parameters
    ----------
    feature_path : str | Path
        Path to stylometric_features.csv.
    output_dir : str | Path
        Directory where statistical test outputs should be saved.
    n_posthoc_features : int
        Number of strongest significant features to use for pairwise
        post-hoc tests.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Overall model-family test results, genre-specific results,
        and pairwise post-hoc results.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_feature_data(feature_path)
    feature_columns = get_feature_columns(df)

    overall_results = run_kruskal_tests_by_model(
        df=df,
        feature_columns=feature_columns,
        group_column="model_family",
    )

    overall_results = apply_fdr_correction(overall_results)

    overall_results = overall_results.sort_values(
        ["significant_fdr_0_05", "p_value_fdr", "epsilon_squared"],
        ascending=[False, True, False],
    ).reset_index(drop=True)

    genre_results = run_genre_specific_tests(
        df=df,
        feature_columns=feature_columns,
    )

    top_posthoc_features = (
        overall_results[overall_results["significant_fdr_0_05"] == True]
        .sort_values(["epsilon_squared", "p_value_fdr"], ascending=[False, True])
        .head(n_posthoc_features)["feature"]
        .tolist()
    )

    posthoc_results = run_pairwise_posthoc_tests(
        df=df,
        features=top_posthoc_features,
        group_column="model_family",
    )

    overall_path = output_dir / "kruskal_model_family_tests.csv"
    genre_path = output_dir / "kruskal_model_family_tests_by_genre.csv"
    posthoc_path = output_dir / "pairwise_posthoc_mannwhitney_tests.csv"

    overall_results.to_csv(overall_path, index=False)
    genre_results.to_csv(genre_path, index=False)
    posthoc_results.to_csv(posthoc_path, index=False)

    summarize_statistical_results(
        overall_results=overall_results,
        genre_results=genre_results,
        posthoc_results=posthoc_results,
    )

    print(f"\nSaved overall statistical tests to: {overall_path}")
    print(f"Saved genre-specific statistical tests to: {genre_path}")
    print(f"Saved pairwise post-hoc tests to: {posthoc_path}")

    return overall_results, genre_results, posthoc_results