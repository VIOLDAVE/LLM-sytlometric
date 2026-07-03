"""
Classification modeling module for LLM stylometric fingerprinting.

This module tests whether stylometric features can predict the model family
that generated a text.

The key methodological requirement is prompt-aware cross-validation:
the same prompt_id must not appear in both train and test folds. This prevents
the classifier from learning prompt-specific content instead of model-specific
style.
"""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
#from sklearn.metrics import confusion_matrix
from sklearn.base import BaseEstimator

from src.config import FEATURE_DIR, OUTPUT_DIR
from src.metrics import (
    compute_classification_metrics,
    build_confusion_matrix_df,
    build_classification_report_df,
)


FEATURE_PATH = FEATURE_DIR / "final" / "stylometric_features.csv"
CLASSIFICATION_OUTPUT_DIR = OUTPUT_DIR / "classification"


CONTENT_ORIENTED_FEATURES = [
    "stylo_type_token_ratio",
    "stylo_hapax_legomena_ratio",
    "stylo_repeated_word_ratio",
    "stylo_yules_k",
    "stylo_simpson_diversity",
    "stylo_honores_r",
    "stylo_mattr_50",
    "stylo_function_word_count",
    "stylo_function_word_ratio",
    "stylo_transition_marker_count",
    "stylo_transition_marker_ratio",
    "stylo_hedging_marker_count",
    "stylo_hedging_marker_ratio",
    "stylo_ai_marker_count",
    "stylo_ai_marker_ratio",
    "stylo_flesch_reading_ease",
    "stylo_gunning_fog",
]


STRUCTURE_FORMAT_FEATURES = [
    "stylo_word_count",
    "stylo_character_count",
    "stylo_sentence_count",
    "stylo_avg_word_length",
    "stylo_avg_sentence_length",
    "stylo_sentence_length_std",
    "stylo_sentence_length_min",
    "stylo_sentence_length_max",
    "stylo_sentence_length_range",
    "stylo_short_sentence_ratio",
    "stylo_long_sentence_ratio",
    "stylo_punctuation_count",
    "stylo_comma_count",
    "stylo_period_count",
    "stylo_question_mark_count",
    "stylo_exclamation_count",
    "stylo_semicolon_count",
    "stylo_colon_count",
    "stylo_dash_count",
    "stylo_quote_count",
    "stylo_punctuation_per_100_words",
    "stylo_comma_per_100_words",
    "stylo_period_per_100_words",
    "stylo_uppercase_word_count",
    "stylo_titlecase_word_count",
    "stylo_uppercase_char_ratio",
]


def load_classification_data(path: str | Path = FEATURE_PATH) -> pd.DataFrame:
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
    Return all stylometric feature columns.

    Parameters
    ----------
    df : pd.DataFrame
        Feature dataframe.

    Returns
    -------
    list[str]
        Stylometric feature columns.
    """
    return [column for column in df.columns if column.startswith("stylo_")]


def build_feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Build feature sets for classification experiments.

    The feature sets compare:
    - all stylometric features;
    - content-oriented features;
    - structure/format features;
    - all features excluding structure/format features.

    Parameters
    ----------
    df : pd.DataFrame
        Feature dataframe.

    Returns
    -------
    dict[str, list[str]]
        Feature-set name mapped to feature columns.
    """
    all_features = get_feature_columns(df)

    available_content_features = [
        feature for feature in CONTENT_ORIENTED_FEATURES
        if feature in df.columns
    ]

    available_structure_features = [
        feature for feature in STRUCTURE_FORMAT_FEATURES
        if feature in df.columns
    ]

    structure_set = set(available_structure_features)

    without_structure_features = [
        feature for feature in all_features
        if feature not in structure_set
    ]

    feature_sets = {
        "all_stylometric_features": all_features,
        "content_oriented_features": available_content_features,
        "structure_format_features": available_structure_features,
        "without_structure_format_features": without_structure_features,
    }

    return {
        name: columns
        for name, columns in feature_sets.items()
        if len(columns) > 0
    }


def build_classifiers(random_state: int = 42) -> dict[str, BaseEstimator]:
    """
    Build classifiers used in the experiment.

    Returns
    -------
    dict[str, BaseEstimator]
        Classifier name mapped to sklearn estimator.
    """
    logistic_regression = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=random_state,
                ),
            ),
        ]
    )

    random_forest = RandomForestClassifier(
        n_estimators=500,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
        max_depth=None,
    )

    return {
        "logistic_regression": logistic_regression,
        "random_forest": random_forest,
    }


def validate_required_columns(
    df: pd.DataFrame,
    target_column: str,
    group_column: str,
) -> None:
    """
    Validate target and group columns.

    Parameters
    ----------
    df : pd.DataFrame
        Feature dataframe.
    target_column : str
        Target label column.
    group_column : str
        Group column for prompt-aware cross-validation.
    """
    missing_columns = [
        column for column in [target_column, group_column]
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if df[target_column].isna().any():
        raise ValueError(f"Target column contains missing values: {target_column}")

    if df[group_column].isna().any():
        raise ValueError(f"Group column contains missing values: {group_column}")


def evaluate_classifier_with_group_cv(
    df: pd.DataFrame,
    feature_columns: list[str],
    classifier: BaseEstimator,
    classifier_name: str,
    feature_set_name: str,
    target_column: str = "model_family",
    group_column: str = "prompt_id",
    n_splits: int = 5,
) -> tuple[dict[str, float | str], np.ndarray, np.ndarray]:
    """
    Evaluate a classifier using prompt-aware GroupKFold cross-validation.

    Parameters
    ----------
    df : pd.DataFrame
        Feature dataframe.
    feature_columns : list[str]
        Feature columns.
    classifier : object
        Sklearn classifier or pipeline.
    classifier_name : str
        Classifier name.
    feature_set_name : str
        Feature-set name.
    target_column : str
        Target column, usually model_family.
    group_column : str
        Group column, usually prompt_id.
    n_splits : int
        Number of cross-validation folds.

    Returns
    -------
    tuple[dict[str, float | str], np.ndarray, np.ndarray]
        Metric summary, true labels, predicted labels.
    """
    validate_required_columns(df, target_column, group_column)

    X = df[feature_columns].astype(float)
    y = df[target_column].astype(str).to_numpy()
    groups = df[group_column].astype(str).to_numpy()

    unique_groups = np.unique(groups)

    if len(unique_groups) < n_splits:
        raise ValueError(
            f"Not enough prompt groups for {n_splits}-fold CV. "
            f"Found {len(unique_groups)} groups."
        )

    cv = GroupKFold(n_splits=n_splits)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        y_pred = cross_val_predict(
            classifier,
            X,
            y,
            groups=groups,
            cv=cv,
            n_jobs=None,
        )

    metrics = compute_classification_metrics(
        y_true=y,
        y_pred=y_pred,
        model_name=classifier_name,
        feature_set=feature_set_name,
    )

    metrics["n_features"] = len(feature_columns)
    metrics["n_observations"] = len(df)
    metrics["n_prompt_groups"] = len(unique_groups)
    metrics["n_splits"] = n_splits

    return metrics, y, y_pred


def save_evaluation_outputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    classifier_name: str,
    feature_set_name: str,
    output_dir: Path,
) -> None:
    """
    Save confusion matrix and classification report.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    labels : list[str]
        Ordered model-family labels.
    classifier_name : str
        Classifier name.
    feature_set_name : str
        Feature-set name.
    output_dir : Path
        Output directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_name = f"{classifier_name}_{feature_set_name}"

    confusion_df = build_confusion_matrix_df(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
    )

    report_df = build_classification_report_df(
        y_true=y_true,
        y_pred=y_pred,
    )

    confusion_df.to_csv(output_dir / f"confusion_matrix_{safe_name}.csv")
    report_df.to_csv(output_dir / f"classification_report_{safe_name}.csv")


def fit_random_forest_importance(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = "model_family",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Fit a random forest on the full dataset and extract feature importances.

    This is used for interpretation after cross-validation evaluation.
    The evaluation metrics are still based on prompt-aware cross-validation.

    Parameters
    ----------
    df : pd.DataFrame
        Feature dataframe.
    feature_columns : list[str]
        Feature columns.
    target_column : str
        Target column.
    random_state : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Random forest feature importance table.
    """
    X = df[feature_columns].astype(float)
    y = df[target_column].astype(str).to_numpy()

    model = RandomForestClassifier(
        n_estimators=500,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
    )

    model.fit(X, y)

    importance_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    importance_df = importance_df.reset_index(drop=True)

    return importance_df


def compute_feature_family_summary(
    importance_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize random-forest importance by feature family.

    Parameters
    ----------
    importance_df : pd.DataFrame
        Feature importance table.

    Returns
    -------
    pd.DataFrame
        Feature-family importance summary.
    """
    family_rules = {
        "length_structure": [
            "word_count",
            "character_count",
            "sentence",
            "avg_word_length",
        ],
        "punctuation": [
            "punctuation",
            "comma",
            "period",
            "question",
            "exclamation",
            "semicolon",
            "colon",
            "dash",
            "quote",
        ],
        "casing": [
            "uppercase",
            "titlecase",
        ],
        "lexical_diversity": [
            "type_token",
            "hapax",
            "repeated",
            "yules",
            "simpson",
            "honores",
            "mattr",
        ],
        "markers_function_words": [
            "function_word",
            "transition",
            "hedging",
            "ai_marker",
        ],
        "readability": [
            "flesch",
            "gunning",
        ],
    }

    def assign_family(feature: str) -> str:
        for family, keywords in family_rules.items():
            if any(keyword in feature for keyword in keywords):
                return family
        return "other"

    importance_df = importance_df.copy()
    importance_df["feature_family"] = importance_df["feature"].apply(assign_family)

    family_summary = (
        importance_df.groupby("feature_family")
        .agg(
            total_importance=("importance", "sum"),
            mean_importance=("importance", "mean"),
            n_features=("feature", "count"),
        )
        .sort_values("total_importance", ascending=False)
        .reset_index()
    )

    return family_summary


def run_classification_modeling(
    feature_path: str | Path = FEATURE_PATH,
    output_dir: str | Path = CLASSIFICATION_OUTPUT_DIR,
    target_column: str = "model_family",
    group_column: str = "prompt_id",
    n_splits: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run the full classification modeling pipeline.

    Parameters
    ----------
    feature_path : str | Path
        Path to stylometric_features.csv.
    output_dir : str | Path
        Output directory.
    target_column : str
        Target column.
    group_column : str
        Group column for prompt-aware cross-validation.
    n_splits : int
        Number of GroupKFold splits.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Performance summary, feature importance table, feature-family summary.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_classification_data(feature_path)
    validate_required_columns(df, target_column, group_column)

    feature_sets = build_feature_sets(df)
    classifiers = build_classifiers()

    labels = sorted(df[target_column].dropna().astype(str).unique())

    performance_rows = []

    print("Classification modeling summary")
    print("=" * 60)
    print(f"Dataset shape: {df.shape}")
    print(f"Target column: {target_column}")
    print(f"Group column: {group_column}")
    print(f"Number of target classes: {len(labels)}")
    print(f"Number of prompt groups: {df[group_column].nunique()}")
    print(f"Cross-validation: GroupKFold with {n_splits} folds")

    for feature_set_name, feature_columns in feature_sets.items():
        print(f"\nFeature set: {feature_set_name}")
        print(f"Number of features: {len(feature_columns)}")

        for classifier_name, classifier in classifiers.items():
            print(f"Evaluating {classifier_name}...")

            metrics, y_true, y_pred = evaluate_classifier_with_group_cv(
                df=df,
                feature_columns=feature_columns,
                classifier=classifier,
                classifier_name=classifier_name,
                feature_set_name=feature_set_name,
                target_column=target_column,
                group_column=group_column,
                n_splits=n_splits,
            )

            performance_rows.append(metrics)

            save_evaluation_outputs(
                y_true=y_true,
                y_pred=y_pred,
                labels=labels,
                classifier_name=classifier_name,
                feature_set_name=feature_set_name,
                output_dir=output_dir,
            )

    performance_df = pd.DataFrame(performance_rows).sort_values(
        ["macro_f1", "accuracy"],
        ascending=False,
    )

    all_features = get_feature_columns(df)

    random_forest_importance = fit_random_forest_importance(
        df=df,
        feature_columns=all_features,
        target_column=target_column,
    )

    feature_family_summary = compute_feature_family_summary(
        random_forest_importance,
    )

    performance_path = output_dir / "model_performance_summary.csv"
    importance_path = output_dir / "feature_importance_random_forest_all_features.csv"
    family_summary_path = output_dir / "feature_family_importance_summary.csv"

    performance_df.to_csv(performance_path, index=False)
    random_forest_importance.to_csv(importance_path, index=False)
    feature_family_summary.to_csv(family_summary_path, index=False)

    print("\nModel performance summary:")
    display_columns = [
        "model_name",
        "feature_set",
        "n_features",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
    ]

    performance_display = performance_df[display_columns].copy()

    for column in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]:
        performance_display[column] = performance_display[column].round(3)

    print(performance_display.to_string(index=False))

    print(f"\nSaved performance summary to: {performance_path}")
    print(f"Saved random-forest feature importance to: {importance_path}")
    print(f"Saved feature-family importance summary to: {family_summary_path}")

    return performance_df, random_forest_importance, feature_family_summary