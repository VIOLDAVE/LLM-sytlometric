"""
Reusable metric utilities for the LLM stylometric fingerprinting project.

This module contains helper functions for classification evaluation.
Model training belongs in classification.py, while reusable metric summaries
belong here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    feature_set: str,
) -> dict[str, float | str]:
    """
    Compute standard classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    model_name : str
        Name of classifier.
    feature_set : str
        Name of feature set used.

    Returns
    -------
    dict[str, float | str]
        Classification metric summary.
    """
    return {
        "model_name": model_name,
        "feature_set": feature_set,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }


def build_confusion_matrix_df(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
) -> pd.DataFrame:
    """
    Create a labelled confusion matrix dataframe.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    labels : list[str]
        Ordered class labels.

    Returns
    -------
    pd.DataFrame
        Confusion matrix dataframe.
    """
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    return pd.DataFrame(
        matrix,
        index=[f"true_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    )


def build_classification_report_df(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """
    Create a classification report dataframe.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.

    Returns
    -------
    pd.DataFrame
        Classification report as dataframe.
    """
    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    return pd.DataFrame(report).transpose()