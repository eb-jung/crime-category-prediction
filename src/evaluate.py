"""
evaluate.py
-----------
Evaluation utilities for multi-class crime category prediction.

Primary metric: multiclass log loss (matches the original Kaggle competition).
Secondary metric: accuracy (easier to communicate).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, classification_report


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray,
             n_classes: int, label_names: list[str] | None = None) -> dict:
    """
    Compute accuracy and log loss, padding probability matrix for any classes
    absent from the training sample.

    Returns a dict with keys: accuracy, log_loss, report_df.
    """
    all_classes = np.arange(n_classes)

    # Pad columns for classes not seen during training
    if y_proba.shape[1] < n_classes:
        present = np.unique(y_pred) if y_proba.shape[1] == len(np.unique(y_pred)) else np.arange(y_proba.shape[1])
        missing = np.setdiff1d(all_classes, present)
        eps = 1e-9
        extra = np.full((y_proba.shape[0], len(missing)), eps)
        y_proba = np.insert(y_proba, missing, extra, axis=1)
        y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)

    acc = accuracy_score(y_true, y_pred)
    ll = log_loss(y_true, y_proba, labels=all_classes)

    all_labels = list(range(n_classes))
    report = classification_report(
        y_true, y_pred,
        labels=all_labels,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    report_df = (
        pd.DataFrame(report)
        .T
        .drop(index=["accuracy", "macro avg", "weighted avg"], errors="ignore")
        .sort_values("support", ascending=False)
    )

    return {"accuracy": acc, "log_loss": ll, "report_df": report_df}


def summarise_results(results: dict[str, dict]) -> pd.DataFrame:
    """
    Build a tidy comparison table from a dict of {model_name: evaluate() output}.

    Returns a DataFrame sorted by log loss (ascending).
    """
    rows = [
        {"Model": name, "Accuracy": r["accuracy"], "Log Loss": r["log_loss"]}
        for name, r in results.items()
    ]
    return (
        pd.DataFrame(rows)
        .sort_values("Log Loss")
        .reset_index(drop=True)
        .assign(
            Accuracy=lambda d: d["Accuracy"].map("{:.1%}".format),
            **{"Log Loss": lambda d: d["Log Loss"].map("{:.4f}".format)},
        )
    )
