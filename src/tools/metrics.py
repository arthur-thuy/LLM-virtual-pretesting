"""Metrics."""

# standard library imports
from typing import Any

# related third party imports
import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import accuracy_score

# local application/library specific imports
# /


def compute_metrics(
    y_val_pred: ArrayLike,
    y_val_true: ArrayLike,
    y_val_student: ArrayLike,
) -> dict[str, Any]:
    """Compute metrics.

    Parameters
    ----------
    y_val_pred : ArrayLike
        Predicted values.
    y_val_true : ArrayLike
        True values.
    y_val_student : ArrayLike
        Student values.

    Returns
    -------
    dict[str, Any]
        Metrics
    """
    # compute metrics
    metrics = {
        "acc_student_pred": accuracy_score(y_true=y_val_student, y_pred=y_val_pred),
        "acc_true_student": accuracy_score(y_true=y_val_true, y_pred=y_val_student),
        "acc_true_pred": accuracy_score(y_true=y_val_true, y_pred=y_val_pred),
        "prop_invalid": np.mean(y_val_pred == -1),
    }
    return metrics
