"""Evaluation."""

# standard library imports
import time
from typing import Any, Literal, Optional

# related third party imports
import structlog
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.metrics import accuracy_score
from pydantic import BaseModel
from langfuse import Langfuse

# local application/library specific imports
from tools.constants import Q_CORRECT_OPTION_ID, S_OPTION_ID
from tools.utils import format_time, BatchCallback
from prompt.json_schema import validate_output

# set up logger
logger = structlog.get_logger(__name__)


def predict(
    chain,
    data: list,
    prefix: Literal["val", "test"],
    structured: bool,
    json_schema: BaseModel,
    langfuse_handler,
) -> dict:
    """Predict.

    Parameters
    ----------
    chain : _type_
        Langchain chain
    data : list
        Data to predict
    prefix : Literal["val", "test"]
        Prefix.
    structured : bool
        Whether model supports structured output
    json_schema : BaseModel
        Pydantic schema
    langfuse_handler : _type_
        Langfuse handler for Langchain callback

    Returns
    -------
    dict
        Logs
    """
    logger.info("Predict - start", split=prefix)
    pred_start_time = time.time()
    cb = BatchCallback(len(data))  # init callback
    preds_raw = chain.batch(data, config={"callbacks": [cb, langfuse_handler]})
    cb.progress_bar.close()
    if structured:
        # get all raw outputs
        preds_raw = [output["raw"] for output in preds_raw]
    preds_validated = validate_output(preds_raw, schema=json_schema)
    pred_time = time.time() - pred_start_time
    logger.info("Predict - end", time=format_time(pred_time))
    logs = {
        f"{prefix}_preds_raw": preds_raw,
        f"{prefix}_preds_validated": preds_validated,
        f"{prefix}_pred_time": pred_time,
    }
    return logs


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


def evaluate(
    preds_validated: list,
    dataset: pd.DataFrame,
    prefix: Literal["val", "test"],
    langfuse_session: Langfuse,
    trace_id: Optional[str] = None,
) -> dict:
    """Evaluate.

    Parameters
    ----------
    preds_validated : list
        Validated predictions.
    dataset : pd.Dataframe
        Dataset.
    prefix : Literal["val", "test"]
        Prefix.
    langfuse_session : Langfuse
        Langfuse session.
    trace_id : Optional[str] = None
        Langfuse trace ID. If None, not logged.

    Returns
    -------
    dict
        Logs
    """
    logger.info("Evaluate - start", split=prefix)
    # NOTE: "student_answer" refers to the structured output class
    y_val_pred = np.array([output.student_answer for output in preds_validated])
    y_val_student = dataset[S_OPTION_ID].to_numpy()
    y_val_true = dataset[Q_CORRECT_OPTION_ID].to_numpy()
    metrics = compute_metrics(
        y_val_pred=y_val_pred,
        y_val_true=y_val_true,
        y_val_student=y_val_student,
    )
    logger.info("Evaluate - end", accuracy=metrics["acc_student_pred"])
    logs = {
        f"{prefix}_metrics": metrics,
        f"{prefix}_y_pred": y_val_pred,
        f"{prefix}_y_true": y_val_true,
        f"{prefix}_y_student": y_val_student,
    }
    if trace_id is not None:
        langfuse_session.score(
            trace_id=trace_id,
            name=f"{prefix}_accuracy",
            value=metrics["acc_student_pred"],
            data_type="NUMERIC",
        )
        langfuse_session.score(
            trace_id=trace_id,
            name=f"{prefix}_prop_invalid",
            value=metrics["prop_invalid"],
            data_type="NUMERIC",
        )
    return logs
