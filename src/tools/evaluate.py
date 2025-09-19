"""Evaluation."""

# standard library imports
import time
from typing import Any, Literal, Optional

# related third party imports
import numpy as np
import pandas as pd
import structlog
import scipy
from langfuse import Langfuse
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    root_mean_squared_error,
    balanced_accuracy_score,
    mean_absolute_error,
)

# local application/library specific imports
from prompt.utils import validate_output
from tools.constants import (
    Q_CORRECT_OPTION_ID,
    Q_DIFFICULTY,
    QUESTION_ID,
    S_OPTION_CORRECT,
    S_OPTION_ID,
    STUDENT_ID,
    STUDENT_LEVEL_GROUP,
    DIFFICULTY_MAX,
    DIFFICULTY_MIN,
)
from tools.utils import BatchCallback, format_time

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


def eval_metric_monotonicity(
    y_true: NDArray,
    y_llm: NDArray,
    student_level_group: NDArray,
    y_student: Optional[NDArray] = None,
    student_group_correctness: Optional[NDArray] = None,
    only_kt: bool = False,
) -> float:
    """Evaluate the model monotonicity across different student levels.

    Adapted from EMNLP paper Luca

    Parameters
    ----------
    y_true : NDArray
        Ground truth labels for the questions.
    y_llm : NDArray
        LLM's answers for the questions.
    student_level_group : NDArray
        Student levels for the interactions.
    y_student : Optional[NDArray]
        Student's answers for the questions. None if roleplaying.
    student_group_correctness : Optional[NDArray]
        Pre-computed correctness of the student group.

    Returns
    -------
    float
        Monotonicity score of the model's performance.
    """
    # checks
    none_count = sum([y_student is None, student_group_correctness is None])
    if none_count != 1:
        # raise ValueError(
        #     "Exactly one of 'y_student' or 'student_group_correctness' must be None"
        # )
        student_group_correctness = [0.0, 0.25, 0.5, 0.75, 1.0]
        logger.warning(
            "No 'student_group_correctness' provided. Using default correctness values."
        )

    if only_kt:
        # directly outputs whether LLM is correct
        llm_correct = y_llm.copy()
    else:
        llm_correct = np.equal(y_llm, y_true)
    # compute student_group_correctness from val set (replication)
    # or take array from train set (roleplay)
    if student_group_correctness is None:
        student_correct = np.equal(y_student, y_true)
        df = pd.DataFrame(
            {
                "student_level_group": student_level_group,
                "student_correct": student_correct,
                "llm_correct": llm_correct,
            }
        )
        df["student_level_group"] = df["student_level_group"].astype(int)
        df = df.sort_values(
            "student_level_group", ascending=True
        )  # NOTE: inputs are digits because using unformatted dataset
        student_group_correctness = (
            df.groupby("student_level_group", sort=False)["student_correct"]
            .mean()
            .to_numpy()
        )
    else:
        df = pd.DataFrame(
            {
                "student_level_group": student_level_group,
                "llm_correct": llm_correct,
            }
        )
        df["student_level_group"] = df["student_level_group"].astype(int)
        df = df.sort_values(
            "student_level_group", ascending=True
        )  # NOTE: inputs are digits because using unformatted dataset

    llm_group_correctness = (
        df.groupby("student_level_group", sort=False)["llm_correct"].mean().to_numpy()
    )
    assert len(student_group_correctness) == len(
        llm_group_correctness
    ), "Number of student groups and LLM groups must match"

    try:
        correlation_score = scipy.stats.linregress(
            llm_group_correctness, student_group_correctness
        ).rvalue
        penalty_non_monotonicity = np.sum(
            [
                (
                    np.sqrt(
                        np.abs(llm_group_correctness[i + 1] - llm_group_correctness[i])
                    )
                    if llm_group_correctness[i + 1] < llm_group_correctness[i]
                    else 0.0
                )
                for i in range(len(llm_group_correctness) - 1)
            ]
        )
        score = (correlation_score - penalty_non_monotonicity).item()
    except ValueError as e:
        logger.error(
            "Error computing monotonicity score",
            error=str(e),
            llm_group_correctness=llm_group_correctness,
            student_group_correctness=student_group_correctness,
        )
        score = np.nan
    return score, llm_group_correctness


def compute_answer_distr(y_val_pred: ArrayLike) -> ArrayLike:
    """Compute answer distribution.

    Parameters
    ----------
    y_val_pred : ArrayLike
        Predicted values.

    Returns
    -------
    ArrayLike
        Answer distribution over options 1-4.
    """
    unique, counts = np.unique(y_val_pred, return_counts=True)
    dict_counts = {int(k): int(v) for k, v in zip(unique, counts)}
    # fill in value 0 if one of keys 1-4 is missing
    for k in range(1, 5):
        if k not in dict_counts:
            dict_counts[k] = 0
    print(dict_counts)
    return np.array([dict_counts[k]/y_val_pred.shape[0] for k in range(1, 5)])


def compute_metrics_replication(
    y_val_pred: ArrayLike,
    y_val_true: ArrayLike,
    y_val_student: ArrayLike,
    only_kt: bool = False,
) -> dict[str, Any]:
    """Compute metrics of student replication.

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
    # helpers for knowledge tracing
    student_correct = np.equal(y_val_student, y_val_true)
    if only_kt:
        # directly outputs whether LLM is correct
        llm_correct = y_val_pred.copy()
    else:
        llm_correct = np.equal(y_val_pred, y_val_true)

    # compute metrics
    kt_metrics = {
        # knowledge tracing
        "acc_kt": accuracy_score(y_true=student_correct, y_pred=llm_correct),
        "bal_acc_kt": balanced_accuracy_score(
            y_true=student_correct, y_pred=llm_correct
        ).item(),
        "f1_kt": f1_score(y_true=student_correct, y_pred=llm_correct, average="binary"),
        "student_correctness": np.mean(student_correct).item(),
        "llm_correctness": np.mean(llm_correct).item(),
    }
    if only_kt:
        metrics = kt_metrics
    else:
        # only compute these of non-KT
        non_kt_metrics = {
            "acc": accuracy_score(y_true=y_val_student, y_pred=y_val_pred),
            "bal_acc": balanced_accuracy_score(
                y_true=y_val_student, y_pred=y_val_pred
            ).item(),
            "prop_invalid": np.mean(y_val_pred == -1),
            # F1 micro
            "f1_micro": f1_score(
                y_true=y_val_student, y_pred=y_val_pred, average="micro"
            ),
            # F1 macro
            "f1_macro": f1_score(
                y_true=y_val_student, y_pred=y_val_pred, average="macro"
            ),
            # F1 weighted
            "f1_weighted": f1_score(
                y_true=y_val_student, y_pred=y_val_pred, average="weighted"
            ),
            # answer distribution
            "answer_distr": compute_answer_distr(y_val_pred=y_val_pred),
        }
        metrics = {**kt_metrics, **non_kt_metrics}
    return metrics


def evaluate_replication(
    preds_validated: list,
    dataset: pd.DataFrame,
    prefix: Literal["val", "test"],
    langfuse_session: Langfuse,
    trace_id: Optional[str] = None,
    only_kt: bool = False,
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
    if only_kt:
        y_val_pred = np.array([output.student_correct for output in preds_validated])
    else:
        y_val_pred = np.array([output.student_answer for output in preds_validated])
    y_val_student = dataset[S_OPTION_ID].to_numpy()
    y_val_true = dataset[Q_CORRECT_OPTION_ID].to_numpy()
    student_ids = dataset[STUDENT_ID].to_numpy()
    metrics = compute_metrics_replication(
        y_val_pred=y_val_pred,
        y_val_true=y_val_true,
        y_val_student=y_val_student,
        only_kt=only_kt,
    )
    if not only_kt:
        logger.info(
            "Evaluate - end",
            acc=round(metrics["acc"], 2),
            acc_kt=round(metrics["acc_kt"], 2),
            bal_acc=round(metrics["bal_acc"], 2),
            bal_acc_kt=round(metrics["bal_acc_kt"], 2),
            f1_macro=round(metrics["f1_macro"], 2),
            correctness_llm=round(metrics["llm_correctness"], 2),
            correctness_student=round(metrics["student_correctness"], 2),
        )
    else:
        logger.info(
            "Evaluate - end (KT)",
            acc_kt=round(metrics["acc_kt"], 2),
            bal_acc_kt=round(metrics["bal_acc_kt"], 2),
            f1_kt=round(metrics["f1_kt"], 2),
            correctness_llm=round(metrics["llm_correctness"], 2),
            correctness_student=round(metrics["student_correctness"], 2),
        )

    # if trace_id is not None:
    #     langfuse_session.score(
    #         trace_id=trace_id,
    #         name=f"{prefix}_f1_macro",
    #         value=metrics["f1_macro"],
    #         data_type="NUMERIC",
    #     )
    #     langfuse_session.score(
    #         trace_id=trace_id,
    #         name=f"{prefix}_prop_invalid",
    #         value=metrics["prop_invalid"],
    #         data_type="NUMERIC",
    #     )
    metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}
    preds = {
        f"{prefix}_y_pred": y_val_pred,
        f"{prefix}_y_true": y_val_true,
        f"{prefix}_y_student": y_val_student,
        f"{prefix}_student_ids": student_ids,
    }
    return metrics, preds


def compute_metrics_roleplay(
    y_val_pred: ArrayLike,
    y_val_true: ArrayLike,
    student_level_group: ArrayLike,
    student_group_correctness: Optional[ArrayLike] = None,
    # question_ids: ArrayLike,
    # prop_df: pd.DataFrame,
    only_kt: bool = False,
) -> dict[str, Any]:
    """Compute metrics for student roleplaying.

    Parameters
    ----------
    y_val_pred : ArrayLike
        Predicted values.
    y_val_true : ArrayLike
        True values.
    student_level_group : ArrayLike
        Student levels for the interactions.
    student_group_correctness : ArrayLike
        Pre-computed correctness of the student group.
    question_ids : ArrayLike
        Question IDs.
    prop_df : pd.DataFrame
        A DataFrame containing the student proportions of each answer option.

    Returns
    -------
    dict[str, Any]
        Metrics
    """
    if only_kt:
        # directly outputs whether LLM is correct
        llm_correct = y_val_pred.copy()
    else:
        llm_correct = np.equal(y_val_pred, y_val_true)
    # compute metrics
    monotonicity, llm_group_correctness = eval_metric_monotonicity(
        y_true=y_val_true,
        y_llm=y_val_pred,
        student_level_group=student_level_group,
        y_student=None,
        student_group_correctness=student_group_correctness,
        only_kt=only_kt,
    )
    kt_metrics = {
        # knowledge tracing
        "llm_correctness": np.mean(llm_correct).item(),
        "monotonicity": monotonicity,
        "llm_group_correctness": llm_group_correctness,
    }
    if only_kt:
        metrics = kt_metrics
    else:
        # only compute these of non-KT
        non_kt_metrics = {
            "prop_invalid": np.mean(y_val_pred == -1),
            # answer distribution
            "answer_distr": compute_answer_distr(y_val_pred=y_val_pred),
            # "distractor_alignment": eval_distractor_alignment(
            #     y_true_array=y_val_true,
            #     y_llm_array=y_val_pred,
            #     student_level_group_array=student_level_group,
            #     question_id_array=question_ids,
            #     prop_df=prop_df,
            # ).item(),
        }
        metrics = {**kt_metrics, **non_kt_metrics}
    return metrics


def evaluate_roleplay(
    preds_validated: list,
    dataset: pd.DataFrame,
    prefix: Literal["val", "test"],
    student_group_correctness: Optional[NDArray] = None,
    only_kt: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate.

    Parameters
    ----------
    preds_validated : list
        Validated predictions.
    dataset : pd.Dataframe
        Dataset.
    prefix : Literal["val", "test"]
        Prefix.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]]
        Metrics and predictions.
    """
    logger.info("Evaluate - start", split=prefix)
    if only_kt:
        y_val_pred = np.array([output.student_correct for output in preds_validated])
    else:
        # NOTE: "student_answer" refers to the structured output class
        y_val_pred = np.array([output.student_answer for output in preds_validated])

    y_val_true = dataset[Q_CORRECT_OPTION_ID].to_numpy()
    student_level_group = dataset[STUDENT_LEVEL_GROUP].to_numpy()
    # question_ids = dataset[QUESTION_ID].to_numpy()

    # # read proportions
    # df_prop = pd.read_csv("../data/platinum/dbe_kt22_proportions_val.csv")
    # df_prop["dict"] = df_prop["dict"].apply(eval)

    metrics = compute_metrics_roleplay(
        y_val_pred=y_val_pred,
        y_val_true=y_val_true,
        student_level_group=student_level_group,
        student_group_correctness=student_group_correctness,
        # question_ids=question_ids,
        # prop_df=df_prop,
    )
    if not only_kt:
        logger.info(
            "Evaluate - end",
            correctness_llm=round(metrics["llm_correctness"], 2),
            monotonicity=round(metrics["monotonicity"], 2),
            llm_group_correctness=[
                round(x.item(), 2) for x in metrics["llm_group_correctness"]
            ],
            answer_distr=[
                round(x.item(), 2) for x in metrics["answer_distr"]
            ],
            # distr_alignment=round(metrics["distractor_alignment"], 2),
        )
    else:
        logger.info(
            "Evaluate - end",
            correctness_llm=round(metrics["llm_correctness"], 2),
            monotonicity=round(metrics["monotonicity"], 2),
            llm_group_correctness=[
                round(x.item(), 2) for x in metrics["llm_group_correctness"]
            ],
        )
    metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}
    preds = {
        f"{prefix}_y_pred": y_val_pred,
        f"{prefix}_y_true": y_val_true,
        f"{prefix}_student_level_group": student_level_group,
    }
    return metrics, preds


def evaluate_q_difficulty(
    preds_validated: list,
    dataset: pd.DataFrame,
    prefix: Literal["val", "test"],
    difficulty_range: tuple[float, float],
    only_kt: bool = False,
):
    from tools.irt_estimator import irt_estimation

    logger.info("Evaluate question difficulty - start", split=prefix)
    # prepare data for IRT estimation
    if only_kt:
        y_val_pred = np.array([output.student_correct for output in preds_validated])
        # directly outputs whether LLM is correct
        llm_correct = y_val_pred.copy()
    else:
        y_val_pred = np.array([output.student_answer for output in preds_validated])
        y_val_true = dataset[Q_CORRECT_OPTION_ID].to_numpy()
        llm_correct = np.equal(y_val_pred, y_val_true)

    df = pd.DataFrame(
        {
            STUDENT_ID: dataset[STUDENT_LEVEL_GROUP],
            QUESTION_ID: dataset[QUESTION_ID],
            S_OPTION_CORRECT: llm_correct.astype(int),
        }
    )
    # Compute IRT parameters
    _, difficulty_dict, _ = irt_estimation(interactions_df=df)
    # rescale difficulty for specific dataset
    logger.info(
        "Rescaling difficulty range",
        old_range=(DIFFICULTY_MIN, DIFFICULTY_MAX),
        new_range=difficulty_range,
    )
    new_min_diff, new_max_diff = difficulty_range
    for key, value in difficulty_dict.items():
        difficulty_dict[key] = (
            (value - DIFFICULTY_MIN) / (DIFFICULTY_MAX - DIFFICULTY_MIN)
        ) * (new_max_diff - new_min_diff) + new_min_diff

    df_q_tmp = dataset.copy()
    # NOTE: only retain unique set of question IDs, because repeated 5 times in dataset!
    df_q_tmp = df_q_tmp.drop_duplicates(subset=[QUESTION_ID]).reset_index(drop=True)
    df_q_tmp["q_diff_pred"] = df_q_tmp[QUESTION_ID].map(difficulty_dict)

    # compute RMSE and MAE
    metrics = {
        f"{prefix}_rmse": root_mean_squared_error(
            y_true=df_q_tmp[Q_DIFFICULTY].to_numpy(),
            y_pred=df_q_tmp["q_diff_pred"].to_numpy(),
        ),
        f"{prefix}_mae": mean_absolute_error(
            y_true=df_q_tmp[Q_DIFFICULTY].to_numpy(),
            y_pred=df_q_tmp["q_diff_pred"].to_numpy(),
        ),
    }
    logger.info(
        "Evaluate question difficulty - end",
        split=prefix,
        rmse=round(metrics[f"{prefix}_rmse"], 2),
        mae=round(metrics[f"{prefix}_mae"], 2),
    )
    preds = {
        f"{prefix}_y_pred": df_q_tmp["q_diff_pred"].to_numpy(),
        f"{prefix}_y_true": df_q_tmp[Q_DIFFICULTY].to_numpy(),
        f"{prefix}_df_input": df,
    }
    return metrics, preds


def alignment_score_single(y_true: int, y_llm: int, dict_props: dict) -> float:
    """Calculate the alignment score for a single question.

    Parameters
    ----------
    y_true : int
        The true answer.
    y_llm : int
        The LLM predicted answer.
    dict_props : dict
        A dictionary mapping answer options to student proportions.

    Returns
    -------
    float
        The alignment score.
    """
    llm_answer_incorrect = y_true != y_llm
    if llm_answer_incorrect:
        dict_tmp = dict_props.copy()
        prop_answer_llm = dict_tmp[y_llm]
        # remove correct idx from dict
        dict_tmp.pop(y_true, None)
        idx_most_popular_distractor = max(dict_tmp, key=dict_tmp.get)
        prop_most_popular_distractor = dict_tmp[idx_most_popular_distractor]
        # calculate score
        try:
            score = prop_answer_llm / prop_most_popular_distractor
        except ZeroDivisionError:
            score = 0.0
    else:
        score = np.nan
    return score


def eval_distractor_alignment(
    y_true_array: NDArray,
    y_llm_array: NDArray,
    student_level_group_array: NDArray,
    question_id_array: NDArray,
    prop_df: pd.DataFrame,
) -> float:
    """Evaluate the alignment of distractor answers.

    Parameters
    ----------
    y_true_array : NDArray
        The true answers.
    y_llm_array : NDArray
        The LLM predicted answers.
    student_level_group_array : NDArray
        The student level groups.
    question_id_array : NDArray
        The question IDs.
    prop_df : pd.DataFrame
        A DataFrame containing the student proportions of each answer option.

    Returns
    -------
    float
        The mean alignment score.
    """
    # convert to int
    student_level_group_array = student_level_group_array.astype(int)

    scores = []
    for y_true, y_llm, student_level_group, question_id in zip(
        y_true_array, y_llm_array, student_level_group_array, question_id_array
    ):
        dict_tmp = (
            prop_df[
                (prop_df["question_id"] == question_id)
                & (prop_df["student_level_group"] == student_level_group)
            ]["dict"]
            .item()
            .copy()
        )
        score = alignment_score_single(y_true=y_true, y_llm=y_llm, dict_props=dict_tmp)
        scores.append(score)

    # compute mean and ignore NaNs
    mean_score = np.nanmean(scores) if scores else 0.0
    return mean_score
