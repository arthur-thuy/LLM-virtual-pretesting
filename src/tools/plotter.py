"""File with plotting functionalities."""

# standard library imports
import os
from typing import Optional, Literal

# related third party imports
import matplotlib.pyplot as plt
import structlog
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib.patches import Rectangle

# local application/library specific imports
from tools.analyzer import get_config_id_print
from tools.utils import ensure_dir, read_pickle

# set up logger
logger = structlog.get_logger(__name__)


# def plot_student_level_performance(  # TODO: remove
#     exp_name: str,
#     config_id: str,
#     run_id: int = 1,
#     metric: str = "val_accuracy",
#     config2legend: Optional[dict] = None,
#     metric2legend: Optional[dict] = None,
#     legend_exact: bool = False,
#     save: bool = False,
#     savefig_kwargs: Optional[dict] = None,
# ):
#     """Plot answer performance for each student level.

#     Parameters
#     ----------
#     exp_name : str
#         Experiment name
#     config_id : str
#         Configuration ID
#     run_id : int, optional
#         Run ID, by default 1
#     metric : str, optional
#         Metric, by default "val_accuracy"
#     config2legend : Optional[dict], optional
#         Mapping from config to legend name, by default None
#     metric2legend : Optional[dict], optional
#         Mapping from metric to legend name, by default None
#     legend_exact : bool, optional
#         Whether to find exact match in config2legend, by default False
#     save : bool, optional
#         Whether to save the table, by default False
#     save_kwargs : Optional[dict], optional
#         Dictionary with save arguments, by default None
#     """
#     output_path = os.path.join("output", exp_name, config_id, f"run_{run_id}.pickle")
#     metrics = read_pickle(output_path)["metrics_answers"][f"answers_{metric}"]

#     # pretty config id
#     config_id_print = get_config_id_print(
#         config_id=config_id, config2legend=config2legend, exact=legend_exact
#     )
#     metric_name_print = get_config_id_print(
#         config_id=metric, config2legend=metric2legend, exact=legend_exact
#     )

#     _, ax = plt.subplots(1, 1, figsize=(10, 6))
#     # Extract levels and values
#     levels = list(metrics.keys())
#     values = list(metrics.values())

#     # Create lineplot
#     ax.plot(levels, values, marker="o", linestyle="-", linewidth=2, markersize=8)

#     # Set x-ticks to be the student levels
#     ax.set_xticks(range(len(levels)), levels)

#     # Set labels and title
#     ax.set_xlabel("Student Level")
#     ax.set_ylabel(metric_name_print)
#     ax.set_title(config_id_print)

#     ax.grid(True, linestyle="--", alpha=0.7)
#     if save:
#         plt.tight_layout()
#         ensure_dir(os.path.dirname(savefig_kwargs["fname"]))
#         plt.savefig(**savefig_kwargs)


def plot_llm_student_confusion(
    preds_dict: dict,
    config_id: str = None,
    normalize: str = "all",
    diagonal: bool = True,
    save: bool = False,
    savefig_kwargs: Optional[dict] = None,
):
    """Plot confusion matrix for LLM student predictions.

    Parameters
    ----------
    preds_dict : dict
        Dictionary containing predictions and true labels for students.
    config_id : str, optional
        Configuration ID, by default None
    normalize : str, optional
        How to normalize the confusion matrix, by default "all"
    diagonal : bool, optional
        Whether to highlight the diagonal, by default True
    save : bool, optional
        Whether to save the plot, by default False
    savefig_kwargs : Optional[dict], optional
        Dictionary with save arguments, by default None
    """
    list_acc_llm_student = []
    list_acc_student_true = []
    student_ids = preds_dict["student_ids"]
    for student_id in np.unique(student_ids):
        idx_student = np.where(student_ids == student_id)[0]
        y_pred_filter = preds_dict["y_pred"][idx_student]
        y_student_filter = preds_dict["y_student"][idx_student]
        y_true_filter = preds_dict["y_true"][idx_student]
        acc_llm_student = accuracy_score(y_true=y_student_filter, y_pred=y_pred_filter)
        acc_student_true = accuracy_score(y_true=y_true_filter, y_pred=y_student_filter)
        list_acc_llm_student.append(acc_llm_student)
        list_acc_student_true.append(acc_student_true)

    # same labels for both axes
    list_acc_llm_student = [f"{i:.2f}" for i in list_acc_llm_student]
    list_acc_student_true = [f"{i:.2f}" for i in list_acc_student_true]
    labels = set(list_acc_llm_student + list_acc_student_true)
    labels = sorted(labels)  # sort for better readability

    # get confusion matrix
    conf_matrix = confusion_matrix(
        list_acc_student_true,
        list_acc_llm_student,
        labels=None,
        normalize=normalize,
    )

    _, ax = plt.subplots(figsize=(len(labels), len(labels)))
    sns.heatmap(
        conf_matrix,
        annot=True,
        xticklabels=labels,
        yticklabels=labels,
        # labels=labels,
        cmap="Blues",
        ax=ax,
        vmin=0.0,
        vmax=1.0,
        cbar=False,
        fmt=".2f",
    )
    # Add borders to diagonal cells
    if diagonal:
        for i in range(len(conf_matrix)):
            for i in range(len(conf_matrix)):
                ax.add_patch(
                    Rectangle((i, i), 1, 1, fill=False, edgecolor="black", lw=2)
                )
    ax.set(
        xlabel="LLM -> student correctness",
        ylabel="Student -> true correctness",
    )
    ax.set_title((None if save else config_id), fontsize=6)
    ax.invert_yaxis()  # NOTE: labels read bottom-to-top
    ax.xaxis.label.set_size(15)
    ax.yaxis.label.set_size(15)
    ax.title.set_size(9)
    # ax.set_xticklabels(xticklabels, fontsize=13)
    # ax.set_yticklabels(yticklabels, fontsize=13)
    if save:
        plt.tight_layout()
        ensure_dir(os.path.dirname(savefig_kwargs["fname"]))
        plt.savefig(**savefig_kwargs)
    plt.show()
    plt.show()


def plot_kt_confusion(
    preds_dict: dict,
    config_id: str = None,
    normalize: str = "all",
    diagonal: bool = True,
    save: bool = False,
    savefig_kwargs: Optional[dict] = None,
):
    """Plot confusion matrix from knowledge tracing perspective.

    Parameters
    ----------
    preds_dict : dict
        Dictionary containing predictions and true labels for students.
    config_id : str, optional
        Configuration ID, by default None
    normalize : str, optional
        How to normalize the confusion matrix, by default "all"
    diagonal : bool, optional
        Whether to highlight the diagonal, by default True
    save : bool, optional
        Whether to save the plot, by default False
    savefig_kwargs : Optional[dict], optional
        Dictionary with save arguments, by default None
    """
    student_correct = preds_dict["y_student"] == preds_dict["y_true"]
    llm_correct = preds_dict["y_pred"] == preds_dict["y_true"]

    # get confusion matrix
    conf_matrix = confusion_matrix(
        student_correct,
        llm_correct,
        labels=None,
        normalize=normalize,
    )

    _, ax = plt.subplots()
    sns.heatmap(
        conf_matrix,
        annot=True,
        # xticklabels=labels,
        # yticklabels=labels,
        # labels=labels,
        cmap="Blues",
        ax=ax,
        vmin=0.0,
        vmax=1.0,
        cbar=False,
        fmt=".2f",
    )
    # Add borders to diagonal cells
    if diagonal:
        for i in range(len(conf_matrix)):
            for i in range(len(conf_matrix)):
                ax.add_patch(
                    Rectangle((i, i), 1, 1, fill=False, edgecolor="black", lw=2)
                )
    ax.set(
        xlabel="LLM -> true correctness",
        ylabel="Student -> true correctness",
    )
    ax.set_title((None if save else config_id), fontsize=6)
    ax.invert_yaxis()  # NOTE: labels read bottom-to-top
    ax.xaxis.label.set_size(15)
    ax.yaxis.label.set_size(15)
    ax.title.set_size(9)
    # ax.set_xticklabels(xticklabels, fontsize=13)
    # ax.set_yticklabels(yticklabels, fontsize=13)
    if save:
        plt.tight_layout()
        ensure_dir(os.path.dirname(savefig_kwargs["fname"]))
        plt.savefig(**savefig_kwargs)
    plt.show()
    plt.show()


def plot_level_correctness(
    preds_dict: dict,
    problem_type: Literal["replicate", "roleplay"],
    config_id: str = None,
    save: bool = False,
    savefig_kwargs: Optional[dict] = None,
):
    """Plot student & LLM correctness per level.

    Parameters
    ----------
    preds_dict : dict
        Dictionary containing predictions and true labels for students.
    problem_type : Literal["replicate", "roleplay"]
        Type of problem, either "replicate" or "roleplay".
        If "replicate", uses `_plot_level_correctness_replicate`.
        If "roleplay", uses `_plot_level_correctness_roleplay`.
    config_id : str, optional
        Configuration ID, by default None
    save : bool, optional
        Whether to save the plot, by default False
    savefig_kwargs : Optional[dict], optional
        Dictionary with save arguments, by default None
    """
    if problem_type == "replicate":
        _plot_level_correctness_replicate(
            preds_dict=preds_dict,
            config_id=config_id,
            save=save,
            savefig_kwargs=savefig_kwargs,
        )
    elif problem_type == "roleplay":
        _plot_level_correctness_roleplay(
            preds_dict=preds_dict,
            config_id=config_id,
            save=save,
            savefig_kwargs=savefig_kwargs,
        )
    else:
        raise ValueError(f"Unknown problem type: {problem_type}")


def _plot_level_correctness_replicate(
    preds_dict: dict,
    config_id: str = None,
    save: bool = False,
    savefig_kwargs: Optional[dict] = None,
):
    """Plot student & LLM correctness per level.

    Parameters
    ----------
    preds_dict : dict
        Dictionary containing predictions and true labels for students.
    config_id : str, optional
        Configuration ID, by default None
    save : bool, optional
        Whether to save the plot, by default False
    savefig_kwargs : Optional[dict], optional
        Dictionary with save arguments, by default None
    """
    student_correct = preds_dict["y_student"] == preds_dict["y_true"]
    llm_correct = preds_dict["y_pred"] == preds_dict["y_true"]

    # Create a DataFrame with the data you already have
    df = pd.DataFrame(
        {
            "student_level_group": preds_dict["student_level_group"],
            "student_correct": student_correct,
            "llm_correct": llm_correct,
        }
    )
    student_group_correctness = df.groupby("student_level_group", sort=False)[
        "student_correct"
    ].mean()
    llm_group_correctness = df.groupby("student_level_group", sort=False)[
        "llm_correct"
    ].mean()

    _, ax = plt.subplots()
    student_group_correctness.plot(kind="line", ax=ax, label="Student")
    llm_group_correctness.plot(kind="line", ax=ax, label="LLM")
    ax.set(
        xlabel="Student levels",
        ylabel="MCQ correctness",
    )
    ax.set_ylim(-0.05, 1.05)
    ax.set_title((None if save else config_id), fontsize=9)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.7)
    if save:
        plt.tight_layout()
        ensure_dir(os.path.dirname(savefig_kwargs["fname"]))
        plt.savefig(**savefig_kwargs)
    plt.show()


def _plot_level_correctness_roleplay(
    preds_dict: dict,
    config_id: str = None,
    save: bool = False,
    savefig_kwargs: Optional[dict] = None,
):
    """Plot LLM correctness per level.

    Parameters
    ----------
    preds_dict : dict
        Dictionary containing predictions and true labels for students.
    config_id : str, optional
        Configuration ID, by default None
    save : bool, optional
        Whether to save the plot, by default False
    savefig_kwargs : Optional[dict], optional
        Dictionary with save arguments, by default None
    """
    llm_correct = preds_dict["y_pred"] == preds_dict["y_true"]

    # Create a DataFrame with the data you already have
    df = pd.DataFrame(
        {
            "student_level_group": preds_dict["student_level_group"],
            "llm_correct": llm_correct,
        }
    )
    llm_group_correctness = df.groupby("student_level_group", sort=False)[
        "llm_correct"
    ].mean()

    student_group_correctness = pd.DataFrame(
        {
            "student_level_group": df.groupby("student_level_group", sort=False)[
                "student_level_group"
            ],
            "student_correct": preds_dict["student_group_correctness"],
        }
    )

    _, ax = plt.subplots()
    student_group_correctness.plot(kind="line", ax=ax, label="Student")
    llm_group_correctness.plot(kind="line", ax=ax, label="LLM")
    ax.set(
        xlabel="Student levels",
        ylabel="MCQ correctness",
    )
    ax.set_ylim(-0.05, 1.05)
    ax.set_title((None if save else config_id), fontsize=9)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.7)
    if save:
        plt.tight_layout()
        ensure_dir(os.path.dirname(savefig_kwargs["fname"]))
        plt.savefig(**savefig_kwargs)
    plt.show()
