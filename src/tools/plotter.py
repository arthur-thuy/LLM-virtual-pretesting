"""File with plotting functionalities."""

# standard library imports
import os
from typing import Optional

# related third party imports
import matplotlib.pyplot as plt
import structlog

# local application/library specific imports
from tools.analyzer import get_config_id_print
from tools.utils import ensure_dir, read_pickle

# set up logger
logger = structlog.get_logger(__name__)


def plot_student_level_performance(
    exp_name: str,
    config_id: str,
    run_id: int = 1,
    metric: str = "val_accuracy",
    config2legend: Optional[dict] = None,
    metric2legend: Optional[dict] = None,
    legend_exact: bool = False,
    save: bool = False,
    savefig_kwargs: Optional[dict] = None,
):
    """Plot answer performance for each student level.

    Parameters
    ----------
    exp_name : str
        Experiment name
    config_id : str
        Configuration ID
    run_id : int, optional
        Run ID, by default 1
    metric : str, optional
        Metric, by default "val_accuracy"
    config2legend : Optional[dict], optional
        Mapping from config to legend name, by default None
    metric2legend : Optional[dict], optional
        Mapping from metric to legend name, by default None
    legend_exact : bool, optional
        Whether to find exact match in config2legend, by default False
    save : bool, optional
        Whether to save the table, by default False
    save_kwargs : Optional[dict], optional
        Dictionary with save arguments, by default None
    """
    output_path = os.path.join("output", exp_name, config_id, f"run_{run_id}.pickle")
    metrics = read_pickle(output_path)["metrics_answers"][f"answers_{metric}"]

    # pretty config id
    config_id_print = get_config_id_print(
        config_id=config_id, config2legend=config2legend, exact=legend_exact
    )
    metric_name_print = get_config_id_print(
        config_id=metric, config2legend=metric2legend, exact=legend_exact
    )

    _, ax = plt.subplots(1, 1, figsize=(10, 6))
    # Extract levels and values
    levels = list(metrics.keys())
    values = list(metrics.values())

    # Create lineplot
    ax.plot(levels, values, marker="o", linestyle="-", linewidth=2, markersize=8)

    # Set x-ticks to be the student levels
    ax.set_xticks(range(len(levels)), levels)

    # Set labels and title
    ax.set_xlabel("Student Level")
    ax.set_ylabel(metric_name_print)
    ax.set_title(config_id_print)

    ax.grid(True, linestyle="--", alpha=0.7)
    if save:
        plt.tight_layout()
        ensure_dir(os.path.dirname(savefig_kwargs["fname"]))
        plt.savefig(**savefig_kwargs)
