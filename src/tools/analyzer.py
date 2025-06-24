"""File with analyzer functionalities."""

# standard library imports
import glob
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Tuple

# related third party imports
import numpy as np
import pandas as pd
import scipy
import structlog
from numpy.typing import NDArray
from tabulate import tabulate

# local application/library specific imports
from tools.utils import ensure_dir, read_pickle, write_pickle

# set up logger
logger = structlog.get_logger(__name__)


def get_output_paths(experiment: str, config_ids: list[str]) -> list[str]:
    """Get paths to output files.

    Parameters
    ----------
    experiment : str
        Experiment name
    config_ids : list[str]
        List of config IDs
    Returns
    -------
    list[str]
        List of output paths
    """
    # paths
    output_dir = os.path.join("output", experiment)
    output_paths = [
        os.path.join(output_dir, f"{config_id}.pickle") for config_id in config_ids
    ]

    return output_paths


def mean_stderror(ary: NDArray, axis: Optional[int] = None) -> Tuple[NDArray, NDArray]:
    """Calculate mean and standard error from array.

    Parameters
    ----------
    ary : NDArray
        Output array containing metrics
    axis : Union[Any, None], optional
        Axis to average over, by default None
    Returns
    -------
    Tuple[NDArray, NDArray]
        Tuple of mean and standard error
    """
    mean = np.mean(ary, axis=axis)
    stderror = scipy.stats.sem(ary, ddof=1, axis=axis)
    return mean, stderror


def _invert_runs_metrics(
    results_dict: dict[str, dict[str, Any]],
) -> dict[str, list[Any]]:
    """Invert dict 'run : metric' to 'metric : run'.

    Parameters
    ----------
    results_dict : dict[str, dict[str, Any]]
        Dict with 'run : metric'
    Returns
    -------
    dict[str, list[Any]]
        Inverted dict with 'metric : run'
    """
    keys = list(results_dict["run_1"].keys())
    inverted_dict: dict[str, list] = {key: [] for key in keys}
    for run in results_dict.keys():
        for key in keys:
            inverted_dict[key].append(results_dict[run][key])
    return inverted_dict


def _default_to_regular(d):
    """Convert defaultdict to regular dict."""
    if isinstance(d, defaultdict):
        d = {k: _default_to_regular(v) for k, v in d.items()}
    return d


def aggregate_metrics(
    results_raw: dict[str, dict[str, dict[str, list[float]]]],
) -> tuple[dict, dict]:
    """Aggregate metrics.

    Concat logs of different epochs and aggregate train metrics over the different runs,
    compute mean and stderror.

    Parameters
    ----------
    results_raw : dict[str, dict[str, dict[str, list[float]]]]
        Results saved by experiment

    Returns
    -------
    tuple[dict, dict]
        Tuple of unaggregated and aggregated metrics dicts
    """
    results_unagg = results_raw
    results_agg: defaultdict = defaultdict(lambda: defaultdict(dict))
    for config_id, config_value in results_raw.items():
        results_unagg[config_id] = config_value
        hist_dict_inv = _invert_runs_metrics(config_value)
        for metric_key, metric_value in hist_dict_inv.items():
            mean, stderr = mean_stderror(np.array(metric_value), axis=0)
            results_agg[config_id][metric_key]["mean"] = mean
            results_agg[config_id][metric_key]["stderr"] = stderr
    results_agg = _default_to_regular(results_agg)
    return results_unagg, results_agg


def get_metrics(
    experiment: str, config_ids: list[str]
) -> tuple[dict[str, dict[str, dict[Any, Any]]], dict[str, dict[str, dict[str, Any]]]]:
    """Read output data and get dataset metrics (computed in analysis script).

    Parameters
    ----------
    experiment : str
        Experiment name
    config_ids : list[str]
        List of config IDs
    Returns
    -------
    tuple
        Tuple of unaggregated and aggregated metrics dicts
    """
    # paths
    output_paths = get_output_paths(experiment, config_ids)

    # metric
    results_raw = {}

    # get dict with metrics for all configs
    for config_id, output_path in zip(config_ids, output_paths):
        logger.info("Loading checkpoint", output_path=output_path)
        metrics_dict = read_pickle(output_path)["metrics"]
        results_raw[config_id] = metrics_dict

    results_unagg, results_agg = aggregate_metrics(results_raw)

    return results_unagg, results_agg


def get_results_dict(
    exp_name: str,
    config_ids: list[str],
    run_id: Optional[int] = None,
) -> dict:
    """Get results dictionary for all configs, averaged over runs.

    Parameters
    ----------
    exp_name : str
        Experiment name
    config_ids : list[str]
        List of config ids
    run_id : Optional[int], optional
        Run id, by default None

    Returns
    -------
    dict
        Dictionary with evaluation results
    """
    results_unagg, results_agg = get_metrics(exp_name, config_ids)
    results_tmp = results_agg if run_id is None else results_unagg[f"run_{run_id}"]
    return results_tmp


def get_config_id_print(
    config_id: str, config2legend: Optional[dict] = None, exact: bool = False
) -> str:
    """Get the legend name for a given config_id.

    Parameters
    ----------
    config_id : str
        Configuration ID.
    config2legend : dict
        Dictionary mapping config_id to legend name.
    exact : bool, optional
        Whether to find exact match in config2legend, by default False

    Returns
    -------
    str
        Legend name for the given config id
    """
    if config2legend is None:
        return config_id

    if exact:
        return config2legend.get(config_id, config_id)
    else:
        config_id = config_id
        for key, value in config2legend.items():
            if config_id.startswith(key):
                config_id = value
        return config_id


def print_table_from_dict(
    eval_dict: dict,
    exp_name: Optional[str] = None,
    exclude_metrics: Optional[list[str]] = None,
    config2legend: Optional[dict] = None,
    metric2legend: Optional[dict] = None,
    legend_exact: bool = False,
    decimals: int = 4,
    save: bool = False,
    save_kwargs: Optional[dict] = None,
) -> None:
    """Print table of results, for all configs, averaged over runs.

    Parameters
    ----------
    eval_dict : dict
        Dictionary with evaluation results
    exp_name : Optional[str], optional
        Experiment name, by default None
    exclude_metrics : Optional[list[str]], optional
        List of metrics to exclude, by default None
    config2legend : Optional[dict], optional
        Mapping from config to legend name, by default None
    metric2legend : Optional[dict], optional
        Mapping from metric to legend name, by default None
    legend_exact : bool, optional
        Whether to find exact match in config2legend, by default False
    decimals : int, optional
        Number of decimals, by default 4
    save : bool, optional
        Whether to save the table, by default False
    save_kwargs : Optional[dict], optional
        Dictionary with save arguments, by default None

    Raises
    ------
    ValueError
        If experiment name is not provided when saving
    ValueError
        If type is unknown
    """
    if save and exp_name is None:
        raise ValueError("Please provide the experiment name.")
    if exclude_metrics is None:
        exclude_metrics = []

    # get list of metric names
    metric_names = []
    config_ids = list(eval_dict.keys())
    for metric_key in eval_dict[config_ids[0]].keys():
        if metric_key in exclude_metrics:
            continue
        metric_name_print = get_config_id_print(
            config_id=metric_key, config2legend=metric2legend, exact=legend_exact
        )
        metric_names.append(metric_name_print)

    table = [["Config"] + metric_names]
    # iterate over configs and append rows to table
    for config_id in eval_dict.keys():
        config_id_print = get_config_id_print(
            config_id=config_id, config2legend=config2legend, exact=legend_exact
        )
        row = [config_id_print]
        for metric_key, metric_value in eval_dict[config_id].items():
            if metric_key in exclude_metrics:
                continue
            if isinstance(metric_value, dict):
                entry = f"{metric_value['mean']:.{decimals}f} ± {metric_value['stderr']:.{decimals}f}"
            elif isinstance(metric_value, float):
                entry = f"{metric_value:.{decimals}f}"
            else:
                raise ValueError("Unknown type.")
            row.append(entry)
        table.append(row)
    tabu_table = tabulate(table, headers="firstrow", tablefmt="psql")

    if save:
        ensure_dir(os.path.dirname(save_kwargs["fname"]))
        with open(save_kwargs["fname"], "w", encoding="UTF-8") as text_file:
            text_file.write(tabu_table)
    print(tabu_table)


def print_df_from_dict(
    eval_dict: dict,
    exp_name: Optional[str] = None,
    exclude_metrics: Optional[list[str]] = None,
    config2legend: Optional[dict] = None,
    metric2legend: Optional[dict] = None,
    legend_exact: bool = False,
    save: bool = False,
    save_kwargs: Optional[dict] = None,
) -> pd.DataFrame:
    """Get df of results, for all configs, averaged over runs.

    Parameters
    ----------
    eval_dict : dict
        Dictionary with evaluation results
    exp_name : Optional[str], optional
        Experiment name, by default None
    exclude_metrics : Optional[list[str]], optional
        List of metrics to exclude, by default None
    config2legend : Optional[dict], optional
        Mapping from config to legend name, by default None
    metric2legend : Optional[dict], optional
        Mapping from metric to legend name, by default None
    legend_exact : bool, optional
        Whether to find exact match in config2legend, by default False
    decimals : int, optional
        Number of decimals, by default 4
    save : bool, optional
        Whether to save the table, by default False
    save_kwargs : Optional[dict], optional
        Dictionary with save arguments, by default None

    Returns
    -------
    pd.DataFrame
        DataFrame with the results

    Raises
    ------
    ValueError
        If experiment name is not provided when saving
    ValueError
        If type is unknown
    """
    if save and exp_name is None:
        raise ValueError("Please provide the experiment name.")
    if exclude_metrics is None:
        exclude_metrics = []

    # get list of metric names
    metric_names = []
    config_ids = list(eval_dict.keys())
    for metric_key in eval_dict[config_ids[0]].keys():
        if metric_key in exclude_metrics:
            continue
        metric_name_print = get_config_id_print(
            config_id=metric_key, config2legend=metric2legend, exact=legend_exact
        )
        metric_names.append(metric_name_print)

    # prepare multiindex for columns
    num_metrics = len(metric_names)
    arrays = [
        [name for name in metric_names for _ in range(2)],
        ["mean", "stderr"] * num_metrics,
    ]
    tuples = list(zip(*arrays))
    index = pd.MultiIndex.from_tuples(tuples, names=["first", "second"])

    table = []
    config_names = []
    # iterate over configs and append rows to table
    for config_id in eval_dict.keys():
        config_id_print = get_config_id_print(
            config_id=config_id, config2legend=config2legend, exact=legend_exact
        )
        config_names.append(config_id_print)
        row = []
        for metric_key, metric_value in eval_dict[config_id].items():
            if metric_key in exclude_metrics:
                continue
            if isinstance(metric_value, dict):
                entry = [metric_value["mean"], metric_value["stderr"]]
            elif isinstance(metric_value, float):
                entry = [metric_value, None]
            else:
                raise ValueError("Unknown type.")
            row.extend(entry)
        table.append(row)

    df = pd.DataFrame(table, index=config_names, columns=index)
    df.index.name = "config_id"
    df.columns.names = ["metric", "statistic"]

    if save:
        ensure_dir(os.path.dirname(save_kwargs["fname"]))
        df.to_csv(save_kwargs["fname"], index=True)
    return df


def merge_run_results(output_dir: str) -> list:
    """Merge all results from different runs within a config.

    Parameters
    ----------
    output_dir : str
        Path to output directory

    Returns
    -------
    list
        List with run IDs
    """
    output_paths = glob.glob(os.path.join(output_dir, "*.pickle"))

    # check if overlap in run IDs
    run_id_list = [
        re.search(r"run_(\d+)", output_path).group(1) for output_path in output_paths
    ]
    run_id_list = [int(run_id) for run_id in run_id_list]
    run_id_list.sort()
    if len(run_id_list) != len(set(run_id_list)):
        raise ValueError("Overlap in run IDs!")

    all_metrics = {}
    for output_path in output_paths:
        run_n = re.search(r"run_(\d+)", output_path).group(1)
        result = read_pickle(output_path)

        all_metrics[f"run_{run_n}"] = result["metrics"]

    path = Path(output_dir)
    logger.info(
        f"Merging runs {run_id_list} in: "
        f"{os.path.join(path.parent, f'{path.name}.pickle')}"
    )
    write_pickle(
        {
            "metrics": all_metrics,
        },
        save_dir=path.parent,
        fname=path.name,
    )
    return run_id_list


def merge_all_results(experiment: str, config_ids: list[str]) -> dict:
    """Merge results for all configs.

    Parameters
    ----------
    experiment : str
        Experiment name
    config_ids : list[str]
        List of config IDs

    Returns
    -------
    dict
        Dictionary with config_id to run_id list
    """
    run_id_dict = {}
    # paths
    output_dir = os.path.join("output", experiment)
    output_paths = [os.path.join(output_dir, config_id) for config_id in config_ids]

    for output_path in output_paths:
        run_id_list = merge_run_results(output_path)
        run_id_dict[Path(output_path).name] = run_id_list
    return run_id_dict


def create_config_id_print(config_id: str) -> str:
    """Create a human-readable string from a config_id.

    Parameters
    ----------
    config_id : str
        The config_id to transform.

    Returns
    -------
    str
       A human-readable string.
    """
    pattern = r"([^~]+)~T([0-9.]+)~S([^~]+)~F(\d+)"
    match = re.match(pattern, config_id)

    if match:
        model_name = match.group(1)
        temperature = float(match.group(2))
        selector = match.group(3)
        few_shot = int(match.group(4))

        config_id_print = (
            f"{model_name} ({selector}, {few_shot}-shot, temp {temperature})"
        )
    else:
        config_id_print = config_id

    return config_id_print
