"""Build file for example formatter."""

# standard library imports
# /

# related third party imports
import structlog
from yacs.config import CfgNode
import pandas as pd

# local application/library specific imports
from tools.registry import Registry

EXAMPLE_FORMATTER_REGISTRY = Registry()

logger = structlog.get_logger(__name__)


def build_example_formatter(
    example_formatter_cfg: CfgNode,
    datasets: dict[str, pd.DataFrame],
    is_interaction: bool,
) -> dict[str, pd.DataFrame]:
    """Build the example formatter.

    Parameters
    ----------
    example_formatter_cfg : CfgNode
        Config node for the example formatter.
    datasets : dict[str, pd.DataFrame]
        Dict of datasets, per split.
    is_interaction : bool
        Whether the datasets are interaction datasets.
        If not, they are questions datasets.

    Returns
    -------
    dict
        Dict of formatted datasets, per split.
    """
    logger.info(
        "Building example formatter",
        name=example_formatter_cfg.NAME,
        splits=list(datasets.keys()),
    )
    formatter = EXAMPLE_FORMATTER_REGISTRY[example_formatter_cfg.NAME]
    dataset_fmt = dict()
    for split, df in datasets.items():
        dataset_fmt[split] = formatter(dataset=df, is_interaction=is_interaction)
    return dataset_fmt
