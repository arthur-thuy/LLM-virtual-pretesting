"""Build file for example formatter."""

# standard library imports
# /

# related third party imports
import structlog
from yacs.config import CfgNode
import pandas as pd

# local application/library specific imports
from tools.registry import Registry
from tools.constants import TRAIN, VALIDATION, TEST

EXAMPLE_FORMATTER_REGISTRY = Registry()

logger = structlog.get_logger(__name__)


def build_example_formatter(
    example_formatter_cfg: CfgNode, datasets: dict[str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    """Build the example formatter.

    Parameters
    ----------
    example_formatter_cfg : CfgNode
        Config node for the example formatter.
    datasets : dict[str, pd.DataFrame]
        Dict of datasets, per split.

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
    dataset_fmt = {
        TRAIN: formatter(datasets[TRAIN]),
        VALIDATION: formatter(datasets[VALIDATION]),
        TEST: formatter(datasets[TEST]),
    }
    return dataset_fmt
