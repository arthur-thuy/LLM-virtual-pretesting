"""Build file for data loader."""

# standard library imports
from typing import Dict

# related third party imports
import structlog
import pandas as pd
from yacs.config import CfgNode

# local application/library specific imports
from data_loader.data_loader import DataLoader
from tools.constants import SILVER_DIR

logger = structlog.get_logger(__name__)


def build_dataset(loader_cfg: CfgNode, seed: int) -> Dict[str, pd.DataFrame]:
    """Build the dataset.

    Parameters
    ----------
    loader_cfg : CfgNode
        Data loader config object
    seed : int
        Random seed

    Returns
    -------
    Dict[str, pd.DataFrame]
        Train/Val/Test interaction dataframes
    """
    logger.info("Building dataset", name=loader_cfg.NAME)
    datasets = DataLoader(
        read_dir=SILVER_DIR,
        dataset_name=loader_cfg.NAME,
        join_key=loader_cfg.JOIN_KEY,
        read_split=True,
    )
    return datasets
