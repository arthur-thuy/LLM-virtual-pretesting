"""Build file for data loader."""

# standard library imports
from typing import Dict

# related third party imports
import structlog
import pandas as pd
from yacs.config import CfgNode

# local application/library specific imports
from data_loader.data_loader import DataLoader
from tools.constants import SILVER_DIR, GOLD_DIR

logger = structlog.get_logger(__name__)


def build_replicate_dataset(loader_cfg: CfgNode) -> Dict[str, pd.DataFrame]:
    """Build the replication dataset.

    Parameters
    ----------
    loader_cfg : CfgNode
        Data loader config object

    Returns
    -------
    Dict[str, pd.DataFrame]
        Train/Val/Test interaction dataframes
    """
    logger.info("Building dataset", name=loader_cfg.NAME)
    data_loader = DataLoader(
        read_dir=SILVER_DIR,
        write_dir=GOLD_DIR,
        dataset_name=loader_cfg.NAME,
    )
    interactions = data_loader.read_splitted_interactions()
    return interactions


def build_roleplay_dataset(loader_cfg: CfgNode) -> Dict[str, pd.DataFrame]:
    """Build the roleplay dataset.

    Parameters
    ----------
    loader_cfg : CfgNode
        Data loader config object

    Returns
    -------
    Dict[str, pd.DataFrame]
        Train/Val/Test interaction dataframes
    """
    logger.info("Building roleplay dataset", name=loader_cfg.NAME)
    data_loader = DataLoader(
        read_dir=SILVER_DIR,
        write_dir=GOLD_DIR,
        dataset_name=loader_cfg.NAME,
    )
    questions = data_loader.read_splitted_questions()
    interact_train = data_loader.read_splitted_train_interactions()
    return questions, interact_train
