# standard library imports
import os
from typing import Dict

# related third party imports
import structlog
import pandas as pd
from sklearn.model_selection import train_test_split

# local application/library specific imports
from tools.constants import INTERACT_ID, TEST, TRAIN, VALIDATION, OPTIONS_TEXT
from tools.utils import set_seed

logger = structlog.get_logger(__name__)


class DataLoader:
    def __init__(self, read_dir: str, dataset_name: str) -> None:
        """Constructor."""
        self.df = self._read_data(read_dir, dataset_name)

    def _read_data(self, read_dir: str, dataset_name: str) -> pd.DataFrame:
        """Read data from disk.

        Parameters
        ----------
        read_dir : str
            Directory to read from
        dataset_name : str
            Name of the dataset

        Returns
        -------
        pd.DataFrame
            Dataframe
        """
        df = pd.read_csv(os.path.join(read_dir, f"{dataset_name}.csv"))
        # convert string back to list
        df[OPTIONS_TEXT] = df[OPTIONS_TEXT].apply(eval)
        return df

    def split_data(self, train_size: float, test_size: float, seed: int) -> Dict[str, pd.DataFrame]:
        """Split data into train, validation, and test sets.

        Parameters
        ----------
        train_size : float
            Fraction of the dataset to include in the train split
        test_size : float
            Fraction of the dataset to include in the test split
        seed : int
            Random seed

        Returns
        -------
        Dict[str, pd.DataFrame]
            Train/Val/Test datasets
        """
        # seed
        set_seed(seed)

        # train-validation-test split
        idx_trainval, idx_test = train_test_split(
            self.df[INTERACT_ID], test_size=test_size
        )
        idx_train, idx_val = train_test_split(
            idx_trainval, test_size=1 - train_size / (1 - test_size)
        )
        splits = {TRAIN: idx_train, VALIDATION: idx_val, TEST: idx_test}

        dataset = dict()
        for split in [TRAIN, VALIDATION, TEST]:
            dataset[split] = self.df[self.df[INTERACT_ID].isin(splits[split])].copy()
            logger.info(
                f"Creating {split} split",
                num_interactions=len(dataset[split]),
            )
        return dataset
