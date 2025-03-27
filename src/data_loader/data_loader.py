# standard library imports
import os
from typing import Dict

# related third party imports
import structlog
import pandas as pd
from sklearn.model_selection import train_test_split

# local application/library specific imports
from tools.constants import (
    INTERACT_ID,
    TEST,
    TRAIN,
    VALIDATION,
    Q_OPTION_TEXTS,
    VALSMALL,
    VALLARGE,
)
from tools.utils import set_seed

logger = structlog.get_logger(__name__)


class DataLoader:
    def __init__(
        self, read_dir: str, dataset_name: str, join_key: str, read_split: bool
    ) -> None:
        """Constructor."""
        self.df = self._read_data(read_dir, dataset_name, join_key, read_split)
        self.read_dir = read_dir
        self.dataset_name = dataset_name

    def _read_data(
        self, read_dir: str, dataset_name: str, join_key: str, read_split: bool
    ) -> pd.DataFrame:
        """Read data from disk.

        Parameters
        ----------
        read_dir : str
            Directory to read from
        dataset_name : str
            Name of the dataset
        join_key : str
            Key to join on
        read_split : bool
            Whether to read the pre-splitted interactions data

        Returns
        -------
        pd.DataFrame
            Dataframe
        """
        # questions
        df_questions = pd.read_csv(
            os.path.join(read_dir, f"{dataset_name}_questions.csv")
        )
        # convert string back to list
        df_questions[Q_OPTION_TEXTS] = df_questions[Q_OPTION_TEXTS].apply(eval)
        if read_split:
            datasets = dict()
            # read pre-splitted interactions
            for split in [TRAIN, VALIDATION, TEST]:
                df_interactions = pd.read_csv(
                    os.path.join(read_dir, f"{dataset_name}_interactions_{split}.csv")
                )
                datasets[split] = pd.merge(df_interactions, df_questions, on=join_key)
                logger.info(
                    f"Reading {split} split",
                    num_interactions=len(datasets[split]),
                )
            return datasets
        else:
            # interactions
            df_interactions = pd.read_csv(
                os.path.join(read_dir, f"{dataset_name}_interactions.csv")
            )
            df = pd.merge(df_interactions, df_questions, on=join_key)

            return df

    def split_data(
        self, train_size: float, test_size: float, seed: int, save: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """Split interactions into train, validation, and test sets.

        Parameters
        ----------
        train_size : float
            Fraction of the dataset to include in the train split
        test_size : float
            Fraction of the dataset to include in the test split
        seed : int
            Random seed
        save : bool
            Whether to save the splits to disk

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
        idx_valsmall, idx_vallarge = train_test_split(idx_val, test_size=5 / 6)
        splits = {
            TRAIN: idx_train,
            VALSMALL: idx_valsmall,
            VALLARGE: idx_vallarge,
            TEST: idx_test,
        }

        datasets = dict()
        for split in [TRAIN, VALSMALL, VALLARGE, TEST]:
            datasets[split] = self.df[self.df[INTERACT_ID].isin(splits[split])].copy()
            if save:
                datasets[split].to_csv(
                    os.path.join(
                        self.read_dir, f"{self.dataset_name}_interactions_{split}.csv"
                    ),
                    index=False,
                )
            logger.info(
                f"Creating {split} split",
                num_interactions=len(datasets[split]),
                saved=save,
            )

        return datasets
