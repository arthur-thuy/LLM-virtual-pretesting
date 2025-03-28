# standard library imports
import os
from typing import Union

# related third party imports
import structlog
import pandas as pd
from sklearn.model_selection import train_test_split

# local application/library specific imports
from tools.constants import (
    INTERACT_ID,
    TEST,
    TRAIN,
    Q_OPTION_TEXTS,
    VALSMALL,
    VALLARGE,
)
from tools.utils import set_seed

logger = structlog.get_logger(__name__)


class DataLoader:
    def __init__(self, read_dir: str, dataset_name: str) -> None:
        """Constructor."""
        self.read_dir = read_dir
        self.dataset_name = dataset_name

    def _read_questions(self) -> pd.DataFrame:
        # questions
        df_questions = pd.read_csv(
            os.path.join(self.read_dir, f"{self.dataset_name}_questions.csv")
        )
        # convert string back to list
        df_questions[Q_OPTION_TEXTS] = df_questions[Q_OPTION_TEXTS].apply(eval)
        return df_questions

    def read_splitted_data(
        self, join_key: str
    ) -> dict[str, pd.DataFrame]:
        """Read data from disk.

        Parameters
        ----------
        join_key : str
            Key to join on

        Returns
        -------
        dict[str, pd.DataFrame]
            Dictionary of dataframes for each split
        """
        # questions
        df_questions = self._read_questions()

        datasets = dict()
        # read pre-splitted interactions
        for split in [TRAIN, VALSMALL, VALLARGE, TEST]:
            df_interactions = pd.read_csv(
                os.path.join(self.read_dir, f"{self.dataset_name}_interactions_{split}.csv")
            )
            datasets[split] = pd.merge(df_interactions, df_questions, on=join_key)
            logger.info(
                f"Reading {split} split",
                num_interactions=len(datasets[split]),
            )
        return datasets

    def split_data(
        self,
        val_size: Union[float, int],
        test_size: Union[float, int],
        seed: int,
    ) -> None:
        """Split interactions into train, validation (small & large), and test sets.

        Parameters
        ----------
        val_size : Union[float, int]
            Fraction of the dataset to include in the val split
        test_size : Union[float, int]
            Fraction of the dataset to include in the test split
        seed : int
            Random seed
        """
        assert type(val_size) is type(
            test_size
        ), "val_size and test_size must be of the same type, either int or float"

        # interactions
        df_interactions = pd.read_csv(
            os.path.join(self.read_dir, f"{self.dataset_name}_interactions.csv")
        )

        # seed
        set_seed(seed)

        # train-validation-test split
        idx_trainval, idx_test = train_test_split(
            df_interactions[INTERACT_ID], test_size=test_size
        )
        if isinstance(val_size, int):
            idx_train, idx_val = train_test_split(idx_trainval, test_size=val_size)
        else:
            idx_train, idx_val = train_test_split(
                idx_trainval, test_size=val_size / (1 - test_size)
            )
        idx_valsmall, idx_vallarge = train_test_split(idx_val, test_size=(5 / 6))
        splits = {
            TRAIN: idx_train,
            VALSMALL: idx_valsmall,
            VALLARGE: idx_vallarge,
            TEST: idx_test,
        }

        datasets = dict()
        for split in [TRAIN, VALSMALL, VALLARGE, TEST]:
            datasets[split] = df_interactions[
                df_interactions[INTERACT_ID].isin(splits[split])
            ].copy()
            datasets[split].to_csv(
                os.path.join(
                    self.read_dir, f"{self.dataset_name}_interactions_{split}.csv"
                ),
                index=False,
            )
            logger.info(
                f"Writing {split} split",
                num_interactions=len(datasets[split]),
            )
