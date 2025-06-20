# standard library imports
import os
from typing import Union, Optional
from statistics import multimode

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
    VALIDATION,
    QUESTION_ID,
    KC,
)
from tools.utils import set_seed
from tools.irt_estimator import compute_student_levels

logger = structlog.get_logger(__name__)


class DataLoader:
    def __init__(self, read_dir: str, write_dir: str, dataset_name: str) -> None:
        """Constructor."""
        self.read_dir = read_dir
        self.write_dir = write_dir
        self.dataset_name = dataset_name

    def _read_questions(self) -> pd.DataFrame:
        # questions
        df_questions = pd.read_csv(
            os.path.join(self.read_dir, f"{self.dataset_name}_questions.csv")
        )
        # convert string back to list
        df_questions[Q_OPTION_TEXTS] = df_questions[Q_OPTION_TEXTS].apply(eval)
        return df_questions

    def read_splitted_data(self, join_key: str) -> dict[str, pd.DataFrame]:
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
                os.path.join(
                    self.write_dir, f"{self.dataset_name}_interactions_{split}.csv"
                )
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
        ## first split the question_ids
        q_ids_trainval, q_ids_test = train_test_split(
            df_interactions[QUESTION_ID].unique(), test_size=0.20
        )
        q_ids_train, q_ids_val = train_test_split(
            q_ids_trainval, test_size=0.2 / (1 - 0.20)
        )
        ## then split the interactions
        idx_train = df_interactions[df_interactions[QUESTION_ID].isin(q_ids_train)][
            INTERACT_ID
        ].tolist()
        idx_val = df_interactions[df_interactions[QUESTION_ID].isin(q_ids_val)][
            INTERACT_ID
        ]
        if isinstance(val_size, int):
            idx_val = idx_val.sample(val_size).tolist()
        else:
            frac_val = min(
                1, val_size * len(df_interactions[QUESTION_ID]) / len(idx_val)
            )
            idx_val = idx_val.sample(frac=frac_val).tolist()
        idx_valsmall, idx_vallarge = train_test_split(idx_val, test_size=(5 / 6))
        idx_test = df_interactions[df_interactions[QUESTION_ID].isin(q_ids_test)][
            INTERACT_ID
        ]
        if isinstance(test_size, int):
            idx_test = idx_test.sample(test_size).tolist()
        else:
            frac_test = min(
                1, test_size * len(df_interactions[QUESTION_ID]) / len(idx_test)
            )
            idx_test = idx_test.sample(frac=frac_test).tolist()

        splits = {
            TRAIN: idx_train,
            VALSMALL: idx_valsmall,
            VALLARGE: idx_vallarge,
            TEST: idx_test,
        }

        # create dataframes for each split
        datasets = dict()
        for split in splits.keys():
            datasets[split] = df_interactions[
                df_interactions[INTERACT_ID].isin(splits[split])
            ].copy()

        # compute IRT on the train set
        datasets[TRAIN] = compute_student_levels(df_interactions=datasets[TRAIN])

        # save data
        for split in splits.keys():
            datasets[split].to_csv(
                os.path.join(
                    self.write_dir, f"{self.dataset_name}_interactions_{split}.csv"
                ),
                index=False,
            )
            logger.info(
                f"Writing {split} split",
                num_interactions=len(datasets[split]),
                num_distinct_questions=len(
                    datasets[split][QUESTION_ID].unique(),
                ),
                num_distinct_students=len(
                    datasets[split]["student_id"].unique(),
                ),
            )


class DataLoaderRoleplay:
    def __init__(self, read_dir: str, write_dir: str, dataset_name: str) -> None:
        """Constructor."""
        self.read_dir = read_dir
        self.write_dir = write_dir
        self.dataset_name = dataset_name

    def _read_questions(self) -> pd.DataFrame:
        # questions
        df_questions = pd.read_csv(
            os.path.join(self.read_dir, f"{self.dataset_name}_questions.csv")
        )
        # convert string back to list
        df_questions[Q_OPTION_TEXTS] = df_questions[Q_OPTION_TEXTS].apply(eval)
        df_questions[KC] = df_questions[KC].apply(eval)
        return df_questions

    def read_splitted_data(self) -> dict[str, pd.DataFrame]:
        """Read data from disk.

        Returns
        -------
        dict[str, pd.DataFrame]
            Dictionary of dataframes for each split
        """
        questions = dict()
        # read pre-splitted questions
        for split in [TRAIN, VALIDATION, TEST]:
            df_questions_tmp = pd.read_csv(
                os.path.join(
                    self.write_dir,
                    f"{self.dataset_name}_roleplay_questions_{split}.csv",
                )
            )
            # convert string back to list
            df_questions_tmp[Q_OPTION_TEXTS] = df_questions_tmp[Q_OPTION_TEXTS].apply(
                eval
            )
            questions[split] = df_questions_tmp
            logger.info(
                f"Reading {split} split questions",
                num_questions=len(questions[split]),
            )
        # read train interactions
        interact_train = pd.read_csv(
            os.path.join(
                self.write_dir, f"{self.dataset_name}_roleplay_interactions_train.csv"
            )
        )
        # convert string back to list
        interact_train[Q_OPTION_TEXTS] = interact_train[Q_OPTION_TEXTS].apply(eval)

        logger.info(
            "Reading train split interactions",
            num_interactions=len(interact_train),
            num_distinct_questions=len(
                interact_train[QUESTION_ID].unique(),
            ),
            num_distinct_students=len(
                interact_train["student_id"].unique(),
            ),
        )
        return questions, interact_train

    def split_data(
        self,
        val_size: float,
        test_size: float,
        split_interactions: bool,  # True for DBE-KT22, False for CFE
        stratified: bool,  # True for DBE-KT22
        seed: int,
        join_key: Optional[str] = None,
    ) -> None:
        """Split interactions into train, validation, and test sets.

        Parameters
        ----------
        val_size : float
            Fraction of the dataset to include in the val split
        test_size : float
            Fraction of the dataset to include in the test split
        seed : int
            Random seed
        """
        # questions
        df_questions = self._read_questions()

        # interactions
        df_interactions = pd.read_csv(
            os.path.join(self.read_dir, f"{self.dataset_name}_interactions.csv")
        )

        # seed
        set_seed(seed)

        # train-validation-test split
        # split the question_ids (stratified or not)
        if stratified:
            q_ids_train, q_ids_val, q_ids_test = self._stratified_split(
                df_questions, val_size, test_size
            )
        else:
            q_ids_trainval, q_ids_test = train_test_split(
                df_questions[QUESTION_ID].unique(), test_size=test_size
            )
            q_ids_train, q_ids_val = train_test_split(
                q_ids_trainval, test_size=val_size / (1 - test_size)
            )
        q_splits = {
            TRAIN: q_ids_train,
            VALIDATION: q_ids_val,
            TEST: q_ids_test,
        }

        # writing questions
        for split in q_splits.keys():
            q_split = df_questions[
                df_questions[QUESTION_ID].isin(q_splits[split])
            ].copy()
            q_split.to_csv(
                os.path.join(
                    self.write_dir,
                    f"{self.dataset_name}_roleplay_questions_{split}.csv",
                ),
                index=False,
            )
            logger.info(
                f"Writing {split} split questions",
                num_questions=len(q_split),
            )

        # writing interactions
        if split_interactions:
            # filter out the train questions
            interact_train = df_interactions[
                df_interactions[QUESTION_ID].isin(q_splits[TRAIN])
            ].copy()
        else:
            # we can keep all interactions
            interact_train = df_interactions.copy()
        if join_key is not None:
            interact_train = pd.merge(
                interact_train, df_questions, on=join_key, how="left"
            )

        # compute IRT on the train set
        interact_train = compute_student_levels(df_interactions=interact_train)

        interact_train.to_csv(
            os.path.join(
                self.write_dir, f"{self.dataset_name}_roleplay_interactions_train.csv"
            ),
            index=False,
        )
        logger.info(
            "Writing train split interactions",
            num_interactions=len(interact_train),
            num_distinct_questions=len(
                interact_train[QUESTION_ID].unique(),
            ),
            num_distinct_students=len(
                interact_train["student_id"].unique(),
            ),
        )

    def _stratified_split(
        self, df_questions: pd.DataFrame, val_size: float, test_size: float
    ) -> tuple[list, list, list]:
        # Get the most common KC for each question (for stratification)
        # If a question has multiple KCs, use the most frequent one in the dataset
        def get_primary_kc(kc_list):
            kc_mode = multimode(kc_list)
            if len(kc_mode) == 1:
                return kc_mode[0]
            if len(kc_mode) > 1:
                # If there are multiple modes, return the one with highest count in the dataset  # noqa
                all_kcs = []
                for kc_list in df_questions[KC]:
                    all_kcs.extend(kc_list)

                # Count the occurrences of each knowledge component
                kc_counts = pd.Series(all_kcs).value_counts().reset_index()
                kc_counts.columns = ["knowledgecomponent_id", "count"]

                # Sort by count in descending order
                kc_counts = kc_counts.sort_values("count", ascending=False)
                kc_frequencies = {
                    kc: kc_counts[kc_counts["knowledgecomponent_id"] == kc][
                        "count"
                    ].values[0]
                    for kc in kc_mode
                    if kc in kc_counts["knowledgecomponent_id"].values
                }
                return max(kc_frequencies, key=kc_frequencies.get)

        df_questions["primary_kc"] = df_questions[KC].apply(get_primary_kc)

        # First split: train and temp (val+test)
        train_questions, temp_questions = train_test_split(
            df_questions,
            test_size=val_size + test_size,
            stratify=df_questions["primary_kc"],
        )

        # Second split: val and test from temp
        val_size_adjusted = val_size / (val_size + test_size)
        val_questions, test_questions = train_test_split(
            temp_questions,
            test_size=(1 - val_size_adjusted),
            stratify=temp_questions["primary_kc"],
        )

        # TODO: remove
        # # Verify KC distribution
        # for split_name, split_df in [
        #     ("Train", train_questions),
        #     ("Val", val_questions),
        #     ("Test", test_questions),
        # ]:
        #     kc_in_split = []
        #     for kc_list in split_df[KC]:
        #         kc_in_split.extend(kc_list)

        #     # Get the KCs in this split
        #     kc_dist = pd.Series(kc_in_split).value_counts()
        #     print(f"\n{split_name} KC distribution:")
        #     print(kc_dist)

        # Drop the added columns and return to original format
        q_ids_train = train_questions[QUESTION_ID].tolist()
        q_ids_val = val_questions[QUESTION_ID].tolist()
        q_ids_test = test_questions[QUESTION_ID].tolist()

        return q_ids_train, q_ids_val, q_ids_test
