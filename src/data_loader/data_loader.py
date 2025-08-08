# standard library imports
import os
from statistics import multimode
from typing import Optional

# related third party imports
import pandas as pd
import structlog
import numpy as np
from sklearn.model_selection import train_test_split

# local application/library specific imports
from tools.constants import (
    INTERACT_ID,
    KC,
    Q_OPTION_TEXTS,
    QUESTION_ID,
    TEST,
    TRAIN,
    VALIDATION,
    VALLARGE,
    VALSMALL,
)
from tools.utils import set_seed
# from tools.irt_estimator import compute_student_levels, group_student_levels

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
        if set([Q_OPTION_TEXTS, KC]).issubset(df_questions.columns):
            # convert string back to list
            df_questions[Q_OPTION_TEXTS] = df_questions[Q_OPTION_TEXTS].apply(eval)
            df_questions[KC] = df_questions[KC].apply(eval)
        return df_questions

    def read_splitted_questions(self) -> dict[str, pd.DataFrame]:
        """
        Read splitted questions from disk.

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
                    f"{self.dataset_name}_questions_{split}.csv",
                )
            )
            if set([Q_OPTION_TEXTS, KC]).issubset(df_questions_tmp.columns):
                # convert string back to list
                df_questions_tmp[Q_OPTION_TEXTS] = df_questions_tmp[
                    Q_OPTION_TEXTS
                ].apply(eval)
                df_questions_tmp[KC] = df_questions_tmp[KC].apply(eval)
            questions[split] = df_questions_tmp
            logger.info(
                f"Reading {split} split questions",
                num_questions=len(questions[split]),
            )
        return questions

    def read_splitted_train_interactions(self) -> pd.DataFrame:
        # read train interactions
        interact_train = pd.read_csv(
            os.path.join(
                self.write_dir,
                f"{self.dataset_name}_interactions_train.csv",
            )
        )
        if set([Q_OPTION_TEXTS, KC]).issubset(interact_train.columns):
            # convert string back to list
            interact_train[Q_OPTION_TEXTS] = interact_train[Q_OPTION_TEXTS].apply(eval)
            interact_train[KC] = interact_train[KC].apply(eval)

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
        return interact_train

    def read_splitted_interactions(self) -> dict[str, pd.DataFrame]:
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
        interactions = dict()
        # read pre-splitted interactions
        for split in [TRAIN, VALSMALL, VALLARGE, TEST]:
            df_interactions_tmp = pd.read_csv(
                os.path.join(
                    self.write_dir, f"{self.dataset_name}_interactions_{split}.csv"
                )
            )
            if set([Q_OPTION_TEXTS, KC]).issubset(df_interactions_tmp.columns):
                # convert string back to list
                df_interactions_tmp[Q_OPTION_TEXTS] = df_interactions_tmp[
                    Q_OPTION_TEXTS
                ].apply(eval)
                df_interactions_tmp[KC] = df_interactions_tmp[KC].apply(eval)
            interactions[split] = df_interactions_tmp
            logger.info(
                f"Reading {split} split",
                num_interactions=len(interactions[split]),
            )
        return interactions

    def split_data(
        self,
        val_size_question: float,
        test_size_question: float,
        split_interactions: bool,  # True for DBE-KT22, False for CFE
        stratified: bool,  # True for DBE-KT22
        seed: int,
        val_size_interact: Optional[int] = None,
        valsmall_size_interact: Optional[int] = None,
        test_size_interact: Optional[int] = None,
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
        if split_interactions and (
            val_size_interact is None
            or valsmall_size_interact is None
            or test_size_interact is None
        ):
            raise ValueError(
                "All interaction split sizes must be specified when splitting."
            )
        # questions
        df_questions = self._read_questions()

        # interactions
        df_interactions = pd.read_csv(
            os.path.join(self.read_dir, f"{self.dataset_name}_interactions.csv")
        )

        # seed
        set_seed(seed)

        # train-validation-test split
        #################
        ### QUESTIONS ###
        #################
        # split the question_ids (stratified or not)
        if stratified:
            q_ids_train, q_ids_val, q_ids_test = self._stratified_split(
                df_questions, val_size_question, test_size_question
            )
        else:
            q_ids_trainval, q_ids_test = train_test_split(
                df_questions[QUESTION_ID].unique(), test_size=test_size_question
            )
            if val_size_question / (1 - test_size_question) == 1.0:
                # if there is no train set
                q_ids_train = np.array([])
                q_ids_val = q_ids_trainval
            else:
                # split train and validation
                q_ids_train, q_ids_val = train_test_split(
                    q_ids_trainval,
                    test_size=val_size_question / (1 - test_size_question),
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
                    f"{self.dataset_name}_questions_{split}.csv",
                ),
                index=False,
            )
            logger.info(
                f"Writing {split} split questions",
                num_questions=len(q_split),
            )

        ####################
        ### INTERACTIONS ###
        ####################
        if split_interactions:
            # filter out the train questions
            i_ids_train = df_interactions[
                df_interactions[QUESTION_ID].isin(q_ids_train)
            ][INTERACT_ID].tolist()
            i_ids_val = df_interactions[df_interactions[QUESTION_ID].isin(q_ids_val)][
                INTERACT_ID
            ]
            i_ids_val = i_ids_val.sample(val_size_interact).tolist()
            # get valsmall and vallarge splits
            i_ids_vallarge, i_ids_valsmall = train_test_split(
                i_ids_val, test_size=valsmall_size_interact
            )
            i_ids_test = df_interactions[df_interactions[QUESTION_ID].isin(q_ids_test)][
                INTERACT_ID
            ]
            # test
            i_ids_test = i_ids_test.sample(test_size_interact).tolist()

            interact_splits = {
                TRAIN: i_ids_train,
                VALSMALL: i_ids_valsmall,
                VALLARGE: i_ids_vallarge,
                TEST: i_ids_test,
            }
        else:
            # we can keep all interactions
            interact_splits = {
                TRAIN: df_interactions[INTERACT_ID].tolist(),
            }

        # writing interactions
        for split in interact_splits.keys():
            interactions_tmp = df_interactions[
                df_interactions[INTERACT_ID].isin(interact_splits[split])
            ].copy()
            if join_key is not None:
                # merge with questions to get more information
                interactions_tmp = pd.merge(
                    interactions_tmp, df_questions, on=join_key, how="left"
                )
            interactions_tmp.to_csv(
                os.path.join(
                    self.write_dir,
                    f"{self.dataset_name}_interactions_{split}.csv",
                ),
                index=False,
            )
            logger.info(
                f"Writing {split} split",
                num_interactions=len(interactions_tmp),
                num_distinct_questions=len(
                    interactions_tmp[QUESTION_ID].unique(),
                ),
                num_distinct_students=len(
                    interactions_tmp["student_id"].unique(),
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
        trainval_questions, test_questions = train_test_split(
            df_questions,
            test_size=test_size,
            stratify=df_questions["primary_kc"],
        )

        # Second split: val and test from temp
        train_questions, val_questions = train_test_split(
            trainval_questions,
            test_size=val_size / (1 - test_size),
            stratify=trainval_questions["primary_kc"],
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

    # def add_student_levels(self, num_groups: int) -> None:  # TODO: remove
    #     """Add student levels to the interactions.

    #     Parameters
    #     ----------
    #     num_groups : int
    #         The number of groups to create for student levels.
    #     """
    #     logger.info(
    #         "Adding student levels",
    #         num_groups=num_groups,
    #     )
    #     interactions_splits = self.read_splitted_interactions()
    #     interactions_splits[TRAIN] = compute_student_levels(interactions_splits[TRAIN])
    #     interactions_splits[TRAIN] = group_student_levels(
    #         interactions_splits[TRAIN], num_groups=num_groups
    #     )
    #     interactions_splits = self.apply_levels_to_eval(
    #         interactions_splits,
    #     )

    #     for split_name, split_value in interactions_splits.items():
    #         split_value.to_csv(
    #             os.path.join(
    #                 self.write_dir,
    #                 f"{self.dataset_name}_interactions_{split_name}.csv",
    #             ),
    #             index=False,
    #         )
    #         logger.info(
    #             f"Writing {split_name} split",
    #             num_interactions=len(split_value),
    #             num_distinct_questions=len(
    #                 split_value[QUESTION_ID].unique(),
    #             ),
    #             num_distinct_students=len(
    #                 split_value["student_id"].unique(),
    #             ),
    #         )

    # def apply_levels_to_eval(
    #     self, interactions: dict[pd.DataFrame]
    # ) -> dict[pd.DataFrame]:
    #     """Apply student levels to validation and test sets."""
    #     # apply student levels to validation and test sets
    #     for split in [VALSMALL, VALLARGE, TEST]:
    #         interactions[split] = interactions[split].merge(
    #             interactions[TRAIN][[STUDENT_ID, STUDENT_LEVEL_GROUP]].drop_duplicates(
    #                 STUDENT_ID
    #             ),
    #             on=STUDENT_ID,
    #             how="left",
    #         )

    #         # assert that all interactions have a student level
    #         print(f"{interactions[split][STUDENT_LEVEL_GROUP].isnull().sum()=}")
    #         assert (
    #             interactions[split][STUDENT_LEVEL_GROUP].notnull().all()
    #         ), f"Missing student levels in {split} split"

    #     return interactions
