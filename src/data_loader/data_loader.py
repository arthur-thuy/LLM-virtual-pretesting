# standard library imports
import os
from statistics import multimode
from typing import Optional

# related third party imports
import pandas as pd
import structlog
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler

# local application/library specific imports
from tools.constants import (
    KC,
    Q_OPTION_TEXTS,
    QUESTION_ID,
    TEST,
    TRAIN,
    VALIDATION,
    VALLARGE,
    VALSMALL,
    STUDENT_ID,
    STUDENT_LEVEL_GROUP,
    STUDENT_LEVEL,
)
from tools.utils import set_seed
from tools.irt_estimator import compute_student_levels, group_student_levels

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
        if Q_OPTION_TEXTS in df_questions.columns:
            # convert string back to list
            df_questions[Q_OPTION_TEXTS] = df_questions[
                Q_OPTION_TEXTS
            ].apply(eval)
        if KC in df_questions.columns:
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
            if df_questions_tmp.empty:
                continue
            if Q_OPTION_TEXTS in df_questions_tmp.columns:
                # convert string back to list
                df_questions_tmp[Q_OPTION_TEXTS] = df_questions_tmp[
                    Q_OPTION_TEXTS
                ].apply(eval)
            if KC in df_questions_tmp.columns:
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
        if Q_OPTION_TEXTS in interact_train.columns:
            # convert string back to list
            interact_train[Q_OPTION_TEXTS] = interact_train[
                Q_OPTION_TEXTS
            ].apply(eval)
        if KC in interact_train.columns:
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
            if Q_OPTION_TEXTS in df_interactions_tmp.columns:
                # convert string back to list
                df_interactions_tmp[Q_OPTION_TEXTS] = df_interactions_tmp[
                    Q_OPTION_TEXTS
                ].apply(eval)
            if KC in df_interactions_tmp.columns:
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

            df_i_train = df_interactions[df_interactions[QUESTION_ID].isin(q_ids_train)]
            df_i_val = df_interactions[df_interactions[QUESTION_ID].isin(q_ids_val)]
            df_i_test = df_interactions[df_interactions[QUESTION_ID].isin(q_ids_test)]

            # ensure that all students in val and test interactions are present in train
            df_i_val = self.ensure_presence_in_train(df_i_train, df_i_val)
            df_i_test = self.ensure_presence_in_train(df_i_train, df_i_test)

            # add student levels to each set, computed from train interactions
            interact_splits_df = {
                TRAIN: df_i_train,
                VALIDATION: df_i_val,
                TEST: df_i_test,
            }
            interact_splits_df = self.add_student_levels(
                interact_splits_df, num_groups=5
            )

            # subsample sets and stratify on student level?
            #  undersample majority classes from "student_level_group"
            rus = RandomUnderSampler(sampling_strategy="not minority")
            # val
            df_i_val_rus, _ = rus.fit_resample(
                interact_splits_df[VALIDATION],
                interact_splits_df[VALIDATION][STUDENT_LEVEL_GROUP],
            )
            _, df_i_val = train_test_split(
                df_i_val_rus,
                test_size=val_size_interact,
                stratify=df_i_val_rus[STUDENT_LEVEL_GROUP],
            )
            # get valsmall and vallarge splits
            # TODO: stratify on student level?
            df_i_vallarge, df_i_valsmall = train_test_split(
                df_i_val,
                test_size=valsmall_size_interact,
                stratify=df_i_val[STUDENT_LEVEL_GROUP],
            )
            # test
            df_i_test_rus, _ = rus.fit_resample(
                interact_splits_df[TEST], interact_splits_df[TEST][STUDENT_LEVEL_GROUP]
            )
            _, df_i_test = train_test_split(
                df_i_test_rus,
                test_size=test_size_interact,
                stratify=df_i_test_rus[STUDENT_LEVEL_GROUP],
            )

            interact_splits_df = {
                TRAIN: interact_splits_df[TRAIN],
                VALSMALL: df_i_valsmall,
                VALLARGE: df_i_vallarge,
                TEST: df_i_test,
            }
        else:
            # we can keep all interactions
            interact_splits_df = {
                TRAIN: df_interactions
            }

        # writing interactions
        for split_name, split_df in interact_splits_df.items():
            interactions_tmp = split_df.copy()
            if join_key is not None:
                # merge with questions to get more information
                interactions_tmp = pd.merge(
                    interactions_tmp, df_questions, on=join_key, how="left"
                )
            interactions_tmp.to_csv(
                os.path.join(
                    self.write_dir,
                    f"{self.dataset_name}_interactions_{split_name}.csv",
                ),
                index=False,
            )
            logger.info(
                f"Writing {split_name} split",
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

    def ensure_presence_in_train(
        self, interact_train: pd.DataFrame, interact_eval: pd.DataFrame
    ) -> pd.DataFrame:
        """Ensure all students in the eval set are present in the training set."""
        missing_students = interact_eval[STUDENT_ID][
            ~interact_eval[STUDENT_ID].isin(interact_train[STUDENT_ID])
        ].unique()
        if missing_students.size > 0:
            logger.warning(
                f"Removing missing students in training set: {missing_students}"
            )
            interact_eval = interact_eval[
                ~interact_eval[STUDENT_ID].isin(missing_students)
            ]
        return interact_eval

    def add_student_levels(
        self, interactions_splits: dict[str, pd.DataFrame], num_groups: int
    ) -> None:
        """Add student levels to the interactions.

        Parameters
        ----------
        num_groups : int
            The number of groups to create for student levels.
        """
        logger.info(
            "Adding student levels",
            num_groups=num_groups,
        )
        interactions_splits[TRAIN] = compute_student_levels(interactions_splits[TRAIN])
        interactions_splits[TRAIN] = group_student_levels(
            interactions_splits[TRAIN], num_groups=num_groups
        )
        interactions_splits[TRAIN] = interactions_splits[TRAIN].drop(
            STUDENT_LEVEL, axis=1
        )
        interactions_splits = self.apply_levels_to_eval(
            interactions_splits,
        )

        return interactions_splits

    def apply_levels_to_eval(
        self, interactions: dict[str, pd.DataFrame]
    ) -> dict[pd.DataFrame]:
        """Apply student levels to validation and test sets."""
        # apply student levels to validation and test sets
        for split in [VALIDATION, TEST]:
            interactions[split] = interactions[split].merge(
                interactions[TRAIN][[STUDENT_ID, STUDENT_LEVEL_GROUP]].drop_duplicates(
                    STUDENT_ID
                ),
                on=STUDENT_ID,
                how="left",
            )

            # assert that all interactions have a student level
            # print(f"{interactions[split][STUDENT_LEVEL_GROUP].isnull().sum()=}")
            assert (
                interactions[split][STUDENT_LEVEL_GROUP].notnull().all()
            ), f"Missing student levels in {split} split"

        return interactions
