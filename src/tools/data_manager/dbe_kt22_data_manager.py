"""DBE-KT22 data manager."""

# standard library imports
import os
from typing import Dict

# related third party imports
import numpy as np
import pandas as pd
import structlog
from sklearn.model_selection import train_test_split

# local application/library specific imports
from tools.data_manager.utils import count_answer_options
from tools.utils import set_seed
from tools.constants import (
    CORRECT_ANSWER,
    TEST,
    TRAIN,
    VALIDATION,
)

QUESTION_ID = "question_id"
STUDENT_ID = "student_id"
INTERACT_ID = "interact_id"
Q_TEXT = "q_text"
STUDENT_ANSWER = "student_answer"
Q_DIFFICULTY = "q_difficulty"
OPTIONS_TEXT = "options_text"

logger = structlog.get_logger(__name__)


class DBEKT22Datamanager:

    def __init__(self):
        """No constructor."""
        pass

    def build_dataset(
        self,
        data_dir: str,
        output_data_dir: str,
        save_dataset: bool = True,
        train_size: float = 0.6,
        test_size: float = 0.25,
        random_state: int = 42,
    ) -> Dict[str, pd.DataFrame]:
        set_seed(random_state)
        df = self._preprocess_datasets(data_dir=data_dir)
        # train-validation-test split
        idx_trainval, idx_test = train_test_split(
            df[INTERACT_ID], test_size=test_size
        )
        idx_train, idx_val = train_test_split(
            idx_trainval, test_size=1 - train_size / (1 - test_size)
        )
        splits = {TRAIN: idx_train, VALIDATION: idx_val, TEST: idx_test}

        dataset = dict()
        for split in [TRAIN, VALIDATION, TEST]:
            dataset[split] = df[df[INTERACT_ID].isin(splits[split])].copy()
            logger.info(
                f"Creating {split} split",
                num_interactions=len(dataset[split]),
            )
            if save_dataset:
                dataset[split].to_csv(
                    os.path.join(output_data_dir, f"dbe_kt22_{split}.csv"), index=False
                )
        return dataset

    def _process_row(self, row, df_q: pd.DataFrame, df_q_choice: pd.DataFrame) -> str:
        # NOTE: get answer option texts, shuffle them and number them
        q_text = df_q[df_q["id"] == row[QUESTION_ID]].iloc[0]["question_rich_text"]
        df_q_choice_text = df_q_choice[
            df_q_choice[QUESTION_ID] == row[QUESTION_ID]
        ].sample(frac=1)
        assert (  # TODO: remove this if we relax the number of answer options
            len(df_q_choice_text) == 4
        ), f"Expected 4 answer options, got {len(df_q_choice_text)}"
        df_q_choice_text["option_position"] = np.arange(1, len(df_q_choice_text) + 1)

        student_answer = df_q_choice_text[
            df_q_choice_text["id"] == row.answer_choice_id
        ].iloc[0]["option_position"]
        correct_answer = df_q_choice_text[df_q_choice_text["is_correct"] == True].iloc[  # noqa: E712
            0
        ]["option_position"]
        q_difficulty = df_q[df_q["id"] == row[QUESTION_ID]].iloc[0]["difficulty"]
        options_text = [
            df_q_choice_text[df_q_choice_text["option_position"] == i].iloc[0][
                "choice_text"
            ]
            for i in range(1, len(df_q_choice_text) + 1)
        ]

        return q_text, options_text, student_answer, correct_answer, q_difficulty

    def _preprocess_datasets(self, data_dir: str) -> pd.DataFrame:

        # student-question interactions
        df_interact = pd.read_csv(os.path.join(data_dir, "Transaction.csv"))
        # NOTE: omit questions where hint is used
        df_interact = df_interact[df_interact["hint_used"] == False]  # noqa: E712
        df_interact = df_interact.rename(columns={"answer_state": "answer_correct"})
        df_interact = df_interact[
            ["id", STUDENT_ID, QUESTION_ID, "answer_choice_id", "answer_correct"]
        ]

        # TODO: remove this filtering
        df_interact = df_interact[df_interact["student_id"] == 5]

        # answer options
        df_question_choice = pd.read_csv(os.path.join(data_dir, "Question_Choices.csv"))

        # questions
        df_question = pd.read_csv(os.path.join(data_dir, "Questions.csv"))
        df_question["num_answer_options"] = df_question.apply(
            lambda row: count_answer_options(row, df_question_choice), axis=1
        )
        df_question = df_question[
            df_question["num_answer_options"] == 4
        ]  # TODO: can we keep this?

        # only keep question_id's that are in the df_question
        df_interact = df_interact[df_interact[QUESTION_ID].isin(df_question["id"])]

        df = pd.DataFrame()
        df[[INTERACT_ID, QUESTION_ID, STUDENT_ID]] = df_interact[
            ["id", QUESTION_ID, STUDENT_ID]
        ]

        (
            df[Q_TEXT],
            df[OPTIONS_TEXT],
            df[STUDENT_ANSWER],
            df[CORRECT_ANSWER],
            df[Q_DIFFICULTY],
        ) = zip(
            *df_interact.apply(
                self._process_row, df_q=df_question, df_q_choice=df_question_choice, axis=1
            )
        )

        return df
