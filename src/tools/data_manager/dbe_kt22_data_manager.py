"""DBE-KT22 data manager."""

# standard library imports
import os
from typing import Optional

# related third party imports
import numpy as np
import pandas as pd
import structlog

# local application/library specific imports
from tools.data_manager.utils import count_answer_options
from tools.utils import set_seed
from tools.constants import (
    CORRECT_ANSWER,
    INTERACT_ID,
    QUESTION_ID,
    STUDENT_ID,
    Q_TEXT,
    OPTIONS_TEXT,
    STUDENT_ANSWER,
    Q_DIFFICULTY,
)


logger = structlog.get_logger(__name__)


class DBEKT22Datamanager:

    def __init__(self):
        """Constructor."""
        self.name = "dbe_kt22"

    def build_dataset(
        self,
        read_dir: str,
        write_dir: str,
        sample_student_ids: Optional[int] = None,
        save_dataset: bool = True,
        random_state: int = 42,
    ) -> pd.DataFrame:
        set_seed(random_state)
        df = self._preprocess_datasets(
            read_dir=os.path.join(read_dir, self.name),
            sample_student_ids=sample_student_ids,
        )
        if save_dataset:
            output_path = os.path.join(write_dir, f"{self.name}.csv")
            logger.info("Saving dataset", path=output_path)
            df.to_csv(output_path, index=False)

        return df

    def _process_row(
        self, row, df_q: pd.DataFrame, df_q_choice: pd.DataFrame
    ) -> tuple[str, list, int, int, int]:
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
        correct_answer = df_q_choice_text[df_q_choice_text["is_correct"] == True].iloc[
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

    def _preprocess_datasets(
        self, read_dir: str, sample_student_ids: Optional[int] = None
    ) -> pd.DataFrame:

        # student-question interactions
        df_interact = pd.read_csv(os.path.join(read_dir, "Transaction.csv"))
        # NOTE: omit questions where hint is used
        df_interact = df_interact[df_interact["hint_used"] == False]
        df_interact = df_interact.rename(columns={"answer_state": "answer_correct"})
        df_interact = df_interact[
            ["id", STUDENT_ID, QUESTION_ID, "answer_choice_id", "answer_correct"]
        ]

        if sample_student_ids:
            # randomly select student ids
            student_ids = np.random.choice(
                df_interact[STUDENT_ID].unique(), size=sample_student_ids, replace=False
            )
            df_interact = df_interact[df_interact[STUDENT_ID].isin(student_ids)]

        # answer options
        df_question_choice = pd.read_csv(os.path.join(read_dir, "Question_Choices.csv"))

        # questions
        df_question = pd.read_csv(os.path.join(read_dir, "Questions.csv"))
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
                self._process_row,
                df_q=df_question,
                df_q_choice=df_question_choice,
                axis=1,
            )
        )

        return df
