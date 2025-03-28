"""DBE-KT22 data manager."""

# standard library imports
import os
import re
from typing import Optional

# related third party imports
import numpy as np
import pandas as pd
import structlog

# local application/library specific imports
from tools.data_manager.utils import count_answer_options
from tools.utils import set_seed
from tools.constants import (
    INTERACT_ID,
    QUESTION_ID,
    STUDENT_ID,
    Q_TEXT,
    Q_DIFFICULTY,
    Q_OPTION_IDS,
    Q_OPTION_TEXTS,
    Q_CORRECT_OPTION_ID,
    Q_CONTEXT_TEXT,
    Q_CONTEXT_ID,
    Q_DISCRIMINATION,
    S_OPTION_ID,
    S_OPTION_CORRECT,
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
        df_questions = self._preprocess_questions(
            read_dir=os.path.join(read_dir, self.name)
        )
        df_interactions = self._preprocess_interactions(
            read_dir=os.path.join(read_dir, self.name),
            df_questions=df_questions,
            sample_student_ids=sample_student_ids,
        )
        df_questions = df_questions.drop(columns=[Q_OPTION_IDS])
        if save_dataset:
            output_path = os.path.join(write_dir, f"{self.name}_questions.csv")
            logger.info("Saving questions dataset", path=output_path)
            df_questions.to_csv(output_path, index=False)

            output_path = os.path.join(write_dir, f"{self.name}_interactions.csv")
            logger.info(
                "Saving interactions dataset",
                path=output_path,
                num_interactions=len(df_interactions),
            )
            df_interactions.to_csv(output_path, index=False)

        return df_questions, df_interactions

    def _process_question_row(
        self, row, df_q_choice: pd.DataFrame
    ) -> tuple[list, list, int]:
        # NOTE: get answer option texts, shuffle them
        df_q_choice_text = df_q_choice[
            df_q_choice[QUESTION_ID] == row[QUESTION_ID]
        ].sample(frac=1)
        assert (
            df_q_choice_text.empty == False
        ), f"Empty df_q_choice_text for {row[QUESTION_ID]}"
        # get info
        options_texts = df_q_choice_text["choice_text"].tolist()
        options_ids = df_q_choice_text["id"].tolist()
        correct_answer_id = df_q_choice_text[
            df_q_choice_text["is_correct"] == True
        ].iloc[0]["id"]
        context_text = None
        context_id = None
        q_discrimination = None
        return (
            options_ids,
            options_texts,
            correct_answer_id,
            context_text,
            context_id,
            q_discrimination,
        )

    def _preprocess_questions(self, read_dir: str) -> pd.DataFrame:
        # answer options
        df_question_choice = pd.read_csv(os.path.join(read_dir, "Question_Choices.csv"))

        # questions
        df_question = pd.read_csv(os.path.join(read_dir, "Questions.csv"))
        df_question["num_answer_options"] = df_question.apply(
            lambda row: count_answer_options(row, df_question_choice), axis=1
        )

        df = pd.DataFrame()
        df[[QUESTION_ID, Q_TEXT, Q_DIFFICULTY, "num_answer_options"]] = df_question[
            ["id", "question_rich_text", "difficulty", "num_answer_options"]
        ]

        (
            df[Q_OPTION_IDS],
            df[Q_OPTION_TEXTS],
            df[Q_CORRECT_OPTION_ID],
            df[Q_CONTEXT_TEXT],
            df[Q_CONTEXT_ID],
            df[Q_DISCRIMINATION],
        ) = zip(
            *df.apply(
                self._process_question_row,
                df_q_choice=df_question_choice,
                axis=1,
            )
        )
        # get position of correct answer in the options
        # NOTE: this one-indexing aligns with one-indexing in prompt
        df[Q_CORRECT_OPTION_ID] = df.apply(
            lambda row: row[Q_OPTION_IDS].index(row[Q_CORRECT_OPTION_ID]) + 1, axis=1
        )

        # regex to get latex code from html link
        pattern = (
            r'<img src="http://latex\.codecogs\.com/gif\.latex\?([^"]+)" border="0"/>'
        )
        df[Q_TEXT] = df[Q_TEXT].apply(lambda row: re.sub(pattern, r"\1", row))
        return df

    def _process_interact_row(self, row, df_q: pd.DataFrame) -> int:
        option_ids = df_q[df_q[QUESTION_ID] == row[QUESTION_ID]].iloc[0][Q_OPTION_IDS]
        assert (
            row[S_OPTION_ID] in option_ids
        ), f"Option id {row[S_OPTION_ID]} not in {option_ids} of question ID {row[QUESTION_ID]}"
        # NOTE: this one-indexing aligns with one-indexing in prompt
        return option_ids.index(row[S_OPTION_ID]) + 1  # NOTE: one-indexing

    def _preprocess_interactions(
        self,
        read_dir: str,
        df_questions: pd.DataFrame,
        sample_student_ids: Optional[int] = None,
    ) -> pd.DataFrame:
        # student-question interactions
        df_interact = pd.read_csv(os.path.join(read_dir, "Transaction.csv"))
        # NOTE: omit records where hint is used
        df_interact = df_interact[df_interact["hint_used"] == False]
        # remove invalid interactions (manually identified)
        df_interact = df_interact[df_interact["id"] != 2878]
        # only keep students that answered all questions
        num_logs = (
            df_interact.drop_duplicates(subset=["question_id", "student_id"])
            .groupby("student_id")["question_id"]
            .count()
            .sort_values(ascending=False)
            .reset_index()
        )
        students_all_q = num_logs[num_logs["question_id"] == 212]["student_id"].tolist()
        df_interact = df_interact[df_interact["student_id"].isin(students_all_q)]

        df_interact = df_interact[
            ["id", STUDENT_ID, QUESTION_ID, "answer_choice_id", "answer_state"]
        ]
        df_interact = df_interact.rename(
            columns={
                "id": INTERACT_ID,
                "answer_state": S_OPTION_CORRECT,
                "answer_choice_id": S_OPTION_ID,
            }
        )

        if sample_student_ids:
            # randomly select student ids
            student_ids = np.random.choice(
                df_interact[STUDENT_ID].unique(), size=sample_student_ids, replace=False
            )
            df_interact = df_interact[df_interact[STUDENT_ID].isin(student_ids)]

        df_interact[S_OPTION_ID] = df_interact.apply(
            self._process_interact_row,
            df_q=df_questions,
            axis=1,
        )

        return df_interact
