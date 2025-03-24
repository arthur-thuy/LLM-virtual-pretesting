"""CUPA data manager.

Adapted from Github repo qdet_utils/data_manager/_mcq_cupa_data_manager.py
"""

# standard library imports
import os
from typing import Literal

# related third party imports
import pandas as pd
import structlog

# local application/library specific imports
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


class CupaDatamanager:

    def __init__(self):
        """Constructor."""
        self.name = "cupa"

    def build_dataset(
        self,
        read_dir: str,
        write_dir: str,
        save_dataset: bool = True,
        random_state: int = 42,
    ) -> pd.DataFrame:
        set_seed(random_state)
        df_questions = self._preprocess_questions(
            read_dir=os.path.join(read_dir, self.name)
        )
        df_interactions = self._preprocess_interactions(
            read_dir=os.path.join(read_dir, self.name),
        )
        if save_dataset:
            output_path = os.path.join(write_dir, f"{self.name}_questions.csv")
            logger.info("Saving dataset", path=output_path)
            df_questions.to_csv(output_path, index=False)

            output_path = os.path.join(write_dir, f"{self.name}_interactions.csv")
            logger.info("Saving dataset", path=output_path)
            df_interactions.to_csv(output_path, index=False)

        return df_questions, df_interactions

    def _preprocess_questions(
        self, read_dir: str, diff_type: Literal["cefr", "irt"] = "irt"
    ) -> pd.DataFrame:

        df = pd.read_json(os.path.join(read_dir, "mcq_data_cupa.jsonl"), lines=True)

        assert len(df) == df["id"].nunique()

        out_list = []
        for _, row in df.iterrows():
            context = (
                row["text"].encode("ascii", "ignore").decode("ascii")
            )  # to fix issue with encoding
            context_id = row["id"]

            for q_idx, q_val in row["questions"].items():

                q_id = f"{context_id}_Q{q_idx}"
                question = (
                    q_val["text"].encode("ascii", "ignore").decode("ascii")
                )  # to fix issue with encoding
                option_0 = (
                    q_val["options"]["a"]["text"]
                    .encode("ascii", "ignore")
                    .decode("ascii")
                )  # to fix issue with encoding
                option_1 = (
                    q_val["options"]["b"]["text"]
                    .encode("ascii", "ignore")
                    .decode("ascii")
                )  # to fix issue with encoding
                option_2 = (
                    q_val["options"]["c"]["text"]
                    .encode("ascii", "ignore")
                    .decode("ascii")
                )  # to fix issue with encoding
                option_3 = (
                    q_val["options"]["d"]["text"]
                    .encode("ascii", "ignore")
                    .decode("ascii")
                )  # to fix issue with encoding
                options = [option_0, option_1, option_2, option_3]
                correct_answer = ord(q_val["answer"]) - ord("a")
                # define difficulty
                if diff_type == "irt":
                    # continuous IRT value
                    difficulty = q_val["diff"]
                else:
                    # dicrete CEFR level: same level for all questions in the same context
                    difficulty = row["level"]
                discrimination = q_val["disc"]

                out_list.append(
                    {
                        QUESTION_ID: q_id,
                        Q_TEXT: question,
                        Q_DIFFICULTY: difficulty,
                        Q_DISCRIMINATION: discrimination,
                        Q_OPTION_IDS: list(range(4)),
                        Q_OPTION_TEXTS: options,
                        Q_CORRECT_OPTION_ID: correct_answer,
                        Q_CONTEXT_TEXT: context,
                        Q_CONTEXT_ID: context_id,
                    }
                )

        out_df = pd.DataFrame(
            out_list,
            columns=[
                QUESTION_ID,
                Q_TEXT,
                Q_DIFFICULTY,
                Q_DISCRIMINATION,
                Q_OPTION_IDS,
                Q_OPTION_TEXTS,
                Q_CORRECT_OPTION_ID,
                Q_CONTEXT_TEXT,
                Q_CONTEXT_ID,
            ],
        )
        return out_df

    def _preprocess_interactions(self, read_dir: str) -> pd.DataFrame:
        df = pd.DataFrame()
        # TODO: implement (using FCE dataset???)
        return df
