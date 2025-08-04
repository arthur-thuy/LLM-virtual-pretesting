"""CUPA data manager.

Adapted from Github repo qdet_utils/data_manager/_mcq_cupa_data_manager.py
"""

# standard library imports
import os
from typing import Literal, Any
from pathlib import Path

# related third party imports
import pandas as pd
import structlog
import pandas as pd
from bs4 import BeautifulSoup

# local application/library specific imports
from tools.constants import (
    INTERACT_ID,
    Q_CONTEXT_ID,
    Q_CONTEXT_TEXT,
    Q_CORRECT_OPTION_ID,
    Q_DIFFICULTY,
    Q_DISCRIMINATION,
    Q_OPTION_TEXTS,
    Q_TEXT,
    QUESTION_ID,
    A_TEXT,
    A_SCORE,
)
from tools.utils import set_seed

logger = structlog.get_logger(__name__)


class CupaCFEDatamanager:

    def __init__(self):
        """Constructor."""
        self.name = "cupacfe"

    def build_dataset(
        self,
        read_dir: str,
        write_dir: str,
        save_dataset: bool = True,
        random_state: int = 42,
    ) -> pd.DataFrame:
        set_seed(random_state)
        df_questions = self._preprocess_questions(
            read_dir=os.path.join(read_dir, "cupa")
        )
        df_interactions = self._preprocess_interactions(
            read_dir=os.path.join(read_dir, "cfe"),
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
                Q_OPTION_TEXTS,
                Q_CORRECT_OPTION_ID,
                Q_CONTEXT_TEXT,
                Q_CONTEXT_ID,
            ],
        )
        return out_df

    def _preprocess_interactions(self, read_dir: str) -> pd.DataFrame:
        # read XML files
        df_raw = xml_folder_to_dataframe_bs(f"{read_dir}/dataset")

        # clean
        df = df_raw.replace("2.3T", "2.3")
        df = df[df["answer1_score"] != "S"]  # Filter out rows with 'S' in answer1_score
        df.answer1_score = pd.to_numeric(df.answer1_score, errors="raise")
        df = df[
            df["answer2_score"] != "5/1"
        ]  # Filter out rows with '5/1' in answer2_score
        df.answer2_score = pd.to_numeric(df.answer2_score, errors="raise")
        df = df.dropna()

        # prepare task 1
        df_task1 = df.copy()
        df_task1["task"] = "1"
        df_task1 = df_task1.drop(columns=["answer2_response", "answer2_score"])
        df_task1 = df_task1.rename(
            columns={"answer1_response": A_TEXT, "answer1_score": A_SCORE}
        )
        # prepare task 2
        df_task2 = df.copy()
        df_task2["task"] = "2"
        df_task2 = df_task2.drop(columns=["answer1_response", "answer1_score"])
        df_task2 = df_task2.rename(
            columns={"answer2_response": A_TEXT, "answer2_score": A_SCORE}
        )
        # combine tasks
        df_explode = pd.concat([df_task1, df_task2], ignore_index=True)
        # only keep task 1
        df_explode = df_explode[df_explode["task"] == "1"]

        # format XML annotations
        df_explode[A_TEXT] = df_explode.apply(
            lambda x: format_xml_annotation(x[A_TEXT]), axis=1
        )
        return df_explode


def extract_xml_data_bs(xml_file_path) -> dict[str, Any]:
    """
    Extract data from XML using BeautifulSoup.
    """
    with open(xml_file_path, "r", encoding="utf-8") as file:
        content = file.read()

    soup = BeautifulSoup(content, "xml")

    data = {
        "filename": os.path.basename(xml_file_path),
        "exam_code": Path(xml_file_path).parent.name,
    }
    data["sortkey"] = soup.learner.head["sortkey"]

    # Extract coded_answer from answer1
    answer1 = soup.find("answer1")
    data["answer1_response"] = str(answer1.find("coded_answer"))
    answer1_score = answer1.find("exam_score")
    data["answer1_score"] = answer1_score.get_text() if answer1_score else None

    # Extract coded_answer from answer2
    answer2 = soup.find("answer2")
    if answer2 is None:
        data["answer2_response"] = None
        data["answer2_score"] = None
    else:
        data["answer2_response"] = str(answer2.find("coded_answer"))
        answer2_score = answer2.find("exam_score")
        data["answer2_score"] = answer2_score.get_text() if answer2_score else None

    # Extract candidate info
    candidate = soup.find("candidate")
    data["student_score"] = float(candidate.find("score").get_text())
    data["student_language"] = candidate.find("language").get_text()
    data["student_age"] = (
        candidate.find("age").get_text() if candidate.find("age") else None
    )

    return data


def xml_folder_to_dataframe_bs(folder_path):
    """
    Convert XML files to DataFrame using BeautifulSoup.
    """
    xml_files = []
    exam_codes = os.listdir(folder_path)
    for exam_code in exam_codes:
        xml_files.extend(list(Path(folder_path).glob(f"{exam_code}/*.xml")))

    # print(xml_files)

    if not xml_files:
        print(f"No XML files found in {folder_path}")
        return pd.DataFrame()

    data_list = []
    for xml_file in xml_files:
        data = extract_xml_data_bs(xml_file)
        data_list.append(data)

    return pd.DataFrame(data_list)


def format_xml_annotation(xml_string: str) -> str:
    """
    Format XML annotations by replacing <NS> tags with their content.
    """
    soup = BeautifulSoup(xml_string, "xml")

    # Process each paragraph
    transformed_paragraphs = []
    for p in soup.find_all("p"):
        for ns in p.find_all("NS"):
            ns_type = ns.get("type")
            incorrect = ns.find("i")
            correct = ns.find("c")
            # Prepare formatted correction string
            correction = f'{incorrect.get_text() if incorrect else ""} <{ns_type} "{correct.get_text() if correct else ""}">'
            # print(correction)
            ns.replace_with(correction)

        transformed_paragraphs.append(p.get_text(separator=" ", strip=True))

    # Join with newlines
    transformed_text = "\n".join(transformed_paragraphs)
    return transformed_text
