"""Utils file."""

# standard library imports
# /

# related third party imports
import pandas as pd
import structlog

# local application/library specific imports
# /

# set up logger
logger = structlog.get_logger(__name__)


def count_answer_options(row, df_question_choice: pd.DataFrame) -> int:
    """Count the number of answer options for a question.

    Parameters
    ----------
    row : _type_
        Dataframe row.
    df_question_choice : pd.DataFrame
        Dataframe with question choices.

    Returns
    -------
    int
        Number of answer options.
    """
    return len(df_question_choice[df_question_choice["question_id"] == row["id"]])


def bring_correct_option_forward(row):
    """
    Bring the correct option to the front of the option_texts list.
    """
    correct_option_index = row["correct_option_id"] - 1  # Convert to zero-based index
    row["option_texts"].insert(0, row["option_texts"].pop(correct_option_index))
    row["correct_option_id"] = 1  # Update correct option index to 1 (first position)
    return row
