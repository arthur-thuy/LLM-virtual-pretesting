"""Utils file."""

# standard library imports
# /

# related third party imports
import pandas as pd
import structlog

# local application/library specific imports
from tools.constants import (
    Q_CORRECT_OPTION_ID,
    Q_OPTION_TEXTS,
    S_OPTION_ID,
)


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


def bring_correct_option_forward(row, is_interaction: bool):
    """
    Bring the correct option to the front of the option_texts list.
    """
    # Convert to zero-indexing
    correct_option_index = row[Q_CORRECT_OPTION_ID] - 1
    if is_interaction:
        student_option_index = row[S_OPTION_ID] - 1

    # Get option texts and pop the correct option
    if is_interaction:
        student_option_text = row[Q_OPTION_TEXTS][student_option_index]
    correct_option_text = row[Q_OPTION_TEXTS].pop(correct_option_index)

    # Insert the correct option at the front
    row[Q_OPTION_TEXTS].insert(0, correct_option_text)
    # Find new indices from text
    row[Q_CORRECT_OPTION_ID] = row[Q_OPTION_TEXTS].index(correct_option_text)
    assert row[Q_CORRECT_OPTION_ID] == 0, "Correct option should be at index 0"
    # Update student option ID based on the new ordering
    if is_interaction:
        row[S_OPTION_ID] = row[Q_OPTION_TEXTS].index(student_option_text)

    # convert to one-indexing
    row[Q_CORRECT_OPTION_ID] += 1
    if is_interaction:
        row[S_OPTION_ID] += 1
    return row
