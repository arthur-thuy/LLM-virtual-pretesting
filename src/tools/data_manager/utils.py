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


def bring_correct_option_forward(row):
    """
    Bring the correct option to the front of the option_texts list.
    """
    correct_option_index = row[Q_CORRECT_OPTION_ID] - 1  # Convert to zero-based index
    row[Q_OPTION_TEXTS].insert(0, row[Q_OPTION_TEXTS].pop(correct_option_index))
    row[Q_CORRECT_OPTION_ID] = 1  # Update correct option index to 1 (first position)
    row[S_OPTION_ID] = None  # make unusable, but not needed for misconception listing
    return row
