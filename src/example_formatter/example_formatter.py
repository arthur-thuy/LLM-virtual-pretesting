"""Module for example formatter."""

# standard library imports
# /

# related third party imports
import pandas as pd

# local application/library specific imports
from example_formatter.build import EXAMPLE_FORMATTER_REGISTRY
from tools.constants import (
    INPUT,
    OUTPUT,
    Q_TEXT,
    Q_OPTION_TEXTS,
    STUDENT_ID,
    Q_CORRECT_OPTION_ID,
    S_OPTION_ID,
)


@EXAMPLE_FORMATTER_REGISTRY.register("A")
def build_A(dataset: pd.DataFrame) -> pd.DataFrame:
    """Build example formatter A.

    Parameters
    ----------
    datasets : pd.DataFrame
        Dataset.

    Returns
    -------
    pd.DataFrame
        Formatted questions-answer records.
    """
    df_out = pd.DataFrame()
    df_out[INPUT] = dataset.apply(input_fmt_A, axis=1)
    df_out[OUTPUT] = dataset.apply(output_fmt_A, axis=1)
    df_out[STUDENT_ID] = dataset[STUDENT_ID]
    return df_out


def input_fmt_A(row: pd.Series) -> str:
    """Create human-readable input text from a row of a DataFrame.

    Parameters
    ----------
    row : pd.Series
        Row of a DataFrame.

    Returns
    -------
    str
        Human-readable input text.
    """
    # NOTE: this is flexible wrt the number of answer options
    text = f"Question:\n{row[Q_TEXT]}\n\nOptions:\n"
    for i, option in enumerate(row[Q_OPTION_TEXTS]):
        text += f"{i+1}. {option}\n"
    text += f"\nCorrect answer: {row[Q_CORRECT_OPTION_ID]}"
    return text


def output_fmt_A(row: pd.Series) -> str:
    """Create human-readable output text from a row of a DataFrame.

    Parameters
    ----------
    row : pd.Series
        Row of a DataFrame.

    Returns
    -------
    str
        Human-readable output text.
    """
    return f"Student answer: {row[S_OPTION_ID]}"
