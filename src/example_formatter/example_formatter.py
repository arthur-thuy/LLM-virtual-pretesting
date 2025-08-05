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
    Q_CORRECT_OPTION_ID,
    Q_OPTION_TEXTS,
    Q_TEXT,
    S_OPTION_ID,
    Q_CONTEXT_TEXT,
)


@EXAMPLE_FORMATTER_REGISTRY.register("no_quotes")
def build_no_quotes(dataset: pd.DataFrame, is_interaction: bool) -> pd.DataFrame:
    """Build example formatter no_quotes.

    Parameters
    ----------
    datasets : pd.DataFrame
        Dataset.
    is_interaction : bool
        Whether the dataset is an interaction dataset.
        If not, it is a questions dataset.

    Returns
    -------
    pd.DataFrame
        Formatted questions-answer records.
    """

    def input_fmt(row: pd.Series) -> str:
        """Create human-readable input text from a row of a DataFrame."""
        # NOTE: this is flexible wrt the number of answer options
        text = f"Question:\n{row[Q_TEXT]}\n\nOptions:\n"
        for i, option in enumerate(row[Q_OPTION_TEXTS]):
            text += f"{i+1}. {option}\n"
        text += f"\nCorrect answer: {row[Q_CORRECT_OPTION_ID]}"
        return text

    def output_fmt(row: pd.Series) -> str:
        """Create human-readable output text from a row of a DataFrame."""
        return f"Student answer: {row[S_OPTION_ID]}"

    df_out = dataset.copy()
    # Create input and output columns
    df_out[INPUT] = dataset.apply(input_fmt, axis=1)
    if is_interaction:
        df_out[OUTPUT] = dataset.apply(output_fmt, axis=1)
    return df_out


@EXAMPLE_FORMATTER_REGISTRY.register("quotes")
def build_quotes(dataset: pd.DataFrame, is_interaction: bool) -> pd.DataFrame:
    """Build example formatter quotes.

    Parameters
    ----------
    datasets : pd.DataFrame
        Dataset.
    is_interaction : bool
        Whether the dataset is an interaction dataset.
        If not, it is a questions dataset.

    Returns
    -------
    pd.DataFrame
        Formatted questions-answer records.
    """

    def input_fmt(row: pd.Series) -> str:
        """Create human-readable input text from a row of a DataFrame."""
        # NOTE: this is flexible wrt the number of answer options
        text = f'Question:\n"{row[Q_TEXT]}"\n\nOptions:\n'
        for i, option in enumerate(row[Q_OPTION_TEXTS]):
            text += f'{i+1}. "{option}"\n'
        text += f'\nCorrect answer: "{row[Q_CORRECT_OPTION_ID]}"'
        return text

    def output_fmt(row: pd.Series) -> str:
        """Create human-readable output text from a row of a DataFrame."""
        return f'Student answer: "{row[S_OPTION_ID]}"'

    df_out = dataset.copy()
    # Create input and output columns
    df_out[INPUT] = dataset.apply(input_fmt, axis=1)
    if is_interaction:
        df_out[OUTPUT] = dataset.apply(output_fmt, axis=1)
    return df_out


@EXAMPLE_FORMATTER_REGISTRY.register("mcq_reading_quotes")
def build_mcq_reading_quotes(
    dataset: pd.DataFrame, is_interaction: bool
) -> pd.DataFrame:
    """Build example formatter quotes.

    Parameters
    ----------
    datasets : pd.DataFrame
        Dataset.
    is_interaction : bool
        Whether the dataset is an interaction dataset.
        If not, it is a questions dataset.

    Returns
    -------
    pd.DataFrame
        Formatted questions-answer records.
    """

    def input_fmt(row: pd.Series) -> str:
        """Create human-readable input text from a row of a DataFrame."""
        # NOTE: this is flexible wrt the number of answer options
        # TODO: add context text
        text = (
            f"Reading context:\n{row[Q_CONTEXT_TEXT]}"
            f"\n\nQuestion:\n{row[Q_TEXT]}\n\nOptions:\n"
        )
        for i, option in enumerate(row[Q_OPTION_TEXTS]):
            text += f"{i+1}. {option}\n"
        text += f"\nCorrect answer: {row[Q_CORRECT_OPTION_ID]}"
        return text

    def output_fmt(row: pd.Series) -> str:
        """Create human-readable output text from a row of a DataFrame."""
        return f"Student answer: {row[S_OPTION_ID]}"

    df_out = dataset.copy()
    # Create input and output columns
    df_out[INPUT] = dataset.apply(input_fmt, axis=1)
    if is_interaction:
        df_out[OUTPUT] = dataset.apply(output_fmt, axis=1)
    return df_out


@EXAMPLE_FORMATTER_REGISTRY.register("open_reading")
def build_open_reading(dataset: pd.DataFrame, is_interaction: bool) -> pd.DataFrame:
    """Build example formatter quotes.

    Parameters
    ----------
    datasets : pd.DataFrame
        Dataset.
    is_interaction : bool
        Whether the dataset is an interaction dataset.
        If not, it is a questions dataset.

    Returns
    -------
    pd.DataFrame
        Formatted questions-answer records.
    """

    def output_fmt(row: pd.Series) -> str:
        """Create human-readable output text from a row of a DataFrame."""
        return (
            f"Student response (first language {row['student_language']}; "
            f"age {row['student_age']}):\n{row['answer_response']}"
        )

    df_out = dataset.copy()
    # Create input and output columns
    df_out[INPUT] = ""
    df_out[OUTPUT] = dataset.apply(output_fmt, axis=1)
    return df_out
