"""Module for example formatter."""

# standard library imports
# /

# related third party imports
# from xml.parsers.expat import errors
import pandas as pd
import structlog

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
    CFE_ERROR_CODES,
)
from example_selector.utils import extract_errors, format_error_legend

logger = structlog.get_logger(__name__)


# TODO: use letters to denote options instead of integers!
# -> need to change evaluate() function to convert letters back to integers because dataset contains integers)
# def int_to_letter(i: int) -> str:
#     """Convert an integer to a letter starting from 'A'.

#     Parameters
#     ----------
#     i : int
#         Integer to convert (starts at integer 1 -> "A").

#     Returns
#     -------
#     str
#         Letter corresponding to the integer.
#     """
#     return chr(ord('@') + i + 1)


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
        if len(row[Q_OPTION_TEXTS]) > 10:
            logger.warning("Number of options > 10, this is likely a mistake.")
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
        return f"Student response:\n{row['answer_response']}"

    df_out = dataset.copy()
    # Create input and output columns
    df_out[INPUT] = (
        "Write a letter of between 120 and 180 words."  # TODO: make flexible
    )
    df_out[OUTPUT] = dataset.apply(output_fmt, axis=1)
    return df_out


@EXAMPLE_FORMATTER_REGISTRY.register("open_reading_collect_miscons")
def build_open_reading_collect_miscons(
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

    def output_fmt(row: pd.Series) -> str:
        """Create human-readable output text from a row of a DataFrame."""

        errors = extract_errors(row["answer_response"])
        error_legend = {key: CFE_ERROR_CODES[key] for key in list(errors.keys())}
        text_error_legend = format_error_legend(error_legend)
        text = (
            text_error_legend + "\n\n" + f"Student response:\n{row['answer_response']}"
        )
        return text

    df_out = dataset.copy()
    # Create input and output columns
    df_out[INPUT] = (
        "Write a letter of between 120 and 180 words."  # TODO: make flexible
    )
    df_out[OUTPUT] = dataset.apply(output_fmt, axis=1)
    return df_out
