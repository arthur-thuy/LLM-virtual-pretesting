"""Module for creating few-shot prompts."""

# standard library imports
from typing import Callable

# related third party imports
import pandas as pd

# local application/library specific imports
# /


def human_format_input(row: pd.Series) -> str:
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
    text = f"Question:\n{row.q_text}\n\nOptions:\n"
    for i, option in enumerate(row.options_text):
        text += f"{i+1}. {option}\n"
    text += f"\nCorrect answer: {row.correct_answer}"
    return text


def human_format_output(row: pd.Series) -> str:
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
    return f"Student answer: {row.student_answer}"


def apply_prompt_fmt(
    df: pd.DataFrame, input_fmt: Callable, output_fmt: Callable
) -> pd.DataFrame:
    """Apply input and output formatting functions to a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to apply
    input_fmt : Callable
        Function to apply to the input
    output_fmt : Callable
        Function to apply to the output

    Returns
    -------
    pd.DataFrame
        DataFrame with formatted input and output columns
    """
    df_out = pd.DataFrame()
    df_out["input"] = df.apply(input_fmt, axis=1)
    df_out["output"] = df.apply(output_fmt, axis=1)
    df_out["student_id"] = df["student_id"]
    return df_out


def df_to_listdict(df: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame to a list of dictionaries.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    list[dict]
        List of dictionaries.
    """
    list_out = []
    for _, row in df.iterrows():
        dict_out = {colname: row[colname] for colname in df.columns}
        list_out.append(dict_out)
    return list_out
