"""Module for creating few-shot prompts."""

# standard library imports
# /

# related third party imports
import pandas as pd

# local application/library specific imports
# /


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
