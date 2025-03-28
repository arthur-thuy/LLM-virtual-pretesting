"""Module for prompt utils."""

# standard library imports
# /

# related third party imports
import pandas as pd
import structlog
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser

# local application/library specific imports
# /

# set up logger
logger = structlog.get_logger(__name__)


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


def validate_output(outputs: list, schema) -> list:
    """Validate the LLM outputs against the schema.

    Parameters
    ----------
    outputs : list
        List of AIMessages
    schema : _type_
        Pydanctic schema

    Returns
    -------
    list
        List of validated outputs
    """
    logger.info("Validating outputs")
    parser = PydanticOutputParser(pydantic_object=schema)
    outputs_validated = []
    for i, output in enumerate(outputs):
        try:
            output_validated = parser.invoke(output)
        except OutputParserException as e:
            logger.warning("Invalid output", index=i)
            print(e)
            output_validated = schema(explanation="", student_answer=-1)
        outputs_validated.append(output_validated)

    return outputs_validated
