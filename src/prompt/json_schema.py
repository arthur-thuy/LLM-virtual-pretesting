"""Module for structured output."""

# standard library imports
# /

# related third party imports
import structlog
from pydantic import BaseModel, Field
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser

# local application/library specific imports
# /

# set up logger
logger = structlog.get_logger(__name__)


class MCQAnswer(BaseModel):
    """Answer to a multiple-choice question."""

    explanation: str = Field(
        description="Misconception if incorrectly answered; motivation if correctly answered"
    )
    student_answer: int = Field(
        description="The student's answer to the question, as an integer (1-4)"
    )
    # difficulty: str = Field(description="The difficulty level of the question")


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
