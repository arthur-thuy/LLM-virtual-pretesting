"""Module for example formatter."""

# standard library imports
# /

# related third party imports
from pydantic import BaseModel, Field

# local application/library specific imports
from structured_outputter.build import STRUCTURED_OUTPUTTER_REGISTRY


@STRUCTURED_OUTPUTTER_REGISTRY.register("A")
class StrOutputA(BaseModel):
    """Answer to a multiple-choice question."""

    explanation: str = Field(
        description="Misconception if incorrectly answered; motivation if correctly answered"
    )
    student_answer: int = Field(
        description="The student's answer to the question, as an integer (1-4)"
    )
