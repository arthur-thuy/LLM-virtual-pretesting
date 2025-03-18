"""Module for structured output."""

# standard library imports
# /

# related third party imports
from pydantic import BaseModel, Field

# local application/library specific imports
# /


class MCQAnswer(BaseModel):
    """Answer to a multiple-choice question."""

    explanation: str = Field(
        description="Misconception if incorrectly answered; motivation if correctly answered"
    )
    student_answer: int = Field(
        description="The student's answer to the question, as an integer (1-4)"
    )
    # difficulty: str = Field(description="The difficulty level of the question")
