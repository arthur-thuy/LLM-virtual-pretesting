"""Module for example formatter."""

# standard library imports
# /

# related third party imports
from pydantic import BaseModel, Field

# local application/library specific imports
from structured_outputter.build import STRUCTURED_OUTPUTTER_REGISTRY

# NOTE: all structured outputters need a "student_answer" field!


@STRUCTURED_OUTPUTTER_REGISTRY.register("A")
class StrOutputA(BaseModel):
    """Answer to a multiple-choice question."""

    explanation: str = Field(
        description=(
            "Misconception if incorrectly answered; motivation if correctly answered"
        )
    )
    student_answer: int = Field(
        description="The index of the answer selected by the student"
    )


@STRUCTURED_OUTPUTTER_REGISTRY.register("B")
class StrOutputB(BaseModel):
    """Answer to a multiple-choice question."""

    misconception: str = Field(
        description=(
            "The misconceptions (if any) that the student has, "
            "from their previous responses"
        )
    )
    answer_explanation: str = Field(
        description=(
            "The reasoning steps that the student might take to answer the question"
        )
    )
    student_answer: int = Field(
        description="The index of the answer selected by the student"
    )
