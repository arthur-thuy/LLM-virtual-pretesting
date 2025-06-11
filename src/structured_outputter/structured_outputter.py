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


@STRUCTURED_OUTPUTTER_REGISTRY.register("LB_A")
class StrOutputB(BaseModel):
    """Answer to a multiple-choice question."""

    misconception: str = Field(
        description=(
            "The list of possibly misconceptions of the student, if one or more of their previous responses is incorrect, the string 'correct answers' otherwise"
        )
    )
    answer_explanation: str = Field(
        description=(
            "The list of reasoning steps that the student might make to answer the current question (possibly affected by the identified misconceptions)"
        )
    )
    student_answer: int = Field(
        description="The index of the answer selected by the student"
    )


@STRUCTURED_OUTPUTTER_REGISTRY.register("luca_emnlp")
class StrOutputLucaEMNLP(BaseModel):
    """Answer to a multiple-choice question."""

    question_level: float = Field(description="difficulty level of the question")
    answer_explanation: str = Field(
        description=(
            "the list of steps that the students of this level would follow to select "
            "the answer, including the misconceptions that might cause them to make "
            "mistakes"
        )
    )
    student_answer: int = Field(
        description="integer index of the answer chosen by a student of this level"
    )
