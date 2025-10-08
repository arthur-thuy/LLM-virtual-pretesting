"""Module for example formatter."""

# standard library imports
# /

# related third party imports
from pydantic import BaseModel, Field

# local application/library specific imports
from structured_outputter.build import STRUCTURED_OUTPUTTER_REGISTRY

# NOTE: all structured outputters need a "student_answer" field!


@STRUCTURED_OUTPUTTER_REGISTRY.register("teacher")
class StrOutputTeacher(BaseModel):
    """Answer to a multiple-choice question."""

    misconception: str = Field(
        description=(
            "The list of possible misconceptions of the student, if one or more of their previous responses is incorrect; the string 'Only correct answers' otherwise"  # noqa
        )
    )
    answer_explanation: str = Field(
        description=(
            "The list of reasoning steps that the student might follow to answer the current question (possibly affected by the identified misconceptions)"  # noqa
        )
    )
    student_answer: int = Field(
        description="The index of the answer selected by the student"
    )


@STRUCTURED_OUTPUTTER_REGISTRY.register("student")
class StrOutputStudent(BaseModel):
    """Answer to a multiple-choice question."""

    misconception: str = Field(
        description=(
            "The list of your possible misconceptions, if one or more of your previous responses is incorrect; the string 'Only correct answers' otherwise"  # noqa
        )
    )
    answer_explanation: str = Field(
        description=(
            "The list of reasoning steps that you might follow to answer the current question (possibly affected by the identified misconceptions)"  # noqa
        )
    )
    student_answer: int = Field(description="The index of your answer")


@STRUCTURED_OUTPUTTER_REGISTRY.register("student_bool")
class StrOutputStudentBool(BaseModel):
    """Answer to a multiple-choice question."""

    used_misconception: bool = Field(
        description=(
            "Indicates whether a misconception listed above was used in the answer"
        )
    )
    used_skill: bool = Field(
        description=("Indicates whether a skill listed above was used in the answer")
    )
    student_answer: int = Field(description="The index of your answer")


@STRUCTURED_OUTPUTTER_REGISTRY.register("student_bool_nocontext")
class StrOutputStudentBoolNoContext(BaseModel):
    """Answer to a multiple-choice question without context."""

    student_answer: int = Field(description="The index of your answer")


@STRUCTURED_OUTPUTTER_REGISTRY.register("teacher_bool")
class StrOutputTeacherBool(BaseModel):
    """Answer to a multiple-choice question."""

    used_misconception: bool = Field(
        description=(
            "Indicates whether a misconception listed above was used in the answer"
        )
    )
    used_skill: bool = Field(
        description=("Indicates whether a skill listed above was used in the answer")
    )
    student_answer: int = Field(description="The index of the student's answer")


@STRUCTURED_OUTPUTTER_REGISTRY.register("teacher_bool_nocontext")
class StrOutputTeacherBoolNoContext(BaseModel):
    """Answer to a multiple-choice question without context."""

    student_answer: int = Field(description="The index of the student's answer")


# @STRUCTURED_OUTPUTTER_REGISTRY.register("student_miscon")
# class StrOutputStudentMiscon(BaseModel):
#     """Answer to a multiple-choice question."""

#     answer_explanation: str = Field(
#         description=(
#             "If answering correctly, explain how the student would arrive to the answer given the mastered knowledge concepts. "  # noqa
#             "If answering incorrectly, explain how the student would arrive to the answer given the misconceptions."  # noqa
#         )
#     )
#     student_answer: int = Field(description="The index of the student's answer")


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


@STRUCTURED_OUTPUTTER_REGISTRY.register("misconceptions")
class StrOutputMisconceptions(BaseModel):
    """Answer to a multiple-choice question."""

    # TODO: Use letters A, B, C, D to refer to the options

    correct_1_knowledge_concepts: str = Field(
        description=(
            "The list of knowledge concepts that are relevant to the correct answer 1"  # noqa
        )
    )
    distractor_2_misconceptions: str = Field(
        description=(
            "The list of misconceptions that might lead the student to selecting the incorrect answer 2"  # noqa
        )
    )
    distractor_3_misconceptions: str = Field(
        description=(
            "The list of misconceptions that might lead the student to selecting the incorrect answer 3"  # noqa
        )
    )
    distractor_4_misconceptions: str = Field(
        description=(
            "The list of misconceptions that might lead the student to selecting the incorrect answer 4"  # noqa
        )
    )


@STRUCTURED_OUTPUTTER_REGISTRY.register("teacher_kt")
class StrOutputTeacherKT(BaseModel):
    """Answer to a multiple-choice question."""

    student_correct: bool = Field(
        description="Whether the student would answer correctly."
    )


@STRUCTURED_OUTPUTTER_REGISTRY.register("student_kt")
class StrOutputStudentKT(BaseModel):
    """Answer to a multiple-choice question."""

    student_correct: bool = Field(description="Whether you would answer correctly.")


@STRUCTURED_OUTPUTTER_REGISTRY.register("misconceptions_cfe")
class StrOutputMisconceptionsCFE(BaseModel):
    """Answer to a multiple-choice question."""

    skills: list[str] = Field(
        description=(
            "The list of skills that the student demonstrates in their writing"
        )
    )
    misconceptions: list[str] = Field(
        description=(
            "The list of misconceptions that the student demonstrates in their writing"  # noqa
        )
    )
