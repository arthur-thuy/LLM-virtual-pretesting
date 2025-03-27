"""Module for system prompt."""

# standard library imports
# /

# related third party imports
# /

# local application/library specific imports
from prompt.build import SYSTEM_PROMPT_REGISTRY


@SYSTEM_PROMPT_REGISTRY.register("student_A")
def build_student_A() -> str:
    """Build system prompt.

    Returns
    -------
    str
        System prompt
    """
    system_prompt_str = (
        "You are a student working on {exam_type}, containing multiple choice questions. "
        "You are shown a set of questions that you answered earlier in the exam, together with the correct answers and your student answers. "
        "Analyse your responses to the questions and identify the possible misconceptions that led to answering incorrectly. "
        "Inspect the new question and think how you would answer it as a student. "
        "If you answer incorrectly, explain which misconception leads to selecting that answer. "
        "If you answer correctly, explain why you think the answer is correct. "
        "Provide your answer as the integer index of the multiple choice option. "
    )
    return system_prompt_str


@SYSTEM_PROMPT_REGISTRY.register("teacher_A")
def build_teacher_A() -> str:
    """Build system prompt.

    Returns
    -------
    str
        System prompt
    """
    system_prompt_str = (
        "You are an expert teacher preparing a set of multiple choice questions for {exam_type}. "
        "You will be shown a set of students' responses to previous questions. "  # They must be from the *same* or a *similar* student.
        "Analyse the responses to the questions and identify the possible misconceptions that led to the errors. "
        "Consider how those misconceptions might cause the student to make a mistake on this new question. "
        "Finally, from the provided options, select the index of the answer option that the student would select. "
    )
    return system_prompt_str
