"""Module for system prompt."""

# standard library imports
# /

# related third party imports
# /

# local application/library specific imports
from prompt.build import SYSTEM_PROMPT_REGISTRY


@SYSTEM_PROMPT_REGISTRY.register("A")
def build_A() -> str:
    """Build system prompt A.

    Returns
    -------
    str
        System prompt A
    """
    system_prompt_str = (
        "You are a student working on {exam_type}, containing multiple choice questions. "
        "You are shown a set of questions that you answered earlier in the exam, together with the correct answers and your student answers. "
        "Analyse your responses to the questions and identify the possible misconceptions that led to answering incorrectly. "
        "Inspect the new question and think how you would answer it as a student. "
        "If you answer incorrectly, explain which misconception leads to selecting that answer. "
        "If you answer correctly, explain why you think the answer is correct. "
        "Provide your answer as an integer in the range 1-4. "
    )
    return system_prompt_str
