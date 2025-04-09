"""Module for system prompt."""

# standard library imports
# /

# related third party imports
# /

# local application/library specific imports
from prompt.build import PROMPT_REGISTRY
from prompt.utils import prepare_str_output


@PROMPT_REGISTRY.register("student_A")
def build_student_A(few_shot_prompt, native_str_output: bool) -> list:
    # NOTE: do not add a statement about JSON output! -> this is added automatically
    system_prompt_str = (
        "You are a student working on {exam_type}, containing multiple choice questions. "
        "You are shown a set of questions that you answered earlier in the exam, together with the correct answers and your student answers. "
        "Analyse your responses to the questions and identify the possible misconceptions that led to answering incorrectly. "
        "Inspect the new question and think how you would answer it as a student. "
        "If you answer incorrectly, explain which misconception leads to selecting that answer. "
        "If you answer correctly, explain why you think the answer is correct. "
        "Provide your answer as the integer index of the multiple choice option. "
    )

    system_prompt_str = prepare_str_output(system_prompt_str, native_str_output)
    messages = [
        ("system", system_prompt_str),
        few_shot_prompt,
        ("human", "{input}"),
    ]
    return messages


@PROMPT_REGISTRY.register("teacher_A")
def build_teacher_A(few_shot_prompt, native_str_output: bool) -> list:
    # NOTE: do not add a statement about JSON output! -> this is added automatically
    system_prompt_str = (
        "You are an expert teacher preparing a set of multiple choice questions for {exam_type}. "
        "You will be shown a set of students' responses to previous questions, and asked to discuss the possible misconceptions that led to the errors and how these can affect future responses of the student. "
    )
    human1_prompt_str = "You have to discuss the misconceptions which might influence the response of the student to the following question:\n{input}"
    human2_prompt_str = "Discuss the misconceptions of the student and how they might cause them to answer wrongly to the following question:\n{input}"

    system_prompt_str = prepare_str_output(system_prompt_str, native_str_output)
    messages = [
        ("system", system_prompt_str),
        ("human", human1_prompt_str),
        few_shot_prompt,
        ("human", human2_prompt_str),
    ]
    return messages


@PROMPT_REGISTRY.register("teacher_B")
def build_teacher_B(few_shot_prompt, native_str_output: bool) -> list:
    # NOTE: do not add a statement about JSON output! -> this is added automatically
    system_prompt_str = (
        "You are an expert teacher preparing a set of multiple choice questions for {exam_type}. "
        "You will be shown a student's responses to previous questions, and asked to identify the possible misconceptions that led to the errors. "
        "Next, you will be shown a new multiple choice question, and asked to discuss how the student would answer it, keeping in mind the misconceptions identified earlier. "
    )
    human1_prompt_str = "Identify the misconceptions that caused incorrect answers in the following question-answer records:"
    human2_prompt_str = "Inspect this new multiple choice question and discuss how the student would answer it, keeping in mind the misconceptions identified earlier:\n{input}"

    system_prompt_str = prepare_str_output(system_prompt_str, native_str_output)
    messages = [
        ("system", system_prompt_str),
        ("human", human1_prompt_str),
        few_shot_prompt,
        ("human", human2_prompt_str),
    ]
    return messages
