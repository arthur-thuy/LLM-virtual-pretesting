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
        "You are a student working on {exam_type}, containing multiple choice questions. "  # noqa
        "You are shown a set of questions that you answered earlier in the exam, together with the correct answers and your student answers. "  # noqa
        "Analyse your responses to the questions and identify the possible misconceptions that led to answering incorrectly. "  # noqa
        "Inspect the new question and think how you would answer it as a student. "
        "If you answer incorrectly, explain which misconception leads to selecting that answer. "  # noqa
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
        "You are an expert teacher preparing a set of multiple choice questions for {exam_type}. "  # noqa
        "You will be shown a set of students' responses to previous questions, and asked to discuss the possible misconceptions that led to the errors and how these can affect future responses of the student. "  # noqa
    )
    human1_prompt_str = "You have to discuss the misconceptions which might influence the response of the student to the following question:\n{input}"  # noqa
    human2_prompt_str = "Discuss the misconceptions of the student and how they might cause them to answer wrongly to the following question:\n{input}"  # noqa

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
        "You are an expert teacher preparing a set of multiple choice questions for {exam_type}. "  # noqa
        "You will be shown a student's responses to previous questions, and asked to identify the possible misconceptions that led to the errors. "  # noqa
        "Next, you will be shown a new multiple choice question, and asked to discuss how the student would answer it, keeping in mind the misconceptions identified earlier. "  # noqa
    )
    human1_prompt_str = "Identify the misconceptions that caused incorrect answers in the following question-answer records:"  # noqa
    human2_prompt_str = "Inspect this new multiple choice question and discuss how the student would answer it, keeping in mind the misconceptions identified earlier:\n{input}"  # noqa

    system_prompt_str = prepare_str_output(system_prompt_str, native_str_output)
    messages = [
        ("system", system_prompt_str),
        ("human", human1_prompt_str),
        few_shot_prompt,
        ("human", human2_prompt_str),
    ]
    return messages


@PROMPT_REGISTRY.register("teacher_C")
def build_teacher_C(few_shot_prompt, native_str_output: bool) -> list:
    # NOTE: do not add a statement about JSON output! -> this is added automatically
    system_prompt_str = (
        "You are an expert teacher preparing a set of multiple choice questions for {exam_type}. "  # noqa
        "You will be shown a student's responses to previous questions. Identify the possible misconceptions that led to the errors. "  # noqa
        "Next, you will be shown a new multiple choice question. Discuss how the student would answer it, keeping in mind the misconceptions identified earlier. "  # noqa
    )
    human1_prompt_str = "Question-answer records:"
    human2_prompt_str = "New multiple choice question:\n{input}"

    system_prompt_str = prepare_str_output(system_prompt_str, native_str_output)
    messages = [
        ("system", system_prompt_str),
        ("human", human1_prompt_str),
        few_shot_prompt,
        ("human", human2_prompt_str),
    ]
    return messages


@PROMPT_REGISTRY.register("teacher_LB_A")
def build_teacher_D(few_shot_prompt, native_str_output: bool) -> list:
    # NOTE: do not add a statement about JSON output! -> this is added automatically
    system_prompt_str = (
        "You are a teacher curating {exam_type}, and need to hypothesise how specific students would answer to a new question. "
        "You will be shown a student's responses to previous questions; "
        "if one or more of the responses are wrong, list the misconceptions that possibly led to the errors. "  # noqa
        "You will be then shown a new multiple choice question. "
        "Discuss how the student would answer it, and how the misconceptions you identified might cause them to answer wrongly. "  # noqa
    )
    human1_prompt_str = "Question-answer records:"
    human2_prompt_str = "New multiple choice question:\n{input}"

    system_prompt_str = prepare_str_output(system_prompt_str, native_str_output)
    messages = [
        ("system", system_prompt_str),
        ("human", human1_prompt_str),
        few_shot_prompt,
        ("human", human2_prompt_str),
    ]
    return messages


@PROMPT_REGISTRY.register("teacher_LB_B")
def build_teacher_D(few_shot_prompt, native_str_output: bool) -> list:
    # NOTE: do not add a statement about JSON output! -> this is added automatically
    system_prompt_str = (
        "You are a teacher curating {exam_type}, and I want you to provide feedback about a student's responses, as well as discuss how they would likely answer to new questions. "
        "First, you will be shown the student's responses to previous questions; "
        "you need to discuss the possible misconceptions that caused the errors, if any. "  # noqa
        "Then, you will be shown a new multiple choice question, and have to discuss how that same student would answer it. "
        "Specifically, discuss how the misconceptions you have identified might be the cause of new errors. "  # noqa
    )
    human1_prompt_str = "Question-answer records:"
    human2_prompt_str = "New multiple choice question:\n{input}"

    system_prompt_str = prepare_str_output(system_prompt_str, native_str_output)
    messages = [
        ("system", system_prompt_str),
        ("human", human1_prompt_str),
        few_shot_prompt,
        ("human", human2_prompt_str),
    ]
    return messages


@PROMPT_REGISTRY.register("roleplay_teacher_A")
def build_roleplay_teacher_A(few_shot_prompt, native_str_output: bool) -> list:
    # NOTE: do not add a statement about JSON output! -> this is added automatically
    system_prompt_str = (
        "You are an expert teacher preparing a set of multiple choice questions for {exam_type}. "  # noqa
        "You will be shown previous question-answer records from students of level {student_level_group} {student_scale}. Identify the possible misconceptions that led to the errors. "  # noqa
        "Next, you will be shown a new multiple choice question. Discuss how a student of that level would answer it, keeping in mind the misconceptions identified earlier. "  # noqa
    )
    human1_prompt_str = "Question-answer records:"
    human2_prompt_str = "New multiple choice question:\n{input}"

    system_prompt_str = prepare_str_output(system_prompt_str, native_str_output)
    messages = [
        ("system", system_prompt_str),
        ("human", human1_prompt_str),
        few_shot_prompt,
        ("human", human2_prompt_str),
    ]
    return messages


@PROMPT_REGISTRY.register("roleplay_luca_emnlp")
def build_roleplay_luca_emnlp(few_shot_prompt, native_str_output: bool) -> list:
    # NOTE: do not add a statement about JSON output! -> this is added automatically

    system_prompt_str = (
        "You will be shown a multiple choice question from {exam_type}, and the questions in the exam have difficulty levels "  # noqa
        "on a scale from one (very easy) to five (very difficult). "  # TODO: make dynamic  # noqa
        "You must assign a difficulty level to the given multiple choice question, and select the answer choice that a student of level {student_level_group} would pick. "  # noqa
    )
    # TODO: add  {student_scale}
    human_prompt_str = "Question:\n{input}"

    system_prompt_str = prepare_str_output(system_prompt_str, native_str_output)
    # TODO: do not add few_shot_prompt because it is zero-shot!
    messages = [
        ("system", system_prompt_str),
        ("human", human_prompt_str),
    ]
    return messages
