"""Module for system prompt."""

# standard library imports
# /

# related third party imports
# /

# local application/library specific imports
from prompt.build import PROMPT_REGISTRY
from prompt.utils import prepare_str_output


###########################
### REPLICATION PROMPTS ###
###########################

@PROMPT_REGISTRY.register("student_B")
def build_student_B(few_shot_prompt, native_str_output: bool) -> list:
    # NOTE: do not add a statement about JSON output! -> this is added automatically
    system_prompt_str = (
        "You are a student working on {exam_type}, containing multiple choice questions. "  # noqa
        "You will be shown your question-answer records from earlier in the exam, together with the correct answers. "  # noqa
        "Analyze your responses and identify the possible misconceptions that led to your errors, if any. "  # noqa
        "Next, you will be shown a new multiple choice question. "
        "Inspect the new question and think how you would answer it as a student, keeping in mind the misconceptions identified earlier. "  # noqa
        "You can answer incorrectly, if that is what the student is likely to do for this question."  # noqa
    )  # TODO: "the student is" or "you are"?
    # NOTE: instructions on how to answer exactly are given in the JSON structured output
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


@PROMPT_REGISTRY.register("teacher_C")
def build_teacher_C(few_shot_prompt, native_str_output: bool) -> list:
    # NOTE: do not add a statement about JSON output! -> this is added automatically
    system_prompt_str = (
        "You are an expert teacher preparing a set of multiple choice questions for {exam_type}. "  # noqa
        "You will be shown a student's question-answer records from earlier in the exam, together with the correct answers. "  # noqa
        "Analyze the responses and identify the possible misconceptions that led to the errors, if any. "  # noqa
        "Next, you will be shown a new multiple choice question. "
        "Inspect the new question and think how the student would answer it, keeping in mind the misconceptions identified earlier."  # noqa
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
def build_teacher_LB_A(few_shot_prompt, native_str_output: bool) -> list:
    # NOTE: do not add a statement about JSON output! -> this is added automatically
    system_prompt_str = (
        "You are a teacher curating {exam_type}, and need to hypothesise how specific students would answer to a new question. "  # noqa
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
def build_teacher_LB_B(few_shot_prompt, native_str_output: bool) -> list:
    # NOTE: do not add a statement about JSON output! -> this is added automatically
    system_prompt_str = (
        "You are a teacher curating {exam_type}, and I want you to provide feedback about a student's responses, as well as discuss how they would likely answer to new questions. "  # noqa
        "First, you will be shown the student's responses to previous questions; "
        "you need to discuss the possible misconceptions that caused the errors, if any. "  # noqa
        "Then, you will be shown a new multiple choice question, and have to discuss how that same student would answer it. "  # noqa
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


########################
### ROLEPLAY PROMPTS ###
########################


@PROMPT_REGISTRY.register("roleplay_student_B")
def build_roleplay_tudent_B(few_shot_prompt, native_str_output: bool) -> list:
    # NOTE: do not add a statement about JSON output! -> this is added automatically

    system_prompt_str = (
        "You are a student working on {exam_type}, containing multiple choice questions, "  # noqa
        "and have the student level {student_level_group} {student_scale}. "
        "You will be shown your question-answer records from earlier in the exam, together with the correct answers. "  # noqa
        "Analyze your responses and identify the possible misconceptions that led to your errors, if any. "  # noqa
        "Next, you will be shown a new multiple choice question. "
        "Inspect the new question and think how you would answer it as a student, keeping in mind your student level and the misconceptions identified earlier. "  # noqa
        "You can answer incorrectly, if that is what you are likely to do for this question."  # noqa
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


@PROMPT_REGISTRY.register(
    "roleplay_teacher_A"
)  # TODO: this aligns with teacher_C, so rename to "roleplay_teacher_C"?
def build_roleplay_teacher_A(few_shot_prompt, native_str_output: bool) -> list:
    # NOTE: do not add a statement about JSON output! -> this is added automatically
    system_prompt_str = (
        "You are an expert teacher preparing a set of multiple choice questions for {exam_type}. "  # noqa
        "You will be shown question-answer records from a student of level {student_level_group} {student_scale} together with the correct answers. "  # noqa
        "Analyze the responses and identify the possible misconceptions that led to the errors, if any. "  # noqa
        "Next, you will be shown a new multiple choice question. "
        "Inspect the new question and think how the student would answer it, keeping in mind the misconceptions identified earlier."  # noqa
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


@PROMPT_REGISTRY.register("roleplay_teacher_LB_A")
def build_roleplay_teacher_LB_A(few_shot_prompt, native_str_output: bool) -> list:
    # NOTE: do not add a statement about JSON output! -> this is added automatically
    system_prompt_str = (
        "You are a teacher curating {exam_type}, and need to hypothesise how specific students would answer to a new question. "  # noqa
        "You will be shown responses to previous questions from a student of level {student_level_group} {student_scale}; "  # noqa
        "if one or more of the responses are wrong, list the misconceptions that possibly led to the errors. "  # noqa
        "You will be then shown a new multiple choice question. "
        "Discuss how the student of this particular level would answer it, and how the misconceptions you identified might cause them to answer wrongly. "  # noqa
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


@PROMPT_REGISTRY.register("roleplay_teacher_LB_B")
def build_roleplay_teacher_LB_B(few_shot_prompt, native_str_output: bool) -> list:
    # NOTE: do not add a statement about JSON output! -> this is added automatically
    system_prompt_str = (
        "You are a teacher curating {exam_type}, and I want you to provide feedback about a student's responses, as well as discuss how they would likely answer to new questions. "  # noqa
        "First, you will be shown responses to previous questions from a student of level {student_level_group} {student_scale}; "  # noqa
        "you need to discuss the possible misconceptions that caused the errors, if any. "  # noqa
        "Then, you will be shown a new multiple choice question, and have to discuss how the student of this particular level would answer it. "  # noqa
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


@PROMPT_REGISTRY.register("roleplay_luca_emnlp")
def build_roleplay_luca_emnlp(few_shot_prompt, native_str_output: bool) -> list:
    # NOTE: do not add a statement about JSON output! -> this is added automatically

    system_prompt_str = (
        "You will be shown a multiple choice question from {exam_type}, and the questions in the exam have difficulty levels "  # noqa
        "on a scale from one (very easy) to five (very difficult). "
        "You must assign a difficulty level to the given multiple choice question, and select the answer choice that a student of level {student_level_group} would pick. "  # noqa
    )
    human_prompt_str = "Question:\n{input}"

    system_prompt_str = prepare_str_output(system_prompt_str, native_str_output)
    # NOTE: do not add few_shot_prompt because it is zero-shot!
    messages = [
        ("system", system_prompt_str),
        ("human", human_prompt_str),
    ]
    return messages
