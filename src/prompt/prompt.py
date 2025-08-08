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


@PROMPT_REGISTRY.register(
    "replicate_student_tomato"
)  # NOTE: previously "replicate_student_B"
def build_replicate_student_tomato(few_shot_prompt, native_str_output: bool) -> list:
    # NOTE: do not add a statement about JSON output! -> this is added automatically

    system_prompt_str = (
        "You are a student working on a multiple choice exam on {exam_type}. "  # noqa
        "You will be shown your question-answer records from earlier in the exam, together with the correct answers. "  # noqa
        "Analyze your responses and identify the possible misconceptions that led to your errors, if any. "  # noqa
        "Next, you will be shown a new multiple choice question. "
        "Inspect the new question and think how you would answer it, keeping in mind the misconceptions identified earlier. "  # noqa
        "You can answer incorrectly, if that is what you are likely to do for this question."  # noqa
    )
    # NOTE: the JSON structured output provides instructions on how to answer exactly
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


@PROMPT_REGISTRY.register("replicate_student_tomato_studentlevel")
def build_replicate_student_tomato_studentlevel(
    few_shot_prompt, native_str_output: bool
) -> list:
    # NOTE: do not add a statement about JSON output! -> this is added automatically

    system_prompt_str = (
        "You are a student of level {student_level_group} {student_scale} working on a multiple choice exam on {exam_type}. "  # noqa
        "You will be shown your question-answer records from earlier in the exam, together with the correct answers. "  # noqa
        "Analyze your responses and identify the possible misconceptions that led to your errors, if any. "  # noqa
        "Next, you will be shown a new multiple choice question. "
        "Inspect the new question and think how you would answer it as a student of level {student_level_group} {student_scale}, keeping in mind the misconceptions identified earlier. "  # noqa
        "You can answer incorrectly, if that is what you are likely to do for this question."  # noqa
    )
    # NOTE: the JSON structured output provides instructions on how to answer exactly
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


@PROMPT_REGISTRY.register("replicate_student_pepper")
def build_replicate_student_pepper(few_shot_prompt, native_str_output: bool) -> list:
    # NOTE: do not add a statement about JSON output! -> this is added automatically
    system_prompt_str = (
        "You are a student working on an exam on {exam_type}, containing multiple choice questions. "  # noqa
        "You are shown a set of questions that you answered earlier in the exam, together with the correct answers and your student answers. "  # noqa
        "Analyse your responses to the questions and identify the possible misconceptions that led to answering incorrectly. "  # noqa
        "Inspect the new question and think how you would answer it as a student. "
        "You can answer incorrectly, if that is what the student is likely to do for this question. "  # noqa
        "If you answer incorrectly, explain which misconception leads to selecting that answer. "  # noqa
        "If you answer correctly, explain why you think the answer is correct. "
        "Provide your answer as the integer index of the multiple choice option."
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
    "replicate_teacher_onion"
)  # NOTE: previously "replicate_teacher_C"
def build_replicate_teacher_onion(few_shot_prompt, native_str_output: bool) -> list:
    # NOTE: do not add a statement about JSON output! -> this is added automatically
    system_prompt_str = (
        "You are an expert teacher preparing a set of questions for a multiple choice exam on {exam_type}. "  # noqa
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


@PROMPT_REGISTRY.register("replicate_teacher_onion_studentlevel")
def build_replicate_teacher_onion_studentlevel(
    few_shot_prompt, native_str_output: bool
) -> list:
    # NOTE: do not add a statement about JSON output! -> this is added automatically
    system_prompt_str = (
        "You are an expert teacher preparing a set of questions for a multiple choice exam on {exam_type}. "  # noqa
        "You will be shown question-answer records of a student of level {student_level_group} {student_scale} together with the correct answers. "  # noqa
        "Analyze the responses and identify the possible misconceptions that led to the errors, if any. "  # noqa
        "Next, you will be shown a new multiple choice question. "
        "Inspect the new question and discuss how the student of level {student_level_group} {student_scale} would answer it, keeping in mind the misconceptions identified earlier."  # noqa
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
    "replicate_teacher_carrot"
)  # NOTE: previously "replicate_teacher_LB_A"
def build_replicate_teacher_carrot(few_shot_prompt, native_str_output: bool) -> list:
    # NOTE: do not add a statement about JSON output! -> this is added automatically
    system_prompt_str = (
        "You are a teacher curating a multiple choice exam on {exam_type}, and need to hypothesise how specific students would answer to a new question. "  # noqa
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


@PROMPT_REGISTRY.register(
    "replicate_teacher_avocado"
)  # NOTE: previously "replicate_teacher_LB_B"
def build_replicate_teacher_avocado(few_shot_prompt, native_str_output: bool) -> list:
    # NOTE: do not add a statement about JSON output! -> this is added automatically
    system_prompt_str = (
        "You are a teacher curating a multiple choice exam on {exam_type}, and I want you to provide feedback about a student's responses, as well as discuss how they would likely answer to new questions. "  # noqa
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


@PROMPT_REGISTRY.register(
    "roleplay_student_tomato"
)  # NOTE: previously "roleplay_student_B"
def build_roleplay_student_tomato(few_shot_prompt, native_str_output: bool) -> list:
    # NOTE: do not add a statement about JSON output! -> this is added automatically

    system_prompt_str = (
        "You are a student of level {student_level_group} {student_scale} working on a multiple choice exam on {exam_type}."  # noqa
        "You will be shown your question-answer records from earlier in the exam, together with the correct answers. "  # noqa
        "Analyze your responses and identify the possible misconceptions that led to your errors, if any. "  # noqa
        "Next, you will be shown a new multiple choice question. "
        "Inspect the new question and think how you would answer it as a student of level {student_level_group} {student_scale}, keeping in mind your student level and the misconceptions identified earlier. "  # noqa
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


@PROMPT_REGISTRY.register("roleplay_student_pepper")
def build_roleplay_student_pepper(few_shot_prompt, native_str_output: bool) -> list:
    # NOTE: do not add a statement about JSON output! -> this is added automatically
    system_prompt_str = (
        "You are a student of level {student_level_group} {student_scale} working on an exam on {exam_type}, containing multiple choice questions. "  # noqa
        "You are shown a set of questions that you answered earlier in the exam, together with the correct answers and your student answers. "  # noqa
        "Analyse your responses to the questions and identify the possible misconceptions that led to answering incorrectly. "  # noqa
        "Inspect the new question and think how you would answer it as a student of level {student_level_group} {student_scale}. "  # noqa
        "You can answer incorrectly, if that is what the student is likely to do for this question. "  # noqa
        "If you answer incorrectly, explain which misconception leads to selecting that answer. "  # noqa
        "If you answer correctly, explain why you think the answer is correct. "
        "Provide your answer as the integer index of the multiple choice option."
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
    "roleplay_teacher_onion"
)  # NOTE: previously "roleplay_teacher_C"?
def build_roleplay_teacher_onion(few_shot_prompt, native_str_output: bool) -> list:
    # NOTE: do not add a statement about JSON output! -> this is added automatically
    system_prompt_str = (
        "You are an expert teacher preparing a set of questions for a multiple choice exam on {exam_type}. "  # noqa
        "You will be shown question-answer records of a student of level {student_level_group} {student_scale} together with the correct answers. "  # noqa
        "Analyze the responses and identify the possible misconceptions that led to the errors, if any. "  # noqa
        "Next, you will be shown a new multiple choice question. "
        "Inspect the new question and think how the student of level {student_level_group} {student_scale} would answer it, keeping in mind the misconceptions identified earlier."  # noqa
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
    "roleplay_teacher_carrot"
)  # NOTE: previously "roleplay_teacher_LB_A"
def build_roleplay_teacher_carrot(few_shot_prompt, native_str_output: bool) -> list:
    # NOTE: do not add a statement about JSON output! -> this is added automatically
    system_prompt_str = (
        "You are a teacher curating a multiple choice exam on {exam_type}, and need to hypothesise how specific students would answer to a new question. "  # noqa
        "You will be shown responses to previous questions of a student of level {student_level_group} {student_scale}; "  # noqa
        "if one or more of the responses are wrong, list the misconceptions that possibly led to the errors. "  # noqa
        "You will be then shown a new multiple choice question. "
        "Discuss how the student of level {student_level_group} {student_scale} would answer it, and how the misconceptions you identified might cause them to answer wrongly. "  # noqa
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
    "roleplay_teacher_avocado"
)  # NOTE: previously "roleplay_teacher_LB_B"
def build_roleplay_teacher_avocado(few_shot_prompt, native_str_output: bool) -> list:
    # NOTE: do not add a statement about JSON output! -> this is added automatically
    system_prompt_str = (
        "You are a teacher curating a multiple choice exam on {exam_type}, and I want you to provide feedback about a student's responses, as well as discuss how they would likely answer to new questions. "  # noqa
        "First, you will be shown responses to previous questions of a student of level {student_level_group} {student_scale}; "  # noqa
        "you need to discuss the possible misconceptions that caused the errors, if any. "  # noqa
        "Then, you will be shown a new multiple choice question, and have to discuss how the student of level {student_level_group} {student_scale} would answer it. "  # noqa
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
        "You will be shown a multiple choice question from a {exam_type_luca_emnlp}, and the questions in the exam have difficulty levels "  # noqa
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


#################################
### COLLECTING MISCONCEPTIONS ###
#################################


@PROMPT_REGISTRY.register("collect_misconceptions")
def build_collect_misconceptions(few_shot_prompt, native_str_output: bool) -> list:
    # NOTE: do not add a statement about JSON output! -> this is added automatically

    system_prompt_str = (
        "You are shown a multiple choice question of an exam on {exam_type}. "
        "You have to analyse the question as follows:\n"
        "- Skills: for the correct answer option, list the knowledge concepts that the student should know to correctly select it;\n"  # noqa
        "- Misconceptions: for each distractor, list the misconceptions that might lead the student to select it;\n\n"  # noqa
        "Your answers should be very concise; each field should have a maximum of 10 words. "  # noqa
        "The skill description should start with 'Understands' and the misconception description should start with 'Confuses'. "  # noqa
    )
    human_prompt_str = "Multiple choice question:\n{input}"

    system_prompt_str = prepare_str_output(system_prompt_str, native_str_output)
    # NOTE: do not add few_shot_prompt because it is zero-shot!
    messages = [
        ("system", system_prompt_str),
        ("human", human_prompt_str),
    ]
    return messages


#######################################
### REPLICATING WITH MISCONCEPTIONS ###
#######################################


@PROMPT_REGISTRY.register("replicate_student_chocolate")
def build_replicate_student_chocolate(
    few_shot_prompt, native_str_output: bool
) -> list:
    # NOTE: do not add a statement about JSON output! -> this is added automatically
    system_prompt_str = (
        "You are a student working on an exam on {exam_type}, containing multiple choice questions. "  # noqa
        "From your earlier answers on the exam, the teacher has identified a set of knowledge concepts that you master and a set of misconceptions that you have. "  # noqa
        "Inspect the new question and think how you would answer it as a student, keeping in mind your skills and misconceptions. "  # noqa
        "You can answer incorrectly, if that is what the student is likely to do for this question. "  # noqa
        # "If you answer incorrectly, explain which misconception leads to selecting that answer. "  # noqa
        # "If you answer correctly, explain why you think the answer is correct. "
        # "Provide your answer as the integer index of the multiple choice option."
    )
    human1_prompt_str = "New multiple choice question:\n\n{input}"

    system_prompt_str = prepare_str_output(system_prompt_str, native_str_output)
    messages = [
        ("system", system_prompt_str),
        few_shot_prompt,
        ("human", human1_prompt_str),
    ]
    return messages


@PROMPT_REGISTRY.register("replicate_student_chocolate_studentlevel")
def build_replicate_student_chocolate_studentlevel(
    few_shot_prompt, native_str_output: bool
) -> list:
    # NOTE: do not add a statement about JSON output! -> this is added automatically
    system_prompt_str = (
        "You are a student of level {student_level_group} {student_scale} working on an exam on {exam_type}, containing multiple choice questions. "  # noqa
        "From your earlier answers on the exam, the teacher has identified a set of knowledge concepts that you master and a set of misconceptions that you have. "  # noqa
        "Inspect the new question and think how you would answer it as a student of level {student_level_group}, keeping in mind your skills and misconceptions. "  # noqa
        "You can answer incorrectly, if that is what the student is likely to do for this question. "  # noqa
        # "If you answer incorrectly, explain which misconception leads to selecting that answer. "  # noqa
        # "If you answer correctly, explain why you think the answer is correct. "
        # "Provide your answer as the integer index of the multiple choice option."
    )
    human1_prompt_str = "New multiple choice question:\n\n{input}"

    system_prompt_str = prepare_str_output(system_prompt_str, native_str_output)
    messages = [
        ("system", system_prompt_str),
        few_shot_prompt,
        ("human", human1_prompt_str),
    ]
    return messages


@PROMPT_REGISTRY.register("replicate_student_chocolate_zero")
def build_replicate_student_chocolate_zero(
    few_shot_prompt, native_str_output: bool
) -> list:
    # NOTE: do not add a statement about JSON output! -> this is added automatically
    system_prompt_str = (
        "You are a student working on an exam on {exam_type}, containing multiple choice questions. "  # noqa
        "Inspect the new question and think how you would answer it as a student. "  # noqa
        "You can answer incorrectly, if that is what the student is likely to do for this question. "  # noqa
        # "If you answer incorrectly, explain which misconception leads to selecting that answer. "  # noqa
        # "If you answer correctly, explain why you think the answer is correct. "
        # "Provide your answer as the integer index of the multiple choice option."
    )
    human1_prompt_str = "New multiple choice question:\n\n{input}"

    system_prompt_str = prepare_str_output(system_prompt_str, native_str_output)
    messages = [
        ("system", system_prompt_str),
        ("human", human1_prompt_str),
    ]
    return messages


@PROMPT_REGISTRY.register("replicate_student_chocolate_zero_studentlevel")
def build_replicate_student_chocolate_zero_studentlevel(
    few_shot_prompt, native_str_output: bool
) -> list:
    # NOTE: do not add a statement about JSON output! -> this is added automatically
    system_prompt_str = (
        "You are a student of level {student_level_group} {student_scale} working on an exam on {exam_type}, containing multiple choice questions. "  # noqa
        "Inspect the new question and think how you would answer it as a student of level {student_level_group}"  # noqa
        "You can answer incorrectly, if that is what the student is likely to do for this question. "  # noqa
        # "If you answer incorrectly, explain which misconception leads to selecting that answer. "  # noqa
        # "If you answer correctly, explain why you think the answer is correct. "
        # "Provide your answer as the integer index of the multiple choice option."
    )
    human1_prompt_str = "New multiple choice question:\n\n{input}"

    system_prompt_str = prepare_str_output(system_prompt_str, native_str_output)
    messages = [
        ("system", system_prompt_str),
        ("human", human1_prompt_str),
    ]
    return messages
