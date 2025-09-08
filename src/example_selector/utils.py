"""Module for example selector utils."""

# standard library imports
# /

# related third party imports
import structlog

# local application/library specific imports
from tools.utils import read_pickle

# set up logger
logger = structlog.get_logger(__name__)


def assert_correct_option(interactions: list[dict]) -> None:
    """Assert that for each interaction, the correct option is 1."""
    for interaction in interactions:
        if interaction["correct_option_id"] != 1:
            logger.warning(
                f"Correct option for question {interaction['question_id']} "
                f"is not 1, but {interaction['correct_option_id']}"
            )


def get_skills_miscons_from_interactions(
    interactions: list[dict],
) -> tuple[list[str], list[str]]:
    """Get skills and misconceptions from interactions."""
    miscon_mapping = read_pickle("../data/platinum/miscon_mapping.pickle")

    skills = []
    misconceptions = []
    # iterate over selected interactions and get skills and misconceptions
    for interaction in interactions:
        question_id = interaction["question_id"]
        student_option_id = interaction["student_option_id"]

        if student_option_id == 1:
            skill = miscon_mapping[question_id][student_option_id]
            if skill is not None:
                skills.append(skill)
        else:
            misconception = miscon_mapping[question_id][student_option_id]
            if misconception is not None:
                misconceptions.append(misconception)

    return skills, misconceptions


def format_skills_miscons(skills: list[str], misconceptions: list[str]) -> str:
    """Format skills and misconceptions into a string."""
    text = "Mastered knowledge concepts:\n"
    if len(skills) == 0:
        text += "None\n"
    else:
        for skill in skills:
            text += f"- {skill}\n"
    text += "\n\nMisconceptions:\n"
    if len(misconceptions) == 0:
        text += "None\n"
    else:
        for misconception in misconceptions:
            text += f"- {misconception}\n"
    return text
