"""Module for example selector utils."""

# standard library imports
import re

# related third party imports
import structlog

# local application/library specific imports
from tools.utils import read_pickle
from tools.constants import CFE_ERROR_CODES

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


def get_skills_miscons_from_interactions_cfe(
    interaction: dict,
) -> tuple[list[str], list[str]]:
    """Get skills and misconceptions from interactions."""
    miscon_mapping = read_pickle("../data/platinum/miscon_mapping_cfe.pickle")

    skills = miscon_mapping[interaction["interact_id"]]["skills"]
    misconceptions = miscon_mapping[interaction["interact_id"]]["misconceptions"]

    return skills, misconceptions


def format_errors(errors: dict[str, list[dict[str, str]]]) -> str:
    """Format errors into a string."""
    text = "Errors:\n"
    if len(errors) == 0:
        text += "None\n"
    else:
        for error_key, error_list in errors.items():
            # Format the list of mistakes and corrections
            error_pairs = ", ".join(
                [
                    f'"{error["mistake"]}"->"{error["correction"]}"'
                    for error in error_list
                ]
            )
            text += f"- {error_key}: {error_pairs}\n"
    return text


def get_errors_from_interactions(
    interactions: list[dict],
) -> dict[str, int]:
    """Get misconceptions from interactions."""
    text = ""
    for interaction in interactions:
        text += interaction["answer_response"] + "\n"

    errors = extract_errors(text)
    # replace error codes with full meaning
    errors = {CFE_ERROR_CODES.get(key, key): value for key, value in errors.items()}

    return errors


def extract_errors(text: str) -> dict:
    """Extract error annotations from text and return as nested dictionary.

    Parameters
    ----------
    text : str
        Text containing error annotations in format <ERROR_TYPE "mistake"->"correction">

    Returns
    -------
    dict
        Nested dict with error types as keys and mistake/correction pairs as values
    """
    # Pattern to match error annotations: <ERROR_TYPE "mistake"->"correction">
    pattern = r'<(\w+)\s+"([^"]*)"->"([^"]*)">'

    # Find all matches
    matches = re.findall(pattern, text)

    # Create nested dictionary
    error_dict = {}

    for error_type, mistake, correction in matches:
        if error_type not in error_dict:
            error_dict[error_type] = []

        error_dict[error_type].append({"mistake": mistake, "correction": correction})

    return error_dict


def get_error_legend_from_interactions(
    interaction: dict,
) -> dict[str, int]:
    """Get error legend from interactions."""
    errors = extract_errors(interaction["answer_response"])
    error_legend = {key: CFE_ERROR_CODES[key] for key in list(errors.keys())}

    return error_legend


def format_error_legend(error_legend: dict[str, str]) -> str:
    """Format error legend into a string."""
    text = "Full name of relevant error codes:\n"
    if len(error_legend) == 0:
        text += "None\n"
    else:
        for error_key, error_value in error_legend.items():
            text += f"- {error_key}: {error_value}\n"
    return text


def format_snippets_dbe(interactions: list[dict]) -> str:
    """Format snippets from interactions for DBE.

    Parameters
    ----------
    interactions : list[dict]
        List of interactions.

    Returns
    -------
    str
        Formatted text snippets.
    """
    text_snippet = f"\n{'#'*10}\n".join(
        [
            interaction["input"] + "\n\n" + interaction["output"]
            for interaction in interactions
        ]
    )
    return text_snippet
