"""IRT estimation."""

# standard library imports
import logging

# related third party imports
import pandas as pd
import structlog
import pyirt
import numpy as np
from typing import Tuple, Dict

# local application/library specific imports
from tools.constants import (
    DIFFICULTY_MIN,
    DIFFICULTY_MAX,
    DEFAULT_DISCRIMINATION,
    DEFAULT_GUESS,
    QUESTION_ID,
    STUDENT_ID,
    S_OPTION_CORRECT,
    STUDENT_LEVEL,
    STUDENT_LEVEL_GROUP,
)

# set up logger
logger = structlog.get_logger(__name__)
# Override pyirt's logging configuration (which is set to DEBUG)
logging.basicConfig(
    level=logging.INFO,  # Set to INFO or WARNING
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def irt_estimation(
    interactions_df: pd.DataFrame,
    difficulty_range: tuple[float, float] = (DIFFICULTY_MIN, DIFFICULTY_MAX),
    discrimination_range: tuple[float, float] = (
        DEFAULT_DISCRIMINATION,
        DEFAULT_DISCRIMINATION,
    ),
    guess: float = DEFAULT_GUESS,
    mode: str = "debug",
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Estimate IRT parameters.

    Estimates student ability, question difficulty, and (optionally) discrimination
    parameters using Item Response Theory (IRT). If necessary, it adds synthetic
    responses before performing the IRT estimation using the `pyirt` library. This
    happens if certain questions have only correct or incorrect responses (it is a known
    issue of pyirt).

    Parameters
    ----------
    interactions_df : pd.DataFrame
        A DataFrame containing student-question interactions with the following columns:
            - STUDENT_ID: Unique identifier for students.
            - QUESTION_ID: Unique identifier for questions.
            - S_OPTION_CORRECT: Boolean indicating if the student's response was correct.
    difficulty_range : tuple[float, float], optional
        Bounds for the difficulty parameter, by default (DIFFICULTY_MIN, DIFFICULTY_MAX)
    discrimination_range : tuple[float, float], optional
        Bounds for the discrimination parameter, by default (DEFAULT_DISCRIMINATION, DEFAULT_DISCRIMINATION)
    guess : float, optional
        Default guessing parameter for all questions, by default DEFAULT_GUESS

    Returns
    -------
    Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]
        - student_dict: Dictionary mapping student IDs to their estimated ability parameters.
        - difficulty_dict: Dictionary mapping question IDs to their estimated difficulty parameters.
        - discrimination_dict: Dictionary mapping question IDs to their estimated discrimination parameters.

    Raises
    ------
    ValueError
        If there are items with only correct or wrong answers.
    """
    interactions_list = [
        (user, item, correctness)
        for user, item, correctness in interactions_df[
            [STUDENT_ID, QUESTION_ID, S_OPTION_CORRECT]
        ].values
    ]

    # If there are some items with only correct or only wrong answers, pyirt crashes.
    # This manages that scenario by augmenting the dataset with correct and wrong responses of an ideally good and ideally bad student.
    question_count_per_correctness = (
        interactions_df.groupby([QUESTION_ID, S_OPTION_CORRECT]).size().reset_index()
    )
    question_count_per_correctness = (
        question_count_per_correctness.groupby(QUESTION_ID)
        .size()
        .reset_index()
        .rename(columns={0: "cnt"})
    )
    list_q_to_augment = list(
        question_count_per_correctness[question_count_per_correctness["cnt"] == 1][
            QUESTION_ID
        ]
    )
    logger.info(
        "Augmenting questions for IRT estimation", num_questions=len(list_q_to_augment)
    )
    interactions_list.extend([("p_good", itemID, True) for itemID in list_q_to_augment])
    interactions_list.extend([("p_bad", itemID, False) for itemID in list_q_to_augment])

    try:
        item_params, user_params = pyirt.irt(
            interactions_list,
            theta_bnds=difficulty_range,
            beta_bnds=difficulty_range,
            alpha_bnds=discrimination_range,
            in_guess_param={q: guess for q in interactions_df[QUESTION_ID].unique()},
            max_iter=100,
            mode=mode,
        )
    except Exception:
        raise ValueError(
            "Problem in irt_estimation. "
            "Check if there are items with only correct/wrong answers."
        )

    difficulty_dict = dict()
    discrimination_dict = dict()
    for question_id, question_params in item_params.items():
        difficulty_dict[question_id] = -question_params[
            "beta"
        ]  # "-" is to align with conventional IRT interpretations
        discrimination_dict[question_id] = question_params["alpha"]
    student_dict = {x[0]: x[1] for x in user_params.items()}
    return student_dict, difficulty_dict, discrimination_dict


def group_student_levels(
    df_interactions: pd.DataFrame, num_groups: int, student_scale_map: Dict[str, str]
) -> pd.DataFrame:
    """Group students into levels based on their estimated ability.

    Parameters
    ----------
    df_interactions : pd.DataFrame
        DataFrame containing student interactions with columns:
            - STUDENT_ID: Unique identifier for students.
    num_groups : int
        Number of groups to divide students into based on their ability.

    Returns
    -------
    pd.DataFrame
        DataFrame with an additional column 'student_level_group' indicating the group of each student.
    """
    assert set([STUDENT_ID, QUESTION_ID, S_OPTION_CORRECT]).issubset(
        df_interactions.columns
    ), "Missing required columns in interactions DataFrame."
    # Compute IRT parameters
    student_dict, _, _ = irt_estimation(interactions_df=df_interactions)
    df_interactions_tmp = df_interactions.copy()
    df_interactions_tmp[STUDENT_LEVEL] = df_interactions_tmp[STUDENT_ID].map(
        student_dict
    )

    # discretize student levels into groups
    ###################################
    ## equally spaced bins
    diff_range = (DIFFICULTY_MIN, DIFFICULTY_MAX)
    bin_edges = np.histogram_bin_edges(None, bins=num_groups, range=diff_range).astype(
        np.float32
    )
    # NOTE: increase last bin edge by epsilon to include the last value
    bin_edges = np.nextafter(bin_edges, bin_edges + (bin_edges == bin_edges[-1]))

    # find all student levels (digits) and determine for each interaction row
    student_levels_base = list(range(1, num_groups + 1))
    assert (
        len(student_levels_base) == len(bin_edges) - 1
    ), "Number of student levels should be one less than the number of bin edges."
    df_interactions_tmp[STUDENT_LEVEL_GROUP] = np.digitize(
        df_interactions_tmp[STUDENT_LEVEL], bin_edges
    ).astype(str)
    assert (
        len(df_interactions_tmp[STUDENT_LEVEL_GROUP].unique()) <= num_groups
    ), "Number of unique student level groups should not exceed the number of groups."

    # NOTE: map student levels (digits) onto the real values (which can be strings)
    df_interactions_tmp[STUDENT_LEVEL_GROUP] = df_interactions_tmp[
        STUDENT_LEVEL_GROUP
    ].map(student_scale_map)

    # value counts of primary KCs
    level_value_counts = (
        df_interactions_tmp[STUDENT_LEVEL_GROUP].value_counts().reset_index()
    )
    level_value_counts.columns = [STUDENT_LEVEL_GROUP, "count"]
    print(level_value_counts)
    ###################################
    ## equal number of students per group
    # df_interactions_tmp[STUDENT_LEVEL_GROUP] = pd.qcut(
    #     df_interactions_tmp[STUDENT_LEVEL],
    #     q=num_groups,
    #     labels=[str(i) for i in range(1, num_groups + 1)],
    # ).astype(str)
    ###################################
    return df_interactions_tmp


def explode_student_levels(df_questions: pd.DataFrame, num_groups: int) -> pd.DataFrame:
    """Explode student levels into multiple rows for each question.

    Parameters
    ----------
    df_questions : pd.DataFrame
        Questions to be exploded for each student level.
    num_groups : int
        Number of student level groups to create.

    Returns
    -------
    pd.DataFrame
        DataFrame with an additional column 'student_level_group'.
    """
    student_level_groups_list = [str(i) for i in range(1, num_groups + 1)]
    df_tmp = df_questions.copy()
    df_tmp[STUDENT_LEVEL_GROUP] = [student_level_groups_list] * len(df_tmp)
    df_tmp = df_tmp.explode(STUDENT_LEVEL_GROUP)
    return df_tmp.reset_index(drop=True)


def write_student_scale(num_groups: int) -> str:
    """Write the student scale based on the number of groups.

    Parameters
    ----------
    num_groups : int
        Number of student level groups.

    Returns
    -------
    str
        Student scale string.
    """
    scale = f"(1 is the lowest level; {num_groups} is the highest level)"
    return scale
