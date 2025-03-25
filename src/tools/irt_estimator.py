import pandas as pd
from pyirt import irt
from typing import Tuple, Dict

from src.tools.constants import (
    DIFFICULTY_MIN,
    DIFFICULTY_MAX,
    DEFAULT_DISCRIMINATION,
    DEFAULT_GUESS,
    QUESTION_ID,
    STUDENT_ID,
    S_OPTION_CORRECT,
)


def irt_estimation(
        interactions_df: pd.DataFrame,
        difficulty_range=(DIFFICULTY_MIN, DIFFICULTY_MAX),
        discrimination_range=(DEFAULT_DISCRIMINATION, DEFAULT_DISCRIMINATION),
        guess=DEFAULT_GUESS
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Estimates student ability, question difficulty, and (optionally) discrimination parameters using Item Response Theory (IRT).
    If necessary, it adds synthetic responses before performing the IRT estimation using the `pyirt` library. This happens if certain
      questions have only correct or incorrect responses (it is a known issue of pyirt).

    Args:
        interactions_df (pd.DataFrame): A DataFrame containing student-question interactions with the following columns:
            - STUDENT_ID: Unique identifier for students.
            - QUESTION_ID: Unique identifier for questions.
            - S_OPTION_CORRECT: Boolean indicating if the student's response was correct.
        difficulty_range (Tuple[float, float], optional): Bounds for the difficulty parameter.
        discrimination_range (Tuple[float, float], optional): Bounds for the discrimination parameter.
        guess (float, optional): Default guessing parameter for all questions.

    Returns:
        Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
            - student_dict: Dictionary mapping student IDs to their estimated ability parameters.
            - difficulty_dict: Dictionary mapping question IDs to their estimated difficulty parameters.
            - discrimination_dict: Dictionary mapping question IDs to their estimated discrimination parameters.
    """
    interactions_list = [
        (user, item, correctness)
        for user, item, correctness in interactions_df[[STUDENT_ID, QUESTION_ID, S_OPTION_CORRECT]].values
    ]
    
    # If there are some items with only correct or only wrong answers, pyirt crashes. 
    # This manages that scenario by augmenting the dataset with correct and wrong responses of an ideally good and ideally bad student.
    question_count_per_correctness = interactions_df.groupby([QUESTION_ID, S_OPTION_CORRECT]).size().reset_index()
    question_count_per_correctness = question_count_per_correctness.groupby(QUESTION_ID).size().reset_index().rename(columns={0: 'cnt'})
    list_q_to_augment = list(question_count_per_correctness[question_count_per_correctness['cnt'] == 1][QUESTION_ID])
    print('[INFO] %d questions filled in' % len(list_q_to_augment))
    interactions_list.extend([('p_good', itemID, True) for itemID in list_q_to_augment])
    interactions_list.extend([('p_bad', itemID, False) for itemID in list_q_to_augment])

    try:
        item_params, user_params = irt(
            interactions_list,
            theta_bnds=difficulty_range,
            beta_bnds=difficulty_range,
            alpha_bnds=discrimination_range,
            in_guess_param={q: guess for q in interactions_df[QUESTION_ID].unique()},
            max_iter=100
        )
    except Exception:
        raise ValueError("Problem in irt_estimation. Check if there are items with only correct/wrong answers.")
    
    difficulty_dict = dict()
    discrimination_dict = dict()
    for question_id, question_params in item_params.items():
        difficulty_dict[question_id] = -question_params['beta']  # "-" is to align with conventional IRT interpretations
        discrimination_dict[question_id] = question_params["alpha"]
    student_dict = {x[0]: x[1] for x in user_params.items()}
    return student_dict, difficulty_dict, discrimination_dict
