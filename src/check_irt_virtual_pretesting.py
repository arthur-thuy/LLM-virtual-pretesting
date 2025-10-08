"""Script to check IRT parameter estimation from virtual pretesting results."""

# standard library imports
import matplotlib.pyplot as plt

# third party imports
# /

# local imports
from tools.constants import (
    DIFFICULTY_MIN,
    DIFFICULTY_MAX,
    STUDENT_LEVEL_MIN,
    STUDENT_LEVEL_MAX,
    DISCRIMINATION_MIN,
    DISCRIMINATION_MAX,
    GUESS_FACTOR,
)
from tools.irt_estimator import irt_estimation
from tools.utils import read_pickle

# read virtual pretesting output
logs = read_pickle(
    "./output/roleplay_cupacfe_errors_20250912-100608/qwen3:8b~T_0.0~SO_student_bool_nocontext~L_5~SP_student_cfe_errors_level_context~SS_proficiency_5_str~EFQ_mcq_reading_quotes~EFI_open_reading~ES_errors_studentlevel_random1/run_1.pickle"
)
df_input = logs["preds_qdiff"]["val_df_input"]

# compute IRT parameters
_, difficulty_dict, _ = irt_estimation(
    interactions_df=df_input,
    difficulty_range=(-11, 11),
    student_range=(-3, 3),
    discrimination_range=(0, 0),
    guess=0.0,
)

# NOTE: CUP&A requires difficulty_range=(30,110)

# using the default parameters from constants.py
# _, difficulty_dict, _ = irt_estimation(
#     interactions_df=df_input,
#     difficulty_range=(DIFFICULTY_MIN,DIFFICULTY_MAX),
#     student_range=(STUDENT_LEVEL_MIN, STUDENT_LEVEL_MAX),
#     discrimination_range=(DISCRIMINATION_MIN, DISCRIMINATION_MAX),
#     guess=GUESS_FACTOR,
# )

# plot question difficulties
plt.hist(list(difficulty_dict.values()), bins=30, alpha=0.7)
plt.title("Histogram of Difficulty Estimates")
plt.xlabel("Difficulty")
plt.ylabel("Frequency")
plt.grid(axis="y", alpha=0.75)
plt.show()
