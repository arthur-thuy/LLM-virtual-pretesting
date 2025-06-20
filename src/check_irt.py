import matplotlib.pyplot as plt

from data_loader.data_loader import DataLoaderRoleplay
from tools.irt_estimator import irt_estimation
from tools.constants import (
    STUDENT_LEVEL,
    STUDENT_ID,
    SILVER_DIR,
    GOLD_DIR,
)


# load data
data_loader = DataLoaderRoleplay(
    read_dir=SILVER_DIR,
    write_dir=GOLD_DIR,
    dataset_name="dbe_kt22",
)
questions, interact_train = data_loader.read_splitted_data()

student_dict, difficulty_dict, _ = irt_estimation(interactions_df=interact_train)
df_interactions_tmp = interact_train.copy()
df_interactions_tmp[STUDENT_LEVEL] = df_interactions_tmp[STUDENT_ID].map(student_dict)

student_info = (
    df_interactions_tmp.groupby(STUDENT_ID)
    .agg({"student_option_correct": "mean", "question_id": "count"})
    .merge(
        df_interactions_tmp[[STUDENT_ID, STUDENT_LEVEL]].drop_duplicates(
            subset=STUDENT_ID
        ),
        on=STUDENT_ID,
    )
    .rename(
        columns={
            "student_option_correct": "prop_correct",
            "question_id": "num_questions_answered",
        }
    )
)
print(student_info.sort_values(by="prop_correct", ascending=False))


print(student_info.sort_values(by="prop_correct", ascending=True))

# print student levels
_, ax = plt.subplots(1, 1, figsize=(10, 8))
ax.hist(student_info[STUDENT_LEVEL])
plt.show()

# plot question difficulties
_, ax = plt.subplots(1, 1, figsize=(10, 8))
ax.hist(difficulty_dict.values())
plt.show()
