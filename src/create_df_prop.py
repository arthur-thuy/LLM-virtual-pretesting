"""Create a DF with proportions of student options for each question and level group."""

import pandas as pd

# read interactions and validation questions
df_interactions = pd.read_csv(
    "../data/silver/dbe_kt22_interactions.csv"
)  # TODO: do not hardcode
df_q_val = pd.read_csv(
    "../data/gold/dbe_kt22_questions_validation.csv"
)  # TODO: do not hardcode

# filter interactions for validation questions
df_i_val = df_interactions[df_interactions["question_id"].isin(df_q_val["question_id"])]

# get proportions as a DataFrame with options as columns
prop_df = (
    df_i_val.groupby(["question_id", "student_level_group"])["student_option_id"]
    .value_counts(normalize=True)
    .unstack(fill_value=0.0)
    .reset_index()
)
# convert 4 columns to dict
prop_df["dict"] = (
    prop_df.set_index(["question_id", "student_level_group"]).to_dict("index").values()
)
prop_df = prop_df.drop(columns=[1, 2, 3, 4])


# save to CSV
prop_df.to_csv(
    "../data/platinum/dbe_kt22_proportions_val.csv", index=False
)  # TODO: do not hardcode
