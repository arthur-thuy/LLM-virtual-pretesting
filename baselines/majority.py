from typing import List, Tuple
import pandas as pd
import numpy as np


def majority_prediction_answer_correctness(
    df: pd.DataFrame, 
    train_df: pd.DataFrame,
    history_len: int = 1,
    correct_prediction_threshold: float = 0.5
) -> Tuple[List[bool], List[bool]]:
    
    list_predicted_correctness = []
    list_student_option_correct = []

    for student_id, student_option_correct, time in df[['student_id', 'student_option_correct', 'time']].values:
        # Item selector
        examples_df = train_df[(train_df['student_id']==student_id)&(train_df['time']<time)]
        if len(examples_df) < history_len:
            print(
                "[Warning] Training dataset for student %s is smaller (%d) than history len (%d)." 
                % (student_id, len(examples_df), history_len)
            )
            continue
        examples_df = examples_df.sample(n=history_len)

        # Predict answer correctness
        predicted_correctness = examples_df['student_option_correct'].mean() >= correct_prediction_threshold
        list_predicted_correctness.append(predicted_correctness)
        list_student_option_correct.append(student_option_correct)

    if len(list_predicted_correctness) != len(list_student_option_correct):
        raise ValueError("The two lists predictions and correctness do not have the same length.")

    return list_predicted_correctness, list_student_option_correct


if __name__ == '__main__':

    train_df = pd.read_csv('data/gold/dbe_kt22_interactions_train.csv')

    split_to_data_path = {
        'val_small': 'data/gold/dbe_kt22_interactions_valsmall.csv',
        'val_large': 'data/gold/dbe_kt22_interactions_vallarge.csv',
        'test': 'data/gold/dbe_kt22_interactions_test.csv',
    }

    for data_split in split_to_data_path.keys():
        print("Data split: %s" % data_split)
        for history_length in [1, 3, 5]:
            print("HISTORY LENGTH = %d" % history_length)
            df = pd.read_csv(split_to_data_path[data_split])
            predictions, correctness = majority_prediction_answer_correctness(df, train_df, history_len=history_length)
            print("ACC = %.4f" % (np.mean([predictions[i] == correctness[i] for i in range(len(predictions))])))

# Data split: val_small
# HISTORY LENGTH = 1 --> ACC = 0.6907
# HISTORY LENGTH = 3 --> ACC = 0.7500
# HISTORY LENGTH = 5 --> ACC = 0.8000

# Data split: val_large
# HISTORY LENGTH = 1 --> ACC = 0.7045
# HISTORY LENGTH = 3 --> ACC = 0.7528
# HISTORY LENGTH = 5 --> ACC = 0.7244

# Data split: test
# HISTORY LENGTH = 1 --> ACC = 0.7027
# HISTORY LENGTH = 3 --> ACC = 0.7369
# HISTORY LENGTH = 5 --> ACC = 0.7497
