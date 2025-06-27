import logging
import structlog

from typing import List, Tuple
import pandas as pd
import numpy as np


# set up logger
logger = structlog.get_logger(__name__)
# Override pyirt's logging configuration (which is set to DEBUG)
logging.basicConfig(
    level=logging.INFO,  # Set to INFO or WARNING
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def majority_prediction_answer_correctness(
    df: pd.DataFrame, 
    train_df: pd.DataFrame,
    history_len: int = 1,
    keep_only_students_with_support: bool = True,
    correct_prediction_threshold: float = 0.5
) -> Tuple[List[bool], List[bool]]:
    
    list_predicted_correctness = []
    list_student_option_correct = []

    for student_id, student_option_correct, time in df[['student_id', 'student_option_correct', 'time']].values:
        # Item selector
        examples_df = train_df[(train_df['student_id']==student_id)&(train_df['time']<time)]

        if len(examples_df) >= history_len:
            # there are enough training examples from the student
            examples_df = examples_df.sample(n=history_len)
        else:
            # there are not enough trainign examples
            logger.warning(
                "examples_df for student %s is smaller (%d) than history_len (%d)."
                % (student_id, len(examples_df), history_len))
            if keep_only_students_with_support:
                logger.warning("Skipping student %s." % student_id)
                continue
            if 0 < len(examples_df) < history_len:
                # there are some previous responses from the student
                logger.warning("Using the %d available responses for student %s." % (len(examples_df), student_id))
            else:
                # There are no previous responses from that student.
                logger.warning("Doing majority prediction for student %s (no previous responses available)" % student_id)
                examples_df = train_df[train_df['time']<time].sample(n=history_len)
        
        # Predict answer correctness
        predicted_correctness = examples_df['student_option_correct'].mean() >= correct_prediction_threshold
        list_predicted_correctness.append(predicted_correctness)
        list_student_option_correct.append(student_option_correct)

    if len(list_predicted_correctness) != len(list_student_option_correct):
        raise ValueError("The two lists predictions and correctness do not have the same length.")

    return list_predicted_correctness, list_student_option_correct


if __name__ == '__main__':

    n_random_runs = 10

    train_df = pd.read_csv('data/gold/dbe_kt22_interactions_train.csv')

    split_to_data_path = {
        'val_small': 'data/gold/dbe_kt22_interactions_valsmall.csv',
        'val_large': 'data/gold/dbe_kt22_interactions_vallarge.csv',
        'test': 'data/gold/dbe_kt22_interactions_test.csv',
    }

    # dict where I collect all the results.
    # first key is the data split, then the history length.
    results = {}
    for data_split in split_to_data_path.keys():
        results[data_split] = {}
        for history_length in [1, 3, 5]:
            results[data_split][history_length] = []

    for data_split in split_to_data_path.keys():
        print("Data split: %s" % data_split)
        for history_length in [1, 3, 5]:
            print("HISTORY LENGTH = %d" % history_length)
            df = pd.read_csv(split_to_data_path[data_split])
            for _ in range(n_random_runs):
                predictions, correctness = majority_prediction_answer_correctness(df, train_df, history_length)
                # predictions, correctness = majority_prediction_answer_correctness(df, train_df, history_length, keep_only_students_with_support=False)
                accuracy = np.mean([predictions[i] == correctness[i] for i in range(len(predictions))])
                print("--> ACC = %.4f" % accuracy)
                results[data_split][history_length].append(accuracy)

    print(results)
    for data_split in split_to_data_path.keys():
        for history_length in [1, 3, 5]:
            print(
                "%9s | %2d | mean acc: %.4f | std.dev. acc: %.4f" 
                % (data_split, history_length, np.mean(results[data_split][history_length]), 
                   np.std(results[data_split][history_length]))
            )

### Averaging across 10 runs
# Keeping only students with full support:
# val_small |  1 | mean acc: 0.7062 | std.dev. acc: 0.0310
# val_small |  3 | mean acc: 0.7750 | std.dev. acc: 0.0193
# val_small |  5 | mean acc: 0.7632 | std.dev. acc: 0.0207
# val_large |  1 | mean acc: 0.6904 | std.dev. acc: 0.0156
# val_large |  3 | mean acc: 0.7243 | std.dev. acc: 0.0130
# val_large |  5 | mean acc: 0.7293 | std.dev. acc: 0.0096
#      test |  1 | mean acc: 0.6942 | std.dev. acc: 0.0109
#      test |  3 | mean acc: 0.7416 | std.dev. acc: 0.0076
#      test |  5 | mean acc: 0.7486 | std.dev. acc: 0.0065

# Keeping all students:
# val_small |  1 | mean acc: 0.7170 | std.dev. acc: 0.0475
# val_small |  3 | mean acc: 0.7720 | std.dev. acc: 0.0227
# val_small |  5 | mean acc: 0.7700 | std.dev. acc: 0.0195
# val_large |  1 | mean acc: 0.6976 | std.dev. acc: 0.0108
# val_large |  3 | mean acc: 0.7238 | std.dev. acc: 0.0113
# val_large |  5 | mean acc: 0.7418 | std.dev. acc: 0.0096
#      test |  1 | mean acc: 0.6931 | std.dev. acc: 0.0150
#      test |  3 | mean acc: 0.7341 | std.dev. acc: 0.0057
#      test |  5 | mean acc: 0.7502 | std.dev. acc: 0.0064


### Below results for single runs
# RESULTS REMOVING STUDENTS FOR WHICH THERE AREN'T ENOUGH PREVIOUS RESPONSES IN TRAIN_DF 
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

# KEEPING ALL STUDENTS - use (less) responses from the student if available, or random prediction if no previous responses
# Data split: val_small
# HISTORY LENGTH = 1 --> ACC = 0.6600
# HISTORY LENGTH = 3 --> ACC = 0.7700
# HISTORY LENGTH = 5 --> ACC = 0.7900

# Data split: val_large
# HISTORY LENGTH = 1 --> ACC = 0.6960
# HISTORY LENGTH = 3 --> ACC = 0.7340
# HISTORY LENGTH = 5 --> ACC = 0.7240

# Data split: test
# HISTORY LENGTH = 1 --> ACC = 0.7160
# HISTORY LENGTH = 3 --> ACC = 0.7400
# HISTORY LENGTH = 5 --> ACC = 0.7570
