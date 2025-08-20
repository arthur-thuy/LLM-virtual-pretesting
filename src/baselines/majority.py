# standard library imports
import logging
import structlog
from typing import List, Tuple, Optional

# related third party imports
import pandas as pd
import numpy as np
import scipy
from numpy.typing import NDArray
from sklearn.metrics import balanced_accuracy_score

# local application/library specific imports
# from tools.analyzer import mean_stderror
# FIXME: I can't import from the library???


# set up logger
logger = structlog.get_logger(__name__)
logging.basicConfig(
    level=logging.INFO,  # Set to INFO or WARNING
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def mean_stderror(ary: NDArray, axis: Optional[int] = None) -> Tuple[NDArray, NDArray]:
    """Calculate mean and standard error from array.

    Parameters
    ----------
    ary : NDArray
        Output array containing metrics
    axis : Union[Any, None], optional
        Axis to average over, by default None
    Returns
    -------
    Tuple[NDArray, NDArray]
        Tuple of mean and standard error
    """
    mean = np.mean(ary, axis=axis)
    stderror = scipy.stats.sem(ary, ddof=1, axis=axis)
    return mean, stderror


def majority_prediction_answer_correctness(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    history_len: int = 1,
    keep_only_students_with_support: bool = True,
    correct_prediction_threshold: float = 0.5,
) -> Tuple[List[bool], List[bool]]:

    list_predicted_correctness = []
    list_student_option_correct = []

    for student_id, student_option_correct, time in df[
        ["student_id", "student_option_correct", "time"]
    ].values:
        # Item selector
        examples_df = train_df[
            (train_df["student_id"] == student_id) & (train_df["time"] < time)
        ]

        if len(examples_df) >= history_len:
            # there are enough training examples from the student
            examples_df = examples_df.sample(n=history_len)
        else:
            # there are not enough trainign examples
            logger.info(
                "examples_df for student %s is smaller (%d) than history_len (%d)."
                % (student_id, len(examples_df), history_len)
            )
            if keep_only_students_with_support:
                logger.info("Skipping student %s." % student_id)
                continue
            if 0 < len(examples_df) < history_len:
                # there are some previous responses from the student
                logger.info(
                    "Using the %d available responses for student %s."
                    % (len(examples_df), student_id)
                )
            else:
                # There are no previous responses from that student.
                logger.info(
                    "Doing majority prediction for student %s (no previous responses available)"
                    % student_id
                )
                examples_df = train_df[train_df["time"] < time].sample(n=history_len)

        # Predict answer correctness
        predicted_correctness = (
            examples_df["student_option_correct"].mean() >= correct_prediction_threshold
        )
        list_predicted_correctness.append(predicted_correctness)
        list_student_option_correct.append(student_option_correct)

    if len(list_predicted_correctness) != len(list_student_option_correct):
        raise ValueError(
            "The two lists predictions and correctness do not have the same length."
        )

    return list_predicted_correctness, list_student_option_correct


if __name__ == "__main__":

    n_random_runs = 10

    train_df = pd.read_csv("../data/gold/dbe_kt22_interactions_train.csv")

    split_to_data_path = {
        "val_small": "../data/gold/dbe_kt22_interactions_valsmall.csv",
        "val_large": "../data/gold/dbe_kt22_interactions_vallarge.csv",
        "test": "../data/gold/dbe_kt22_interactions_test.csv",
    }

    # dict where I collect all the results.
    # first key is the data split, then the history length.
    results = {}
    for data_split in split_to_data_path.keys():
        results[data_split] = {}
        for history_length in [1, 3, 5]:
            results[data_split][history_length] = {}
            for metric in ["bal_acc", "llm_correctness"]:
                results[data_split][history_length][metric] = []

    for data_split in split_to_data_path.keys():
        print("Data split: %s" % data_split)
        for history_length in [1, 3, 5]:
            print("HISTORY LENGTH = %d" % history_length)
            df = pd.read_csv(split_to_data_path[data_split])
            for _ in range(n_random_runs):
                predictions, correctness = majority_prediction_answer_correctness(
                    df, train_df, history_length, keep_only_students_with_support=True
                )
                # predictions, correctness = majority_prediction_answer_correctness(
                #     df, train_df, history_length, keep_only_students_with_support=False
                # )
                bal_acc = balanced_accuracy_score(
                    y_true=correctness, y_pred=predictions
                )
                llm_correctness = np.mean(predictions)
                results[data_split][history_length]["bal_acc"].append(bal_acc)
                results[data_split][history_length]["llm_correctness"].append(
                    llm_correctness
                )

    print(results)
    for data_split in split_to_data_path.keys():
        for history_length in [1, 3, 5]:
            bal_acc_mean, bal_acc_stderr = mean_stderror(
                results[data_split][history_length]["bal_acc"]
            )
            llm_correctness_mean, llm_correctness_stderr = mean_stderror(
                results[data_split][history_length]["llm_correctness"]
            )
            print(
                "%9s | %2d | (mean +- stderror) bal acc: %.3f +- %.3f | llm correctness : %.3f +- %.3f"
                % (
                    data_split,
                    history_length,
                    bal_acc_mean,
                    bal_acc_stderr,
                    llm_correctness_mean,
                    llm_correctness_stderr,
                )
            )

### Averaging across 10 runs
# Keeping only students with full support:
# val_small |  1 | (mean +- stderror) bal acc: 0.562 +- 0.016 | llm correctness : 0.819 +- 0.011
# val_small |  3 | (mean +- stderror) bal acc: 0.547 +- 0.010 | llm correctness : 0.857 +- 0.008
# val_small |  5 | (mean +- stderror) bal acc: 0.532 +- 0.010 | llm correctness : 0.901 +- 0.007
# val_large |  1 | (mean +- stderror) bal acc: 0.553 +- 0.005 | llm correctness : 0.827 +- 0.005
# val_large |  3 | (mean +- stderror) bal acc: 0.554 +- 0.006 | llm correctness : 0.885 +- 0.005
# val_large |  5 | (mean +- stderror) bal acc: 0.538 +- 0.004 | llm correctness : 0.923 +- 0.002
#      test |  1 | (mean +- stderror) bal acc: 0.532 +- 0.009 | llm correctness : 0.831 +- 0.007
#      test |  3 | (mean +- stderror) bal acc: 0.532 +- 0.007 | llm correctness : 0.878 +- 0.002
#      test |  5 | (mean +- stderror) bal acc: 0.536 +- 0.006 | llm correctness : 0.901 +- 0.004

# Keeping all students:
# val_small |  1 | (mean +- stderror) bal acc: 0.540 +- 0.010 | llm correctness : 0.823 +- 0.009
# val_small |  3 | (mean +- stderror) bal acc: 0.551 +- 0.009 | llm correctness : 0.869 +- 0.008
# val_small |  5 | (mean +- stderror) bal acc: 0.568 +- 0.006 | llm correctness : 0.900 +- 0.009
# val_large |  1 | (mean +- stderror) bal acc: 0.550 +- 0.004 | llm correctness : 0.833 +- 0.004
# val_large |  3 | (mean +- stderror) bal acc: 0.548 +- 0.004 | llm correctness : 0.898 +- 0.003
# val_large |  5 | (mean +- stderror) bal acc: 0.545 +- 0.006 | llm correctness : 0.931 +- 0.003
#      test |  1 | (mean +- stderror) bal acc: 0.539 +- 0.008 | llm correctness : 0.826 +- 0.006
#      test |  3 | (mean +- stderror) bal acc: 0.540 +- 0.004 | llm correctness : 0.886 +- 0.002
#      test |  5 | (mean +- stderror) bal acc: 0.541 +- 0.004 | llm correctness : 0.913 +- 0.002
