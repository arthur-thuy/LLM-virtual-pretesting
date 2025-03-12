"""Constants."""

OPENAI_API_KEY = "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"

# datasets
CUPA = "cupa"

# paths
WRITE_DIR = "../data/processed"

DEV = "dev"
VALIDATION = "validation"
TEST = "test"
TRAIN = "train"
CORRECT_ANSWERS_LIST = "correct_answers_list"
PRED_DIFFICULTY = "predicted_difficulty"
TF_QUESTION_ID = "question_id"
TF_DIFFICULTY = "difficulty"
TF_TEXT = "text"
TF_LABEL = "label"
TF_DESCRIPTION = "description"
TF_ANSWERS = "answers"
TF_CORRECT = "correct"
TF_ANS_ID = "id"

CORRECT_ANSWER = "correct_answer"
OPTIONS = "options"
OPTION_ = "option_"
OPTION_0 = "option_0"
OPTION_1 = "option_1"
OPTION_2 = "option_2"
OPTION_3 = "option_3"
QUESTION = "question"
CONTEXT = "context"
CONTEXT_ID = "context_id"
Q_ID = "q_id"
SPLIT = "split"
DIFFICULTY = "difficulty"
DF_COLS = [
    CORRECT_ANSWER,
    OPTIONS,
    OPTION_0,
    OPTION_1,
    OPTION_2,
    OPTION_3,
    QUESTION,
    CONTEXT,
    CONTEXT_ID,
    Q_ID,
    SPLIT,
    DIFFICULTY,
]
