"""Constants."""

from types import NoneType


# datasets
CUPA = "cupa"

# paths
BRONZE_DIR = "../data/raw"
SILVER_DIR = "../data/processed"

VALIDATION = "validation"
TEST = "test"
TRAIN = "train"

# columns
## questions
QUESTION_ID = "question_id"
Q_TEXT = "q_text"
Q_DIFFICULTY = "q_difficulty"
Q_OPTION_IDS = "option_ids"
Q_OPTION_TEXTS = "option_texts"
Q_CORRECT_OPTION_ID = "correct_option_id"
Q_CONTEXT_TEXT = "context_text"
Q_CONTEXT_ID = "context_id"
Q_DISCRIMINATION = "q_discrimination"

## interactions
INTERACT_ID = "interact_id"
STUDENT_ID = "student_id"
S_OPTION_ID = "student_option_id"
S_OPTION_CORRECT = "student_option_correct"

## example formatter
INPUT = "input"
OUTPUT = "output"


# model
MODEL_STRUCTURED_OUTPUT = {
    "llama3": False,
    "llama3.2": True,
    "olmo2:7b": False,
    "gpt-4o": True,
    "gpt-4o-mini": True,
}

# prompt info
DICT_CEFR_DESCRIPTIONS = {
    "A1": "can understand and use familiar everyday expressions and very basic phrases aimed at the satisfaction of needs of a concrete type; can introduce him/herself and others and can ask and answer questions about personal details such as where he/she lives, people he/she knows and things he/she has; can interact in a simple way provided the other person talks slowly and clearly and is prepared to help.",
    "A2": "can understand sentences and frequently used expressions related to areas of most immediate relevance (e.g. very basic personal and family information, shopping, local geography, employment); can communicate in simple and routine tasks requiring a simple and direct exchange of information on familiar and routine matters; can describe in simple terms aspects of his/her background, immediate environment and matters in areas of immediate need.",
    "B1": "can understand the main points of clear standard input on familiar matters regularly encountered in work, school, leisure, etc; can deal with most situations likely to arise whilst travelling in an area where the language is spoken; can produce simple connected text on topics which are familiar or of personal interest; can describe experiences and events, dreams, hopes & ambitions and briefly give reasons and explanations for opinions and plans.",
    "B2": "can understand the main ideas of complex text on both concrete and abstract topics, including technical discussions in his/her field of specialisation; can interact with a degree of fluency and spontaneity that makes regular interaction with native speakers quite possible without strain for either party; can produce clear, detailed text on a wide range of subjects and explain a viewpoint on a topical issue giving the advantages and disadvantages of various options.",
    "C1": "can understand a wide range of demanding, longer texts, and recognise implicit meaning; can express him/herself fluently and spontaneously without much obvious searching for expressions; can use language flexibly and effectively for social, academic and professional purposes; can produce clear, well-structured, detailed text on complex subjects, showing controlled use of organisational patterns, connectors and cohesive devices.",
    "C2": "can understand with ease virtually everything heard or read; can summarise information from different spoken and written sources, reconstructing arguments and accounts in a coherent presentation; can express him/herself spontaneously, very fluently and precisely, differentiating finer shades of meaning even in more complex situations.",
}
PROMPT_INFO = {
    "dbe_kt22": {
        "exam_type": "a database systems exam (Department of Computer Science)"
    }
}

# configuration types

_VALID_TYPES = {tuple, list, str, int, float, bool, NoneType}
