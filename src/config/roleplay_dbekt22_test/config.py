"""Default config file."""

from yacs.config import CfgNode as CN

_C = CN()

# seed for reproducibility
_C.SEED = 42

# number of runs
_C.RUNS = 3

# context type
_C.CONTEXT_TYPE = "snippets_misconceptions"

# model architecture
_C.MODEL = CN()
# model name
_C.MODEL.NAME = "qwen3:8b"
# model temperature
_C.MODEL.TEMPERATURE = 0.0
# max tokens
_C.MODEL.MAX_TOKENS = 1024
# timeout (openai and anthropic only)
_C.MODEL.TIMEOUT = None
# max retries (openai and anthropic only)
_C.MODEL.MAX_RETRIES = None

# structured output
_C.STRUCTURED_OUTPUTTER = CN()
# structured output name
_C.STRUCTURED_OUTPUTTER.NAME = "student_bool_nocontext"

# data loader
_C.LOADER = CN()
# dataset name
_C.LOADER.NAME = "dbe_kt22"
# dataset join key
_C.LOADER.JOIN_KEY = "question_id"
# run sets
_C.LOADER.RUN_VAL = False
_C.LOADER.RUN_LARGE_VAL = False
_C.LOADER.RUN_TEST = True
# difficulty range
_C.LOADER.DIFFICULTY_RANGE = (-5, 5)

# example formatter
_C.EXAMPLE_FORMATTER = CN()
# interactions
_C.EXAMPLE_FORMATTER.INTERACTIONS = CN()
_C.EXAMPLE_FORMATTER.INTERACTIONS.NAME = "quotes"
# questions
_C.EXAMPLE_FORMATTER.QUESTIONS = CN()
_C.EXAMPLE_FORMATTER.QUESTIONS.NAME = "quotes"

# example selector
_C.EXAMPLE_SELECTOR = CN()
# example selector name
_C.EXAMPLE_SELECTOR.NAME = "both_cfe_studentlevel_random"  # TODO: change
# number of examples to select
_C.EXAMPLE_SELECTOR.NUM_EXAMPLES = 1
# example selector embedding model
_C.EXAMPLE_SELECTOR.EMBEDDING = "text-embedding-3-large"

# system prompt
_C.PROMPT = CN()
# system prompt name
_C.PROMPT.NAME = "teacher_dbe_miscons_level_context"

_C.ROLEPLAY = CN()
# number of student levels to simulate
_C.ROLEPLAY.NUM_STUDENT_LEVELS = 5
# student level scale
_C.ROLEPLAY.STUDENT_SCALE = "proficiency_5_str"


def get_cfg_defaults() -> CN:
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    return _C.clone()
