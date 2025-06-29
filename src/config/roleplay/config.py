"""Default config file."""

from yacs.config import CfgNode as CN

_C = CN()

# seed for reproducibility
_C.SEED = 42

# number of runs
_C.RUNS = 1

# model architecture
_C.MODEL = CN()
# model name
_C.MODEL.NAME = "o3-mini"
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
_C.STRUCTURED_OUTPUTTER.NAME = "teacher"

# data loader
_C.LOADER = CN()
# dataset name
_C.LOADER.NAME = "dbe_kt22"
# dataset join key
_C.LOADER.JOIN_KEY = "question_id"
# run large validation set
_C.LOADER.RUN_LARGE_VAL = False

# example formatter
_C.EXAMPLE_FORMATTER = CN()
# example formatter name
_C.EXAMPLE_FORMATTER.NAME = "quotes"

# example selector
_C.EXAMPLE_SELECTOR = CN()
# example selector name
_C.EXAMPLE_SELECTOR.NAME = "studentlevel_random"
# number of examples to select
_C.EXAMPLE_SELECTOR.NUM_EXAMPLES = 3
# example selector embedding model
_C.EXAMPLE_SELECTOR.EMBEDDING = "text-embedding-3-large"

# systemprompt
_C.PROMPT = CN()
# system prompt name
_C.PROMPT.NAME = "replicate_teacher_onion"

_C.ROLEPLAY = CN()
# number of student levels to simulate
_C.ROLEPLAY.NUM_STUDENT_LEVELS = 5
# student level scale
_C.ROLEPLAY.STUDENT_SCALE = "proficiency_5_str"


def get_cfg_defaults() -> CN:
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    return _C.clone()
