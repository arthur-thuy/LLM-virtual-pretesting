"""Default config file."""

from yacs.config import CfgNode as CN

_C = CN()

# seed for reproducibility
_C.SEED = 42

# number of runs
_C.RUNS = 1

# context type
_C.CONTEXT_TYPE = ""

# model architecture
_C.MODEL = CN()
# model name
_C.MODEL.NAME = "o4-mini-2025-04-16"  # "qwen3:8b"  # 
# model temperature
_C.MODEL.TEMPERATURE = 1.0
# max tokens
_C.MODEL.MAX_TOKENS = None
# timeout (openai and anthropic only)
_C.MODEL.TIMEOUT = None
# max retries (openai and anthropic only)
_C.MODEL.MAX_RETRIES = None

# structured output
_C.STRUCTURED_OUTPUTTER = CN()
# structured output name
_C.STRUCTURED_OUTPUTTER.NAME = "misconceptions_cfe"

# data loader
_C.LOADER = CN()
# dataset name
_C.LOADER.NAME = "cupacfe"
# dataset join key
_C.LOADER.JOIN_KEY = None

# example formatter
_C.EXAMPLE_FORMATTER = CN()
# interactions
_C.EXAMPLE_FORMATTER.INTERACTIONS = CN()
_C.EXAMPLE_FORMATTER.INTERACTIONS.NAME = "open_reading_collect_miscons"
# # questions
# _C.EXAMPLE_FORMATTER.QUESTIONS = CN()
# _C.EXAMPLE_FORMATTER.QUESTIONS.NAME = "mcq_reading_quotes"

# system prompt
_C.PROMPT = CN()
# system prompt name
_C.PROMPT.NAME = "collect_misconceptions_cfe"


def get_cfg_defaults() -> CN:
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    return _C.clone()
