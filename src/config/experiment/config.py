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
_C.MODEL.NAME = "llama3"
# model provider
_C.MODEL.PROVIDER = "ollama"
# model temperature
_C.MODEL.TEMPERATURE = 0.0
# max tokens
_C.MODEL.MAX_TOKENS = None
# timeout (openai and anthropic only)
_C.MODEL.TIMEOUT = None
# max retries (openai and anthropic only)
_C.MODEL.MAX_RETRIES = None

# structured output
_C.STRUCTURED_OUTPUTTER = CN()
# structured output name
_C.STRUCTURED_OUTPUTTER.NAME = "B"

# data loader
_C.LOADER = CN()
# dataset name
_C.LOADER.NAME = "dbe_kt22"
# dataset join key
_C.LOADER.JOIN_KEY = "question_id"
# train size
_C.LOADER.TRAIN_SIZE = 0.6
# test size
_C.LOADER.TEST_SIZE = 0.25

# example formatter
_C.EXAMPLE_FORMATTER = CN()
# example formatter name
_C.EXAMPLE_FORMATTER.NAME = "A"

# example selector
_C.EXAMPLE_SELECTOR = CN()
# example selector name
_C.EXAMPLE_SELECTOR.NAME = "random"
# number of examples to select
_C.EXAMPLE_SELECTOR.NUM_EXAMPLES = 3

# systemprompt
_C.SYSTEM_PROMPT = CN()
# system prompt name
_C.PROMPT.SYSTEM.NAME = "student_A"



def get_cfg_defaults() -> CN:
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    return _C.clone()
