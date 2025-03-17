"""Default config file."""

from yacs.config import CfgNode as CN

_C = CN()

# seed for reproducibility
_C.SEED = 42


# model architecture
_C.MODEL = CN()
# model name
_C.MODEL.NAME = "llama3.2"
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

# data loader
_C.LOADER = CN()
# dataset name
_C.LOADER.NAME = "dbe_kt22"
# train size
_C.LOADER.TRAIN_SIZE = 0.6
# test size
_C.LOADER.TEST_SIZE = 0.25


def get_cfg_defaults() -> CN:
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    return _C.clone()
