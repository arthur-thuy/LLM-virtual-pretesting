"""Build file for models."""

# standard library imports
# /

# related third party imports
import structlog
from yacs.config import CfgNode

# local application/library specific imports
from tools.registry import Registry

MODEL_PROVIDER_REGISTRY = Registry()

logger = structlog.get_logger(__name__)


def build_model(model_cfg: CfgNode):
    """Build the model.

    Parameters
    ----------
    model_cfg : CfgNode
        Model config object

    Returns
    -------
    model
        Model
    """
    logger.info("Building model", name=model_cfg.NAME, provider=model_cfg.PROVIDER)
    model = MODEL_PROVIDER_REGISTRY[model_cfg.PROVIDER](model_cfg)
    return model
