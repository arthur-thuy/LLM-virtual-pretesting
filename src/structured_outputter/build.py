"""Build file for example formatter."""

# standard library imports
# /

# related third party imports
import structlog
from yacs.config import CfgNode
from pydantic import BaseModel

# local application/library specific imports
from tools.registry import Registry

STRUCTURED_OUTPUTTER_REGISTRY = Registry()

logger = structlog.get_logger(__name__)


def build_structured_outputter(structured_outputter_cfg: CfgNode) -> BaseModel:
    """Build the structured outputter.

    Parameters
    ----------
    structured_outputter_cfg : CfgNode
        Config node for the structured output.

    Returns
    -------
    BaseModel
        Pydantic model
    """
    logger.info(
        "Building structured outputter",
        name=structured_outputter_cfg.NAME,
    )
    pydantic_class = STRUCTURED_OUTPUTTER_REGISTRY[structured_outputter_cfg.NAME]
    return pydantic_class
