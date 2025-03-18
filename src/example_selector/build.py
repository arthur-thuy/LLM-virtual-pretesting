"""Build file for example selectors."""

# standard library imports
# /

# related third party imports
import structlog
from yacs.config import CfgNode

# local application/library specific imports
from tools.registry import Registry

EXAMPLE_SELECTOR_REGISTRY = Registry()

logger = structlog.get_logger(__name__)


def build_example_selector(example_selector_cfg: CfgNode, examples: list[dict]) -> None:
    """Build the example selector.

    Parameters
    ----------
    example_selector_cfg : CfgNode
        Config node for the example selector.
    examples : list[dict]
        List of examples.

    Returns
    -------
    TODO
        Example selector object
    """
    logger.info(
        "Building example selector",
    )
    model = EXAMPLE_SELECTOR_REGISTRY[example_selector_cfg.NAME](example_selector_cfg, examples)
    return model
