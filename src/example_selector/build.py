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


def build_example_selector(cfg: CfgNode, examples: list[dict]) -> None:
    """Build the example selector.

    Parameters
    ----------
    cfg : CfgNode
        Config node.
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
    (selector, input_vars) = EXAMPLE_SELECTOR_REGISTRY[cfg.EXAMPLE_SELECTOR.NAME](
        cfg, examples
    )
    return (selector, input_vars)
