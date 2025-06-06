"""Build file for student scale."""

# standard library imports
# /

# related third party imports
import structlog
from yacs.config import CfgNode

# local application/library specific imports
from tools.registry import Registry

STUDENT_SCALE_REGISTRY = Registry()

logger = structlog.get_logger(__name__)


def build_student_scale(
    cfg: CfgNode,
) -> dict[str, str]:
    """Build the roleplay dataset.

    Parameters
    ----------
    loader_cfg : CfgNode
        Data loader config object

    Returns
    -------
    Dict[str, pd.DataFrame]
        Train/Val/Test interaction dataframes
    """
    logger.info(
        "Building student scale",
        num_levels=cfg.ROLEPLAY.NUM_STUDENT_LEVELS,
        student_scale=cfg.ROLEPLAY.STUDENT_SCALE,
    )
    # NOTE: mapping is "digits as strings -> strings of student levels"
    student_scale_map, student_scale_str = STUDENT_SCALE_REGISTRY[
        cfg.ROLEPLAY.STUDENT_SCALE
    ](
        num_groups=cfg.ROLEPLAY.NUM_STUDENT_LEVELS,
    )
    logger.info(
        "Student scale built",
        map=student_scale_map,
        string=student_scale_str,
    )
    return student_scale_map, student_scale_str
