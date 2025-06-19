"""File with configurator functionalities."""

# standard library imports
import importlib
import os
import datetime
from pathlib import Path
from typing import Any, Union

# related third party imports
import structlog
import yaml
from yacs.config import CfgNode

# local application/library specific imports
from tools.utils import ensure_dir
from tools.constants import MODEL_STRUCTURED_OUTPUT, _VALID_TYPES

# set up logger
logger = structlog.get_logger(__name__)


def save_config(cfg: CfgNode, save_dir: str, fname: str) -> None:
    """Save config.

    Parameters
    ----------
    cfg : CfgNode
        Config object
    save_dir : str
        Directory to save to
    fname : str
        Filename to save to
    """
    ensure_dir(os.path.join(save_dir, "config"))
    fpath = os.path.join(save_dir, "config", f"{fname}.yaml")
    cfg.dump(stream=open(fpath, "w", encoding="utf-8"))


def _merge_config(config_path: Union[str, Path]) -> CfgNode:
    """Load single config file from path.

    Parameters
    ----------
    config_path : Union[str, Path]
        Config file path.

    Returns
    -------
    CfgNode
        Config object.
    """
    logger.info(f"Loading config from '{config_path}'")
    exp_name = os.path.basename(os.path.dirname(config_path))
    config_exp = importlib.import_module(f"config.{exp_name}.config")
    cfg = config_exp.get_cfg_defaults()
    cfg.merge_from_file(config_path)
    return cfg


def _add_derived_configs(
    cfg: CfgNode, config_dir: Union[str, Path], freeze: bool = True
) -> CfgNode:
    """Add derived config variables at runtime.

    Parameters
    ----------
    cfg : CfgNode
        Config object.
    config_dir : Union[str, Path]
        Config directory.
    freeze : bool, optional
        Whether to freeze config object, by default True

    Returns
    -------
    CfgNode
        Config object with derived variables.
    """
    # add derived config variables at runtime
    cfg.ID = create_config_id(cfg)
    cfg.ID_ROLEPLAY = create_roleplay_config_id(cfg)
    cfg.OUTPUT_DIR = os.path.join(".", "output", config_dir)
    cfg.OUTPUT_DIR_ROLEPLAY = os.path.join(".", "output", f"roleplay_{config_dir}")
    cfg.MODEL.NATIVE_STRUCTURED_OUTPUT = MODEL_STRUCTURED_OUTPUT[cfg.MODEL.NAME]
    if freeze:
        cfg.freeze()
    return cfg


def load_configs(fpath: str, freeze: bool = True) -> tuple[CfgNode, ...]:
    """Load one or more config files from path.

    Parameters
    ----------
    fpath : str
        Path to config file or directory.
    freeze : bool, optional
        Whether to freeze config objects, by default True

    Returns
    -------
    tuple[CfgNode, ...]
        Tuple of config objects.

    Raises
    ------
    ValueError
        When fpath is not a valid directory.
    """
    # check if path is valid directory
    config_path_full = Path(os.path.join("config", fpath))
    if not (config_path_full.exists() and config_path_full.is_dir()):
        raise ValueError(f"Invalid config dirname (base): {config_path_full}")
    config_paths = list(config_path_full.glob("*.yaml"))
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = f"{fpath}_{timestamp}"
    # load config files
    configs = tuple([_merge_config(config_path) for config_path in config_paths])
    # add derived config variables
    configs = tuple(
        [_add_derived_configs(config, output_dir, freeze) for config in configs]
    )
    return configs


def create_config_id(cfg: CfgNode) -> str:
    """Create identifier for config.

    Parameters
    ----------
    cfg : CfgNode
        Config object.

    Returns
    -------
    str
        Config identifier.
    """
    cfg_id = cfg.MODEL.NAME
    cfg_id += f"~T_{cfg.MODEL.TEMPERATURE}"
    cfg_id += f"~SO_{cfg.STRUCTURED_OUTPUTTER.NAME}"
    cfg_id += f"~SP_{cfg.PROMPT.NAME}"
    cfg_id += f"~EF_{cfg.EXAMPLE_FORMATTER.NAME}"
    cfg_id += f"~ES_{cfg.EXAMPLE_SELECTOR.NAME}{cfg.EXAMPLE_SELECTOR.NUM_EXAMPLES}"
    return cfg_id


def create_roleplay_config_id(cfg: CfgNode) -> str:
    """Create identifier for config during roleplaying.

    Parameters
    ----------
    cfg : CfgNode
        Config object.

    Returns
    -------
    str
        Config identifier.
    """
    cfg_id = cfg.MODEL.NAME
    cfg_id += f"~T_{cfg.MODEL.TEMPERATURE}"
    cfg_id += f"~SO_{cfg.STRUCTURED_OUTPUTTER.NAME}"
    cfg_id += f"~L_{cfg.ROLEPLAY.NUM_STUDENT_LEVELS}"
    cfg_id += f"~SP_{cfg.PROMPT.NAME}"
    cfg_id += f"~SS_{cfg.ROLEPLAY.STUDENT_SCALE}"
    cfg_id += f"~EF_{cfg.EXAMPLE_FORMATTER.NAME}"
    cfg_id += f"~ES_{cfg.EXAMPLE_SELECTOR.NAME}{cfg.EXAMPLE_SELECTOR.NUM_EXAMPLES}"
    return cfg_id


def get_configs_out(experiment: str) -> tuple[dict[str, Any], ...]:
    """Get configs from output directory.

    Parameters
    ----------
    experiment : str
        Experiment name

    Returns
    -------
    Tuple[Dict[str, Any], ...]
        Tuple of config dicts
    """
    config_dir = Path(os.path.join("output", experiment, "config"))
    config_paths = list(config_dir.glob("*.yaml"))

    def _read_yaml_config(config_path: Union[str, Path]) -> dict:
        with open(config_path, "r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        return config

    configs = tuple([_read_yaml_config(config_path) for config_path in config_paths])
    return configs


def get_config_ids(configs: tuple[dict]) -> list[Any]:
    """Get config IDs from configs.

    Parameters
    ----------
    configs : Tuple[dict]
        Tuple of config dicts

    Returns
    -------
    List[Any]
        Tuple of config IDs
    """
    config_ids = [cfg["ID"] for cfg in configs]
    return config_ids


def check_cfg(cfg: CfgNode) -> None:
    """Check config for logical errors.

    Parameters
    ----------
    cfg : CfgNode
        Config object.

    Raises
    ------
    ValueError
        If NUM_STUDENT_LEVELS is less than 3.
    ValueError
        If structured outputter and prompt do not match.
    """
    if cfg.ROLEPLAY.NUM_STUDENT_LEVELS < 3:
        raise ValueError(
            "ROLEPLAY.NUM_STUDENT_LEVELS must be at least 3, "
            f"got {cfg.ROLEPLAY.NUM_STUDENT_LEVELS}"

    if "student" in cfg.PROMPT.NAME:
        if "student" not in cfg.STRUCTURED_OUTPUTTER.NAME:
            raise ValueError(
                "Both structured outputter and prompt should be of the same type, "
                "either 'student' or 'teacher'."
            )
    if "teacher" in cfg.PROMPT.NAME:
        if "teacher" not in cfg.STRUCTURED_OUTPUTTER.NAME:
            raise ValueError(
                "Both structured outputter and prompt should be of the same type, "
                "either 'student' or 'teacher'."
            )


def convert_to_dict(cfg_node, key_list=[]):
    """Convert a config node to dictionary"""
    if not isinstance(cfg_node, CfgNode):
        if type(cfg_node) not in _VALID_TYPES:
            print(
                "Key {} with value {} is not a valid type; valid types: {}".format(
                    ".".join(key_list), type(cfg_node), _VALID_TYPES
                ),
            )
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = convert_to_dict(v, key_list + [k])
        return cfg_dict
