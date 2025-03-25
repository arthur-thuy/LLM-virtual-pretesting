"""Utils file."""

# standard library imports
import os
import random
import subprocess
import sys
import time
import pickle
from pathlib import Path
from typing import Union, Optional, Any

# related third party imports
import dotenv
import click
import numpy as np
import structlog
import torch
import matplotlib.pyplot as plt
from torch.backends import cudnn
from yacs.config import CfgNode
from uuid import UUID
from tqdm.auto import tqdm
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


# local application/library specific imports
# /

# set up logger
logger = structlog.get_logger(__name__)


def ensure_dir(dirname: Union[Path, str]) -> None:
    """Ensure directory exists.

    Parameters
    ----------
    dirname : Union[Path, str]
        Directory to check/create.
    """
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def get_save_path(save_dir: str, fname: str) -> str:
    """Get save path.

    Parameters
    ----------
    save_dir : str
        Directory to save to
    fname : str
        Filename to save to

    Returns
    -------
    str
        Full path to save to
    """
    ensure_dir(save_dir)
    fpath = os.path.join(save_dir, f"{fname}.pickle")
    return fpath


def write_pickle(data: dict, save_dir: str, fname: str) -> None:
    """Write data to pickle file.

    Parameters
    ----------
    data : dict
        Data to save
    save_dir : str
        Directory to save to
    fname : str
        Filename to save to
    """
    fpath = get_save_path(save_dir, fname)
    with open(fpath, "wb") as f:
        pickle.dump(data, f)


def read_pickle(fpath: str) -> dict:
    """Read data from pickle file.

    Parameters
    ----------
    fpath : str
        Path to pickle file

    Returns
    -------
    dict
        Loaded data
    """
    with open(fpath, "rb") as f:
        data = pickle.load(f)
    return data


def set_seed(seed: Optional[int] = None) -> None:
    """Set seed for reproducibility.

    Parameters
    ----------
    seed : int
        Seed value
    """
    if seed is None:
        logger.warning("No seed set")
    else:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)
        logger.info(f"Set seed ({seed})")


def print_elapsed_time(start_time: float, run_id: int) -> None:
    """Print elapsed time for each experiment of acquiring.

    Parameters
    ----------
    start_time : float
        Starting time (in time.time())
    run_id : int
        Run iteration
    """
    elp = time.time() - start_time
    print(f"Run {run_id} finished: {format_time(elp=elp)}")


def format_time(elp: float, format: str = "H:M:S") -> str:
    """Format time elapsed.

    Parameters
    ----------
    elp : float
        Time elapsed
    format : str, optional
        Format to return, by default "H:M:S"

    Returns
    -------
    str
        Formatted time elapsed
    """
    if format == "H:M:S":
        return f"{int(elp//3600)}:{int(elp % 3600//60)}:{int(elp % 60)}"
    elif format == "M:S":
        return f"{int((elp % 3600//60)+60*(elp//3600))}:{int(elp % 60)}"
    else:
        raise ValueError(f"Unknown format: {format}")


def delete_previous_content(cfg: CfgNode) -> None:
    """Delete previous content in output directory.

    Parameters
    ----------
    cfg : CfgNode
        Config
    """
    if os.path.isdir(cfg.OUTPUT_DIR):
        if not os.listdir(cfg.OUTPUT_DIR):
            logger.info("Directory is empty and will be removed")
        else:
            if click.confirm(
                f"Proceed to delete previous content in {cfg.OUTPUT_DIR}?",
                default=False,
            ):
                logger.info("Deleting previous content...")
                subprocess.run(
                    f"rm -r {os.path.join(cfg.OUTPUT_DIR, '*')}",
                    shell=True,
                    check=False,
                )
            else:
                logger.warning("Abort process")
                sys.exit(1)


def activate_latex(sans_serif: bool = False):
    """Activate latex for matplotlib."""
    if sans_serif:
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "Helvetica",
                "text.latex.preamble": r"\usepackage[cm]{sfmath}",
            }
        )
    else:
        plt.rcParams.update(
            {"text.usetex": True, "font.family": "Computer Modern Roman"}
        )


def deactivate_latex():
    """Deactivate latex for matplotlib."""
    plt.rcParams.update(
        {"text.usetex": False, "font.family": "DejaVu Sans", "text.latex.preamble": ""}
    )


class BatchCallback(BaseCallbackHandler):
    def __init__(self, total: int):
        super().__init__()
        self.count = 0
        self.progress_bar = tqdm(total=total)  # define a progress bar

    # Override on_llm_end method. This is called after every response from LLM
    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        self.count += 1
        self.progress_bar.update(1)


def load_env(env_path: str) -> None:
    """Load environment variables from file.

    Parameters
    ----------
    env_path : str
        Path to environment file
    """
    # Reload the variables in your '.env' file (override the existing variables)
    dotenv.load_dotenv(env_path, override=True)
    logger.info(f"Loaded environment variables from {env_path}")
