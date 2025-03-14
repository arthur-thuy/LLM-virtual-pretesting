"""Module to prepare the datasets for the experiments."""

# standard library imports
# /

# related third party imports
import structlog

# local application/library specific imports
from tools.constants import (
    WRITE_DIR,
)
from tools.data_manager import DBEKT22Datamanager

# set logger
logger = structlog.get_logger()


def main():
    """Run the data preparation."""
    # DBE-KT22
    logger.info("Starting preparation DBE-KT22")
    cupa_data_dir = "../data/raw/dbe-kt22"
    dbekt22_dm = DBEKT22Datamanager()
    dbekt22_dm.build_dataset(cupa_data_dir, WRITE_DIR)


if __name__ == "__main__":
    main()
