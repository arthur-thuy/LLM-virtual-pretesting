"""Module to prepare the datasets for the experiments."""

# standard library imports
# /

# related third party imports
import structlog

# local application/library specific imports
from tools.constants import (
    BRONZE_DIR,
    SILVER_DIR,
)
from tools.data_manager import DBEKT22Datamanager

# set logger
logger = structlog.get_logger()


def main():
    """Run the data preparation."""
    # DBE-KT22
    logger.info("Starting preparation DBE-KT22")
    dbekt22_dm = DBEKT22Datamanager()
    # TODO: remove this if want to use all students
    SAMPLE_STUDENT_IDS = 25  # 100
    _ = dbekt22_dm.build_dataset(
        read_dir=BRONZE_DIR, write_dir=SILVER_DIR, sample_student_ids=SAMPLE_STUDENT_IDS
    )


if __name__ == "__main__":
    main()
