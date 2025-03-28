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
from tools.data_manager import DBEKT22Datamanager, CupaDatamanager
from data_loader.data_loader import DataLoader

# set logger
logger = structlog.get_logger()


def main():
    """Run the data preparation."""
    # DBE-KT22
    logger.info("Starting preparation DBE-KT22")
    dbekt22_dm = DBEKT22Datamanager()
    # TODO: remove this if want to use all students
    SAMPLE_STUDENT_IDS = None  # 25  # 100
    _, _ = dbekt22_dm.build_dataset(
        read_dir=BRONZE_DIR, write_dir=SILVER_DIR, sample_student_ids=SAMPLE_STUDENT_IDS
    )
    # create fixed split
    data_loader = DataLoader(
        read_dir=SILVER_DIR,
        dataset_name="dbe_kt22",
    )
    data_loader.split_data(
        val_size=600,
        test_size=1000,
        seed=42,
    )

    # # CUPA
    # logger.info("Starting preparation CUPA")
    # cupa_dm = CupaDatamanager()
    # _, _ = cupa_dm.build_dataset(read_dir=BRONZE_DIR, write_dir=SILVER_DIR)


if __name__ == "__main__":
    main()
