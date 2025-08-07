"""Module to prepare the datasets for the experiments."""

# standard library imports
# /

# related third party imports
import structlog

# local application/library specific imports
from data_loader.data_loader import DataLoader
from tools.constants import (
    BRONZE_DIR,
    GOLD_DIR,
    SILVER_DIR,
)
from tools.data_manager import CupaCFEDatamanager, DBEKT22Datamanager  # noqa

# set logger
logger = structlog.get_logger()


def main():
    """Run the data preparation."""

    # DBE-KT22
    logger.info("Starting preparation DBE-KT22")
    # NOTE: already built
    dbekt22_dm = DBEKT22Datamanager()
    _, _ = dbekt22_dm.build_dataset(read_dir=BRONZE_DIR, write_dir=SILVER_DIR)
    logger.info("Processing DBE-KT22 for student behaviour replication and roleplay")
    # create fixed split
    data_loader = DataLoader(
        read_dir=SILVER_DIR,
        write_dir=GOLD_DIR,
        dataset_name="dbe_kt22",
    )
    data_loader.split_data(
        val_size_question=0.15,
        test_size_question=0.25,
        val_size_interact=600,
        valsmall_size_interact=100,
        test_size_interact=500,
        split_interactions=True,
        stratified=True,
        seed=43,
        join_key="question_id",
    )

    # # CUPA-CFE
    # logger.info("Starting preparation CUPA-CFE")
    # cupacfe_dm = CupaCFEDatamanager()
    # _, _ = cupacfe_dm.build_dataset(read_dir=BRONZE_DIR, write_dir=SILVER_DIR)
    # # create fixed split
    # data_loader = DataLoader(
    #     read_dir=SILVER_DIR,
    #     write_dir=GOLD_DIR,
    #     dataset_name="cupacfe",
    # )
    # data_loader.split_data(
    #     val_size_question=0.25,
    #     test_size_question=0.75,
    #     split_interactions=False,  # only train interactions
    #     stratified=False,
    #     seed=42,
    #     join_key=None,  # No join key for CUPA-CFE
    # )


if __name__ == "__main__":
    main()
