"""Module to prepare the datasets for the experiments."""

# standard library imports
import os

# NOTE: load environment variables
from tools.utils import load_env  # isort:skip

load_env(os.path.join("..", ".env"))  # noqa

# related third party imports
import structlog

# local application/library specific imports
from data_loader.data_loader import DataLoader
from tools.constants import (
    EMBEDDING_PROVIDER,
    GOLD_DIR,
    RATE_LIMIT,
    SILVER_DIR,
)
from tools.vector_db import populate_vector_store, prepare_empty_vector_store

# set logger
logger = structlog.get_logger()

EMBEDDING_NAMES = ["llama3", "text-embedding-3-large", "gemini-embedding-exp-03-07"]


def main() -> None:
    # DBE-KT22
    logger.info("Starting preparation DBE-KT22")
    data_loader = DataLoader(
        read_dir=SILVER_DIR,
        write_dir=GOLD_DIR,
        dataset_name="dbe_kt22",
    )
    train_dataset = data_loader.read_splitted_train_interactions()

    for embedding_name in EMBEDDING_NAMES:
        # prepare vector store
        vector_store = prepare_empty_vector_store(
            index_name=embedding_name,
            embedding_name=embedding_name,
            namespace="dbe_kt22",
        )
        # populate vector store
        populate_vector_store(
            vector_store=vector_store,
            data=train_dataset,
            requests_per_minute=RATE_LIMIT[EMBEDDING_PROVIDER[embedding_name]],
        )


if __name__ == "__main__":
    main()
