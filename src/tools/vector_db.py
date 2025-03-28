"""File with vector DB functionalities."""

# standard library imports
import os
import sys

# related third party imports
import click
import structlog
import pandas as pd
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec, Index

# local application/library specific imports
from model.build import build_embedding
from tools.constants import (
    EMBEDDINGS_DIM,
    EMBEDDING_PROVIDER,
    QUESTION_ID,
    Q_TEXT,
)

# set up logger
logger = structlog.get_logger()


def namespace_exists(index: Index, namespace: str) -> bool:
    """Check if a namespace exists in the index.

    Parameters
    ----------
    index :
        Index object
    namespace : str
        Namespace name

    Returns
    -------
    bool
        True if the namespace exists, False otherwise.
    """
    namespaces = index.describe_index_stats()["namespaces"]
    return namespace in namespaces


def prepare_empty_vector_store(
    index_name: str, embedding_name: str, namespace: str
) -> PineconeVectorStore:
    """Get an empty Pinecode vector store.

    Parameters
    ----------
    index_name : str
        Index name
    embedding_name : str
        Embedding name
    namespace : str
        Index namespace

    Returns
    -------
    PineconeVectorStore
        The Pinecone vector store.
    """
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

    embedding = build_embedding(
        embedding_name=embedding_name, provider=EMBEDDING_PROVIDER[embedding_name]
    )

    if pc.has_index(index_name):
        if namespace_exists(pc.Index(index_name), namespace):
            logger.info(
                "Namespace already exists in index",
                index_name=index_name,
                namespace=namespace,
            )
            vector_store = PineconeVectorStore(
                index=pc.Index(index_name), embedding=embedding, namespace=namespace
            )
            if click.confirm(
                "Proceed to delete existing namespace?",
                default=False,
            ):
                logger.info("Deleting namespace...")
                vector_store.delete(delete_all=True)
            else:
                logger.warning("Abort process")
                sys.exit(1)
        else:
            logger.info(
                "Namespace does not exist in index",
                index_name=index_name,
                namespace=namespace,
            )
            vector_store = PineconeVectorStore(
                index=pc.Index(index_name), embedding=embedding, namespace=namespace
            )
    else:
        logger.info(
            "Creating index",
            index_name=index_name,
            dimension=EMBEDDINGS_DIM[embedding_name],
        )
        pc.create_index(
            name=index_name,
            dimension=EMBEDDINGS_DIM[embedding_name],
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        # while not pc.describe_index(index_name).status["ready"]:  # TODO: remove
        #     time.sleep(1)

        vector_store = PineconeVectorStore(
            index=pc.Index(index_name), embedding=embedding, namespace=namespace
        )
    return vector_store


def populate_vector_store(
    vector_store: PineconeVectorStore,
    data: pd.DataFrame,
) -> None:
    """Populate the vector store with documents.

    Parameters
    ----------
    vector_store : PineconeVectorStore
        The vector store to populate.
    data : pd.DataFrame
        The data to populate the vector store with.
    """
    vector_input_df = data.drop_duplicates(subset=QUESTION_ID)

    vector_input_doc = [
        Document(
            page_content=row[Q_TEXT],
            metadata={
                QUESTION_ID: str(row[QUESTION_ID]),
            },
        )
        for _, row in vector_input_df.iterrows()
    ]
    vector_input_id = vector_input_df[QUESTION_ID].astype(str).tolist()
    logger.info(
        "Adding documents to vector store",
        num_docs=len(vector_input_doc),
    )
    _ = vector_store.add_documents(documents=vector_input_doc, ids=vector_input_id)


def get_vector_store(
    index_name: str, embedding_name: str, namespace: str
) -> PineconeVectorStore:
    """Get the Pinecode vector store.

    Parameters
    ----------
    index_name : str
        Index name
    embedding_name : str
        Embedding name
    namespace : str
        Index namespace

    Returns
    -------
    PineconeVectorStore
        The Pinecone vector store.

    Raises
    ------
    ValueError
        If the index does not exist.
    """
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    if not pc.has_index(index_name):
        raise ValueError(f"Index {index_name} does not exist.")
    index = pc.Index(index_name)
    embedding = build_embedding(
        embedding_name=embedding_name, provider=EMBEDDING_PROVIDER[embedding_name]
    )
    vector_store = PineconeVectorStore(
        index=index, embedding=embedding, namespace=namespace
    )
    logger.info(
        "Loaded Pinecone vector store", index_name=index_name, namespace=namespace
    )
    return vector_store
