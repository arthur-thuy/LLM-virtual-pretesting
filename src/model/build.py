"""Build file for models."""

# standard library imports
# /

# related third party imports
import structlog
from yacs.config import CfgNode
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

# local application/library specific imports
from tools.registry import Registry
from tools.constants import MODEL_PROVIDER

MODEL_PROVIDER_REGISTRY = Registry()

logger = structlog.get_logger(__name__)


def build_model(model_cfg: CfgNode):
    """Build the model.

    Parameters
    ----------
    model_cfg : CfgNode
        Model config object

    Returns
    -------
    model
        Model
    """
    logger.info(
        "Building model", name=model_cfg.NAME, provider=MODEL_PROVIDER[model_cfg.NAME]
    )
    model = MODEL_PROVIDER_REGISTRY[MODEL_PROVIDER[model_cfg.NAME]](model_cfg)
    return model


def build_embedding(embedding_name: str, provider: str) -> Embeddings:
    """Get the embedding function.

    Parameters
    ----------
    embedding_name : str
        Embedding name
    provider : str
        Embedding provider

    Returns
    -------
    Embeddings
        The embedding function.
    """
    if provider == "ollama":
        return OllamaEmbeddings(model=embedding_name)
    elif provider == "openai":
        return OpenAIEmbeddings(model=embedding_name)
    elif provider == "anthropic":
        raise ValueError(f"{provider} does not support embeddings in their API.")
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")
