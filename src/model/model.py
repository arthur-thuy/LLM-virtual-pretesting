"""Module for building models."""

# standard library imports
# /

# related third party imports
from yacs.config import CfgNode
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

# local application/library specific imports
from model.build import MODEL_PROVIDER_REGISTRY


@MODEL_PROVIDER_REGISTRY.register("openai")
def build_openai_model(model_cfg: CfgNode) -> ChatOpenAI:
    if model_cfg.NAME.startswith('o'):
        model = ChatOpenAI(
            model=model_cfg.NAME,
            max_tokens=model_cfg.MAX_TOKENS,
            timeout=model_cfg.TIMEOUT,
            max_retries=model_cfg.MAX_RETRIES,
        )
    else:
        model = ChatOpenAI(
            model=model_cfg.NAME,
            temperature=model_cfg.TEMPERATURE,
            max_tokens=model_cfg.MAX_TOKENS,
            timeout=model_cfg.TIMEOUT,
            max_retries=model_cfg.MAX_RETRIES,
        )
    return model


@MODEL_PROVIDER_REGISTRY.register("anthropic")
def build_anthropic_model(model_cfg: CfgNode) -> ChatAnthropic:
    model = ChatAnthropic(
        model=model_cfg.NAME,
        temperature=model_cfg.TEMPERATURE,
        max_tokens=1024,  # NOTE: model_cfg.MAX_TOKENS,
        timeout=model_cfg.TIMEOUT,
        max_retries=2,  # NOTE: ignoring model_cfg.MAX_RETRIES,
    )
    return model


@MODEL_PROVIDER_REGISTRY.register("ollama")
def build_ollama_model(model_cfg: CfgNode) -> ChatOllama:
    model = ChatOllama(
        model=model_cfg.NAME,
        temperature=model_cfg.TEMPERATURE,
        num_predict=model_cfg.MAX_TOKENS,
        format="json",
    )
    return model


@MODEL_PROVIDER_REGISTRY.register("google")
def build_google_model(model_cfg: CfgNode) -> ChatGoogleGenerativeAI:
    model = ChatGoogleGenerativeAI(
        model=model_cfg.NAME,
        temperature=model_cfg.TEMPERATURE,
        max_tokens=model_cfg.MAX_TOKENS,
        timeout=model_cfg.TIMEOUT,
        # NOTE: leave max_retries at default value (6)
    )
    return model
