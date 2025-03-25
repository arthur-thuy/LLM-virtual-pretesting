"""Module for building models."""

# standard library imports
# /

# related third party imports
from yacs.config import CfgNode
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_anthropic import ChatAnthropic

# local application/library specific imports
from model.build import MODEL_PROVIDER_REGISTRY


@MODEL_PROVIDER_REGISTRY.register("openai")
def build_openai(model_cfg: CfgNode) -> ChatOpenAI:
    model = ChatOpenAI(
        model=model_cfg.NAME,
        temperature=model_cfg.TEMPERATURE,
        max_tokens=model_cfg.MAX_TOKENS,
        timeout=model_cfg.TIMEOUT,
        max_retries=model_cfg.MAX_RETRIES,
    )
    return model


@MODEL_PROVIDER_REGISTRY.register("anthropic")
def build_anthropic(model_cfg: CfgNode) -> ChatAnthropic:
    model = ChatAnthropic(
        model=model_cfg.NAME,
        temperature=model_cfg.TEMPERATURE,
        max_tokens=model_cfg.MAX_TOKENS,
        timeout=model_cfg.TIMEOUT,
        max_retries=model_cfg.MAX_RETRIES,
        # TODO: does it support json mode?
    )
    return model


@MODEL_PROVIDER_REGISTRY.register("ollama")
def build_ollama(model_cfg: CfgNode) -> ChatOllama:
    model = ChatOllama(
        model=model_cfg.NAME,
        temperature=model_cfg.TEMPERATURE,
        num_predict=model_cfg.MAX_TOKENS,
        format="json",
    )
    return model
