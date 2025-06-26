"""Module for building models."""

# standard library imports
# /

# related third party imports
from langchain_anthropic import ChatAnthropic
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from yacs.config import CfgNode

# local application/library specific imports
from model.build import MODEL_PROVIDER_REGISTRY
from tools.constants import MODEL_RATE_LIMIT


@MODEL_PROVIDER_REGISTRY.register("openai")
def build_openai_model(model_cfg: CfgNode) -> ChatOpenAI:
    if model_cfg.NAME.startswith("o"):
        if MODEL_RATE_LIMIT[model_cfg.NAME] is not None:
            rate_limiter = InMemoryRateLimiter(
                requests_per_second=MODEL_RATE_LIMIT[model_cfg.NAME],
                check_every_n_seconds=0.1,
                max_bucket_size=1,
            )
        else:
            rate_limiter = None
        model = ChatOpenAI(
            model=model_cfg.NAME,
            max_tokens=model_cfg.MAX_TOKENS,
            timeout=model_cfg.TIMEOUT,
            max_retries=model_cfg.MAX_RETRIES,
            rate_limiter=rate_limiter,
        )
    else:
        if MODEL_RATE_LIMIT[model_cfg.NAME] is not None:
            rate_limiter = InMemoryRateLimiter(
                requests_per_second=MODEL_RATE_LIMIT[model_cfg.NAME],
                check_every_n_seconds=0.1,
                max_bucket_size=1,
            )
        else:
            rate_limiter = None
        model = ChatOpenAI(
            model=model_cfg.NAME,
            temperature=model_cfg.TEMPERATURE,
            max_tokens=model_cfg.MAX_TOKENS,
            timeout=model_cfg.TIMEOUT,
            max_retries=model_cfg.MAX_RETRIES,
            rate_limiter=rate_limiter,
        )
    return model


@MODEL_PROVIDER_REGISTRY.register("anthropic")
def build_anthropic_model(model_cfg: CfgNode) -> ChatAnthropic:
    if MODEL_RATE_LIMIT[model_cfg.NAME] is not None:
        rate_limiter = InMemoryRateLimiter(
            requests_per_second=MODEL_RATE_LIMIT[model_cfg.NAME],
            check_every_n_seconds=0.1,
            max_bucket_size=1,
        )
    else:
        rate_limiter = None
    model = ChatAnthropic(
        model=model_cfg.NAME,
        temperature=model_cfg.TEMPERATURE,
        max_tokens=1024,  # NOTE: model_cfg.MAX_TOKENS,
        timeout=model_cfg.TIMEOUT,
        max_retries=2,  # NOTE: ignoring model_cfg.MAX_RETRIES,
        rate_limiter=rate_limiter,
    )
    return model


@MODEL_PROVIDER_REGISTRY.register("ollama")
def build_ollama_model(model_cfg: CfgNode) -> ChatOllama:
    if MODEL_RATE_LIMIT[model_cfg.NAME] is not None:
        rate_limiter = InMemoryRateLimiter(
            requests_per_second=MODEL_RATE_LIMIT[model_cfg.NAME],
            check_every_n_seconds=0.1,
            max_bucket_size=1,
        )
    else:
        rate_limiter = None
    model = ChatOllama(
        model=model_cfg.NAME,
        temperature=model_cfg.TEMPERATURE,
        num_predict=model_cfg.MAX_TOKENS,
        format="json",
        rate_limiter=rate_limiter,
    )
    return model


@MODEL_PROVIDER_REGISTRY.register("google")
def build_google_model(model_cfg: CfgNode) -> ChatGoogleGenerativeAI:
    if MODEL_RATE_LIMIT[model_cfg.NAME] is not None:
        rate_limiter = InMemoryRateLimiter(
            requests_per_second=MODEL_RATE_LIMIT[model_cfg.NAME],
            check_every_n_seconds=0.1,
            max_bucket_size=1,
        )
    else:
        rate_limiter = None
    model = ChatGoogleGenerativeAI(
        model=model_cfg.NAME,
        temperature=model_cfg.TEMPERATURE,
        max_tokens=model_cfg.MAX_TOKENS,
        timeout=model_cfg.TIMEOUT,
        # NOTE: leave max_retries at default value (6)
        rate_limiter=rate_limiter,
    )
    return model
