"""Execute at import time, needed for decorators to work."""

# standard library imports
# /

# related third party imports
# /

# local application/library specific imports
from model.build import MODEL_PROVIDER_REGISTRY, build_model
from model.model import build_openai_model, build_anthropic_model, build_ollama_model
