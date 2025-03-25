"""Execute at import time, needed for decorators to work."""

# standard library imports
# /

# related third party imports
# /

# local application/library specific imports
from prompt.build import build_prompt, SYSTEM_PROMPT_REGISTRY
from prompt.system_prompt import build_A, build_B
