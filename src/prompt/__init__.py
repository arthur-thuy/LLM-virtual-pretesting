"""Execute at import time, needed for decorators to work."""

# standard library imports
# /

# related third party imports
# /

# local application/library specific imports
from prompt.build import PROMPT_REGISTRY, build_prompt
from prompt.prompt import build_student_pepper_level_context
