"""Execute at import time, needed for decorators to work."""

# standard library imports
# /

# related third party imports
# /

# local application/library specific imports
from prompt.build import build_prompt, PROMPT_REGISTRY
from prompt.prompt import build_replicate_student_tomato
