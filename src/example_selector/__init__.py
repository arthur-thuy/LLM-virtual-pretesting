"""Execute at import time, needed for decorators to work."""

# standard library imports
# /

# related third party imports
# /

# local application/library specific imports
from example_selector.build import build_example_selector
from example_selector.example_selector import (
    RandomExampleSelector,
    StudentIDRandomExampleSelector,
)
