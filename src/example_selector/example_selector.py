"""Module for example selectors."""

# standard library imports
import random

# related third party imports
from langchain_core.example_selectors.base import BaseExampleSelector

# local application/library specific imports
# /


class RandomExampleSelector(BaseExampleSelector):
    def __init__(self, examples: list[dict], k: int) -> None:
        """Initialize the example selector.

        Parameters
        ----------
        examples :  list[dict]
            List of examples
        k : int
            k-shot prompting
        """
        self.examples = examples
        self.k = k

    def add_example(self, example):
        self.examples.append(example)

    def select_examples(self, input_variables):
        # Sample k random examples without replacement
        # (if k > len(examples), returns all examples in random order)
        k = min(self.k, len(self.examples))
        return random.sample(self.examples, k)
