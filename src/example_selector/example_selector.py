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

    def select_examples(self, input_variables) -> list[dict]:
        # Sample k random examples without replacement
        # (if k > len(examples), returns all examples in random order)
        k = min(self.k, len(self.examples))
        return random.sample(self.examples, k)


class StudentIDExampleSelector(BaseExampleSelector):
    def __init__(self, examples: list[dict], k: int):
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
        # student_id of target student
        student_id = input_variables["student_id"]

        match_idx = []
        # Iterate through each example
        for idx, example in enumerate(self.examples):
            if example["student_id"] == student_id:
                match_idx.append(idx)

        # Sample k random examples without replacement
        # (if k > len(examples), returns all examples in random order)
        k = min(self.k, len(match_idx))
        idx_out = random.sample(match_idx, k)
        return [self.examples[i] for i in idx_out]
