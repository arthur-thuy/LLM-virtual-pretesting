"""Module for example selectors."""

# standard library imports
import random

# related third party imports
from langchain_core.example_selectors.base import BaseExampleSelector
from yacs.config import CfgNode

# local application/library specific imports
from example_selector.build import EXAMPLE_SELECTOR_REGISTRY


@EXAMPLE_SELECTOR_REGISTRY.register("random")
def build_random(
    example_selector_cfg: CfgNode, examples: list[dict]
) -> BaseExampleSelector:
    input_vars = []
    selector = RandomExampleSelector(
        examples=examples,
        k=example_selector_cfg.NUM_EXAMPLES,
    )
    return (selector, input_vars)


@EXAMPLE_SELECTOR_REGISTRY.register("studentid_random")
def build_studentid_random(
    example_selector_cfg: CfgNode, examples: list[dict]
) -> BaseExampleSelector:
    input_vars = ["student_id"]
    selector = StudentIDRandomExampleSelector(
        examples=examples,
        k=example_selector_cfg.NUM_EXAMPLES,
    )
    return (selector, input_vars)


class RandomExampleSelector(BaseExampleSelector):
    """Randomly select examples."""
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


class StudentIDRandomExampleSelector(BaseExampleSelector):
    """Filter examples of the same student_id and randomly select."""
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
