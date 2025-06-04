"""Module for example selectors."""

# standard library imports
import random

# related third party imports
import structlog
import numpy as np
from langchain_core.example_selectors.base import BaseExampleSelector
from yacs.config import CfgNode

# local application/library specific imports
from example_selector.build import EXAMPLE_SELECTOR_REGISTRY
from tools.vector_db import get_vector_store

logger = structlog.get_logger()


@EXAMPLE_SELECTOR_REGISTRY.register("random")
def build_random(
    cfg: CfgNode, examples: list[dict], q_ids_train: list[int]
) -> BaseExampleSelector:
    input_vars = []
    selector = RandomExampleSelector(
        examples=examples,
        k=cfg.EXAMPLE_SELECTOR.NUM_EXAMPLES,
    )
    return (selector, input_vars)


@EXAMPLE_SELECTOR_REGISTRY.register("studentid_random")
def build_studentid_random(
    cfg: CfgNode, examples: list[dict], q_ids_train: list[int]
) -> BaseExampleSelector:
    input_vars = ["student_id"]
    selector = StudentIDRandomExampleSelector(
        examples=examples,
        k=cfg.EXAMPLE_SELECTOR.NUM_EXAMPLES,
    )
    return (selector, input_vars)


@EXAMPLE_SELECTOR_REGISTRY.register("studentid_semantic")
def build_studentid_semantic(
    cfg: CfgNode, examples: list[dict], q_ids_train: list[int]
) -> BaseExampleSelector:
    input_vars = ["student_id", "question_id", "q_text"]
    selector = StudentIDSemanticExampleSelector(
        examples=examples,
        k=cfg.EXAMPLE_SELECTOR.NUM_EXAMPLES,
        embedding=cfg.EXAMPLE_SELECTOR.EMBEDDING,
        namespace=cfg.LOADER.NAME,
    )
    return (selector, input_vars)


@EXAMPLE_SELECTOR_REGISTRY.register("studentid_recency")
def build_studentid_recency(
    cfg: CfgNode, examples: list[dict], q_ids_train: list[int]
) -> BaseExampleSelector:
    input_vars = ["student_id", "time"]
    selector = StudentIDRecencyExampleSelector(
        examples=examples,
        k=cfg.EXAMPLE_SELECTOR.NUM_EXAMPLES,
    )
    return (selector, input_vars)


@EXAMPLE_SELECTOR_REGISTRY.register("studentlevel_random")
def build_studentlevel_random(
    cfg: CfgNode, examples: list[dict], q_ids_train: list[int]
) -> BaseExampleSelector:
    """Build a random example selector based on student levels."""
    input_vars = ["student_level_group"]
    selector = StudentLevelRandomExampleSelector(
        examples=examples,
        q_ids_train=q_ids_train,
        k=cfg.EXAMPLE_SELECTOR.NUM_EXAMPLES,
    )
    return (selector, input_vars)


@EXAMPLE_SELECTOR_REGISTRY.register("studentlevel_semantic")
def build_studentlevel_semantic(
    cfg: CfgNode, examples: list[dict], q_ids_train: list[int]
) -> BaseExampleSelector:
    """Build a semantic example selector based on student levels."""
    input_vars = ["student_level_group", "question_id", "q_text"]
    selector = StudentLevelSemanticExampleSelector(
        examples=examples,
        q_ids_train=q_ids_train,
        k=cfg.EXAMPLE_SELECTOR.NUM_EXAMPLES,
        embedding=cfg.EXAMPLE_SELECTOR.EMBEDDING,
        namespace=cfg.LOADER.NAME,
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
        time = input_variables["time"]

        # find all interactions of this student, before the current time
        student_interactions = [
            interact
            for interact in self.examples
            if interact["student_id"] == student_id and interact["time"] <= time
        ]
        # NOTE: there can be max 1 interaction per question_id and student_id

        # randomly select from questions
        k = min(self.k, len(student_interactions))
        interactions_selected = random.sample(student_interactions, k)

        return interactions_selected


class StudentIDSemanticExampleSelector(BaseExampleSelector):
    """Filter examples of the same student_id and select based on semantic similarity."""

    def __init__(self, examples: list, k: int, embedding: str, namespace: str) -> None:
        """Initialize the example selector.

        Parameters
        ----------
        examples :  list[dict]
            List of examples
        k : int
            k-shot prompting
        embedding : str
            The name of the embedding model.
        namespace : str
            The namespace of the Pinecone index.
        """
        self.examples = examples
        self.k = k

        self.vectorstore = get_vector_store(
            index_name=embedding,
            embedding_name=embedding,
            namespace=namespace,
        )

    def add_example(self, example: list) -> None:
        self.examples.append(example)

    def select_examples(self, input_variables: dict) -> list[dict[str, str]]:
        """Select examples based on semantic similarity.

        Parameters
        ----------
        input_variables : dict[str, str]
            A dict containing info about a single observation.

        Returns
        -------
        list[dict[str, str]]
            The selected examples.
        """
        # information of target observation
        student_id = input_variables["student_id"]
        question_id = input_variables["question_id"]
        q_text = input_variables["q_text"]
        time = input_variables["time"]

        # find all questions answered by this student, before the current time
        student_interactions = [
            interact
            for interact in self.examples
            if interact["student_id"] == student_id and interact["time"] <= time
        ]
        q_answered = set([interact["question_id"] for interact in student_interactions])
        q_answered = list(
            map(str, q_answered - {question_id})
        )  # NOTE: remove current question_id

        # semantic search on question text
        results = self.vectorstore.similarity_search(
            query=q_text,
            k=self.k,
            filter={"question_id": {"$in": q_answered}},
        )
        question_ids_selected = list(
            map(int, [res.metadata["question_id"] for res in results])
        )

        # find interactions of selected question_ids and student_id
        interactions_selected = [
            interact
            for interact in self.examples
            if (
                interact["question_id"] in question_ids_selected
                and interact["student_id"] == student_id
            )
        ]
        return interactions_selected
        # NOTE: can decide to only return input and output
        # return [
        #     {"input": interact["input"], "output": interact["output"]}
        #     for interact in interactions_selected
        # ]


class StudentIDRecencyExampleSelector(BaseExampleSelector):
    """Filter examples of the same student_id and select based on recency."""

    def __init__(self, examples: list, k: int) -> None:
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

    def add_example(self, example: list) -> None:
        self.examples.append(example)

    def select_examples(self, input_variables: dict) -> list[dict[str, str]]:
        """Select examples based on semantic similarity.

        Parameters
        ----------
        input_variables : dict[str, str]
            A dict containing info about a single observation.

        Returns
        -------
        list[dict[str, str]]
            The selected examples.
        """
        # information of target observation
        student_id = input_variables["student_id"]
        time = input_variables["time"]

        # find all questions answered by this student
        student_interactions = [
            interact
            for interact in self.examples
            if interact["student_id"] == student_id
        ]
        interact_times = [interact["time"] for interact in student_interactions]
        interact_times = np.array(interact_times)
        diff = time - interact_times

        # Get indices of interactions sorted by recency (smallest time difference first)
        # Only consider interactions that happened before the current time (diff > 0)
        valid_indices = np.where(diff > 0)[0]
        if len(valid_indices) == 0:
            # No previous interactions found
            logger.warning("No previous interactions found for this student")

        # Sort valid indices by time difference (ascending)
        sorted_indices = valid_indices[np.argsort(diff[valid_indices])]

        # Select the k most recent interactions
        recent_k_indices = sorted_indices[: min(self.k, len(sorted_indices))]

        # Get the selected interactions
        selected_interactions = [student_interactions[i] for i in recent_k_indices]

        return selected_interactions


class StudentLevelRandomExampleSelector(BaseExampleSelector):
    """Filter examples of the same student level and randomly select."""

    def __init__(self, examples: list[dict], q_ids_train: list[int], k: int) -> None:
        """Initialize the example selector.

        Parameters
        ----------
        examples :  list[dict]
            List of examples
        q_ids_train : list[int]
            List of question IDs in the training set.
        k : int
            k-shot prompting
        """
        self.examples = examples
        self.q_ids_train = q_ids_train
        self.k = k

    def add_example(self, example):
        self.examples.append(example)

    def select_examples(self, input_variables):
        # student_level_group of target student
        student_level_group = input_variables["student_level_group"]

        # find all interactions of this student level group
        student_interactions = [
            interact
            for interact in self.examples
            if interact["student_level_group"] == student_level_group
            and interact["question_id"] in self.q_ids_train
        ]
        if len(student_interactions) == 0:
            logger.warning(
                f"No interactions found for student level group {student_level_group}"
            )
            return []

        # randomly select from questions
        k = min(self.k, len(student_interactions))
        interactions_selected = random.sample(student_interactions, k)

        return interactions_selected


class StudentLevelSemanticExampleSelector(BaseExampleSelector):
    """Filter examples of the same student level and select based on semantic similarity."""

    def __init__(
        self,
        examples: list,
        q_ids_train: list[int],
        k: int,
        embedding: str,
        namespace: str,
    ) -> None:
        """Initialize the example selector.

        Parameters
        ----------
        examples :  list[dict]
            List of examples
        q_ids_train : list[int]
            List of question IDs in the training set.
        k : int
            k-shot prompting
        embedding : str
            The name of the embedding model.
        namespace : str
            The namespace of the Pinecone index.
        """
        self.examples = examples
        self.q_ids_train = q_ids_train
        self.k = k

        self.vectorstore = get_vector_store(
            index_name=embedding,
            embedding_name=embedding,
            namespace=namespace,
        )

    def add_example(self, example: list) -> None:
        self.examples.append(example)

    def select_examples(self, input_variables: dict) -> list[dict[str, str]]:
        """Select examples based on semantic similarity.

        Parameters
        ----------
        input_variables : dict[str, str]
            A dict containing info about a single observation.

        Returns
        -------
        list[dict[str, str]]
            The selected examples.
        """
        # information of target observation
        student_level_group = input_variables["student_level_group"]
        q_text = input_variables["q_text"]

        # find all questions answered by this student level group
        student_interactions = [
            interact
            for interact in self.examples
            if interact["student_level_group"] == student_level_group
            and interact["question_id"] in self.q_ids_train
        ]

        if len(student_interactions) == 0:
            logger.warning(
                f"No interactions found for student level group {student_level_group}"
            )
            return []

        # semantic search on question text
        results = self.vectorstore.similarity_search(
            query=q_text,
            k=self.k,
            filter={
                "question_id": {
                    "$in": [
                        interact["question_id"] for interact in student_interactions
                    ]
                }
            },
        )

        question_ids_selected = list(
            map(int, [res.metadata["question_id"] for res in results])
        )

        # find interactions of selected question_ids and student_level_group
        interactions_selected = []
        for question_id in question_ids_selected:
            # select random interaction for each question_id
            interactions_for_question = [
                interact
                for interact in self.examples
                if (
                    interact["question_id"] == question_id
                    and interact["student_level_group"] == student_level_group
                )
            ]
            selected_interaction = random.sample(interactions_for_question)
            interactions_selected.append(selected_interaction)

        assert len(interactions_selected) == len(question_ids_selected)

        return interactions_selected
