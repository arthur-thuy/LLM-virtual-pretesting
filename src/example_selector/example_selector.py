"""Module for example selectors."""

# standard library imports
import random
from typing import Optional, Literal

# related third party imports
import numpy as np
import pandas as pd
import structlog
from langchain_core.example_selectors.base import BaseExampleSelector
from yacs.config import CfgNode

# local application/library specific imports
from example_selector.build import EXAMPLE_SELECTOR_REGISTRY
from example_selector.utils import (
    assert_correct_option,
    format_skills_miscons,
    get_skills_miscons_from_interactions,
    get_errors_from_interactions,
    format_errors,
    get_error_legend_from_interactions,
    format_error_legend,
    get_skills_miscons_from_interactions_cfe,
    format_snippets_dbe,
)
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
    input_vars = ["student_id", "time"]
    selector = StudentIDRandomExampleSelector(
        examples=examples,
        k=cfg.EXAMPLE_SELECTOR.NUM_EXAMPLES,
        return_value="snippets",
    )
    return (selector, input_vars)


@EXAMPLE_SELECTOR_REGISTRY.register("both_dbe_studentid_random")
def build_both_dbe_studentid_random(
    cfg: CfgNode, examples: list[dict], q_ids_train: list[int]
) -> BaseExampleSelector:
    input_vars = ["student_id", "time"]
    selector = StudentIDRandomExampleSelector(
        examples=examples,
        k=cfg.EXAMPLE_SELECTOR.NUM_EXAMPLES,
        return_value="both_dbe",
    )
    return (selector, input_vars)


@EXAMPLE_SELECTOR_REGISTRY.register("studentid_semantic")
def build_studentid_semantic(
    cfg: CfgNode, examples: list[dict], q_ids_train: list[int]
) -> BaseExampleSelector:
    input_vars = ["student_id", "question_id", "q_text", "time"]
    selector = StudentIDSemanticExampleSelector(
        examples=examples,
        k=cfg.EXAMPLE_SELECTOR.NUM_EXAMPLES,
        embedding=cfg.EXAMPLE_SELECTOR.EMBEDDING,
        namespace=cfg.LOADER.NAME,
        return_value="snippets",
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


@EXAMPLE_SELECTOR_REGISTRY.register("studentid_kc_primary")
def build_studentid_kc_primary(
    cfg: CfgNode, examples: list[dict], q_ids_train: list[int]
) -> BaseExampleSelector:
    input_vars = ["student_id", "primary_kc", "time"]
    selector = StudentIDKCPrimaryExampleSelector(
        examples=examples,
        k=cfg.EXAMPLE_SELECTOR.NUM_EXAMPLES,
    )
    return (selector, input_vars)


@EXAMPLE_SELECTOR_REGISTRY.register("studentid_kc_exact")
def build_studentid_kc_exact(
    cfg: CfgNode, examples: list[dict], q_ids_train: list[int]
) -> BaseExampleSelector:
    input_vars = ["student_id", "question_id", "knowledge_components", "time"]
    selector = StudentIDKCExactExampleSelector(
        examples=examples,
        k=cfg.EXAMPLE_SELECTOR.NUM_EXAMPLES,
        return_value="snippets",
    )
    return (selector, input_vars)


@EXAMPLE_SELECTOR_REGISTRY.register("both_dbe_studentid_kc_exact")
def build_both_dbe_studentid_kc_exact(
    cfg: CfgNode, examples: list[dict], q_ids_train: list[int]
) -> BaseExampleSelector:
    input_vars = ["student_id", "question_id", "knowledge_components", "time"]
    selector = StudentIDKCExactExampleSelector(
        examples=examples,
        k=cfg.EXAMPLE_SELECTOR.NUM_EXAMPLES,
        return_value="both_dbe",
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
        return_value="snippets",
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


@EXAMPLE_SELECTOR_REGISTRY.register("studentlevel_kc_primary")
def build_studentlevel_kc_primary(
    cfg: CfgNode, examples: list[dict], q_ids_train: list[int]
) -> BaseExampleSelector:
    """Build a random example selector based on student levels."""
    input_vars = ["student_level_group", "primary_kc"]
    selector = StudentLevelKCPrimaryExampleSelector(
        examples=examples,
        q_ids_train=q_ids_train,
        k=cfg.EXAMPLE_SELECTOR.NUM_EXAMPLES,
    )
    return (selector, input_vars)


@EXAMPLE_SELECTOR_REGISTRY.register("studentlevel_kc_exact")
def build_studentlevel_kc_exact(
    cfg: CfgNode, examples: list[dict], q_ids_train: list[int]
) -> BaseExampleSelector:
    """Build a random example selector based on student levels."""
    input_vars = ["student_level_group", "knowledge_components", "question_id"]
    selector = StudentLevelKCExactExampleSelector(
        examples=examples,
        q_ids_train=q_ids_train,
        k=cfg.EXAMPLE_SELECTOR.NUM_EXAMPLES,
        return_value="snippets",
    )
    return (selector, input_vars)


@EXAMPLE_SELECTOR_REGISTRY.register("miscon_studentid_random")
def build_miscon_studentid_random(
    cfg: CfgNode, examples: list[dict], q_ids_train: list[int]
) -> BaseExampleSelector:
    input_vars = ["student_id", "time"]
    selector = StudentIDRandomExampleSelector(
        examples=examples,
        k=cfg.EXAMPLE_SELECTOR.NUM_EXAMPLES,
        return_value="misconceptions",
    )
    return (selector, input_vars)


@EXAMPLE_SELECTOR_REGISTRY.register("miscon_studentid_semantic")
def build_miscon_studentid_semantic(
    cfg: CfgNode, examples: list[dict], q_ids_train: list[int]
) -> BaseExampleSelector:
    input_vars = ["student_id", "question_id", "q_text", "time"]
    selector = StudentIDSemanticExampleSelector(
        examples=examples,
        k=cfg.EXAMPLE_SELECTOR.NUM_EXAMPLES,
        embedding=cfg.EXAMPLE_SELECTOR.EMBEDDING,
        namespace=cfg.LOADER.NAME,
        return_value="misconceptions",
    )
    return (selector, input_vars)


@EXAMPLE_SELECTOR_REGISTRY.register("miscon_studentid_kc_exact")
def build_miscon_studentid_kc_exact(
    cfg: CfgNode, examples: list[dict], q_ids_train: list[int]
) -> BaseExampleSelector:
    input_vars = ["student_id", "question_id", "knowledge_components", "time"]
    selector = StudentIDKCExactExampleSelector(
        examples=examples,
        k=cfg.EXAMPLE_SELECTOR.NUM_EXAMPLES,
        return_value="misconceptions",
    )
    return (selector, input_vars)


@EXAMPLE_SELECTOR_REGISTRY.register("miscon_studentlevel_random")
def build_miscon_studentlevel_random(
    cfg: CfgNode, examples: list[dict], q_ids_train: list[int]
) -> BaseExampleSelector:
    """Build a random example selector based on student levels."""
    input_vars = ["student_level_group"]
    selector = StudentLevelRandomExampleSelector(
        examples=examples,
        q_ids_train=q_ids_train,
        k=cfg.EXAMPLE_SELECTOR.NUM_EXAMPLES,
        return_value="misconceptions",
    )
    return (selector, input_vars)


@EXAMPLE_SELECTOR_REGISTRY.register("miscon_studentlevel_kc_exact")
def build_miscon_studentlevel_kc_exact(
    cfg: CfgNode, examples: list[dict], q_ids_train: list[int]
) -> BaseExampleSelector:
    """Build a random example selector based on student levels."""
    input_vars = ["student_level_group", "knowledge_components", "question_id"]
    selector = StudentLevelKCExactExampleSelector(
        examples=examples,
        q_ids_train=q_ids_train,
        k=cfg.EXAMPLE_SELECTOR.NUM_EXAMPLES,
        return_value="misconceptions",
    )
    return (selector, input_vars)


@EXAMPLE_SELECTOR_REGISTRY.register("errors_studentlevel_random")
def build_errors_studentlevel_random(
    cfg: CfgNode, examples: list[dict], q_ids_train: list[int]
) -> BaseExampleSelector:
    """Build a random example selector based on student levels."""
    input_vars = ["student_level_group"]
    selector = StudentLevelRandomExampleSelector(
        examples=examples,
        q_ids_train=q_ids_train,
        k=cfg.EXAMPLE_SELECTOR.NUM_EXAMPLES,
        return_value="errors",
    )
    return (selector, input_vars)


@EXAMPLE_SELECTOR_REGISTRY.register("both_cfe_studentlevel_random")
def build_both_cfe_studentlevel_random(
    cfg: CfgNode, examples: list[dict], q_ids_train: list[int]
) -> BaseExampleSelector:
    """Build a random example selector based on student levels."""
    input_vars = ["student_level_group"]
    selector = StudentLevelRandomExampleSelector(
        examples=examples,
        q_ids_train=q_ids_train,
        k=cfg.EXAMPLE_SELECTOR.NUM_EXAMPLES,
        return_value="both_cfe",
    )
    return (selector, input_vars)


@EXAMPLE_SELECTOR_REGISTRY.register("both_dbe_studentlevel_random")
def build_both_dbe_studentlevel_random(
    cfg: CfgNode, examples: list[dict], q_ids_train: list[int]
) -> BaseExampleSelector:
    """Build a random example selector based on student levels."""
    input_vars = ["student_level_group"]
    selector = StudentLevelRandomExampleSelector(
        examples=examples,
        q_ids_train=q_ids_train,
        k=cfg.EXAMPLE_SELECTOR.NUM_EXAMPLES,
        return_value="both_dbe",
    )
    return (selector, input_vars)


@EXAMPLE_SELECTOR_REGISTRY.register("both_dbe_studentlevel_kc_exact")
def build_both_dbe_studentlevel_kc_exact(
    cfg: CfgNode, examples: list[dict], q_ids_train: list[int]
) -> BaseExampleSelector:
    """Build a random example selector based on student levels."""
    input_vars = ["student_level_group", "knowledge_components", "question_id"]
    selector = StudentLevelKCExactExampleSelector(
        examples=examples,
        q_ids_train=q_ids_train,
        k=cfg.EXAMPLE_SELECTOR.NUM_EXAMPLES,
        return_value="both_dbe",
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

    def __init__(
        self,
        examples: list[dict],
        k: int,
        return_value: Literal["snippets", "misconceptions", "errors"],
    ) -> None:
        """Initialize the example selector.

        Parameters
        ----------
        examples :  list[dict]
            List of examples
        k : int
            k-shot prompting
        return_miscons : bool
            Whether to return misconceptions or not.
        """
        self.examples = examples
        self.k = k
        self.return_value = return_value

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
        if len(student_interactions) == 0:
            logger.warning(
                f"No interactions found for student {student_id} before time {time}"
            )
            return []

        # randomly select from questions
        k = min(self.k, len(student_interactions))
        interactions_selected = random.sample(student_interactions, k)

        if self.return_value == "snippets":
            # if we don't want to return misconceptions, just return the interactions
            return interactions_selected
        elif self.return_value == "misconceptions":
            # assert that for each interaction, correct option is 1
            assert_correct_option(interactions_selected)

            skills, misconceptions = get_skills_miscons_from_interactions(
                interactions_selected
            )
            text = format_skills_miscons(skills, misconceptions)

            # NOTE: return list of length 1 with dict with key "skills_misconceptions"
            return [{"skills_misconceptions": text}]
        elif self.return_value == "both_dbe":
            # assert that for each interaction, correct option is 1
            assert_correct_option(interactions_selected)

            skills, misconceptions = get_skills_miscons_from_interactions(
                interactions_selected
            )
            text_miscons = format_skills_miscons(skills, misconceptions)

            text_snippet = format_snippets_dbe(interactions_selected)

            return [{"snippets": text_snippet, "misconceptions": text_miscons}]


class StudentIDSemanticExampleSelector(BaseExampleSelector):
    """Filter examples of the same student_id and select based on semantic similarity."""  # noqa

    def __init__(
        self,
        examples: list,
        k: int,
        embedding: str,
        namespace: str,
        return_value: Literal["snippets", "misconceptions", "errors"],
    ) -> None:
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
        return_miscons : bool
            Whether to return misconceptions or not.
        """
        self.examples = examples
        self.k = k
        self.return_value = return_value

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

        if len(student_interactions) == 0:
            logger.warning(f"No interactions found for student {student_id}")
            return []

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
            for interact in student_interactions
            if interact["question_id"] in question_ids_selected
        ]

        if self.return_value == "snippets":
            # if we don't want to return misconceptions, just return the interactions
            return interactions_selected
        elif self.return_value == "misconceptions":
            # assert that for each interaction, correct option is 1
            assert_correct_option(interactions_selected)

            skills, misconceptions = get_skills_miscons_from_interactions(
                interactions_selected
            )
            text = format_skills_miscons(skills, misconceptions)

            # NOTE: return list of length 1 with dict with key "skills_misconceptions"
            return [{"skills_misconceptions": text}]


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


class StudentIDKCPrimaryExampleSelector(BaseExampleSelector):
    """Filter on student_id and primary KC."""

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
        """Select examples based on knowledge component.

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
        kc = input_variables["primary_kc"]
        time = input_variables["time"]

        # find all questions answered by this student on the KC, before the current time
        student_interactions = [
            interact
            for interact in self.examples
            if interact["student_id"] == student_id
            and kc == interact["primary_kc"]
            and interact["time"] <= time
        ]

        if len(student_interactions) == 0:
            logger.warning(
                f"No interactions found for student {student_id} with primary KC {kc}"
            )
            return []

        # randomly select from questions, not sampling the same question multiple times
        k = min(self.k, len(student_interactions))

        interactions_selected = []
        question_ids_selected = []
        for _ in range(k):
            # select random interaction for each question on the KC
            # + no duplicate questions
            interactions_filtered = [
                interact
                for interact in student_interactions
                if interact["question_id"] not in question_ids_selected
            ]
            selected_interaction = random.sample(interactions_filtered, 1)[0]
            interactions_selected.append(selected_interaction)
            question_ids_selected.append(selected_interaction["question_id"])

        if len(interactions_selected) < self.k:
            # if we selected fewer interactions than requested, log a warning
            logger.warning(
                "Selected fewer interactions than requested",
                requested=self.k,
                selected=k,
            )

        return interactions_selected


class StudentIDKCExactExampleSelector(BaseExampleSelector):
    """Filter on student_id and entire list of KCs."""

    def __init__(
        self,
        examples: list,
        k: int,
        return_value: Literal["snippets", "misconceptions", "errors"],
    ) -> None:
        """Initialize the example selector.

        Parameters
        ----------
        examples :  list[dict]
            List of examples
        k : int
            k-shot prompting
        return_miscons : bool
            Whether to return misconceptions or not.
        """
        self.examples = examples
        self.k = k
        self.return_value = return_value

    def add_example(self, example: list) -> None:
        self.examples.append(example)

    def select_examples(self, input_variables: dict) -> list[dict[str, str]]:
        """Select examples based on knowledge component.

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
        kc = input_variables["knowledge_components"]
        time = input_variables["time"]

        # print(f"{kc=}")  # TODO: remove

        # find all interactions of this student, before the current time
        student_interactions = [
            interact
            for interact in self.examples
            if interact["student_id"] == student_id and interact["time"] <= time
        ]

        if len(student_interactions) == 0:
            logger.warning(f"No interactions found for student {student_id}")
            return []

        q_answered = set([interact["question_id"] for interact in student_interactions])
        q_answered.discard(question_id)  # remove current question_id

        # compute jaccard similarity for each question_id in q_answered
        q_jaccard_sim = {}
        for interact in student_interactions:
            if (
                interact["question_id"] in q_answered
                and interact["question_id"] not in q_jaccard_sim
            ):
                # compute Jaccard similarity
                jaccard_sim = compute_jaccard_similarity(
                    other_kcs=interact["knowledge_components"],
                    target_kcs=kc,
                )
                q_jaccard_sim[interact["question_id"]] = jaccard_sim
        # convert to df
        df = pd.DataFrame(q_jaccard_sim.items(), columns=["question_id", "jaccard_sim"])
        # sort questions by Jaccard similarity, in descending order
        # NOTE: if same value, select randomly
        df_sorted = df.sample(frac=1).sort_values(by="jaccard_sim", ascending=False)

        # randomly select from questions, not sampling the same question multiple times
        k = min(self.k, len(df_sorted.index))

        # find interactions of selected question_ids and student_id
        question_ids_selected = df_sorted["question_id"].tolist()[:k]
        interactions_selected = [
            interact
            for interact in student_interactions
            if interact["question_id"] in question_ids_selected
        ]

        # print(
        #     f"{[interact['knowledge_components'] for interact in interactions_selected]=}"  # noqa
        # )  # TODO: remove

        if len(interactions_selected) < self.k:
            # if we selected fewer interactions than requested, log a warning
            logger.warning(
                "Selected fewer interactions than requested",
                requested=self.k,
                selected=k,
            )

        if self.return_value == "snippets":
            # if we don't want to return misconceptions, just return the interactions
            return interactions_selected
        elif self.return_value == "misconceptions":
            # assert that for each interaction, correct option is 1
            assert_correct_option(interactions_selected)

            skills, misconceptions = get_skills_miscons_from_interactions(
                interactions_selected
            )
            text = format_skills_miscons(skills, misconceptions)

            # NOTE: return list of length 1 with dict with key "skills_misconceptions"
            return [{"skills_misconceptions": text}]
        elif self.return_value == "both_dbe":
            # assert that for each interaction, correct option is 1
            assert_correct_option(interactions_selected)

            skills, misconceptions = get_skills_miscons_from_interactions(
                interactions_selected
            )
            text_miscons = format_skills_miscons(skills, misconceptions)

            text_snippet = format_snippets_dbe(interactions_selected)

            return [{"snippets": text_snippet, "misconceptions": text_miscons}]


def compute_jaccard_similarity(other_kcs: list[int], target_kcs: list[int]) -> float:
    """
    Compute Jaccard similarity between two sets of knowledge components.
    """
    set_row = set(other_kcs)
    set_target = set(target_kcs)

    intersection = len(set_row.intersection(set_target))
    union = len(set_row.union(set_target))

    if union == 0:
        return 0.0  # Avoid division by zero
    return intersection / union


class StudentLevelRandomExampleSelector(BaseExampleSelector):
    """Filter examples of the same student level and randomly select."""

    def __init__(
        self,
        examples: list[dict],
        k: int,
        return_value: Literal["snippets", "misconceptions", "errors"],
        q_ids_train: Optional[list[int]] = None,
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
        return_miscons : bool
            Whether to return misconceptions or not.
        """
        self.examples = examples
        self.q_ids_train = q_ids_train
        self.k = k
        self.return_value = return_value

    def add_example(self, example):
        self.examples.append(example)

    def select_examples(self, input_variables):
        # student_level_group of target student
        student_level_group = input_variables["student_level_group"]

        # find all interactions of this student level group
        if self.q_ids_train is None:
            student_interactions = [
                interact
                for interact in self.examples
                if interact["student_level_group"] == student_level_group
            ]
        else:
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

        interactions_selected = []
        question_ids_selected = []
        for _ in range(k):
            # select random interaction for each question_id
            interactions_filtered = [
                interact
                for interact in student_interactions
                if interact["question_id"] not in question_ids_selected
            ]
            selected_interaction = random.sample(interactions_filtered, 1)[0]
            interactions_selected.append(selected_interaction)
            question_ids_selected.append(selected_interaction["question_id"])

        if len(interactions_selected) < self.k:
            # if we selected fewer interactions than requested, log a warning
            logger.warning(
                "Selected fewer interactions than requested",
                requested=self.k,
                selected=k,
            )

        if self.return_value == "snippets":
            # if we don't want to return misconceptions, just return the interactions
            return interactions_selected
        elif self.return_value == "misconceptions":
            # assert that for each interaction, correct option is 1
            assert_correct_option(interactions_selected)

            skills, misconceptions = get_skills_miscons_from_interactions(
                interactions_selected
            )
            text = format_skills_miscons(skills, misconceptions)

            # NOTE: return list of length 1 with dict with key "skills_misconceptions"
            return [{"skills_misconceptions": text}]
        elif self.return_value == "errors":
            # get errors from an open-ended response
            errors = get_errors_from_interactions(interactions_selected)
            text = format_errors(errors)
            # NOTE: return list of length 1 with dict with key "errors"
            return [{"skills_misconceptions": text}]  # NOTE: need this key for prompt
        elif self.return_value == "both_cfe":
            assert len(interactions_selected) == 1  # TODO: relax this for DBE-KT22
            interaction_selected = interactions_selected[0]
            # get skills and misconceptions from mapping
            skills, misconceptions = get_skills_miscons_from_interactions_cfe(
                interaction_selected
            )
            text_miscons = format_skills_miscons(skills, misconceptions)

            error_legend = get_error_legend_from_interactions(interaction_selected)
            text_error_legend = format_error_legend(error_legend)
            text_snippet = text_error_legend + "\n\n" + interaction_selected["output"]

            return [{"snippets": text_snippet, "misconceptions": text_miscons}]
        elif self.return_value == "both_dbe":
            # assert that for each interaction, correct option is 1
            assert_correct_option(interactions_selected)

            skills, misconceptions = get_skills_miscons_from_interactions(
                interactions_selected
            )
            text_miscons = format_skills_miscons(skills, misconceptions)

            text_snippet = format_snippets_dbe(interactions_selected)

            return [{"snippets": text_snippet, "misconceptions": text_miscons}]


class StudentLevelSemanticExampleSelector(BaseExampleSelector):
    """Filter examples of the same student level and select based on semantic similarity."""  # noqa

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
                for interact in student_interactions
                if interact["question_id"] == question_id
            ]
            selected_interaction = random.sample(interactions_for_question, 1)[0]
            interactions_selected.append(selected_interaction)

        assert len(interactions_selected) == len(question_ids_selected)

        if len(interactions_selected) < self.k:
            # if we selected fewer interactions than requested, log a warning
            logger.warning(
                "Selected fewer interactions than requested",
                requested=self.k,
                selected=len(interactions_selected),
            )

        return interactions_selected


class StudentLevelKCPrimaryExampleSelector(BaseExampleSelector):
    """Filter examples of the same student level and randomly select from that primary KC."""  # noqa

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
        kc = input_variables["primary_kc"]

        # find all interactions of this student level group
        student_interactions = [
            interact
            for interact in self.examples
            if interact["student_level_group"] == student_level_group
            and interact["question_id"] in self.q_ids_train
            and kc == interact["primary_kc"]
        ]
        if len(student_interactions) == 0:
            logger.warning(
                f"No interactions found for student level group {student_level_group} "
                f"with KC {kc}"
            )
            return []

        # randomly select from questions
        k = min(self.k, len(student_interactions))

        interactions_selected = []
        question_ids_selected = []
        for _ in range(k):
            # select random interaction for each question_id
            interactions_filtered = [
                interact
                for interact in student_interactions
                if interact["question_id"] not in question_ids_selected
            ]
            selected_interaction = random.sample(interactions_filtered, 1)[0]
            interactions_selected.append(selected_interaction)
            question_ids_selected.append(selected_interaction["question_id"])

        if len(interactions_selected) < self.k:
            # if we selected fewer interactions than requested, log a warning
            logger.warning(
                "Selected fewer interactions than requested",
                requested=self.k,
                selected=k,
            )

        return interactions_selected


class StudentLevelKCExactExampleSelector(BaseExampleSelector):
    """Filter examples of the same student level and randomly select from that exact KC."""  # noqa

    def __init__(
        self,
        examples: list[dict],
        q_ids_train: list[int],
        k: int,
        return_value: Literal["snippets", "misconceptions", "errors"],
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
        return_miscons : bool
            Whether to return misconceptions or not.
        """
        self.examples = examples
        self.q_ids_train = q_ids_train
        self.k = k
        self.return_value = return_value

    def add_example(self, example):
        self.examples.append(example)

    def select_examples(self, input_variables):
        # student_level_group of target student
        student_level_group = input_variables["student_level_group"]
        question_id = input_variables["question_id"]
        kc = input_variables["knowledge_components"]

        # print(f"{kc=}")  # TODO: remove

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

        q_answered = set([interact["question_id"] for interact in student_interactions])
        q_answered.discard(question_id)  # remove current question_id

        # compute jaccard similarity for each question_id in q_answered
        q_jaccard_sim = {}
        for interact in student_interactions:
            if (
                interact["question_id"] in q_answered
                and interact["question_id"] not in q_jaccard_sim
            ):
                # compute Jaccard similarity
                jaccard_sim = compute_jaccard_similarity(
                    other_kcs=interact["knowledge_components"],
                    target_kcs=kc,
                )
                q_jaccard_sim[interact["question_id"]] = jaccard_sim
        # convert to df
        df = pd.DataFrame(q_jaccard_sim.items(), columns=["question_id", "jaccard_sim"])
        # sort questions by Jaccard similarity, in descending order
        # NOTE: if same value, select randomly
        df_sorted = df.sample(frac=1).sort_values(by="jaccard_sim", ascending=False)

        # randomly select from questions, not sampling the same question multiple times
        k = min(self.k, len(df_sorted.index))

        # find interactions of selected question_ids and student_id
        question_ids_selected = df_sorted["question_id"].tolist()[:k]

        # find interactions of selected question_ids and student_level_group
        interactions_selected = []
        for question_id in question_ids_selected:
            # select random interaction for each question_id
            interactions_for_question = [
                interact
                for interact in student_interactions
                if interact["question_id"] == question_id
            ]
            selected_interaction = random.sample(interactions_for_question, 1)[0]
            interactions_selected.append(selected_interaction)

        assert len(interactions_selected) == len(question_ids_selected)

        # print(
        #     f"{[interact['knowledge_components'] for interact in interactions_selected]=}"  # noqa
        # )  # TODO: remove

        if len(interactions_selected) < self.k:
            # if we selected fewer interactions than requested, log a warning
            logger.warning(
                "Selected fewer interactions than requested",
                requested=self.k,
                selected=k,
            )

        if self.return_value == "snippets":
            # if we don't want to return misconceptions, just return the interactions
            return interactions_selected
        elif self.return_value == "misconceptions":
            # assert that for each interaction, correct option is 1
            assert_correct_option(interactions_selected)

            skills, misconceptions = get_skills_miscons_from_interactions(
                interactions_selected
            )
            text = format_skills_miscons(skills, misconceptions)

            # NOTE: return list of length 1 with dict with key "skills_misconceptions"
            return [{"skills_misconceptions": text}]
        elif self.return_value == "both_dbe":
            # assert that for each interaction, correct option is 1
            assert_correct_option(interactions_selected)

            skills, misconceptions = get_skills_miscons_from_interactions(
                interactions_selected
            )
            text_miscons = format_skills_miscons(skills, misconceptions)

            text_snippet = format_snippets_dbe(interactions_selected)

            return [{"snippets": text_snippet, "misconceptions": text_miscons}]
