"""Build file for system prompt."""

# standard library imports
from typing import Optional

# related third party imports
import structlog
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)
from pydantic import BaseModel
from yacs.config import CfgNode

# local application/library specific imports
from example_selector.build import build_example_selector
from tools.constants import PROMPT_INFO
from tools.registry import Registry

PROMPT_REGISTRY = Registry()

logger = structlog.get_logger(__name__)


def build_prompt(
    cfg: CfgNode,
    examples: list[dict],
    struc_output: BaseModel,
    student_scale_str: str,
    q_ids_train: Optional[list[int]] = None,
) -> ChatPromptTemplate:
    """Build the prompt.

    Parameters
    ----------
    cfg : CfgNode
        Config node.
    examples : list[dict]
        List of examples.
    struc_output : BaseModel
        Pydantic model for structured output.
    student_scale_str : str
        String representation of the student scale.
    q_ids_train : Optional[list[int]], optional
        List of question IDs for training interactions, by default None

    Returns
    -------
    ChatPromptTemplate
        Prompt object
    """
    logger.info(
        "Building prompt",
        system_prompt=cfg.PROMPT.NAME,
        native_structured_output=cfg.MODEL.NATIVE_STRUCTURED_OUTPUT,
        example_selector=cfg.EXAMPLE_SELECTOR.NAME,
        num_examples=cfg.EXAMPLE_SELECTOR.NUM_EXAMPLES,
    )

    # Set up a parser (not used if model supports structured output)
    parser = PydanticOutputParser(pydantic_object=struc_output)

    # build few_shot_prompt
    if cfg.EXAMPLE_SELECTOR.NUM_EXAMPLES == 0:
        # If no examples are used, we can skip the example selector
        few_shot_prompt = None
    else:
        example_selector, input_vars = build_example_selector(
            cfg, examples=examples, q_ids_train=q_ids_train
        )
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            # The input variables select the values to pass to the example_selector
            input_variables=input_vars,
            example_selector=example_selector,
            # each example is 2 messages: 1 human, 1 AI
            example_prompt=ChatPromptTemplate.from_messages(
                [("human", "{input}"), ("ai", "{output}")]
            ),
        )

    # get the messages list
    messages = PROMPT_REGISTRY[cfg.PROMPT.NAME](
        few_shot_prompt=few_shot_prompt,
        native_str_output=cfg.MODEL.NATIVE_STRUCTURED_OUTPUT,
    )
    # Create the template from the messages list
    final_prompt = ChatPromptTemplate.from_messages(messages).partial(
        format_instructions=parser.get_format_instructions(),
        exam_type=PROMPT_INFO[cfg.LOADER.NAME]["exam_type"],
        exam_type_luca_emnlp=PROMPT_INFO[cfg.LOADER.NAME]["exam_type_luca_emnlp"],
        student_scale=student_scale_str,
    )
    # NOTE: unused variables are simply ignored

    return final_prompt, parser
