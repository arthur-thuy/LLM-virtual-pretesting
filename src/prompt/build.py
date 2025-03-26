"""Build file for system prompt."""

# standard library imports
# /

# related third party imports
import structlog
from yacs.config import CfgNode
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel

# local application/library specific imports
from tools.registry import Registry
from example_selector.build import build_example_selector
from tools.constants import PROMPT_INFO

SYSTEM_PROMPT_REGISTRY = Registry()

logger = structlog.get_logger(__name__)


def build_prompt(cfg: CfgNode, examples: list[dict], str_output: BaseModel) -> ChatPromptTemplate:
    """Build the prompt.

    Parameters
    ----------
    cfg : CfgNode
        Config node.
    examples : list[dict]
        List of examples.
    str_output : BaseModel
        Pydantic model for structured output.

    Returns
    -------
    ChatPromptTemplate
        Prompt object
    """
    logger.info(
        "Building prompt",
        system_prompt=cfg.SYSTEM_PROMPT.NAME,
        native_structured_output=cfg.MODEL.NATIVE_STRUCTURED_OUTPUT,
        example_selector=cfg.EXAMPLE_SELECTOR.NAME,
        num_examples=cfg.EXAMPLE_SELECTOR.NUM_EXAMPLES,
    )
    # build system prompt string
    system_prompt_str = SYSTEM_PROMPT_REGISTRY[cfg.SYSTEM_PROMPT.NAME]()
    # Set up a parser (not used if model supports structured output)
    parser = PydanticOutputParser(pydantic_object=str_output)
    if not cfg.MODEL.NATIVE_STRUCTURED_OUTPUT:
        system_prompt_str += "Wrap the output in `json` tags\n{format_instructions}"

    # build few_shot_prompt
    example_selector, input_vars = build_example_selector(
        cfg.EXAMPLE_SELECTOR, examples=examples
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

    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt_str),
            few_shot_prompt,
            ("human", "{input}"),
        ]
    ).partial(
        format_instructions=parser.get_format_instructions(),
        exam_type=PROMPT_INFO[cfg.LOADER.NAME]["exam_type"],
    )
    # NOTE: unused variables are simply ignored

    return final_prompt, parser
