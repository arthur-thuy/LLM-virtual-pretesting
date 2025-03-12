[How-to guides](https://python.langchain.com/docs/how_to/#prompt-templates)

Prompt templates:
- [Concept](https://python.langchain.com/docs/concepts/prompt_templates/)
- [Conceptual guides](https://python.langchain.com/docs/how_to/#prompt-templates)

Chat models:
- [Conceptual guides](https://python.langchain.com/docs/how_to/#chat-models)
- [Concept](https://python.langchain.com/docs/concepts/chat_models/)
- [few-shot examples](https://python.langchain.com/docs/how_to/few_shot_examples_chat/) -> very good guide!!!

Ollama models:
- [text completion models](https://python.langchain.com/docs/integrations/llms/ollama/)
- [chat completion models](https://python.langchain.com/docs/integrations/chat/ollama/)

[Structured JSON output](https://python.langchain.com/docs/concepts/structured_outputs/)

Notes:
- we need to work with Chat models!
- need to use batching "A method that allows you to batch multiple requests to a chat model together for more efficient processing."
- do we use the "system" role or "assistant" role?
- use custom Example Selectors to determine which examples to show to the model [link](https://python.langchain.com/docs/how_to/example_selectors/) (see also [How to select examples by similarity](https://python.langchain.com/docs/how_to/example_selectors_similarity/))
- [FewShotChatMessagePromptTemplate](https://python.langchain.com/api_reference/core/prompts/langchain_core.prompts.few_shot.FewShotChatMessagePromptTemplate.html#langchain_core.prompts.few_shot.FewShotChatMessagePromptTemplate) combines the prompt template and example selector
