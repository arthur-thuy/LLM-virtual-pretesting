

# Questions implementations Luca

- Did Luca use langchain or other package?
- What are the `prompt_idx` variables? see `utils.py` functions `build_system_message_from_params` and `get_student_levels_from_prompt_idx`



# Own implementation

Elements of prompt:
- System message
    - Give context about type of exam and subject. "You will be shown a multiple choice question from an English reading comprehension exam"
    - Say that it needs to analyze the mistakes in the few-shot examples and that it sould make similar mistakes/misconceptions.
    - Do we say what the level of the student is or should the LLM solely rely on the previous examples?
    - ask to explain misconception in this question. -> we do not have these misconceptions in the data!
    - ask for chain of thought (CoT) why each answer options is correct/incorrect?
    - say how it should answer? What would be the correct answer for a particular student?
- show few-shot examples (TODO: decide how exactly)



TODO: 
- give few-shot examples of a particular student -> what dataset??

# Questions meeting 17/03/2025

Models:
- LLama3: 8B or 70B version?
- OLMo2: 7B or 13B version?