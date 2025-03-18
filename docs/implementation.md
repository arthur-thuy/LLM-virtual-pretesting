

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
- add build functions:
    - config files
    - model
    - system prompt (student level explanation here)
    - example selector for few-shot examples
    - dataset loader (into dataframe with columns: prompt, output)
- give CUPA dataset and ask to find possible misconceptions that students of various levels can make
- add code to pretty print an example
- example selector: within the same student_id, select semantically similar examples
- can we simplify model building by using the [`init_chat_model`](https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html) function from langchain?


# Questions meeting 17/03/2025

- Models:
    - Llama3: 8B or 70B version?
    - OLMo2: 7B or 13B version?
    - always temperature 0.0?
    - 
- Prompting
    - Does it make sense to ask for misconceptions in the correct answer option?
    - Do we tell the roll-playing model what the correct answer is? + show the explanation (see DBE-KT22)
- Datasets
    - What dataset to use? No dataset is perfect?
    - DBE-KT22
        - use this because it is the only dataset with MC questions, student-question pairs and question texts
        - need to compute IRT parameters manually -> provided difficulty is discrete
        - can use rich question text and change the latex urls to normal latex commands (e.g., `<img src="http://latex.codecogs.com/gif.latex?T_{1}" border="0"/>` to `T_{1}`)
        - what do we do with MC questions with more or less than 4 answer options?
        - What to do with the HTML content?
        - Do we use entire dataset? A lot of questions so might be expensive
    - Can we share question_id's and student_id's over train/val/test splits? -> I think yes because we need to find relevant examples in train set to answer the val and test questions
- Small experiment: 
    - olmo2 has much more invalid responses than llama3 (approx 3% vs 0.3%)
    - no difference between random selector and studentid_random -> if a student has a lot of questions in different topics, not enough additional information is given to the model
