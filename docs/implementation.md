# Own implementation

Resources:
- check Kaggle competition [Eedi - Mining Misconceptions in Mathematics](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/overview) for inspiration on how to handle misconceptions. Can check the winning solution in the discussion forum.
- check [End-to-end LLM Workflows Guide](https://www.anyscale.com/blog/end-to-end-llm-workflows-guide?_gl=1*b35e5w*_gcl_au*MjM4MDY3NDkwLjE3NDI5NzMxNzU.)



TODO: 
- add code to pretty print an example
- example selector: within the same student_id, select semantically similar examples
- can we simplify model building by using the [`init_chat_model`](https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html) function from langchain?
- Make sure the data is shuffled before splitting!
- langfuse custom scores ([link](https://langfuse.com/docs/scores/custom)). Does it make sense to log this on a per-observation basis? "correct" and "output_valid". -> don't know how to do it on a per-observation basis. I only implemented on a per batch basis (accuracy)
- example formatter with quotes?


# Questions meeting 17/03/2025

- Research proposal
    - Comments?
- Models:
    - Llama3: 8B or 70B version?
    - OLMo2: 7B or 13B version?
    - always temperature 0.0?
- Prompting
    - Does it make sense to ask for misconceptions in the correct answer option?
    - Do we tell the roll-playing model what the correct answer is? + show the explanation (see DBE-KT22)
- Datasets
    - What dataset to use? No dataset is perfect?
    - DBE-KT22
        - use this because it is the only dataset with MC questions, student-question pairs and question texts
        - need to compute IRT parameters manually -> provided difficulty is discrete. Which package to use?
        - can use rich question text and change the latex urls to normal latex commands (e.g., `<img src="http://latex.codecogs.com/gif.latex?T_{1}" border="0"/>` to `T_{1}`)
        - what do we do with MC questions with more or less than 4 answer options?
        - What to do with the HTML content? E.g., tables.
        - Do we use entire dataset? A lot of questions so might be expensive
    - CUPA
        - We do not have student-answer records? How do we select relevant few-shot examples?
    - Can we share question_id's and student_id's over train/val/test splits? -> I think yes because we need to find relevant examples in train set to answer the val and test questions
- Small experiment: 
    - olmo2 has much more invalid responses than llama3 (approx 3% vs 0.3%)
    - no difference between random selector and studentid_random -> if a student has a lot of questions in different topics, not enough additional information is given to the model
- Example selector with studentID and semantic similarity:
    - What is exact input to vector database? Only question without "correct answer"?
    - How to avoid that embeddings are calculated multiple times? Can we calculate once for all examples and then filter on the student_id?
    - What if unknown student_id at inference time? -> use random student_id
- System prompt
    - Clear enough? 
    - Do we need to restrict the length of the response?
    - Would it help if we make a new field to indicate that the student answer is correct/incorrect?
    - Ask for common misconception in the few-shot examples? Can we ask to show where this mistakes has been made in the few-shot examples?
