# Own implementation

Resources:
- check Kaggle competition [Eedi - Mining Misconceptions in Mathematics](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/overview) for inspiration on how to handle misconceptions. Can check the winning solution in the discussion forum.
- check [End-to-end LLM Workflows Guide](https://www.anyscale.com/blog/end-to-end-llm-workflows-guide?_gl=1*b35e5w*_gcl_au*MjM4MDY3NDkwLjE3NDI5NzMxNzU.)

Notes:
- pinecone for vector database:
    - can use namespaces to get multiple datasets in one index (max 5 indexes in free tier). E.g., index llama3 has 2 namespaces: DBE-KT22 and CUPA
    - given that database only holds questions, it does not depend on the dataset split of the interactions! -> easy to work with


TODO: 
- can we simplify model building by using the [`init_chat_model`](https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html) function from langchain?
- langfuse custom scores ([link](https://langfuse.com/docs/scores/custom)). Does it make sense to log this on a per-observation basis? "correct" and "output_valid". -> don't know how to do it on a per-observation basis. I only implemented on a per batch basis (accuracy) -> see notebook Kate!
- DBE-KT22:
    - LLaSA sampling: they only use students that filled out all 212 questions! -> these students have accuracy of 80% while general student population has accuracy 77%.
    - how to filter out 6 questions (212 -> 206)? -> check number of student answers per question, if too low -> remove question
- from Kate "I'd map student histories or questions as embeddings in train/val/test to ensure subsets are representative."
- can we avoid merging the questions.csv and interactions.csv? See new setup with misconceptions!
- make student persona more explicit. E.g., add background field of study for DBE-KT22
- New misconception prompt:
    1. Select relevant examples from student history (most recent, most similar, mix of correct/incorrect 50/50, history of specific student or of other students with similar abilities (low, medium, high))
    1. Find misconceptions in the examples
    1. Present the target question and ask which misconceptions are relevant
    1. Ask for the student answer and explanation
- passive learning
    - check performance when reducing the dataset size\ 
    DBE-KT22: samples questions or interactions and decide on the stratification\
    CUPA: filter on target CEFR level made by the exam creators
- embedding models: just try which performs best, e.g., BERT sentence embedding, OpenAI, Llama3 (not really linked to LLMs)
- later: experiment with temperature > 0.0. Value 0.0 is easiest but higher values might give better results because this will likely lower the answer correctness, which might bring it closer to the student answer.




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
        - What to do with the HTML tables -> can we render to markdown?
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
