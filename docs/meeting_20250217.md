

# Ranked Probability Score (RPS)


Check last figure in [this blog](https://www.lokad.com/continuous-ranked-probability-score/)\
=> We can make same comparision for RPS, where we change the true y on the x-axis.


# Papers linked by Luca

Criteria:
- LLM capability: Using inherent question-answering capability of the model vs prompt role-playing to ask for low, medium, high-level student capability
- LLM output: Using generative LLMs vs discriminative models (regression/classification head)
- Data: Need question-answer logs vs entirely unsupervised
- Compute: fine-tuning vs prompt engineering

## [Large Language Models are Students at Various Levels: Zero-shot Question Difficulty Estimation](https://aclanthology.org/2024.findings-emnlp.477/)

[Github repo](https://github.com/cuk-nlp/llms-are-students-at-various-levels)

Notes:
- uses generative LLMs
- uses prompting techniques: process of elimination (PoE), chain-of-thought (CoT) and plan-and-solve (PS)\
-> all with zero, 1-, 3-, and 5-shot prompting
- in the zero-short LLaSA model, student abilities is modeled as a discrete variable (high, medium, low)
- QDE target: 
    - regression setting: continuous difficulty -> RMSE and Pearson correlation
    - classification setting: 6 classes (continuous range divided into equal intervals!!!) -> F1 metric
=> implications for QDE_ordinal paper: this paper uses both regression and classification but they use different metrics in both settings so the performance cannot be compared. Here, they do not have the intention to compare the performance of the two settings to find out which model output type is most appropriate for QDE.
- TODO: read paper ["LoRA: Low-Rank Adaptation of Large Language Models"](https://arxiv.org/abs/2106.09685)
- They use the accuracy of the LLMs in question-solving (some are low, medium, high level) and match this to the actual students' performance (also low, medium, high level). In the future, LLMs will only improve performance so we will not have poorly performing models. How do we get low-level output? Use prompt engineering on high-performing models, use smaller versions of high-performing models?

Idea 1:
- They use 65 LLMs and set the temprature to 0. 
- In section 2.2.1 "LLM Cluster Response Aggregation", they say it is challenging to achieve the same question-solving performance with LLM than with real students. The high-performing models have shown potential in substituting a single student with a single LLM; but smaller or outdated models do not achieve human-level question-solving performance.
- I think there is an opportunity to use only the most performant models and use prompt engineering to make them similar to a low,medium,high-level student.
Sensible to set temperature to 0 in LLMs?

## [Using LLMs to simulate students’ responses to exam questions](https://aclanthology.org/2024.findings-emnlp.663/)

[Github repo](https://github.com/lucabenedetto/LLM-student-simulations)

Notes:
- uses generative LLMs
- Prompting
    - also uses temperature 0?
    - only zero-shot prompts
- focus on prompt engineering
- CUP&A target: used continuous estimate from pretesting (instead of CEFR levels)
- setup: quality of LLM question-answering is evaluated directly by assessing the accuracy for various student levels => this is not a pure QDE task where the final difficulty estimate is evaluated
- seems a bit dangerous to say that the behaviour of GPT-4 is bad if you didn't tune the prompt for it
- only brief evaluation of QDE performance (only measured Pearson correlation)
- section 4.5.2: how obtained the target in the range [1,5]?
- each simulated learning answers all the questions in the test + using 5 simulated students (of 5 different levels)

## [Question Difficulty Prediction Based on Virtual Test-Takers and Item Response Theory](https://ceur-ws.org/Vol-3772/paper1.pdf)

Notes:
- they focus on test linking
- Process:
    - fine-tune a QA system on exam questions
    - they first estimate student-level ability using question-answering logs with IRT
    - using the student-level ability values, they estimate the QA systems' ability values with IRT (test linking)
    - for new questions, the question-answering logs of the QA systems are used to estimate the difficulty of the new questions
- not unsupervised because you need:
    - initial question-answering logs from student to estimate the student-level ability!
    - exam questions for fine-tuning
- just like LLaSA, uses inherent question-answering capability of the model instead of fine-tuning to ask for low, medium, high-level students
- conversely to LLaSA, they find that most QA systems are better than human test-takers

## [Field-Testing Multiple-Choice Questions With AI Examinees: English Grammar Items](https://journals.sagepub.com/doi/10.1177/00131644241281053?icid=int.sj-full-text.citing-articles.4)

[Github repo](https://github.com/hotakamaeda/ai_field_testing1)


Notes:
- Strange: all MC questions are restructured so they have an identical stem -> this stem is then removed and the model has to predict the correct answer\
=> I would just use raw input data because this requires a lot of manual preprocessing!
- they use 61 LLMs and fine-tune these. They use the inherent accuracy of the LLMs to estimate the difficulty of the questions.
- they discretize the student ability range \theta (-3, 3) in 61 bins of step 0.1. These levels are assigned to individual LLMs. (how????)\
The LLM was fine-tuned to have \theta=3 and was later modified to achieve lower \theta values using further fine-tuning, by 0.1 each time.
- not unsupervised because we need question-answer logs for fine-tuning
- analysis from psychometrics aspect -> more difficult to follow for me
- the models are not generative! They use a regression/classification head to predict the answer.\


# Proposals

- Question about question-answer logs: What information do we have on the students? Do we have new students in the val and test set?
- Is it necessary to fine-tune the LLMs because this requires questions-answer logs from students?\
- Instruction fine-tuning: use prompt engineering to make the LLMs behave like low, medium, high-level students\
    - question-answering logs of low-level students have more mistakes
    - how to calibrate the LLMs to behave like low-level students?
- Prompt engineering:
    - In-context learning: 1-shot, 3-shot, 5-shot prompts
    - Chain-of-thought prompting: one-shot vs few-shot vs zero-shot
    - Did Luca use a zero-shot CoT approach? e.g., "Let's think step by step" -> he asks to solve the question step by step
    - Self-refinement: 
        - predict-then-refine: can we ask it to refine by giving feedback using a reward model (Predicted answer vs true answer) on a numbered of labeled samples? -> compare this to supervised fine-tuning!
        - deliberate-then-generate (DTG): Ask it to first detect the type of error and to then refine the answer (question-solving)\\
        We could also give random MC answer and default error type to see if it can detect the error type and refine the answer.
    - Ensembling:
        - Model ensembling: using 1 prompt in multiple LLMs
        - Prompt ensembling: using multiple prompts in the same LLM (different demonstrations)
        - Output ensembling: using 1 prompt and 1 LLM and sampling multiple outputs
- RAG
    - Provide previous question-answering logs of specific-level students as a context
    - Ask the model to analyze the behaviour and answer the following questions in the same manner
    - If we have question-answer logs, what is difference with supervised fine-tuning?
- Think about how you can bring domain knowledge of language learning (e.g., false friends) into LLMs
- Soft prompts in LLMs (parameter-efficient fine-tuning): cheaply fine-tune the LLMs to behave like low, medium, high-level students?
- Supervised fine-tuning
    - X is the instruction + user input. For the instruction, does it need to be exactly the same wording in the samples, and exactly the same than what is used at inference?\\
    So we are also doing prompt engineering inside supervised fine-tuning?
    - Use self-instruct techniques to automatically generate more training data
    - Generate outputs with "input inversion" (see book p166): adjust the order of generation of different fields.
- 