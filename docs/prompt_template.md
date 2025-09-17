# Conversational prompt template for misconceptions - teacher version

The aim is to build a prompt template where the LLM either lists the skills and misconceptions live when assessing the student output, or the skills and misconceptions are inserted as context (collected from an LLM in an earlier step).
In a final step, the LLM is asked to answer how a student of a given level would answer a new question, based on the misconceptions and skills identified earlier.

## System message
```python
system_prompt_str = (
        "You are an expert teacher preparing a set of multiple choice exam questions on {exam_type}. "
    )
```

## User message

```python
user_prompt_str = (
    "You have a student in your class of level {student_level_group} {student_scale}. "
    "Consider their earlier open-ended response below: \n\n"
    "{open_ended_response}\n\n"  # TODO: insert actual response (text with annotations + error legend)
    "Inspect the open-ended response and list the skills and misconceptions that the student has. "
    "List up to 3 skills and up to 3 misconceptions, in bullet point format. "
    "Specifically, think about what skills and misconceptions would translate from writing to reading comprehension. "
    "If there are no skills or misconceptions, return 'None'."
)
```

## AI message
> NOTE: this is either generated live by the LLM, or inserted as context from an earlier LLM call

If generated live, we use the LLM output.

If inserted as context, we use the following template.
```python
ai_prompt_str = (
    "Skills:\n"
    "{skills}\n\n"  # TODO: insert actual skills
    "Misconceptions:\n"
    "{misconceptions}"  # TODO: insert actual misconceptions
)
```

## User message
```python
user_prompt_str = (
    "Inspect the following new multiple-choice question:\n"
    "{question_text}\n\n"  # TODO: insert actual question
    "How would the student of level {student_level_group} answer this question? "
    "Think about how the student level relates to the question difficulty. "
    "You can answer incorrectly, if that is what the student is likely to do for this question. "
    # TODO: add the JSON formatting instructions here
)
```




