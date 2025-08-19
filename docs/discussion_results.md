# Discussion results

# Overview of configs

Contextual settings:
- "replication_miscon_valsmall_kt_20250818-153513"
- "replication_miscon_vallarge_kt_20250819-115545"
- "replication_miscon_test_kt_20250819-192630"

Non-contextual settings:
- "replication_miscon_vallarge_kt_nocontext_XXX"
- "replication_miscon_test_kt_nocontext_XXX"

=> both test results should be merged into "replication_miscon_test_kt_merged_XXX"


In density plots, we can see that the plot over all configs aligns well with the student correctness, which directly hints to Multi-LLM simulation. However, this is not the avenue we want to pursue.




# Running configs on test

Should we keep the best student prompt and teacher prompt configuration for each model?

## Class Imbalance

- select appropriate metric: F1-score for multiclass, but micro or macro averaged?
- inspect the proportion of LLM responses who are correct (so true answer). I have the impression that the smaller models tend to always answer the correct option, while the large models have a proportion that is closer to the true proportion in the eval set.\
-> we could plot histograms of answer correctness for each model in the qwen family, and draw a vertical line at the true proportion. Hopefully, we see that the larger models have a distribution that is closer to the true proportion.


## Temperature

- check whether there is a trend for temperature within the models, as opposed to averaging over all the models
- temp 1.0 decreases the LLM correctness significantly! It brings it closer to the student correctness.

## Number of examples

Problem is that some students have a limited number of previous interactions ("random") or previous interactions of relevant knowledge concepts ("kc_exact").
As such, the model may not get the number of examples that it requests.
This might limit the advantage that the configurations with a larger number of examples can get.

=> fix issue with stratifying the train interactions\
=> determine student levels on the train set + bring to eval sets (check that every student in eval set has at least some interactions in train)

## Model

- Analyse impact of varying model size in a family (on answer correctness)
- compare open-weight vs closed LLMs

