# Discussion results

# Overview of configs

Contextual settings:
- "replication_miscon_valsmall_kt_20250818-153513"
- "replication_miscon_vallarge_kt_20250819-115545"
- "replication_miscon_test_kt_20250819-192630"

Non-contextual settings:
- "replication_miscon_vallarge_kt_nocontext_20250819-212500"
- "replication_miscon_test_kt_nocontext_20250820-075221"

=> both test results should be merged into "replication_miscon_test_kt_merged_20250820"


In density plots, we can see that the plot over all configs aligns well with the student correctness, which directly hints to Multi-LLM simulation. However, this is not the avenue we want to pursue.



## Class Imbalance

- select appropriate metric: F1-score for multiclass, but micro or macro averaged?
- inspect the proportion of LLM responses who are correct (so true answer). I have the impression that the smaller models tend to always answer the correct option, while the large models have a proportion that is closer to the true proportion in the eval set.\
-> we could plot histograms of answer correctness for each model in the qwen family, and draw a vertical line at the true proportion. Hopefully, we see that the larger models have a distribution that is closer to the true proportion.


## Temperature

- check whether there is a trend for temperature within the models, as opposed to averaging over all the models
- temp 1.0 decreases the LLM correctness significantly! It brings it closer to the student correctness.

## Model

- Analyse impact of varying model size in a family (on answer correctness)

