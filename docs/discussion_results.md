# Discussion results

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