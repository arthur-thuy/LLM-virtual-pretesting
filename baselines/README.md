# Baselines

This folder contains the scripts to run some baseline models.

The following model is implemented:
* BKT (Bayesian Knowledge Tracing) -- a lightweight approach, can be considered as a weak baseline.

Under development:
* DKT baseline.

## About Neural Baselines
The baselines based on deep neural networks are being implemented with `pykt-toolkit`.
This is a "heavy" library, which requires pytorch. 
Since that's not needed for the rest of the code, pykt-toolkit is not added to the requirements.txt, and it should be
manually installed to run the baselines.
`pip install pykt-toolkit==0.0.38`
