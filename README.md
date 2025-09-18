# BabyAlone

BabyLM Scale Models and Let-Alone Experiments (to appear at EMNLP 2025).



## Installation:

This was written using Python version 3.13.5.



## Scripts:

* preprocess\_babyLM.py: takes the BabyLM data and sentence segments it
* filter\_babyLM.py: filters pretraining data to remove relevant constructions
* pretrain\_babyLM.py: pretrains on specified data split
* make\_templates.py: makes test set templates
* perplexity\_eval.py: runs evaluations on template datasets
* unigramlm.py: Copied from: https://github.com/kanishkamisra/aannalysis takes unigram frequencies and creates unigram lm (for SLOR)
* unigrams.py: Copied from: https://github.com/kanishkamisra/aannalysis to calculate unigram frequencies for lm
