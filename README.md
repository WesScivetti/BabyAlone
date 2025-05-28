# BabyAlone
 BabyLM Scale Models and Let-Alone Experiments (ARR MAY)

 ## Scripts:
 - preprocess_babyLM.py: takes the BabyLM data and sentence segments it
 - filter_babyLM.py: filters pretraining data to remove relevant constructions
 - pretrain_babyLM.py: pretrains on specified data split
 - make_templates.py: makes test set templates
 - perplexity_eval.py: runs evaluations on template datasets
 - unigramlm.py: Copied from: https://github.com/kanishkamisra/aannalysis takes unigram frequencies and creates unigram lm (for SLOR)
 - unigrams.py: Copied from: https://github.com/kanishkamisra/aannalysis to calculate unigram frequencies for lm
