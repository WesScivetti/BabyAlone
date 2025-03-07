import re
import csv
import numpy as np
from transformers import AutoTokenizer

#https://github.com/kanishkamisra/aannalysis/blob/main/src/unigramlm.py

class UnigramLM:

    """COPIED FROM: https://github.com/kanishkamisra/aannalysis/blob/main/src/unigramlm.py"""

    def __init__(self, counts_path, tokenizer):
        self.counts_path = counts_path
        # self.model_name = counts_path.split("/")[-1].split(".")[0]
        self.tokenizer = tokenizer

    def load_counts(self):
        self.counts = {}
        with open(self.counts_path, "r") as f:
            reader = csv.DictReader(f)
            for line in reader:
                self.counts[line["word"]] = int(line["count"])
        self.total_counts = sum(self.counts.values())

    def sentence_log_prob(self, sentence, token_wise=False):
        words = self.tokenizer.tokenize(sentence)
        probs = []
        for word in words:
            if word in self.counts:
                probs.append(self.counts[word] / self.total_counts)
            else:
                probs.append(1 / self.total_counts)

        if token_wise:
            return [np.log(prob) for prob in probs]
        else:
            return np.mean([np.log(prob) for prob in probs])
