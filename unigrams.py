import argparse
import csv
import pathlib
import re
# import utils

from collections import defaultdict, Counter
from joblib import Parallel, delayed
from multiprocessing import Manager
from tqdm import tqdm
from transformers import AutoTokenizer



def main(args):
    """
    THIS ENTIRE FILE IS COPIED FROM MAHOWALD AND MISHRA 2024 FOR CALCULATING UNIGRAM FREQUENCIES
    """

    # read corpus
    with open(args.corpus) as inf:
        corpus = inf.read()

    # tokenize corpus
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def count_word_frequencies(sentence):
        words = tokenizer.tokenize(sentence)
        return Counter(words)

    def merge_counters(counters):
        result = Counter()
        for counter in counters:
            result.update(counter)
        return result

    def count_word_frequencies_in_parallel(sentences):
        # Use joblib's Parallel and delayed to parallelize the word frequency counting
        counters = Parallel(n_jobs=-1)(delayed(count_word_frequencies)(sentence) for sentence in tqdm(sentences))
        
        # Merge individual counters into a single counter
        word_frequencies = merge_counters(counters)
        return word_frequencies

    word_counters = [Counter(tokenizer.tokenize(sent)) for sent in tqdm(corpus)]
   
    word_counts = Counter()
    for counter in word_counters:
        word_counts.update(counter)

    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    output_file = "counts"

    with open(f"{args.output_dir}/{output_file}.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["word", "count"])
        for word, count in word_counts.most_common():
            writer.writerow([word, count])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, required=True, help="Path to corpus file")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    args = parser.parse_args()
    main(args)
    

    


