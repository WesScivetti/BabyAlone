from datasets import Dataset
import glob
import json
import numpy as np
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast, AutoTokenizer

def train_tokenizer(train_dir, tokenizer_save_path):
    """
    Train the tokenizer with the given input dataset (can be full BabyLM train or filtered).
    Tokenizer is a BPE tokenizer. By default, also saves the tokenizer.
    Returns: tokenizer object
    """

    train_data_path = train_dir + "/*.train"
    corpus_files = glob.glob(train_data_path)
    print(f"Training on files: {corpus_files}")

    print("Starting tokenizer training...")

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=corpus_files, vocab_size=16383, min_frequency=2,
                    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])

    print("Tokenizer training complete.")

    tokenizer.save(f"{tokenizer_save_path}/tokenizer.json")

    config = {
        "model_type": "gpt2",  # Must define the model type
        "vocab_size": 16383,
        "special_tokens": ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    }
    with open(f"{tokenizer_save_path}/config.json", "w") as f:
        json.dump(config, f, indent=4)

    print("Tokenizer saved successfully.")

    # Save and reload to make sure it saved correctly (only takes a second)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_save_path, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


# Load and combine all text files
def load_text_lines(file_list, chunk_size=800):
    """chunks the dataset into simple chunks. Will be replaced by more sophisticated chunking."""
    all_text = ""
    for file in file_list:
        with open(file, "r", encoding="utf-8") as f:
            all_text += f.read() + " "  # Concatenate all text with spaces

    # Split into fixed-size chunks
    chunks = [all_text[i: i + chunk_size] for i in range(0, len(all_text), chunk_size)]
    return Dataset.from_dict({"text": chunks})  # Convert to Hugging Face Dataset


def load_text_fancy(file_list, tokenizer, max_seq=256):
    """chunks in a more informed way based on chunks of the maximum sequence length while respecting sentence boundaries"""

    print("chunking stuff")
    chunk_counter = 0
    chunks = []

    curr_chunk = ""
    curr_tok_count = 0

    for f in file_list:
        with open(f, "r", encoding="utf-8") as inf:
            for line in inf:
                sent_toks = tokenizer.encode(line)
                add_count = len(sent_toks)

                if curr_tok_count + add_count < 250:
                    curr_chunk += line
                    curr_tok_count += add_count
                else:
                    chunks.append(curr_chunk)
                    chunk_counter += 1
                    curr_chunk = line
                    curr_tok_count = add_count
                    if chunk_counter % 10000 == 0:
                        print(f"Processed {chunk_counter} chunks so far")

    chunks.append(curr_chunk)  # add last chunk in
    # print(len(chunks))
    return Dataset.from_dict({"text": chunks})

def tokenize_function(examples, tokenizer):
    """
    tokenizes the text input, provides labels that are corresponding to the input_ids
    Labels are input ids since this is just a language modeling task.
    """
    tokenized_output = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)
    input_ids = np.array(tokenized_output["input_ids"])
    labels = np.where(input_ids == tokenizer.pad_token_id, -100, input_ids)
    tokenized_output["labels"] = labels.tolist()

    return tokenized_output