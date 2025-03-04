from argparse import ArgumentParser
from datasets import Dataset
import glob
import json
import numpy as np
import os
import pandas as pd
import random
from tokenizers import ByteLevelBPETokenizer
import torch
from transformers import PreTrainedTokenizerFast, AutoTokenizer, OPTConfig, OPTForCausalLM, TrainingArguments, Trainer, set_seed

from tokenization_utils import train_tokenizer, load_text_lines, load_text_fancy, tokenize_function

# def train_tokenizer(train_dir, tokenizer_save_path):
#     """
#     Train the tokenizer with the given input dataset (can be full BabyLM train or filtered).
#     Tokenizer is a BPE tokenizer. By default, also saves the tokenizer.
#     Returns: tokenizer object
#     """
#
#     train_data_path = train_dir + "/*.train"
#     corpus_files = glob.glob(train_data_path)
#     print(f"Training on files: {corpus_files}")
#
#
#     print("Starting tokenizer training...")
#
#     tokenizer = ByteLevelBPETokenizer()
#     tokenizer.train(files=corpus_files, vocab_size=16383, min_frequency=2,
#                     special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
#
#     print("Tokenizer training complete.")
#
#
#
#     tokenizer.save(f"{tokenizer_save_path}/tokenizer.json")
#
#     config = {
#         "model_type": "gpt2",  # Must define the model type
#         "vocab_size": 16383,
#         "special_tokens": ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
#     }
#     with open(f"{tokenizer_save_path}/config.json", "w") as f:
#         json.dump(config, f, indent=4)
#
#     print("Tokenizer saved successfully.")
#
#     #Save and reload to make sure it saved correctly (only takes a second)
#     tokenizer = AutoTokenizer.from_pretrained(tokenizer_save_path, use_fast=True)
#
#     if tokenizer.pad_token is None:
#       tokenizer.pad_token = tokenizer.eos_token
#
#     return tokenizer
#
# # Load and combine all text files
# def load_text_lines(file_list, chunk_size=800):
#     """chunks the dataset into simple chunks. Will be replaced by more sophisticated chunking."""
#     all_text = ""
#     for file in file_list:
#         with open(file, "r", encoding="utf-8") as f:
#             all_text += f.read() + " "  # Concatenate all text with spaces
#
#     # Split into fixed-size chunks
#     chunks = [all_text[i : i + chunk_size] for i in range(0, len(all_text), chunk_size)]
#     return Dataset.from_dict({"text": chunks})  # Convert to Hugging Face Dataset
#
#
# def load_text_fancy(file_list, tokenizer, max_seq = 256):
#     """chunks in a more informed way based on chunks of the maximum sequence length while respecting sentence boundaries"""
#
#     print("chunking stuff")
#     chunk_counter = 0
#     chunks = []
#
#     curr_chunk = ""
#     curr_tok_count = 0
#
#     for f in file_list:
#         with open(f, "r", encoding="utf-8") as inf:
#             for line in inf:
#                 sent_toks = tokenizer.encode(line)
#                 add_count = len(sent_toks)
#
#                 if curr_tok_count + add_count < 250:
#                     curr_chunk += line
#                     curr_tok_count += add_count
#                 else:
#                     chunks.append(curr_chunk)
#                     chunk_counter += 1
#                     curr_chunk = line
#                     curr_tok_count = add_count
#                     if chunk_counter % 10000 == 0:
#                         print(f"Processed {chunk_counter} chunks so far")
#
#
#     chunks.append(curr_chunk) #add last chunk in
#     #print(len(chunks))
#     return Dataset.from_dict({"text": chunks})
#
# def tokenize_function(examples, tokenizer):
#     """
#     tokenizes the text input, provides labels that are corresponding to the input_ids
#     Labels are input ids since this is just a language modeling task.
#     """
#     tokenized_output = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)
#     input_ids = np.array(tokenized_output["input_ids"])
#     labels = np.where(input_ids == tokenizer.pad_token_id, -100, input_ids)
#     tokenized_output["labels"] = labels.tolist()
#
#     return tokenized_output


def create_model():
    # Define the custom OPT model config
    config = OPTConfig(
        vocab_size=16384,   # Match tokenizer vocab size
        hidden_size=768,     # Model dimension
        ffn_dim=3072,        # Feedforward network dimension
        num_attention_heads=12,
        num_hidden_layers=12,
        max_position_embeddings=256  # Same as OPT default
    )

    # Initialize the model
    model = OPTForCausalLM(config)

    print("OPT-97M model initialized successfully!")
    return model


def create_trainer(output_dir, model, train_dataset, dev_dataset, seed):
    os.makedirs(output_dir + "logs", exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=32,  # Adjust based on GPU memory
        per_device_eval_batch_size=32,
        learning_rate=5e-4,
        weight_decay=0.01,
        save_total_limit=10,
        num_train_epochs=10,
        logging_dir= output_dir + "logs",
        logging_steps=50000,
        report_to="none",  # Change to "wandb" or "tensorboard" if needed
        fp16=True,
        bf16=False,# Enable mixed precision for faster training
        warmup_steps=3200,
        gradient_accumulation_steps=1,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        optim="adamw_torch_fused",
        seed=seed
    )

    print("Training arguments defined!")

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset
    )

    return trainer





#trainer.train()

# import shutil
# shutil.make_archive("tokenizer_10M.zip", 'zip', "./custom-tokenizer")
#
# shutil.make_archive("opt-97m-pretrained_10M_all_chkp6", 'zip', "./opt-97m-pretrained/checkpoint-11928")

def main(train_dir, dev_dir, eval_dir, size, filter, seed):

    print("Making Directory: " + f"models/opt97m_{size}_{filter}_seed{seed}")
    os.makedirs(f"models/opt97m_{size}_{filter}_seed{seed}", exist_ok=True)

    print("Making Directory: " + f"results/opt97m_{size}_{filter}_seed{seed}")
    os.makedirs(f"results/opt97m_{size}_{filter}_seed{seed}", exist_ok=True)

    model_output_path = f"models/opt97m_{size}_{filter}_seed{seed}/chkpts/"
    os.makedirs(model_output_path, exist_ok=True)

    tokenizer_output_path = f"models/opt97m_{size}_{filter}_seed{seed}/tokenizer/"
    os.makedirs(tokenizer_output_path, exist_ok=True)

    results_output_path = f"results/opt97m_{size}_{filter}_seed{seed}/chkpts"
    os.makedirs(results_output_path, exist_ok=True)

    #The seed is set, right?
    seed = int(args.seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


    tokenizer = train_tokenizer(train_dir, tokenizer_output_path)


    corpus_files = glob.glob(train_dir + "/*.train")

    dev_corpus_files = glob.glob(dev_dir + "/*.dev")

    dataset = load_text_fancy(corpus_files, tokenizer)
    eval_dataset = load_text_fancy(dev_corpus_files, tokenizer)


    tokenized_datasets = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True, remove_columns=["text"])

    print("Dataset tokenized successfully!")

    tokenized_dev_datasets = eval_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True, remove_columns=["text"])
    print("Dev Dataset tokenized successfully!")

    model = create_model()

    trainer = create_trainer(model_output_path, model, tokenized_datasets, tokenized_dev_datasets, seed)

    trainer.train()




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train_dir")
    parser.add_argument("--dev_dir")
    parser.add_argument("--eval_dir")
    parser.add_argument("--size")
    parser.add_argument("--seed")
    parser.add_argument("--filter")

    args = parser.parse_args()
    main(args.train_dir, args.dev_dir, args.eval_dir, args.size, args.filter, args.seed)