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
from perplexity_eval import load_perp_dataset



def create_model():
    """
    creates a fresh OPT model with the hyperparams from Mahowald and Mishra 2024
    """
    config = OPTConfig(
        vocab_size=16384,
        hidden_size=768,
        ffn_dim=3072,
        num_attention_heads=12,
        num_hidden_layers=12,
        max_position_embeddings=256
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
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=1e-4,
        weight_decay=0.01,
        save_total_limit=20,
        num_train_epochs=20,
        logging_dir= output_dir + "logs",
        logging_steps=50000,
        report_to="none",
        fp16=True,
        bf16=False,
        warmup_steps=32000,
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