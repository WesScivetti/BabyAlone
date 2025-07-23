#!/usr/bin/env python
"""
continue_pretrain.py
====================
Utility script for **continuing** pre‑training of a previously‑saved OPT‑97M checkpoint on a *new* corpus.

Key points
----------
* **Input** corpora are single TSV files with three columns:
    * ``sents_before`` – optional context that comes *before* the target sentence
    * ``target_sent``  – the focal sentence (required)
    * ``sents_after``  – optional context that comes *after* the target sentence
* The script concatenates the non‑empty parts of each row into a plain‑text line,
  builds a Hugging Face ``datasets.Dataset``, tokenises with the same tokenizer
  used during the original pre‑training, and keeps training from the supplied
  checkpoint.

Typical usage::

    python continue_pretrain.py \
        --checkpoint_dir models/opt97m_10M_nofilt_seed0/chkpts/checkpoint-11928 \
        --train_tsv      new_corpus/train.tsv \
        --dev_tsv        new_corpus/dev.tsv \
        --output_dir     models/opt97m_10M_nofilt_seed0_contd \
        --seed           42
"""

from argparse import ArgumentParser
import os
import random
import numpy as np
import glob
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    OPTForCausalLM,
    TrainingArguments,
    Trainer,
    set_seed,
)

# Re‑use the helpers from the original repo
from tokenization_utils import tokenize_function, load_text_fancy

# ---------------------------------------------------------------------------
# Data loading utilities
# ---------------------------------------------------------------------------

def load_concat_dataset(tsv_path: str) -> Dataset:
    """Read a TSV file and return a :class:`datasets.Dataset` with a single
    **text** column, where each entry is the concatenation of the
    ``sents_before``, ``target_sent``, and ``sents_after`` columns (in that
    order, skipping empty parts).
    """
    df = pd.read_csv(tsv_path, sep="\t", dtype=str, keep_default_na=False)

    def _merge(row):
        parts = [row["sents_before"].strip(), row["target_sent"].strip(), row["sents_after"].strip()]
        parts = [p for p in parts if p]  # drop empties
        return " ".join(parts)

    texts = df.apply(_merge, axis=1).tolist()
    return Dataset.from_dict({"text": texts})

# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def create_trainer(
    output_dir: str,
    model: OPTForCausalLM,
    train_dataset: Dataset,
    dev_dataset: Dataset,
    seed: int,
) -> Trainer:
    """Wrap the model in a Hugging Face :class:`Trainer` with the same
    hyper‑parameters used during the initial pre‑training run."""

    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)

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
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=50_000,
        report_to="none",
        fp16=True,
        bf16=False,
        warmup_steps=32_000,
        gradient_accumulation_steps=1,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        optim="adamw_torch_fused",
        seed=seed,
    )

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
    )

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    # ---------------------------------------------------------------------
    # Reproducibility
    # ---------------------------------------------------------------------
    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)

    # ---------------------------------------------------------------------
    # Tokenizer & Model
    # ---------------------------------------------------------------------
    tokenizer_path = args.tokenizer_dir
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)

    # Ensure a padding token is available (required for `padding="max_length"`)
    if tokenizer.pad_token is None:
        if "<pad>" in tokenizer.get_vocab():
            tokenizer.pad_token = "<pad>"
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})



    model = OPTForCausalLM.from_pretrained(args.checkpoint_dir)

    # ---------------------------------------------------------------------
    # Data
    # ---------------------------------------------------------------------
    train_dataset = load_concat_dataset(args.train_tsv)
    #dev_dataset = load_concat_dataset(args.dev_tsv)

    dev_corpus_files = glob.glob(args.dev_dir + "/simple*.dev")

    #dataset = load_text_fancy(corpus_files, tokenizer)
    dev_dataset = load_text_fancy(dev_corpus_files, tokenizer)


    tokenized_train = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["text"],
    )
    tokenized_dev = dev_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["text"],
    )

    # ---------------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------------
    trainer = create_trainer(args.output_dir, model, tokenized_train, tokenized_dev, seed)
    trainer.train()
    trainer.save_model(args.output_dir)  # final checkpoint
    tokenizer.save_pretrained(os.path.join(args.output_dir, "tokenizer"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True, help="Existing model checkpoint to resume from.")
    parser.add_argument("--train_tsv", required=True, help="TSV file containing the *training* split.")
    parser.add_argument("--dev_dir", required=True, help="TSV file containing the *validation* split.")
    parser.add_argument("--output_dir", required=True, help="Where to write checkpoints, logs, and tokenizer.")
    parser.add_argument("--tokenizer_dir", default=None, help="Load tokenizer from here instead of --checkpoint_dir, if set.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility.")

    main(parser.parse_args())
