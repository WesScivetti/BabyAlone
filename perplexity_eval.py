from argparse import ArgumentParser
import openai
from collections import defaultdict
import glob
from itertools import chain
import numpy as np
import pandas as pd
from tokenizers import ByteLevelBPETokenizer
import torch
from transformers import PreTrainedTokenizerFast, AutoTokenizer, OPTConfig, OPTForCausalLM, TrainingArguments, Trainer, set_seed, AutoModelForCausalLM
from minicons import scorer
from unigramlm import UnigramLM
import re

OPENAI_KEY = "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"

def load_perp_dataset(input_file):
    """
    loads in the let-alone data that will be put into the correct format for reading into the model.
    """
    df = pd.read_csv(input_file, sep="\t")
    return df


def avg_surprisal_minicons(model, tokenizer, let_alone_list, stimuli_list=None):
    """
    calculates the surprisal using minicons
    """


    model = scorer.IncrementalLMScorer(model, tokenizer=tokenizer, device='cuda')

    # surprisals = model.token_score(text_list, surprisal=True, base_two=True)
    # print(surprisals)


    if stimuli_list:
        seq_log_prob = model.conditional_score(let_alone_list, stimuli_list) #log prob
        seq_surprisal = [-s for s in seq_log_prob]

    else:
        seq_surprisal = model.sequence_score(let_alone_list, reduction=lambda x: -x.mean(0))
        seq_surprisal = [s.cpu() for s in seq_surprisal]
    # print(seq_surprisal[0])
    return seq_surprisal

def surprisal_long_list(model, tokenizer, let_alone_list, stimuli_list=None):
    """chunks and returns surprisal of really long list
    can't feed the whole list into minicon in one batch
    COMPUTES CONDITIONAL LOG-PROB of stimuli based on context, but doesn't include context in average."""
    chunk_surps = []
    chunks_context = [let_alone_list[i:i + 32] for i in range(0, len(let_alone_list), 32)]
    if stimuli_list:
        chunks_stimuli = [stimuli_list[i:i + 32] for i in range(0, len(stimuli_list), 32)]
        for chunk_context, chunk_stimuli in zip(chunks_context, chunks_stimuli):
            sups = avg_surprisal_minicons(model, tokenizer, chunk_context, chunk_stimuli)
            chunk_surps.append(sups)
    else:
        for chunk_context in chunks_context:
            sups = avg_surprisal_minicons(model, tokenizer, chunk_context)
            chunk_surps.append(sups)
    return list(chain.from_iterable(chunk_surps))

def logprob_list_unigrams(unigram_lm, stimuli_list):
    """unigram log probability of the sentence"""
    lps = []
    for sent in stimuli_list:
        lp = unigram_lm.sentence_log_prob(sent)
        lps.append(lp)
    #print(sent, lp)
    return lps

def evaluate_npi(model, tokenizer, npi_df, output_dir, unigramlm=None, slor=False):
    """
    evaluate NPI dataset.
    """
    let_alone_list = []
    for r in npi_df.index:
        base_subbed = npi_df.loc[r, "base_subbed"]
        base_no_npi_subbed = npi_df.loc[r, "base_no_npi_subbed"]
        swap_subbed = npi_df.loc[r, "swap_subbed"]
        swap_no_npi_subbed = npi_df.loc[r, "swap_no_npi_subbed"]
        base_and_subbed = npi_df.loc[r, "base_and_subbed"]
        base_and_no_npi_subbed = npi_df.loc[r, "base_and_npi_subbed"]
        swap_and_subbed = npi_df.loc[r, "swap_and_subbed"]
        swap_and_no_npi_subbed = npi_df.loc[r, "swap_and_npi_subbed"]
        let_alone_list.append(base_subbed)
        let_alone_list.append(base_no_npi_subbed)
        let_alone_list.append(swap_subbed)
        let_alone_list.append(swap_no_npi_subbed)
        let_alone_list.append(base_and_subbed)
        let_alone_list.append(base_and_no_npi_subbed)
        let_alone_list.append(swap_and_subbed)
        let_alone_list.append(swap_and_no_npi_subbed)

    let_alone_list_surps = surprisal_long_list(model, tokenizer, let_alone_list)  #

    if slor:
        let_alone_list_surps = [-s for s in let_alone_list_surps]  # change to logprobs
        # print(ls_surps)
        ls_u  = logprob_list_unigrams(unigramlm, let_alone_list)
        # print(ls_u)
        slors = [lm - u for (lm, u) in zip(let_alone_list_surps, ls_u)]

    i = 0
    correct_count = 0
    total_count = 0
    correct_count_diff = 0
    for r in npi_df.index:

        npi_df.loc[r, "SLOR_Base"] = slors[i]
        slore_base = npi_df.loc[r, "SLOR_Base"]
        i += 1
        npi_df.loc[r, "SLOR_Base_No_NPI"] = slors[i]
        slore_base_no_npi = npi_df.loc[r, "SLOR_Base_No_NPI"]
        i += 1

        npi_df.loc[r, "SLOR_Swap"] = slors[i]
        slore_swap = npi_df.loc[r, "SLOR_Swap"]
        i += 1
        npi_df.loc[r, "SLOR_Swap_No_NPI"] = slors[i]
        slore_swap_no_npi = npi_df.loc[r, "SLOR_Swap_No_NPI"]
        i += 1

        npi_df.loc[r, "SLOR_Base_And"] = slors[i]
        slore_base_and = npi_df.loc[r, "SLOR_Base_And"]
        i += 1

        npi_df.loc[r, "SLOR_Base_And_No_NPI"] = slors[i]
        slore_base_and_no_npi = npi_df.loc[r, "SLOR_Base_And_No_NPI"]
        i += 1

        npi_df.loc[r, "SLOR_Swap_And"] = slors[i]
        slore_swap_and = npi_df.loc[r, "SLOR_Swap_And"]
        i += 1

        npi_df.loc[r, "SLOR_Swap_And_No_NPI"] = slors[i]
        slore_swap_and_no_npi = npi_df.loc[r, "SLOR_Swap_And_No_NPI"]
        i += 1

        and_diff_base = slore_base_and_no_npi - slore_base_and
        and_diff_swap = slore_swap_and_no_npi - slore_swap_and

        la_diff_base = slore_base_no_npi - slore_base #should be negative
        la_diff_swap = slore_swap_no_npi - slore_swap #should be negative

        if (and_diff_swap > la_diff_swap) and (and_diff_base > la_diff_base):
            npi_df.loc[r, "Correct_Diff"] = "Y"
            correct_count_diff += 1

        if (slore_base > slore_base_no_npi) and (slore_swap > slore_swap_no_npi):
            npi_df.loc[r, "Correct"] = "Y"
            correct_count += 1
            total_count += 1

        else:
            npi_df.loc[r, "Correct"] = "N"
            total_count += 1

    print(i, len(slors))
    #print(correct_count, total_count, correct_count / total_count)
    print(f"NPI DATASET: {correct_count}/{total_count} correct ({correct_count / total_count})")
    print(f"DIFF OVER LET ALONE: {correct_count_diff}/{total_count} correct ({correct_count_diff / total_count})")


def evaluate_psuedo(model, tokenizer, ps_df, output_dir, unigramlm=None, slor=False):
    """
    evaluate NPI dataset.
    """
    let_alone_list = []
    for r in ps_df.index:
        base_subbed = ps_df.loc[r, "base_subbed"]
        base_ps_subbed = ps_df.loc[r, "base_psuedo_subbed"]
        swap_subbed = ps_df.loc[r, "swap_subbed"]
        swap_ps_subbed = ps_df.loc[r, "swap_psuedo_subbed"]

        base_and_subbed= ps_df.loc[r, "base_and_subbed"]
        base_and_ps_subbed = ps_df.loc[r, "base_and_psuedo_subbed"]
        swap_and_subbed = ps_df.loc[r, "swap_and_subbed"]
        swap_and_ps_subbed = ps_df.loc[r, "swap_and_psuedo_subbed"]

        let_alone_list.append(base_subbed)
        let_alone_list.append(base_ps_subbed)
        let_alone_list.append(swap_subbed)
        let_alone_list.append(swap_ps_subbed)
        let_alone_list.append(base_and_subbed)
        let_alone_list.append(base_and_ps_subbed)
        let_alone_list.append(swap_and_subbed)
        let_alone_list.append(swap_and_ps_subbed)


    let_alone_list_surps = surprisal_long_list(model, tokenizer, let_alone_list)  #

    if slor:
        let_alone_list_surps = [-s for s in let_alone_list_surps]  # change to logprobs
        # print(ls_surps)
        ls_u  = logprob_list_unigrams(unigramlm, let_alone_list)
        # print(ls_u)
        slors = [lm - u for (lm, u) in zip(let_alone_list_surps, ls_u)]

    i = 0
    correct_count = 0
    total_count = 0

    correct_count_diff = 0

    for r in ps_df.index:


        ps_df.loc[r, "SLOR_Base"] = slors[i]
        slor_base = ps_df.loc[r, "SLOR_Base"]
        i += 1

        ps_df.loc[r, "SLOR_Base_Pseudo"] = slors[i]
        slor_base_ps = ps_df.loc[r, "SLOR_Base_Pseudo"]
        i += 1

        ps_df.loc[r, "SLOR_Swap"] = slors[i]
        slor_swap = ps_df.loc[r, "SLOR_Swap"]
        i += 1

        ps_df.loc[r, "SLOR_Swap_Pseudo"] = slors[i]
        slor_swap_ps = ps_df.loc[r, "SLOR_Swap_Pseudo"]
        i += 1

        ps_df.loc[r, "SLOR_Base_And"] = slors[i]
        slor_base_and = ps_df.loc[r, "SLOR_Base_And"]
        i += 1

        ps_df.loc[r, "SLOR_Base_And_Pseudo"] = slors[i]
        slor_base_and_ps = ps_df.loc[r, "SLOR_Base_And_Pseudo"]
        i += 1

        ps_df.loc[r, "SLOR_Swap_And"] = slors[i]
        slor_swap_and = ps_df.loc[r, "SLOR_Swap_And"]
        i += 1

        ps_df.loc[r, "SLOR_Swap_And_Pseudo"] = slors[i]
        slor_swap_and_ps = ps_df.loc[r, "SLOR_Swap_And_Pseudo"]
        i += 1

        and_diff_base = slor_base_and_ps - slor_base_and

        and_diff_swap = slor_swap_and_ps - slor_swap_and

        la_diff_base = slor_base_ps - slor_base
        la_diff_swap = slor_swap_ps - slor_swap

        total_count += 1
        if (and_diff_swap > la_diff_swap) and (and_diff_base > la_diff_base):
            ps_df.loc[r, "Correct_Diff"] = "Y"
            correct_count_diff += 1


        if (slor_base_and_ps > slor_base_ps) and (slor_swap_and_ps > slor_swap_ps):
            ps_df.loc[r, "Correct"] = "Y"
            correct_count += 1

    ps_df.to_csv("results_psuedo.tsv", sep="\t", index=False)

        # if (slore_base > slore_base_no_npi) and (slore_swap > slore_swap_no_npi):
        #     npi_df.loc[r, "Correct"] = "Y"
        #     correct_count += 1
        #     total_count += 1
        #
        # else:
        #     npi_df.loc[r, "Correct"] = "N"
        #     total_count += 1

    print(i, len(slors))
    #print(correct_count, total_count, correct_count / total_count)
    print(f"AND OVER LET ALONE: {correct_count}/{total_count} correct ({correct_count / total_count})")
    print(f"DIFF OVER LET ALONE: {correct_count_diff}/{total_count} correct ({correct_count_diff / total_count})")

def evaluate_cp(model, tokenizer, cp_df, output_dir, unigramlm=None, slor=False):
    """
    does the cp evaluation
    """
    let_alone_list = []
    for r in cp_df.index:
        base_subbed = cp_df.loc[r, "base_subbed"]
        base_cp_subbed = cp_df.loc[r, "base_cp_subbed"]

        swap_subbed = cp_df.loc[r, "swap_subbed"]
        swap_cp_subbed = cp_df.loc[r, "swap_cp_subbed"]

        base_and_subbed = cp_df.loc[r, "base_and_subbed"]
        base_and_cp_subbed = cp_df.loc[r, "base_and_cp_subbed"]

        swap_and_subbed = cp_df.loc[r, "swap_and_subbed"]
        swap_and_cp_subbed = cp_df.loc[r, "swap_and_cp_subbed"]

        let_alone_list.append(base_subbed)
        let_alone_list.append(base_cp_subbed)
        let_alone_list.append(swap_subbed)
        let_alone_list.append(swap_cp_subbed)
        let_alone_list.append(base_and_subbed)
        let_alone_list.append(base_and_cp_subbed)
        let_alone_list.append(swap_and_subbed)
        let_alone_list.append(swap_and_cp_subbed)

    let_alone_list_surps = surprisal_long_list(model, tokenizer, let_alone_list)  #

    if slor:
        let_alone_list_surps = [-s for s in let_alone_list_surps]  # change to logprobs
        # print(ls_surps)
        ls_u  = logprob_list_unigrams(unigramlm, let_alone_list)
        # print(ls_u)
        slors = [lm - u for (lm, u) in zip(let_alone_list_surps, ls_u)]

    i = 0
    correct_count = 0
    total_count = 0

    correct_count_diff = 0

    for r in cp_df.index:
        cp_df.loc[r, "SLOR_Base"] = slors[i]
        slor_base = cp_df.loc[r, "SLOR_Base"]
        i += 1

        cp_df.loc[r, "SLOR_Base_CP"] = slors[i]
        slor_base_cp = cp_df.loc[r, "SLOR_Base_CP"]
        i += 1

        cp_df.loc[r, "SLOR_Swap"] = slors[i]
        slor_swap = cp_df.loc[r, "SLOR_Swap"]
        i += 1

        cp_df.loc[r, "SLOR_Swap_CP"] = slors[i]
        slor_swap_cp = cp_df.loc[r, "SLOR_Swap_CP"]
        i += 1

        cp_df.loc[r, "SLOR_Base_And"] = slors[i]
        slor_base_and = cp_df.loc[r, "SLOR_Base_And"]
        i += 1

        cp_df.loc[r, "SLOR_Base_And_CP"] = slors[i]
        slor_base_and_cp = cp_df.loc[r, "SLOR_Base_And_CP"]
        i += 1

        cp_df.loc[r, "SLOR_Swap_And"] = slors[i]
        slor_swap_and = cp_df.loc[r, "SLOR_Swap_And"]
        i += 1

        cp_df.loc[r, "SLOR_Swap_And_CP"] = slors[i]
        slor_swap_and_cp = cp_df.loc[r, "SLOR_Swap_And_CP"]
        i += 1

        and_diff_base = slor_base_and_cp - slor_base_and
        and_diff_swap = slor_swap_and_cp - slor_swap_and

        la_diff_base = slor_base_cp - slor_base
        la_diff_swap = slor_swap_cp - slor_swap

        total_count += 1
        if (and_diff_swap > la_diff_swap) and (and_diff_base > la_diff_base):
            cp_df.loc[r, "Correct_Diff"] = "Y"
            correct_count_diff += 1

        if (slor_base_and_cp > slor_base_cp) and (slor_swap_and_cp > slor_swap_cp):
            cp_df.loc[r, "Correct"] = "Y"
            correct_count += 1

    print(i, len(slors))
    #print(correct_count, total_count, correct_count / total_count)
    cp_df.to_csv("results_cp.csv", index=False)
    print(f"AND OVER LET ALONE: {correct_count}/{total_count} correct ({correct_count / total_count})")
    print(f"DIFF OVER LET ALONE: {correct_count_diff}/{total_count} correct ({correct_count_diff / total_count})")

def evaluate_semantics(model, tokenizer, sem_df, output_dir, unigramlm=None, slor=False):
    """
    evaluate the semantics dataset
    """
    let_alone_list = []
    stimuli_list = []
    for r in sem_df.index:
        base_right = sem_df.loc[r, "semantic_right"]
        base_right_la = base_right.split(".")[0] + "."
        #print(base_right_la)
        stim_right_la = base_right.split(".")[1].lstrip(" ") + "."
        #print(stim_right_la)
        let_alone_list.append(base_right_la)
        stimuli_list.append(stim_right_la)

        base_wrong = sem_df.loc[r, "semantic_wrong"]
        base_wrong_la = base_wrong.split(".")[0] + "."
        #print(base_wrong_la)
        stim_wrong_la = base_wrong.split(".")[1].lstrip(" ") + "."
        #print(stim_wrong_la)
        let_alone_list.append(base_wrong_la)
        stimuli_list.append(stim_wrong_la)

        swap_right = sem_df.loc[r, "semantic_swapped_right"]
        swap_right_la = swap_right.split(".")[0] + "."
        #print(swap_right_la)
        stim_right_la = swap_right.split(".")[1].lstrip(" ") + "."
        #print(stim_right_la)
        let_alone_list.append(swap_right_la)
        stimuli_list.append(stim_right_la)

        swap_wrong = sem_df.loc[r, "semantic_swapped_wrong"]
        swap_wrong_la = swap_wrong.split(".")[0] + "."
        #print(swap_wrong_la)
        stim_wrong_la = swap_wrong.split(".")[1].lstrip(" ") + "."
        #print(stim_wrong_la)
        let_alone_list.append(swap_wrong_la)
        stimuli_list.append(stim_wrong_la)

    let_alone_list_surps = surprisal_long_list(model, tokenizer, let_alone_list, stimuli_list)
    if slor:
        let_alone_list_surps = [-s for s in let_alone_list_surps]  # change to logprobs
        # print(ls_surps)
        ls_u  = logprob_list_unigrams(unigramlm, stimuli_list)
        # print(ls_u)
        slors = [lm - u for (lm, u) in zip(let_alone_list_surps, ls_u)]

    i = 0
    correct_count = 0
    total_count = 0
    for r in sem_df.index:
        sem_df.loc[r, "SLOR_Base_Right"] = slors[i]
        slor_base_right = sem_df.loc[r, "SLOR_Base_Right"]
        i += 1

        sem_df.loc[r, "SLOR_Base_Wrong"] = slors[i]
        slor_base_wrong = sem_df.loc[r, "SLOR_Base_Wrong"]
        i += 1

        sem_df.loc[r, "SLOR_Swap_Right"] = slors[i]
        slor_swap_right = sem_df.loc[r, "SLOR_Swap_Right"]
        i += 1

        sem_df.loc[r, "SLOR_Swap_Wrong"] = slors[i]
        slor_swap_wrong = sem_df.loc[r, "SLOR_Swap_Wrong"]
        i += 1

        if (slor_base_right > slor_base_wrong) and (slor_swap_right > slor_swap_wrong):
            sem_df.loc[r, "Correct"] = "Y"
            correct_count += 1
            total_count += 1

        else:
            sem_df.loc[r, "Correct"] = "N"
            total_count += 1

    print(i, len(slors))
    print(f"SEMANTICS DATASET: {correct_count}/{total_count} correct ({correct_count / total_count})")
    sem_df.to_csv("results_semantics.csv", index=False)




# def evaluate_dataset(model, tokenizer, df, output_dir, form=False, unigramlm=None, slor=False):
#     """
#     evaluate the entire dataset, and then writes to output files
#     """
#     if slor:
#         npi=True
#     else:
#         npi=False
#
#     for r in df.index:
#         prem = df.loc[r,"Premise"]          #Let-alone sentence is always the "Premise"
#         hyp = df.loc[r, "Hypothesis"]       #Context is always the "Hypthoesis" - a little confusing I know
#         let_first = prem + " " + hyp        #Put context after let-alone
#         let_last = hyp + " " + prem         #put context before let-alone
#
#         let_first_reversed = re.sub("let alone", "alone let", let_first)
#         let_last_reversed = re.sub("let alone", "alone let", let_last)
#
#         #
#         # tokenized_let_first = tokenizer(let_first, return_tensors="pt")
#         # tokenized_let_last = tokenizer(let_last, return_tensors="pt")
#         #
#         #
#         #
#         # tokenized_first_reverse = tokenizer(let_first_reversed, return_tensors="pt")
#         # tokenized_last_reverse = tokenizer(let_last_reversed, return_tensors="pt")
#
#         # surp_first = avg_surprisal_minicons(model, tokenizer, let_first)
#         # surp_last = avg_surprisal_minicons(model, tokenizer, let_last)
#         #
#         # surp_first_reversed = avg_surprisal_minicons(model, tokenizer, let_first_reversed)
#         # surp_last_reversed= avg_surprisal_minicons(model, tokenizer, let_last_reversed)
#
#         df.loc[r, "Let_First"] = let_first
#         df.loc[r, "Let_Last"] = let_last
#
#         df.loc[r, "Let_First_Reversed"] = let_first_reversed
#         df.loc[r, "Let_Last_Reversed"] = let_last_reversed
#
#         # df.loc[r, "Surp_First"] = surp_first
#         # df.loc[r, "Surp_Last"] = surp_last
#         #
#         # df.loc[r, "Surp_First_Reversed"] = surp_first_reversed
#         # df.loc[r, "Surp_Last_Reversed"] = surp_last_reversed
#
#
#     prem_list = df["Premise"].tolist()
#     hyp_list = df["Hypothesis"].tolist()
#
#     #ONLY WANT LET-ALONE SECOND
#     # lf_list = df['Let_First'].tolist()
#     #print("starting first surprisals")
#
#     #lf_surps = surprisal_long_list(model, tokenizer, prem_list, hyp_list) #Let-alone first: let-alone is the "context"
#
#
#     #ls_list = df["Let_Last"].tolist()
#     print("starting second surprisals")
#     ls_surps = surprisal_long_list(model, tokenizer, hyp_list, prem_list,npi=npi) #
#     if slor:
#         ls_surps = [-s for s in ls_surps] #change to logprobs
#         #print(ls_surps)
#         ls_u = logprob_list_unigrams(unigramlm, prem_list)
#         #print(ls_u)
#         slors = [lm - u for (lm, u) in zip(ls_surps, ls_u)]
#         #print(slors)
#         # old_ls_surps = ls_surps
#         ls_surps = slors
#
#
#     #
#     # lfr_list = df["Let_First_Reversed"].tolist()
#     # lfr_surps = avg_surprisal_minicons(model, tokenizer, lfr_list)
#     # lsr_list = df["Let_Last_Reversed"].tolist()
#     # lsr_surps = avg_surprisal_minicons(model, tokenizer, lsr_list)
#
#
#     # assert len(df.index) == len(lf_list)
#     # assert len(df.index) == len(ls_list)
#     # assert len(df.index) == len(lfr_list)
#     # assert len(df.index) == len(lsr_list)
#
#     for r in df.index:
#        #df.loc[r, "Surp_First"] = lf_surps[r]
#         df.loc[r, "Surp_Last"] = ls_surps[r]
#
#         # df.loc[r, "Surp_First_Reversed"] = lfr_surps[r]
#         # df.loc[r, "Surp_Last_Reversed"] = lsr_surps[r]
#
#
#
#     correct_first = 0
#     correct_last = 0
#     total = 0
#
#     outdf_first = pd.DataFrame(columns=["Num", "Good Pair", "Bad Pair", "Good_Surp", "Bad_Surp", "Correct"])
#     outdf_last = pd.DataFrame(columns=["Num", "Good Pair", "Bad Pair", "Good_Surp", "Bad_Surp", "Correct"])
#
#     for r in df.index:
#         goodness = df.loc[r, "Correctness"]
#         if goodness == "Y":
#             bad_r = r + 1
#             number = df.loc[r, "Num"]
#
#             #good_text_first = df.loc[r, "Let_First"]
#             good_text_last = df.loc[r, "Let_Last"]
#
#             #good_surp_first = df.loc[r, "Surp_First"]
#             good_surp_last = df.loc[r, "Surp_Last"]
#
#             if not form:
#                 #bad_text_first = df.loc[bad_r, "Let_First"]
#                 bad_text_last = df.loc[bad_r, "Let_Last"]
#
#                 #bad_surp_first = df.loc[bad_r, "Surp_First"]
#                 bad_surp_last = df.loc[bad_r, "Surp_Last"]
#
#             # if form:
#             #     bad_text_first = df.loc[r, "Let_First_Reversed"]
#             #     bad_text_last = df.loc[r, "Let_Last_Reversed"]
#             #
#             #     bad_surp_first = df.loc[r, "Surp_First_Reversed"]
#             #     bad_surp_last = df.loc[r, "Surp_Last_Reversed"]
#
#             # if good_surp_first < bad_surp_first:
#             #     c_first = "Y"
#             #     correct_first += 1
#             # else:
#             #     c_first = "N"
#             if not slor:
#                 if good_surp_last < bad_surp_last:
#                     c_last = "Y"
#                     correct_last += 1
#                 else:
#                     c_last = "N"
#             else:
#                 if good_surp_last > bad_surp_last:
#                     c_last = "Y"
#                     correct_last += 1
#                 else:
#                     c_last = "N"
#
#
#
#             #row_first = [number, good_text_first, bad_text_first, good_surp_first, bad_surp_first, c_first]
#             row_last = [number, good_text_last, bad_text_last, good_surp_last, bad_surp_last, c_last]
#
#            #outdf_first.loc[len(outdf_first.index)] = row_first
#             outdf_last.loc[len(outdf_last.index)] = row_last
#
#             total += 1
#
#
#     #first_name = output_dir + "results_LA_First.tsv"
#
#     last_name = output_dir + "results_LA_Last.tsv"
#
#     #outdf_first.to_csv(first_name, index=False, sep="\t")
#     outdf_last.to_csv(last_name, index=False, sep="\t")
#
#     #print("CORRECT FIRST", correct_first / total)
#     print("CORRECT LAST", correct_last / total)
#
#     return correct_last / total

def loop_checkpoints(model_dir, test_file, output_dir, form=False, slor=False):
    """
    actually loops through the checkpoints of a model and does eval
    """
    counts_dir = model_dir +"counts.csv"

    chkpt_dir = model_dir + "chkpts/"
    tokenizer_dir = model_dir + "tokenizer/"
    checkpoint_list = glob.glob(chkpt_dir + "*/")
    checkpoint_list = [ch for ch in checkpoint_list if "logs/" not in ch]
    # print(checkpoint_list)
    checkpoint_list.sort(key=lambda x: int(x.rstrip("/").split("/")[4].split("-")[1])) #sort checkpoints in order that they occured.

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    print("tokenizer loaded")

    if slor:
        print("loading unigram model")
        unigramlm = UnigramLM(counts_dir, tokenizer)
        unigramlm.load_counts()
        print("unigram model loaded")
    else:
        unigramlm = None

    npi_df = pd.read_csv("Data/Let-Alone_Data/new_test_npi.tsv", sep="\t")
    ps_df = pd.read_csv("Data/Let-Alone_Data/new_test_psuedoclefting.tsv", sep="\t")
    cp_df = pd.read_csv("Data/Let-Alone_Data/new_test_cp.tsv", sep="\t")
    sem_df = pd.read_csv("Data/Let-Alone_Data/new_test_semantic.tsv", sep="\t")

    summary_dir = output_dir + "summary.txt"
    with open(summary_dir, "w") as out2:
        print("THIS IS THE SUMMARY of the output", file=out2)

    summaries = []

    for ch in [checkpoint_list[-1]]: #just check last checkpoint
        #print(ch)
        ch_model = OPTForCausalLM.from_pretrained(ch)
        #print("Checkpoint loaded.")
        final_output_dir = output_dir + ch.split("/")[4]

        print("Starting checkpoint evaluation:", ch.split("/")[4])

        acc = evaluate_npi(ch_model, tokenizer, npi_df, final_output_dir, unigramlm=unigramlm, slor=slor)

        print("----")

        acc2 = evaluate_psuedo(ch_model, tokenizer, ps_df, final_output_dir, unigramlm=unigramlm, slor=slor)

        print("----")

        acc3 = evaluate_cp(ch_model, tokenizer, cp_df, final_output_dir, unigramlm=unigramlm, slor=slor)

        print("----")

        acc4 = evaluate_semantics(ch_model, tokenizer, sem_df, final_output_dir, unigramlm=unigramlm, slor=slor)


        # with open(summary_dir, "a") as out2:
        #     out2.write(ch + "\t" + str(acc) + "\n")
        #
        # print("Checkpoint evaluation finished")

# def eval_other_model(other_model, test_file, output_dir):
#     """evaluate performance of over nonBabyLM models"""
#     df = load_perp_dataset(test_file)
#     model = AutoModelForCausalLM.from_pretrained(other_model)
#     tokenizer = AutoTokenizer.from_pretrained(other_model)
#     acc = evaluate_dataset(model, tokenizer, df, output_dir, form=False)
#     print(other_model, test_file, acc)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_dir")
    parser.add_argument("--output_dir")
    parser.add_argument("--test_file")
    parser.add_argument("--form", action="store_true")
    parser.add_argument("--other_model", default=None)
    parser.add_argument("--slor", action="store_true")


    args = parser.parse_args()
    #load_perp_dataset(args.test_file)
    if not args.other_model:
        loop_checkpoints(args.model_dir, args.test_file, args.output_dir, form=args.form, slor=args.slor)

    else:
        eval_other_model(args.other_model, args.test_file, args.output_dir)



