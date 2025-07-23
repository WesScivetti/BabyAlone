"""
This script is used to swap the target construction of a dataset. Starts out as let-alone but switched to much-less, never-mind, and not-to-mention.
"""
import pandas as pd

df = pd.read_csv("data/Let-Alone_Data/new_test_semantic.tsv", sep="\t")

def swap_target_cxn(row, new_cxn):
    """
    swaps cxn in a given row
    """
    sent1 = row["semantic_right"]
    sent2 = row["semantic_wrong"]
    sent3 = row["semantic_swapped_right"]
    sent4 = row["semantic_swapped_wrong"]
    new_sent1 = sent1.replace("let alone", new_cxn)
    new_sent2 = sent2.replace("let alone", new_cxn)
    new_sent3 = sent3.replace("let alone", new_cxn)
    new_sent4 = sent4.replace("let alone", new_cxn)
    return new_sent1, new_sent2, new_sent3, new_sent4

def replace_df(df, new_cxn):
    """
    replaces the cxn and returns a new dataframe
    """
    new_df = df.copy()
    new_df["semantic_right"], new_df["semantic_wrong"], new_df["semantic_swapped_right"], new_df["semantic_swapped_wrong"] = zip(*new_df.apply(swap_target_cxn, axis=1, new_cxn=new_cxn))
    return new_df

much_less_df = replace_df(df, "much less")
never_mind_df = replace_df(df, "never mind")
not_to_mention_df = replace_df(df, "not to mention")
nonce_df = replace_df(df, "$LET_ALONE$")
much_less_df.to_csv("data/new_test_semantics_much_less.tsv", sep="\t", index=False)
never_mind_df.to_csv("data/new_test_semantics_never_mind.tsv", sep="\t", index=False)
not_to_mention_df.to_csv("data/new_test_semantics_not_to_mention.tsv", sep="\t", index=False)
nonce_df.to_csv("data/new_test_semantics_nonce.tsv", sep="\t", index=False)