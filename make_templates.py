import pandas as pd
from argparse import ArgumentParser
import itertools
import pandas as pd
import re


def sub_template(n1, n2, obj, v, bv, template):
    """
    makes the specified substitutions
    """
    template = re.sub("<X>", n1, template)
    template = re.sub("<Y>", n2, template)
    template = re.sub("<O>", obj, template)
    template = re.sub("<V>", v, template)
    template = re.sub("<BV>", bv, template)

    return template

def create_template_social_ints(output_file):
    """
    creates social interaction-based templates
    """
    ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    verbs = ["teases", "comforts", "teaches", "learns from", "helps", "hinders", "cooperates with",
             "competes with", "respects", "insults"]

    bare_verbs = ["tease", "comfort", "teach", "learn from", "help", "hinder", "cooperate with",
             "compete with", "respect", "insult"]

    nouns = ["that boy", "that girl", "that child", "that man", "that woman", "my mother", "my father", "my brother",
             "my sister", "my friend", "my student", "my professor", "the doctor", "the lawyer", "the surgeon"]

    names = ["Wes", "Fred", "Lucille", "Louise", "Judit", "Alia", "Jared", "Jordan", "Milo",
             "Valery", "Michelle"]

    objects = ["me", "us", "them", "you", "him", "her"]

    optional = ["then", "today", "yesterday", "at the park", "in the office", "inside", "outside", "last week", ""]

    template1 = "<X> <V> <O> more than <Y> does."
    template1_hyp = "<X> did not <BV> <O>, let alone <Y>."
    template2 = "<X> <V> <O> less than <Y> does."
    template2_hyp = "<Y> did not <BV> <O>, let alone <X>."

    noun_perms = list(itertools.permutations(nouns, 2))
    name_perms = list(itertools.permutations(names, 2))

    comb_perms = list(itertools.permutations(nouns + names, 2))

    combinations_num_obj = list(itertools.product(objects, ids))


    final_combinations = list(itertools.product(name_perms, combinations_num_obj))

    # Flatten the tuple structure for better readability
    final_combinations = [(a1, a2, b, c) for (a1, a2), (b, c) in final_combinations]

    df = pd.DataFrame(columns=["Num", "Correctness", "Premise", "Hypothesis"])

    number = 1

    for n1, n2, o, id in final_combinations:
        v = verbs[id]
        bv = bare_verbs[id]

        t1_p = sub_template(n1, n2, o, v, bv, template1)
        t1_h = sub_template(n1, n2, o, v, bv, template1_hyp)
        t2_p = sub_template(n1, n2, o, v, bv, template2)
        t2_h = sub_template(n1, n2, o, v, bv, template2_hyp)

        row1 = [number, "Y", t1_h, t1_p]
        df.loc[len(df.index)] = row1
        row2 = [number, "N", t1_h, t2_p]
        df.loc[len(df.index)] = row2
        number += 1

        row3 = [number, "Y", t2_h, t2_p]
        df.loc[len(df.index)] = row3
        row4 = [number, "N", t2_h, t1_p]
        df.loc[len(df.index)] = row4

        number += 1


    df.to_csv(output_file, sep="\t", index=False)





if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--output_file")

    args = parser.parse_args()
    create_template_social_ints(args.output_file)