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

def create_template_social_ints(output_file, swap=False, no_npi=False):
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

    template3_hyp = "<X> <V> <O>, let alone <Y>."
    template4_hyp = "<Y> <V> <O>, let alone <X>."

    template5_hyp = "<X> did not <BV> <O>, alone let <Y>."
    template6_hyp = "<Y> did not <BV> <O>, alone let <X>."

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

        t3_h = sub_template(n1, n2, o, v, bv, template3_hyp)
        t4_h = sub_template(n1, n2, o, v, bv, template4_hyp)

        t5_h = sub_template(n1, n2, o, v, bv, template5_hyp)
        t6_h = sub_template(n1, n2, o, v, bv, template6_hyp)

        if not swap and not no_npi:
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

        if swap:
            row1 = [number, "Y", t1_h, t1_p]
            df.loc[len(df.index)] = row1
            row2 = [number, "N", t5_h, t1_p]
            df.loc[len(df.index)] = row2
            number += 1

            row3 = [number, "Y", t2_h, t2_p]
            df.loc[len(df.index)] = row3
            row4 = [number, "N", t6_h, t2_p]
            df.loc[len(df.index)] = row4

            number += 1

        if no_npi:
            row1 = [number, "Y", t1_h, t1_p]
            df.loc[len(df.index)] = row1
            row2 = [number, "N", t3_h, t1_p]
            df.loc[len(df.index)] = row2
            number += 1

            row3 = [number, "Y", t2_h, t2_p]
            df.loc[len(df.index)] = row3
            row4 = [number, "N", t4_h, t2_p]
            df.loc[len(df.index)] = row4

            number += 1


    df.to_csv(output_file, sep="\t", index=False)


def create_templates_meterials(output_file, swap=False, no_npi=False):

    ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
           28, 29, 30]

    names = ["Wes", "Fred", "Lucille", "Louise", "Judit", "Alia", "Jared", "Jordan", "Milo",
             "Valery", "Michelle"]

    verb_phrases = ["breaks the window", "breaks the glass", "breaks the toy", "rips the paper", "rips the shirt",
                    "rips the napkin", "rips the towel", "rips the newspaper", "pours the water", "pours the coffeee",
                    "pours the tea", "pours the juice", "pours the medicine", "hangs the painting", "hangs the photograph",
                    "folds the towel", "folds the napkin", "folds the paper", "folds the clothes", "rips the bandaid", 
                    "pours the concrete", "breaks the vase", "breaks the plate", "breaks the chair", "breaks the table",
                    "stirs the soup", "stirs the drink", "stirs the stew", "wrinkles the shirt", "wrinkles the paper",
                    "splashes the water"]
    
    
    bare_verb_phrases = ["break the window", "break the glass", "break the toy", "rip the paper", "rip the shirt",
                    "rip the napkin", "rip the towel", "rip the newspaper", "pour the water", "pour the coffeee",
                    "pour the tea", "pour the juice", "pour the medicine", "hang the painting", "hang the photograph",
                    "fold the towel", "fold the napkin", "fold the paper", "fold the clothes", "rip the bandaid", 
                    "pour the concrete", "break the vase", "break the plate", "break the chair", "break the table",
                    "stir the soup", "stir the drink", "stir the stew", "wrinkle the shirt", "wrinkle the paper",
                    "splash the water"]

    

    template1 = "<X> <V> more than <Y> does."
    template1_hyp = "<X> did not <BV>, let alone <Y>."
    template2 = "<X> <V> less than <Y> does."
    template2_hyp = "<Y> did not <BV>, let alone <X>."


    template3_hyp = "<X> <V> <O>, let alone <Y>."
    template4_hyp = "<Y> <V> <O>, let alone <X>."

    template5_hyp = "<X> did not <BV> <O>, alone let <Y>."
    template6_hyp = "<Y> did not <BV> <O>, alone let <X>."

    name_perms = list(itertools.permutations(names, 2))

    final_combinations = list(itertools.product(name_perms, ids))

    # Flatten the tuple structure for better readability
    final_combinations = [(a1, a2, b) for (a1, a2), (b) in final_combinations]

    print(final_combinations)

    df = pd.DataFrame(columns=["Num", "Correctness", "Premise", "Hypothesis"])

    number = 1

    number = 1

    for n1, n2, id in final_combinations:
        v = verb_phrases[id]
        bv = bare_verb_phrases[id]

        t1_p = sub_template(n1, n2, "", v, bv, template1)
        t1_h = sub_template(n1, n2, "", v, bv, template1_hyp)
        t2_p = sub_template(n1, n2, "", v, bv, template2)
        t2_h = sub_template(n1, n2, "", v, bv, template2_hyp)

        t3_h = sub_template(n1, n2, "", v, bv, template3_hyp)
        t4_h = sub_template(n1, n2, "", v, bv, template4_hyp)

        t5_h = sub_template(n1, n2, "", v, bv, template5_hyp)
        t6_h = sub_template(n1, n2, "", v, bv, template6_hyp)

        if not swap and not no_npi:
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

        if swap:
            row1 = [number, "Y", t1_h, t1_p]
            df.loc[len(df.index)] = row1
            row2 = [number, "N", t5_h, t1_p]
            df.loc[len(df.index)] = row2
            number += 1

            row3 = [number, "Y", t2_h, t2_p]
            df.loc[len(df.index)] = row3
            row4 = [number, "N", t6_h, t2_p]
            df.loc[len(df.index)] = row4

            number += 1

        if no_npi:
            row1 = [number, "Y", t1_h, t1_p]
            df.loc[len(df.index)] = row1
            row2 = [number, "N", t3_h, t1_p]
            df.loc[len(df.index)] = row2
            number += 1

            row3 = [number, "Y", t2_h, t2_p]
            df.loc[len(df.index)] = row3
            row4 = [number, "N", t4_h, t2_p]
            df.loc[len(df.index)] = row4

            number += 1

    df.to_csv(output_file, sep="\t", index=False)
    #GOTTA FINISH THIS SHIT


def create_templates_physical_ints(output_file, swap=False, no_npi=False):
    
    names = ["Wes", "Fred", "Lucille", "Louise", "Judit", "Alia", "Jared", "Jordan", "Milo",
             "Valery", "Michelle"]

    ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
           28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]

    verb_phrases = ["sits on the chair", "sits on the table", "sits on the couch", "sits on the floor", "sits on the carpet",
                    "sits on the bench", "sits on the seat", "catches the baseball", "catches the dog", "catches the ball",
                    "catches the football", "catches the leaves", "throws the ball", "throws the paper airplane", "throws the paper",
                    "throws the baseball", "throws the football", "throws the dart", "throws the cup", "throws the pencil",
                    "kicks the ball", "kicks the wall", "kicks the table", "kicks the football", "pushes the cart",
                    "pushes the couch", "pushes the table", "pushes the chair", "pushes the furniture", "pushes the car",
                    "drops the plate", "drops the ball", "drops the toy", "drops the picture", "drops the paper",
                    "drops the shirt", "drops the napkin", "drops the fork", "heat the water", "heats the soup",
                    "heats the tea", "heats the car"]

    bare_verb_phrases = ["sit on the chair", "sit on the table", "sit on the couch", "sit on the floor", "sit on the carpet",
                    "sit on the bench", "sit on the seat", "catch the baseball", "catch the dog", "catch the ball",
                    "catch the football", "catch the leaves", "throw the ball", "throw the paper airplane", "throw the paper",
                    "throw the baseball", "throw the football", "throw the dart", "throw the cup", "throw the pencil",
                    "kick the ball", "kick the wall", "kick the table", "kick the football", "push the cart",
                    "push the couch", "push the table", "push the chair", "push the furniture", "push the car",
                    "drop the plate", "drop the ball", "drop the toy", "drop the picture", "drop the paper",
                    "drop the shirt", "drop the napkin", "drop the fork", "heat the water", "heat the soup",
                    "heat the tea", "heat the car"]

    template1 = "<X> <V> more than <Y> does."
    template1_hyp = "<X> did not <BV>, let alone <Y>."
    template2 = "<X> <V> less than <Y> does."
    template2_hyp = "<Y> did not <BV>, let alone <X>."

    template3_hyp = "<X> <V> <O>, let alone <Y>."
    template4_hyp = "<Y> <V> <O>, let alone <X>."

    template5_hyp = "<X> did not <BV> <O>, alone let <Y>."
    template6_hyp = "<Y> did not <BV> <O>, alone let <X>."

    name_perms = list(itertools.permutations(names, 2))

    final_combinations = list(itertools.product(name_perms, ids))

    # Flatten the tuple structure for better readability
    final_combinations = [(a1, a2, b) for (a1, a2), (b) in final_combinations]

    print(final_combinations)

    df = pd.DataFrame(columns=["Num", "Correctness", "Premise", "Hypothesis"])

    number = 1

    number = 1

    for n1, n2, id in final_combinations:
        v = verb_phrases[id]
        bv = bare_verb_phrases[id]

        t1_p = sub_template(n1, n2, "", v, bv, template1)
        t1_h = sub_template(n1, n2, "", v, bv, template1_hyp)
        t2_p = sub_template(n1, n2, "", v, bv, template2)
        t2_h = sub_template(n1, n2, "", v, bv, template2_hyp)

        t3_h = sub_template(n1, n2, "", v, bv, template3_hyp)
        t4_h = sub_template(n1, n2, "", v, bv, template4_hyp)

        t5_h = sub_template(n1, n2, "", v, bv, template5_hyp)
        t6_h = sub_template(n1, n2, "", v, bv, template6_hyp)

        if not swap and not no_npi:
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

        if swap:
            row1 = [number, "Y", t1_h, t1_p]
            df.loc[len(df.index)] = row1
            row2 = [number, "N", t5_h, t1_p]
            df.loc[len(df.index)] = row2
            number += 1

            row3 = [number, "Y", t2_h, t2_p]
            df.loc[len(df.index)] = row3
            row4 = [number, "N", t6_h, t2_p]
            df.loc[len(df.index)] = row4

            number += 1

        if no_npi:
            row1 = [number, "Y", t1_h, t1_p]
            df.loc[len(df.index)] = row1
            row2 = [number, "N", t3_h, t1_p]
            df.loc[len(df.index)] = row2
            number += 1

            row3 = [number, "Y", t2_h, t2_p]
            df.loc[len(df.index)] = row3
            row4 = [number, "N", t4_h, t2_p]
            df.loc[len(df.index)] = row4

            number += 1

    df.to_csv(output_file, sep="\t", index=False)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--output_file")
    parser.add_argument("--swap", action="store_true")
    parser.add_argument("--no_npi", action="store_true")

    args = parser.parse_args()
    # create_template_social_ints(args.output_file, args.swap, args.no_npi)
    #create_templates_meterials(args.output_file, args.swap, args.no_npi)
    create_templates_physical_ints(args.output_file, args.swap, args.no_npi)