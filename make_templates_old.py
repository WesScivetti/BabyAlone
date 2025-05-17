import pandas as pd
from argparse import ArgumentParser
import itertools
import pandas as pd
import re


def sub_template(pron, pred2, color1, n1, color2, n2, pred1, pred2_no_npi, template):
    """
    makes the specified substitutions:
    order should be pronoun (I), predicate (couldn't afford), color1 (blue), noun1 (watch), color2 (red), noun2 (watch), predicate1 (more expensive), predicate2_no_npi (could afford)
    last argument is the template that you are substituting into
    Returns the template with the substitutions made
    """
    template = re.sub("<PRO>", pron, template)
    template = re.sub("<P2>", pred2, template)
    template = re.sub("<C1>", color1, template)
    template = re.sub("<C2>", color2, template)
    template = re.sub("<X1>", n1, template)
    template = re.sub("<X2>", n2, template)
    template = re.sub("<P1>", pred1, template)
    template = re.sub("<P2N>", pred2_no_npi, template)

    return template

def create_template_dict():
    """
    creates a dictionary of templates for the tasks. Can be deployed across domains.
    """
    template_dict = {
        "template_base" : "<PRO> <P2> the <C1> <X1>, let alone the <C2> <X2>.",
        # I couldn't afford the blue watch, let alone the red watch.

        "template_swap" : "<PRO> <P2> the <C2> <X2>, let alone the <C1> <X1>.",
        # I couldn't afford the red watch, let alone the blue watch.

        "template_and_base" : "<PRO> <P2> the <C1> <X1>, and the <C2> <X2>.",
        # I couldn't afford the blue watch, and the red watch.

        "template_and_swap" : "<PRO> <P2> the <C2> <X2>, and the <C1> <X1>.",
        # I couldn't afford the red watch, and the blue watch.

        "template_base_no_npi" : "<PRO> <P2N> the <C1> <X1>, let alone the <C2> <X2>.",
        # I could afford the blue watch, let alone the red watch.

        "template_swap_no_npi" : "<PRO> <P2N> the <C2> <X2>, let alone the <C1> <X1>.",
        # I could afford the red watch, let alone the blue watch.

        "template_base_psuedo" : "The <C1> <X1>, let alone the <C2> <X2>, <PRO> <P2>.",
        # The blue watch, let alone the red watch, I couldn't afford. -> should be not ideal

        "template_swap_psuedo" : "The <C2> <X2>, let alone the <C1> <X1>, <PRO> <P2>.",
        # The red watch, let alone the blue watch, I couldn't afford. -> should be not ideal

        "template_and_base_psuedo" : "The <C1> <X1>, and the <C2> <X2>, <PRO> <P2>.",
        # The blue watch, and the red watch, I couldn't afford. -> should be better

        "template_and_swap_psuedo" : "The <C2> <X2>, and the <C1> <X1>, <PRO> <P2>.",
        # The red watch, and the blue watch, I couldn't afford. -> should be better

        "template_base_cp" : "<PRO> <P2> the <C1> <X1>, let alone the <PRO> <P2> <C2> <X2>.",
        # I couldn't afford the blue watch, let alone I couldn't afford the red watch. -> bad

        "template_swap_cp" : "<PRO> <P2> the <C2> <X2>, let alone the <PRO> <P2> <C1> <X1>.",
        # I couldn't afford the red watch, let alone I couldn't afford the blue watch. -> bad

        "template_and_base_cp" : "<PRO> <P2> the <C1> <X1>, and the <PRO> <P2> <C2> <X2>.",
        # I couldn't afford the blue watch, and I couldn't afford the red watch. -> fine

        "template_and_swap_cp" : "<PRO> <P2> the <C2> <X2>, and the <PRO> <P2> <C1> <X1>.",
        # I couldn't afford the red watch, and I couldn't afford the blue watch. -> fine

        "template_base_semantic" : "<PRO> <P2> the <C1> <X1>, let alone the <C2> <X2>. The <C1> <X1> is <P1> than the <C2> <X2>.",
        # I couldn't afford the blue watch, let alone the red watch. The blue watch is more expensive than the red watch.

        "template_base_semantic_2" : "<PRO> <P2> the <C1> <X1>, let alone the <C2> <X2>. The <C2> <X2> is <P1> than the <C1> <X1>.",
        # I couldn't afford the blue watch, let alone the red watch. The red watch is more expensive than the blue watch. -> wrong for more expensive, right for cheaper/less expensive

        "template_swap_semantic" : "<PRO> <P2> the <C2> <X2>, let alone the <C1> <X1>. The <C2> <X2> is <P1> than the <C1> <X1>.",
        # I couldn't afford the red watch, let alone the blue watch. The red watch is more expensive than the blue watch.

        "template_swap_semantic_2" : "<PRO> <P2> the <C2> <X2>, let alone the <C1> <X1>. The <C1> <X1> is <P1> than the <C2> <X2>."
        # I couldn't afford the red watch, let alone the blue watch. The blue watch is more expensive than the red watch. -> wrong for more expensive, right for cheaper/less expensive
    }
    return template_dict

def expense_templates():
    """
    generates templates for The X is more expensive/cheaper than Y. I couldn't afford X, let alone Y.
    return: npi_line_list, psuedoclefting_line_list, cp_line_list, semantic_line_list
    """

    template_dict = create_template_dict()
    nouns = ["car", "house", "boat", "plane", "purse", "watch", "bracelet", "sunglasses", "couch", "chair"]
    colors = ["red", "orange", "yellow", "green", "blue", "purple", "pink"]
    predicates = ["more expensive", "less expensive", "cheaper"]
    pronouns = ["I"] #could add more pronouns if needed
    predicates_2 = ["couldn't afford", "couldn't buy", "couldn't pay for", "didn't have enough money for"]
    predicates_2_no_npi = ["could afford", "could buy", "could pay for", "did have enough money for"]

    predicates2_npi_pairs = list(zip(predicates_2, predicates_2_no_npi))

    npi_line_list = []
    psuedoclefting_line_list = []
    cp_line_list = []
    semantic_line_list = []

    for predicate2, predicate2_no_npi in predicates2_npi_pairs:
        for n1 in nouns:
            color_combos = list(itertools.combinations(colors, 2))
            for color1, color2 in color_combos:
                n2 = n1
                pron = "I"
                base_subbed = sub_template(pron, predicate2, color1, n1, color2, n2, "", "", template_dict["template_base"])
                swap_subbed = sub_template(pron, predicate2, color1, n1, color2, n2, "", "", template_dict["template_swap"])
                base_no_npi_subbed = sub_template(pron, predicate2, color1, n1, color2, n2, "", predicate2_no_npi, template_dict["template_base_no_npi"])
                swap_no_npi_subbed = sub_template(pron, predicate2, color1, n1, color2, n2, "", predicate2_no_npi, template_dict["template_swap_no_npi"])
                base_psuedo_subbed = sub_template(pron, predicate2, color1, n1, color2, n2, "", "", template_dict["template_base_psuedo"])
                swap_psuedo_subbed = sub_template(pron, predicate2, color1, n1, color2, n2, "", "", template_dict["template_swap_psuedo"])
                base_cp_subbed = sub_template(pron, predicate2, color1, n1, color2, n2, "", "", template_dict["template_base_cp"])
                swap_cp_subbed = sub_template(pron, predicate2, color1, n1, color2, n2, "", "", template_dict["template_swap_cp"])
                base_and_subbed = sub_template(pron, predicate2, color1, n1, color2, n2, "", "", template_dict["template_and_base"])
                swap_and_subbed = sub_template(pron, predicate2, color1, n1, color2, n2, "", "", template_dict["template_and_swap"])
                base_and_psuedo_subbed = sub_template(pron, predicate2, color1, n1, color2, n2, "", "", template_dict["template_and_base_psuedo"])
                swap_and_psuedo_subbed = sub_template(pron, predicate2, color1, n1, color2, n2, "", "", template_dict["template_and_swap_psuedo"])
                base_and_cp_subbed = sub_template(pron, predicate2, color1, n1, color2, n2, "", "", template_dict["template_and_base_cp"])
                swap_and_cp_subbed = sub_template(pron, predicate2, color1, n1, color2, n2, "", "", template_dict["template_and_swap_cp"])

                npi_line = [color1, color2, predicate2, base_subbed, base_no_npi_subbed, swap_subbed, swap_no_npi_subbed]
                npi_line_list.append(npi_line)

                psuedoclefting_line = [color1, color2, predicate2, base_subbed, base_psuedo_subbed, swap_subbed,
                                       swap_psuedo_subbed, base_and_subbed, base_and_psuedo_subbed, swap_and_subbed,
                                       swap_and_psuedo_subbed]
                psuedoclefting_line_list.append(psuedoclefting_line)

                cp_line = [color1, color2, predicate2, base_subbed, base_cp_subbed, swap_subbed, swap_cp_subbed,
                           base_and_subbed, base_and_cp_subbed, swap_and_subbed, swap_and_cp_subbed]
                cp_line_list.append(cp_line)








def create_templates(output_file):
    """
    put all the line lists together and write to files.
    """




def create_template_social_ints(output_file, swap=False, no_npi=False): #deprecated
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


def create_templates_meterials(output_file, swap=False, no_npi=False): #deprecated
    """
    returns
    """

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

    

    template1 = "Usually, <X> <V> more than <Y> does."
    template1_hyp = "This time, <X> did not <BV>, let alone <Y>."
    template2 = "Usually, <X> <V> less than <Y> does."
    template2_hyp = "This time, <Y> did not <BV>, let alone <X>."


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
    """
    deprecated
    """
    
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

    template1 = "Usually, <X> <V> more than <Y> does."
    template1_hyp = "This time, <X> did not <BV>, let alone <Y>."
    template2 = "Usually, <X> <V> less than <Y> does."
    template2_hyp = "This time, <Y> did not <BV>, let alone <X>."

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