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

        "template_and_base_no_npi" : "<PRO> <P2N> the <C1> <X1>, and the <C2> <X2>.",
        # I could afford the blue watch, and the red watch.

        "template_and_swap_no_npi" : "<PRO> <P2N> the <C2> <X2>, and the <C1> <X1>.",
        # I could afford the red watch, and the blue watch.

        "template_base_psuedo" : "The <C1> <X1>, let alone the <C2> <X2>, <PRO> <P2>.",
        # The blue watch, let alone the red watch, I couldn't afford. -> should be not ideal

        "template_swap_psuedo" : "The <C2> <X2>, let alone the <C1> <X1>, <PRO> <P2>.",
        # The red watch, let alone the blue watch, I couldn't afford. -> should be not ideal

        "template_and_base_psuedo" : "The <C1> <X1>, and the <C2> <X2>, <PRO> <P2>.",
        # The blue watch, and the red watch, I couldn't afford. -> should be better

        "template_and_swap_psuedo" : "The <C2> <X2>, and the <C1> <X1>, <PRO> <P2>.",
        # The red watch, and the blue watch, I couldn't afford. -> should be better

        "template_base_cp" : "<PRO> <P2> the <C1> <X1>, let alone <PRO> <P2>  the <C2> <X2>.",
        # I couldn't afford the blue watch, let alone I couldn't afford the red watch. -> bad

        "template_swap_cp" : "<PRO> <P2> the <C2> <X2>, let alone <PRO> <P2> the <C1> <X1>.",
        # I couldn't afford the red watch, let alone I couldn't afford the blue watch. -> bad

        "template_and_base_cp" : "<PRO> <P2> the <C1> <X1>, and <PRO> <P2> the <C2> <X2>.",
        # I couldn't afford the blue watch, and I couldn't afford the red watch. -> fine

        "template_and_swap_cp" : "<PRO> <P2> the <C2> <X2>, and <PRO> <P2> the <C1> <X1>.",
        # I couldn't afford the red watch, and I couldn't afford the blue watch. -> fine

        "template_base_semantic" : "<PRO> <P2> the <C1> <X1>, let alone the <C2> <X2>. The <C1> <X1> <P1> than the <C2> <X2>.",
        # I couldn't afford the blue watch, let alone the red watch. The blue watch is more expensive than the red watch.

        "template_base_semantic_2" : "<PRO> <P2> the <C1> <X1>, let alone the <C2> <X2>. The <C2> <X2> <P1> than the <C1> <X1>.",
        # I couldn't afford the blue watch, let alone the red watch. The red watch is more expensive than the blue watch. -> wrong for more expensive, right for cheaper/less expensive

        "template_swap_semantic" : "<PRO> <P2> the <C2> <X2>, let alone the <C1> <X1>. The <C2> <X2> <P1> than the <C1> <X1>.",
        # I couldn't afford the red watch, let alone the blue watch. The red watch is more expensive than the blue watch.

        "template_swap_semantic_2" : "<PRO> <P2> the <C2> <X2>, let alone the <C1> <X1>. The <C1> <X1> <P1> than the <C2> <X2>."
        # I couldn't afford the red watch, let alone the blue watch. The blue watch is more expensive than the red watch. -> wrong for more expensive, right for cheaper/less expensive
    }
    return template_dict

def make_lines_lists(nouns, colors, predicates, predicates_2, predicates_2_no_npi):
    """
    given lists of fillers, output the line lists for the templates that are filled in
    """
    template_dict = create_template_dict()
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
                base_subbed = sub_template(pron, predicate2, color1, n1, color2, n2, "", "",
                                           template_dict["template_base"])
                swap_subbed = sub_template(pron, predicate2, color1, n1, color2, n2, "", "",
                                           template_dict["template_swap"])
                base_no_npi_subbed = sub_template(pron, predicate2, color1, n1, color2, n2, "", predicate2_no_npi,
                                                  template_dict["template_base_no_npi"])
                swap_no_npi_subbed = sub_template(pron, predicate2, color1, n1, color2, n2, "", predicate2_no_npi,
                                                  template_dict["template_swap_no_npi"])
                base_psuedo_subbed = sub_template(pron, predicate2, color1, n1, color2, n2, "", "",
                                                  template_dict["template_base_psuedo"])
                swap_psuedo_subbed = sub_template(pron, predicate2, color1, n1, color2, n2, "", "",
                                                  template_dict["template_swap_psuedo"])
                base_cp_subbed = sub_template(pron, predicate2, color1, n1, color2, n2, "", "",
                                              template_dict["template_base_cp"])
                swap_cp_subbed = sub_template(pron, predicate2, color1, n1, color2, n2, "", "",
                                              template_dict["template_swap_cp"])
                base_and_subbed = sub_template(pron, predicate2, color1, n1, color2, n2, "", "",
                                               template_dict["template_and_base"])
                swap_and_subbed = sub_template(pron, predicate2, color1, n1, color2, n2, "", "",
                                               template_dict["template_and_swap"])
                base_and_npi_subbed = sub_template(pron, predicate2, color1, n1, color2, n2, "", predicate2_no_npi,
                                                    template_dict["template_and_base_no_npi"])
                swap_and_npi_subbed = sub_template(pron, predicate2, color1, n1, color2, n2, "", predicate2_no_npi,
                                                    template_dict["template_and_swap_no_npi"])
                base_and_psuedo_subbed = sub_template(pron, predicate2, color1, n1, color2, n2, "", "",
                                                      template_dict["template_and_base_psuedo"])
                swap_and_psuedo_subbed = sub_template(pron, predicate2, color1, n1, color2, n2, "", "",
                                                      template_dict["template_and_swap_psuedo"])
                base_and_cp_subbed = sub_template(pron, predicate2, color1, n1, color2, n2, "", "",
                                                  template_dict["template_and_base_cp"])
                swap_and_cp_subbed = sub_template(pron, predicate2, color1, n1, color2, n2, "", "",
                                                  template_dict["template_and_swap_cp"])

                for predicate1 in predicates:
                    # semantic line
                    semantic_subbed = sub_template(pron, predicate2, color1, n1, color2, n2, predicate1, "",
                                                   template_dict["template_base_semantic"])
                    semantic_subbed_2 = sub_template(pron, predicate2, color1, n1, color2, n2, predicate1, "",
                                                     template_dict["template_base_semantic_2"])
                    semantic_swapped_subbed = sub_template(pron, predicate2, color1, n1, color2, n2, predicate1, "",
                                                           template_dict["template_swap_semantic"])
                    semantic_swapped_subbed_2 = sub_template(pron, predicate2, color1, n1, color2, n2, predicate1, "",
                                                             template_dict["template_swap_semantic_2"])

                    if predicate1 in ["is more expensive", "is heavier", "weighs more", "is further away", "is faster", "is quicker", "is hotter"]:
                        semantic_line = [n1, color1, color2, predicate2, predicate1, semantic_subbed_2, semantic_subbed,
                                         semantic_swapped_subbed_2, semantic_swapped_subbed]
                        semantic_line_list.append(semantic_line)
                    else:
                        semantic_line = [n1, color1, color2, predicate2, predicate1, semantic_subbed, semantic_subbed_2,
                                         semantic_swapped_subbed, semantic_swapped_subbed_2]
                        semantic_line_list.append(semantic_line)

                npi_line = [n1, color1, color2, predicate2, base_subbed, base_no_npi_subbed, swap_subbed,
                            swap_no_npi_subbed, base_and_subbed, base_and_npi_subbed, swap_and_subbed, swap_no_npi_subbed]
                npi_line_list.append(npi_line)

                psuedoclefting_line = [n1, color1, color2, predicate2, base_subbed, base_psuedo_subbed, swap_subbed,
                                       swap_psuedo_subbed, base_and_subbed, base_and_psuedo_subbed, swap_and_subbed,
                                       swap_and_psuedo_subbed]
                psuedoclefting_line_list.append(psuedoclefting_line)

                cp_line = [n1, color1, color2, predicate2, base_subbed, base_cp_subbed, swap_subbed, swap_cp_subbed,
                           base_and_subbed, base_and_cp_subbed, swap_and_subbed, swap_and_cp_subbed]
                cp_line_list.append(cp_line)

    return npi_line_list, psuedoclefting_line_list, cp_line_list, semantic_line_list

def expense_templates():
    """
    generates templates for The X is more expensive/cheaper than Y. I couldn't afford X, let alone Y.
    return: npi_line_list, psuedoclefting_line_list, cp_line_list, semantic_line_list
    """


    nouns = ["car", "house", "boat", "plane", "purse", "watch", "bracelet", "sunglasses", "couch", "chair"]
    colors = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "gray", "black", "white"]
    predicates = ["is more expensive", "is less expensive", "is cheaper"]
    predicates_2 = ["couldn't afford", "couldn't buy", "couldn't pay for", "didn't have enough money for"]
    predicates_2_no_npi = ["could afford", "could buy", "could pay for", "did have enough money for"]

    npi_line_list, psuedoclefting_line_list, cp_line_list, semantic_line_list = make_lines_lists(nouns, colors, predicates,
                                                                                                 predicates_2,
                                                                                                 predicates_2_no_npi)
    return npi_line_list, psuedoclefting_line_list, cp_line_list, semantic_line_list


def weight_templates():
    """
    generates templates for I couldn't lift/carry/move/pick up X, let alone Y. X is heavier/lighter than Y.
    """
    nouns = ["box", "bag", "suitcase", "backpack", "crate", "bin", "chair"]
    colors = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "gray", "black", "white"]
    predicates = ["is heavier", "is lighter", "weighs more", "weighs less"]
    predicates2 = ["couldn't lift", "couldn't carry", "couldn't move", "couldn't pick up"]
    predicates2_no_npi = ["could lift", "could carry", "could move", "could pick up"]
    npi_line_list, psuedoclefting_line_list, cp_line_list, semantic_line_list = make_lines_lists(nouns, colors,
                                                                                                 predicates,
                                                                                                 predicates2,
                                                                                                 predicates2_no_npi)
    return npi_line_list, psuedoclefting_line_list, cp_line_list, semantic_line_list


def short_distance_templates():
    nouns = ["pencil", "pen", "highlighter", "paper", "notebook", "book", "folder", "crayon", "paintbrush"]
    colors = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "gray", "black", "white"]
    predicates = ["is further away", "is closer", "is nearer"]
    predicates2 = ["couldn't reach", "couldn't grab", "couldn't touch"]
    predicates2_no_npi = ["could reach", "could grab", "could touch"]
    npi_line_list, psuedoclefting_line_list, cp_line_list, semantic_line_list = make_lines_lists(nouns, colors,
                                                                                                 predicates,
                                                                                                 predicates2,
                                                                                                 predicates2_no_npi)
    return npi_line_list, psuedoclefting_line_list, cp_line_list, semantic_line_list

def long_distance_templates():
    nouns = ["house", "building"]
    colors = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "gray", "black", "white"]
    predicates = ["is further away", "is closer", "is nearer"]
    predicates2 = ["couldn't reach", "couldn't get to", "couldn't walk to"]
    predicates2_no_npi = ["could reach", "could get to", "could walk to"]
    npi_line_list, psuedoclefting_line_list, cp_line_list, semantic_line_list = make_lines_lists(nouns, colors,
                                                                                                 predicates,
                                                                                                 predicates2,
                                                                                                 predicates2_no_npi)
    return npi_line_list, psuedoclefting_line_list, cp_line_list, semantic_line_list

def speed_templates_1():
    nouns = ["car", "boat", "truck", "van"]
    colors = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "gray", "black", "white"]
    predicates = ["is faster", "is slower", "is quicker"]
    predicates2 = ["couldn't pass", "couldn't catch up to", "couldn't overtake"]
    predicates2_no_npi = ["could pass", "could catch up to", "could overtake"]
    npi_line_list, psuedoclefting_line_list, cp_line_list, semantic_line_list = make_lines_lists(nouns, colors,
                                                                                                 predicates,
                                                                                                 predicates2,
                                                                                                 predicates2_no_npi)
    return npi_line_list, psuedoclefting_line_list, cp_line_list, semantic_line_list

def speed_templates_2():
    nouns = ["dog", "cat", "rabbit", "puppy", "bunny", "kitten"]
    colors = ["brown", "tan", "gray", "black", "white"]
    predicates = ["is faster", "is slower", "is quicker"]
    predicates2 = ["couldn't catch", "couldn't keep up with"]
    predicates2_no_npi = ["could catch", "could keep up with"]
    npi_line_list, psuedoclefting_line_list, cp_line_list, semantic_line_list = make_lines_lists(nouns, colors,
                                                                                                 predicates,
                                                                                                 predicates2,
                                                                                                 predicates2_no_npi)
    return npi_line_list, psuedoclefting_line_list, cp_line_list, semantic_line_list


def tea_templates():
    nouns = ["tea"]
    colors = ["red", "white", "green", "black"]
    predicates = ["is hotter"]
    predicates2 = ["couldn't drink", "couldn't sip"]
    predicates2_no_npi = ["could drink", "could sip"]
    npi_line_list, psuedoclefting_line_list, cp_line_list, semantic_line_list = make_lines_lists(nouns, colors,
                                                                                                 predicates,
                                                                                                 predicates2,
                                                                                                 predicates2_no_npi)
    return npi_line_list, psuedoclefting_line_list, cp_line_list, semantic_line_list





def create_templates(output_file):
    """
    put all the line lists together and write to files.
    """
    npi_line_list, psuedoclefting_line_list, cp_line_list, semantic_line_list = expense_templates()

    #combine these lists with weight_templates
    npi_line_list2, psuedoclefting_line_list2, cp_line_list2, semantic_line_list2 = weight_templates()

    npi_line_list3, psuedoclefting_line_list3, cp_line_list3, semantic_line_list3 = short_distance_templates()
    npi_line_list4, psuedoclefting_line_list4, cp_line_list4, semantic_line_list4 = long_distance_templates()
    npi_line_list5, psuedoclefting_line_list5, cp_line_list5, semantic_line_list5 = speed_templates_1()
    npi_line_list6, psuedoclefting_line_list6, cp_line_list6, semantic_line_list6 = speed_templates_2()
    npi_line_list7, psuedoclefting_line_list7, cp_line_list7, semantic_line_list7 = tea_templates()

    npi_line_list = npi_line_list + npi_line_list2 + npi_line_list3 + npi_line_list4 + npi_line_list5 + npi_line_list6 + npi_line_list7
    psuedoclefting_line_list = psuedoclefting_line_list + psuedoclefting_line_list2 + psuedoclefting_line_list3 + psuedoclefting_line_list4 + psuedoclefting_line_list5 + psuedoclefting_line_list6 + psuedoclefting_line_list7
    cp_line_list = cp_line_list + cp_line_list2 + cp_line_list3 + cp_line_list4 + cp_line_list5 + cp_line_list6 + cp_line_list7
    semantic_line_list = semantic_line_list + semantic_line_list2 + semantic_line_list3 + semantic_line_list4 + semantic_line_list5 + semantic_line_list6 + semantic_line_list7



    # create a dataframe for each line list
    npi_df = pd.DataFrame(npi_line_list, columns=["noun", "color1", "color2", "predicate2", "base_subbed", "base_no_npi_subbed", "swap_subbed", "swap_no_npi_subbed", "base_and_subbed", "base_and_npi_subbed", "swap_and_subbed", "swap_and_npi_subbed"])
    psuedoclefting_df = pd.DataFrame(psuedoclefting_line_list, columns=["noun", "color1", "color2", "predicate2", "base_subbed", "base_psuedo_subbed", "swap_subbed", "swap_psuedo_subbed", "base_and_subbed", "base_and_psuedo_subbed", "swap_and_subbed", "swap_and_psuedo_subbed"])
    cp_df = pd.DataFrame(cp_line_list, columns=["noun", "color1", "color2", "predicate2", "base_subbed", "base_cp_subbed", "swap_subbed", "swap_cp_subbed", "base_and_subbed", "base_and_cp_subbed", "swap_and_subbed", "swap_and_cp_subbed"])
    semantic_df = pd.DataFrame(semantic_line_list, columns=["noun", "color1", "color2", "predicate2", "predicate1", "semantic_right", "semantic_wrong", "semantic_swapped_right", "semantic_swapped_wrong"])

    # write to files
    npi_df.to_csv(output_file + "_npi.tsv", sep="\t", index=False)
    psuedoclefting_df.to_csv(output_file + "_psuedoclefting.tsv", sep="\t", index=False)
    cp_df.to_csv(output_file + "_cp.tsv", sep="\t", index=False)
    semantic_df.to_csv(output_file + "_semantic.tsv", sep="\t", index=False)




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--output_file")


    args = parser.parse_args()
    output_file = args.output_file
    create_templates(output_file)
