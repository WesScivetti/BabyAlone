import re
import glob
from argparse import ArgumentParser
import random
from collections import Counter


def count_npi_la(input_dir, output_dir, related=False, all=False):
    """
    Basically, this one checks if there is NPI licensor in a sentence with let or alone. The test is if negation
    is a spurious correlation with let alone. Just checks negation and common adverbial licensors
    """
    npi_licensors = [
        "not", "n't", "no", "never", "none", "nobody", "nothing", "rarely", "seldom", "hardly", "scarcely", "barely",
    ]


    lic_alt   = "|".join(map(re.escape, npi_licensors))


    file_list = glob.glob(input_dir + "/*train")

    negation_total = 0

    negation_let = 0

    negation_alone = 0

    negation_la = 0

    total_la = 0

    total_let = 0

    total_alone = 0

    total = 0

    for fname in file_list:
        outfname = output_dir + fname.split("/")[3].split(".")[0] + "_noLA.train"

        with open(outfname, "w") as outf:
            with open(fname) as inf:
                for line in inf:
                    if not related and not all:
                        total += 1
                        pattern = rf"\b({lic_alt})\b.*?\blet\b \balone\b"
                        if re.search(pattern, line, flags=re.I):
                            #print("NPI found before let alone")
                            negation_la += 1

                        if re.search(rf"\blet \balone\b", line, flags=re.I):
                            total_la += 1

                        pattern = rf"\b({lic_alt})\b.*?\blet\b"
                        if re.search(pattern, line, flags=re.I):
                            #print("NPI found before let alone")
                            negation_let += 1

                        if re.search(rf"\blet\b", line, flags=re.I):
                            total_let += 1

                        pattern = rf"\b({lic_alt})\b.*?\balone\b"
                        if re.search(pattern, line, flags=re.I):
                            #print("NPI found before let alone")
                            negation_alone += 1

                        if re.search(rf"\balone\b", line, flags=re.I):
                            total_alone += 1

                        pattern = rf"\b({lic_alt})\b"
                        if re.search(pattern, line, flags=re.I):
                            #print("NPI found before let alone")
                            negation_total += 1

    print("TOTAL", total)
    print("TOTAL NEG", negation_total)
    print("PROPORTION", negation_total/total)
    print("TOTAL LA", total_la)
    print("TOTAL NEG LA ", negation_la)
    print("PROPORTION LA ", negation_la / total_la)
    print("TOTAL LET", total_let)
    print("TOTAL NEG LET", negation_let)
    print("PROPORTION LET", negation_let/total_let)
    print("TOTAL ALONE", total_alone)
    print("TOTAL NEG ALONE", negation_alone)
    print("PROPORTION ALONE", negation_alone/total_alone)

def filter_let(input_dir, output_dir):
    """
     filters out sentences with 'let' and counts them
     """

    file_list = glob.glob(input_dir + "/*train")

    filtered_counter = 0

    for fname in file_list:
        outfname = output_dir + fname.split("/")[3].split(".")[0] + "_nolet.train"

        with open(outfname, "w") as outf:
            with open(fname) as inf:
                for line in inf:
                    if "let" in line:
                        print(line)
                        filtered_counter += 1

                    else:
                        outf.write(line)

    print(f"Filtered out {filtered_counter} sentences")

def filter_alone(input_dir, output_dir):
    """
     filters out sentences with 'let' and counts them
     """

    file_list = glob.glob(input_dir + "/*train")

    filtered_counter = 0

    for fname in file_list:
        outfname = output_dir + fname.split("/")[3].split(".")[0] + "_noalone.train"

        with open(outfname, "w") as outf:
            with open(fname) as inf:
                for line in inf:
                    if "alone" in line:
                        print(line)
                        filtered_counter += 1

                    else:
                        outf.write(line)

    print(f"Filtered out {filtered_counter} sentences")

def counter(input_dir, output_dir):
    """
    counts the number of word types in the input directory
    """

    file_list = glob.glob(input_dir + "/*train")

    counter = Counter()

    for fname in file_list:
        print("WORKING ON COUNTING", fname)
        with open(fname) as inf:
            for line in inf:
                words = line.split()
                words = [w.strip(".,!?;:()[]\"'") for w in words]
                counter.update(words)
    with open("counter.txt", "w") as outf:
        for word, count in counter.most_common():
            outf.write(f"{word}\t{count}\n")

def filter_let_or_alone(input_dir, output_dir):
    """
     filters out sentences with 'let' and counts them
     """

    file_list = glob.glob(input_dir + "/*train")

    filtered_counter = 0

    for fname in file_list:
        outfname = output_dir + fname.split("/")[3].split(".")[0] + "_nolet_or_alone.train"

        with open(outfname, "w") as outf:
            with open(fname) as inf:
                for line in inf:
                    if "let" in line:
                        print(line)
                        filtered_counter += 1

                    elif "alone" in line:
                        print(line)
                        filtered_counter += 1

                    else:
                        outf.write(line)

    print(f"Filtered out {filtered_counter} sentences")


def filter_letalone(input_dir, output_dir, related=False, all=False):
    """
    filters out let-alone sentences and counts them
    """
    
    file_list = glob.glob(input_dir + "/*train")

    filtered_counter = 0

    for fname in file_list:
        outfname = output_dir + fname.split("/")[3].split(".")[0] + "_noLA.train"

        with open(outfname, "w") as outf:
            with open(fname) as inf:
                for line in inf:
                    if not related and not all:
                        if "let alone" in line:
                            print(line)
                            filtered_counter += 1

                        else:
                            outf.write(line)

                    elif related:
                        if "let alone" in line:
                            print(line)
                            filtered_counter += 1

                        if "never mind" in line:
                            print(line)
                            filtered_counter += 1
                        elif "much less" in line and "than" not in line:
                            print(line)
                            filtered_counter += 1
                        elif "not to mention" in  line:
                            print(line)
                            filtered_counter += 1

                        else:
                            outf.write(line)

                    elif all:
                        if "let alone" in line:
                            print(line)
                            filtered_counter += 1

                        if "never mind" in line:
                            print(line)
                            filtered_counter += 1
                        elif "much less" in line:
                            print(line)
                            filtered_counter += 1
                        elif "not to mention" in  line:
                            print(line)
                            filtered_counter += 1

                        elif "more" in line and "than" in line:
                            print(line)
                            filtered_counter += 1

                        elif "less" in line and "than" in line:
                            print(line)
                            filtered_counter += 1

                        elif re.search(r"\W[A-Za-z]*er\W", line) and re.search("\Wthan\W", line):
                            if not re.search(r"\Wrather than\W", line) and not re.search(r"\Wother than\W", line):
                                print("Comparative")
                                print(line)
                                print("------")
                                filtered_counter += 1

                        else:
                            outf.write(line)




    print(f"Filtered out {filtered_counter} sentences")


def filter_random(input_dir, output_dir, count):
    file_list = glob.glob(input_dir + "/*train")

    count = int(count)
    filtered_counter = 0
    all_counter = 0
    not_done = True

    for fname in file_list:
        outfname = output_dir + fname.split("/")[3].split(".")[0] + "_noLA.train"

        with open(outfname, "w") as outf:
            with open(fname) as inf:
                for line in inf:
                    if filtered_counter == count:
                        if not_done == True:
                            print(f"Filtered out {filtered_counter} sentences")
                            not_done = False
                        
                        outf.write(line)

                    else:
                        i = random.randint(1, 100)
                        if i != 42:
                            outf.write(line)
                        else:
                            filtered_counter += 1

                    all_counter += 1



    print(f"Total sents: {all_counter}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_dir")
    parser.add_argument("--output_dir")
    parser.add_argument("--random", action='store_true')
    parser.add_argument("--count", default=0)
    parser.add_argument("--related", action='store_true')
    parser.add_argument("--all", action="store_true")

    args = parser.parse_args()
    if args.random == False:
        counter(args.input_dir, args.output_dir)
        #count_npi_la(args.input_dir, args.output_dir, related=args.related, all=args.all)
        #filter_letalone(args.input_dir, args.output_dir, related=args.related, all=args.all)
        # filter_let_or_alone(args.input_dir, args.output_dir)


    if args.random == True:
        filter_random(args.input_dir, args.output_dir, args.count)