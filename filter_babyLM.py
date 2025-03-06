import re
import glob
from argparse import ArgumentParser
import random

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
        filter_letalone(args.input_dir, args.output_dir, related=args.related, all=args.all)

    if args.random == True:
        filter_random(args.input_dir, args.output_dir, args.count)