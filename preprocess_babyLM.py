import re
import glob
from argparse import ArgumentParser
import stanza

print("loading stanza model")
nlp = stanza.Pipeline(lang="en", processors="tokenize", batch_size=32)
print("model loaded")

def preprocess_babyLM(input_dir, output_dir):
    """
    used for removing the speaker id things from childes and switchboard.
    Outputs one sentence per line. This is useful for filtering, when things are removed at the sentence level.
    """
    file_list = glob.glob(input_dir + "/*train")

    for fname in file_list:
        print(fname)
        outfname = output_dir + fname.split("/")[3].split(".")[0] + "_filtered.train"

        with open(outfname, "w") as outf:
            with open(fname) as inf:
                if ("switchboard" not in fname) and ("childes" not in fname):
                    if ("gutenberg" not in fname) and ("simple_wiki" not in fname):
                        text = inf.read()
                        outf.write(text)
                    else:
                        print("doing stanza stuff")
                        print(fname)

                        #chunk the document into manageable stanza chunks
                        lines = inf.readlines()

                        chunks = []
                        idx = 0
                        print(len(lines))
                        while len(lines) > idx + 50000:
                            chunk = "\n".join(lines[idx:idx+50000])
                            chunks.append(chunk)
                            idx += 50000
                        else: #last chunk
                            print("last chunk")
                            chunk = "\n".join(lines[idx:])
                            chunks.append(chunk)

                        print(len(chunks))


                        for chunk in chunks:
                            doc = nlp(chunk)  # Process the chunk
                            sentences = [sentence.text.rstrip("\n") for sentence in doc.sentences]  # Extract sentences
                            for s in sentences:
                                outf.write(s+"\n")
                            print("done with a chunk")
                            print("----")
                else:
                        for line in inf:
                            # print(line)
                            if "\t" in line:
                                l = line.split("\t")[-1]
                                outf.write(l)
                            else:
                                outf.write(line)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_dir")
    parser.add_argument("--output_dir")
    args = parser.parse_args()
    preprocess_babyLM(args.input_dir, args.output_dir)
