import pandas as pd

def load_perp_dataset(input_file):
    """
    loads in the let-alone data that will be put into the correct format for reading into the model.
    """
    df = pd.read_csv(input_file, sep="\t")
    print(df.head())

