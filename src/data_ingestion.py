import pandas as pd

def load_data(path="data/emails.csv"):
    df = pd.read_csv(path)
    df = df.drop_duplicates()
    df = df.dropna()

    return df