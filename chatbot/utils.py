import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def preprocess_text(text):
    return text.lower().strip()
