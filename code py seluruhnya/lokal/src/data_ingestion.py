import pandas as pd


class DataIngestion:

    def __init__(self, filepath):
        self.filepath = filepath

    def load_data(self):

        df = pd.read_csv(self.filepath)

        print("Dataset loaded successfully")
        print(f"Shape: {df.shape}")

        return df