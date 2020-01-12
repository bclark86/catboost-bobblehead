import os
import pandas as pd

DATA_DIR = 'data'

def load_bobbleheads_data(data_path=DATA_DIR):
    """
    Load data from folder
    """

    csv_path = os.path.join(data_path, 'raw', 'bobbleheads.csv')
    return pd.read_csv(csv_path)