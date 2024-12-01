import polars as pl
import numpy as np

def load_df(path: str):
    dataset = pl.read_csv(path, schema_overrides={"one_hot_encoding": pl.String})
     
    # Extract pixel features except for the last 2 columns (letter, one_hot_encoding)
    input_X = dataset[:, 1:-2].to_numpy()
    output_y = np.array([list(map(int, list(row))) for row in dataset["one_hot_encoding"].to_numpy()])

    return input_X, output_y
