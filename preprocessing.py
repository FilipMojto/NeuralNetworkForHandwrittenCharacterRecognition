import polars as pl
from argparse import ArgumentParser

TRAINING_DF_FILE_PATH = './data/emnist-balanced-train.csv'
TESTING_DF_FILE_PATH = './data/emnist-balanced-test.csv'

PREPROCESSED_TRAIN_DF_PATH = "./data/preprocessed-train.csv"
PREPROCESSED_TEST_DF_PATH = "./data/preprocessed-test.csv"


LOWER_BOUND_LETTER = 'M'
UPPER_BOUND_LETTER = 'V'
SHIFT = 48
DF_LIMIT = 4000
TESTING_DF_LIMIT = 800
COLUMNS = ['class'] + [f'pixel {i}' for i in range(784)]

LOWER_BOUND = ord(LOWER_BOUND_LETTER) - SHIFT
UPPER_BOUND = ord(UPPER_BOUND_LETTER) - SHIFT

parser = ArgumentParser(
    prog="Data Processing Script",
    description="Effectively preprocessed data for Neural Network training."
)

# preprocessing params
# parser.add_argument("-p", "--preprocess", action="store_true", help="If provided training&testing datasets are preprocessed before training.")

parser.add_argument("--input", required=False, default=TRAINING_DF_FILE_PATH, help="A custom path to a training dataset csv file.")
# parser.add_argument("--testing-dataset", required=False, default=INPUT_TEST_DF_FILE_PATH, help="A custom path to a testing dataset csv file.")
parser.add_argument("--output", required=False, default=PREPROCESSED_TRAIN_DF_PATH)

parser.add_argument("--limit", required=False, default=DF_LIMIT)

parser.add_argument("--min-letter", type=str, required=False, default=LOWER_BOUND_LETTER, help="User can set the lower bound letter for preprocessing.")
parser.add_argument("--max-letter", type=str, required=False, default=UPPER_BOUND_LETTER, help="User can set the upper bound letter for preprocessing.")


def preprocess_df(df_path: str, lower_bound: int, upper_bound: int, row_limit: int, save_to_csv: bool = False, output_file: str = "./data/preprocessed_df.csv"):
    # Read the CSV file, treating the first row as data, not headers
    df = pl.read_csv(
        df_path, 
        has_header=False, 
        new_columns=COLUMNS
    )
    
    # here we map the class of each row into its repective letter and create a new column
    df = df.with_columns((pl.col("class") + SHIFT).map_elements(chr, return_dtype=pl.String).alias("letter"))

        
    # Filter rows based on the class column (for letters M-V only)
    filtered_df = df.filter(
        (pl.col("class") >= lower_bound) & (pl.col("class") <= upper_bound)
    ).with_columns(
        (pl.col("class") + SHIFT).map_elements(chr, return_dtype=pl.String).alias("letter")
    )

    # Normalize all pixel columns (excluding 'class' and 'letter')
    normalized_df = filtered_df.with_columns(
        [
            (pl.col(col_name) / 255).alias(col_name)
            for col_name in filtered_df.columns
            if col_name not in {"class", "letter"}
        ]
    )

    # Group by 'class' and sample rows within each group to enforce a balanced limit
    grouped_df = (
        normalized_df
        .group_by("class")
        .head(row_limit // (upper_bound - lower_bound + 1))  # Proportional rows per class
    )

    # Generate one-hot encoded column as an array for the 'class' column
    unique_classes = grouped_df["class"].unique().to_list()
    

    def create_one_hot_encoding_str(class_value):
    # Generate the one-hot encoding as a string
        one_hot_str = "".join("1" if i == class_value else "0" for i in unique_classes)
        return one_hot_str
    



    # Create a one-hot encoded array for each row
    one_hot_array_col = pl.col("class").map_elements(
        lambda class_value: create_one_hot_encoding_str(class_value), return_dtype=pl.String
    ).alias("one_hot_encoding")

    

    # Add the one-hot encoded array column to the dataframe
    one_hot_encoded_df = grouped_df.with_columns([one_hot_array_col])
    # print(one_hot_encoded_df.head())

    if save_to_csv:
        one_hot_encoded_df.write_csv(output_file)
    
    return one_hot_encoded_df

# # Read the CSV file, treating the first row as data, not headers
# emnist_letters_train_df = pl.read_csv(
#     TRAINING_DF_FILE_PATH, 
#     has_header=False, 
#     new_columns=COLUMNS
# )

# emnist_letters_train_df = emnist_letters_train_df.with_columns((pl.col("class") + SHIFT).map_elements(chr, return_dtype=pl.String).alias("letter"))

# emnist_letters_test_df = pl.read_csv(
#     TESTING_DF_FILE_PATH, 
#     has_header=False, 
#     new_columns=COLUMNS
# )

# emnist_letters_test_df = emnist_letters_test_df.with_columns((pl.col("class") + SHIFT).map_elements(chr, return_dtype=pl.String).alias("letter"))

# # Preprocess the training and testing DataFrames
# emnist_letters_train_df = preprocess_df(
#     df=emnist_letters_train_df,
#     lower_bound=LOWER_BOUND,
#     upper_bound=UPPER_BOUND,
#     row_limit=TRAINING_DF_LIMIT
# )

# emnist_letters_test_df = preprocess_df(
#     df=emnist_letters_test_df,
#     lower_bound=LOWER_BOUND,
#     upper_bound=UPPER_BOUND,
#     row_limit=TESTING_DF_LIMIT
# )

# Print the grouped DataFrame (count rows per class)
# emnist_letters_train_df = emnist_letters_train_df.group_by("class").agg(pl.len())
# emnist_letters_test_df = emnist_letters_test_df.group_by("class").agg(pl.len())

# print("Training Data:")
# print(emnist_letters_train_df)

# print("Testing Data:")
# print(emnist_letters_test_df)

# emnist_letters_train_df.write_csv(PREPROCESSED_TRAIN_DF_PATH)
# emnist_letters_test_df.write_csv(PREPROCESSED_TEST_DF_PATH)


if __name__ == "__main__":
    args = parser.parse_args()
    
  
    print("preprocessing ... ", end="")
    preprocess_df(df_path=args.input,
                    lower_bound=ord(args.min_letter) - SHIFT, upper_bound=ord(args.max_letter) - SHIFT,
                    row_limit=args.limit,
                    save_to_csv=True, output_file=args.output)
    
    # preprocess_df(df_path=args.testing_dataset,
    #                 lower_bound=ord(args.min_letter) - SHIFT,
    #                 upper_bound=ord(args.max_letter) - SHIFT,
    #                 row_limit=TESTING_DF_LIMIT, save_to_csv=True,
    #                 output_file=PREPROCESSED_TEST_DF_PATH)
    
    print("success!")
    