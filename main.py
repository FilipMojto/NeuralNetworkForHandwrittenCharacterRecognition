import argparse

import numpy as np
import polars as pl

from model import NeuralNetwork
from preprocessing import preprocess_df, SHIFT
from training import train_neural_network
from testing import test_neural_network


INPUT_TRAIN_DF_FILE_PATH = './data/emnist-balanced-train.csv'
INPUT_TEST_DF_FILE_PATH = './data/emnist-balanced-test.csv'

PREPROCESSED_TRAIN_DF_FILE_PATH =  "./data/preprocessed-train.csv"
PREPROCESSED_TEST_DF_FILE_PATH = './data/preprocessed-test.csv'

LOWER_BOUND_LETTER = 'M'
UPPER_BOUND_LETTER = 'V'

TRAINING_DF_LIMIT = 4000
TESTING_DF_LIMIT = 800

# Network Architecture
input_size = 784  # Number of pixels in the input layer
HIDDEN_LAYER_NEURONS = 60  # Number of neurons in the hidden layer

# Training the model
DEF_ACCURACY_MIN = 0.7
DEF_LEARNING_RATE = 0.01
DEF_EPOCH_MAX = 1000  # Set a high limit for epochs


parser = argparse.ArgumentParser(
    prog="HLR Neural Network Script",
    description="This script provides commands for effective resource loading and Neural Network training&testing processes."
)

# preprocessing params
parser.add_argument("-p", "--preprocess", action="store_true", help="If provided training&testing datasets are preprocessed before training.")
parser.add_argument("--min-letter", type=str, required=False, default=LOWER_BOUND_LETTER, help="User can set the lower bound letter for preprocessing.")
parser.add_argument("--max-letter", type=str, required=False, default=UPPER_BOUND_LETTER, help="User can set the upper bound letter for preprocessing.")

# training model params
parser.add_argument("-a", "--accuracy-min", type=float, required=False, default=DEF_ACCURACY_MIN, help="User can set a required minimum of model's accuracy, must be within [0, 1].")
parser.add_argument("-e", "--epoch-max", type=int, required=False, default=DEF_EPOCH_MAX, help="User can set a maximum of epochs that will be carried out during model's training.")
parser.add_argument("-l", "--learning-rate", type=float, required=False, default=DEF_LEARNING_RATE, help="User can set a custom learning rate of the model.")

args = parser.parse_args()


# Load and preprocess the dataset
# def load_data(file_path):
#     data = pl.read_csv(file_path, schema_overrides={"one_hot_encoding": pl.String})
    
#     # Extract pixel features except for the last 2 columns (letter, one_hot_encoding)
#     X = data[:, 1:-2].to_numpy()
#     # Here we convert the one-hot-encoding string into an numpy array
#     y_one_hot = np.array([list(map(int, list(row))) for row in data["one_hot_encoding"].to_numpy()])
    
#     return X, y_one_hot


def load_df(path: str):
    dataset = pl.read_csv(path, schema_overrides={"one_hot_encoding": pl.String})
     
    # Extract pixel features except for the last 2 columns (letter, one_hot_encoding)
    input_X = dataset[:, 1:-2].to_numpy()
    output_y = np.array([list(map(int, list(row))) for row in dataset["one_hot_encoding"].to_numpy()])

    return input_X, output_y


if __name__ == "__main__":

    if args.preprocess:
        print("preprocessing ...")
        preprocess_df(df_path=INPUT_TRAIN_DF_FILE_PATH,
                      lower_bound=ord(args.lower_letter) - SHIFT, upper_bound=ord(args.upper_letter) - SHIFT,
                      row_limit=TRAINING_DF_LIMIT,
                      save_to_csv=True, output_file=PREPROCESSED_TRAIN_DF_FILE_PATH)
        
        preprocess_df(df_path=INPUT_TEST_DF_FILE_PATH,
                      lower_bound=ord(args.lower_letter) - SHIFT,
                      upper_bound=ord(args.upper_letter) - SHIFT,
                      row_limit=TESTING_DF_LIMIT, save_to_csv=True,
                      output_file=PREPROCESSED_TEST_DF_FILE_PATH)


    
    # Load training data
    input_X, output_y = load_df(path=PREPROCESSED_TRAIN_DF_FILE_PATH)
    # input_X = input_X / 1.0  # Normalize features to range [0, 1]
    
    # Load testing data
    test_X, test_y = load_df(path=PREPROCESSED_TEST_DF_FILE_PATH)
    # test_X = test_X / 1.0  # Normalize test features
    
    # Initialize the model
    input_size = input_X.shape[1]  # Number of pixels in the input layer
    HIDDEN_LAYER_NEURONS = 60  # Number of neurons in the hidden layer
    output_size = output_y.shape[1]  # Number of unique classes
    model = NeuralNetwork(input_size, HIDDEN_LAYER_NEURONS, output_size)
    

    train_neural_network(model, input_X, output_y, args.learning_rate, args.epoch_max, acc_limit=args.accuracy_min)
    
    # Test the model
    test_neural_network(model, test_X, test_y)