import argparse

from model import NeuralNetwork
from training import train_neural_network
from testing import accuracy
from utils import load_df


PREPROCESSED_TRAIN_DF_FILE_PATH =  "./data/preprocessed-train.csv"
DEF_SAVE_FILE_PATH = './models/model_state.pkl'


# Network Architecture
input_size = 784  # Number of pixels in the input layer
# MAX 128
HIDDEN_LAYER_NEURONS = 128  # Number of neurons in the hidden layer

# Training the model
DEF_ACCURACY_MIN = 0.7
DEF_LEARNING_RATE = 0.01
DEF_EPOCH_MAX = 1500  # Set a high limit for epochs


parser = argparse.ArgumentParser(
    prog="HLR Neural Network Script",
    description="This script provides commands for effective resource loading and Neural Network training&testing processes."
)

# files

parser.add_argument("--load", required=False, help="If provided, model and its state are loaded from a file. Otherwise, new model is created.")
parser.add_argument("--save", required=False, default=DEF_SAVE_FILE_PATH, help="Provide a specific path to a file where the model will be stored.")

# model configuration params
parser.add_argument("--hidden-neurons", type=int, required=False, default=HIDDEN_LAYER_NEURONS, help="Provide a specific amount of hidden neurons.")

# training model params
parser.add_argument("-a", "--accuracy-min", type=float, required=False, default=DEF_ACCURACY_MIN, help="User can set a required minimum of model's accuracy, must be within [0, 1].")
parser.add_argument("-e", "--epoch-max", type=int, required=False, default=DEF_EPOCH_MAX, help="User can set a maximum of epochs that will be carried out during model's training.")
parser.add_argument("-l", "--learning-rate", type=float, required=False, default=DEF_LEARNING_RATE, help="User can set a custom learning rate of the model.")

# plotting params
parser.add_argument("--plot-softmax", action='store_true', help="If true, softmax activation function is plotted.")
parser.add_argument("--plot-loss", action='store_true', help="If true, loss function is plotted.")
parser.add_argument("--plot-ROC", action='store_true', help="If true, ROC is plotted.")


args = parser.parse_args()


if __name__ == "__main__":
    
    # Load training data
    input_X, output_y = load_df(path=PREPROCESSED_TRAIN_DF_FILE_PATH)

    # Initialize the model
    input_size = input_X.shape[1]  # Number of pixels in the input layer

    output_size = output_y.shape[1]  # Number of unique classes
    # model = NeuralNetwork(input_size, args.hidden_neurons, output_size)
    model = NeuralNetwork.load_model(args.load) if args.load else NeuralNetwork(input_size, args.hidden_neurons, output_size)
    

    train_neural_network(model, input_X, output_y,
                         args.learning_rate, args.epoch_max, acc_limit=args.accuracy_min,
                         plt_loss=args.plot_loss,
                         plt_softmax=args.plot_softmax)
    print(accuracy(model.a2, output_y))

    model.save_model(file_path=DEF_SAVE_FILE_PATH)