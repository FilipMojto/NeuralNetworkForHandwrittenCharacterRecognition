import argparse

from model import NeuralNetwork
from testing import test_neural_network
from utils import load_df


MODEL_STATE_FILE_PATH = './models/model_state.pkl'
TEST_DF_FILE_PATH = "./data/preprocessed-test.csv"

parser = argparse.ArgumentParser(
	prog="Neural Network Tester",
	description="Test your Neural Network effectively using this script!"
)

parser.add_argument("--load", required=False, default=MODEL_STATE_FILE_PATH, help="Input file path (.pkl) from which the model is loaded.")
parser.add_argument("--test", required=False, default=TEST_DF_FILE_PATH, help="Input test file (.csv) upon which the model is tested.")
parser.add_argument("--plot-ROC", required=False, action='store_true', help="Print a ROC curve plot.")
parser.add_argument("--plot-softmax", required=False, action="store_true", help="Plot a Softmax function of the model during testing.")


if __name__ == "__main__":
	args = parser.parse_args()

	network = NeuralNetwork.load_model(file_path=args.load)

	input_X, output_y = load_df(args.test)

	test_neural_network(model=network, X_test=input_X, y_test=output_y, plt_softmax=args.plot_softmax, plt_ROC=args.plot_ROC)
