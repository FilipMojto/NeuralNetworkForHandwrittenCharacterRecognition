import argparse

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import seaborn as sns

from preprocessing import preprocess_df
from neuralnetwork import NeuralNetwork
from preprocessing import SHIFT



INPUT_TRAIN_DF_FILE_PATH = './data/emnist-balanced-train.csv'
INPUT_TEST_DF_FILE_PATH = './data/emnist-balanced-test.csv'

PREPROCESSED_TRAIN_DF_FILE_PATH =  "./data/preprocessed-train.csv"
PREPROCESSED_TEST_DF_FILE_PATH = './data/preprocessed-test.csv'

LOWER_BOUND_LETTER = 'M'
UPPER_BOUND_LETTER = 'V'

# LOWER_BOUND = ord(LOWER_BOUND_LETTER) - SHIFT
# UPPER_BOUND = ord(UPPER_BOUND_LETTER) - SHIFT

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
def load_data(file_path):
    data = pl.read_csv(file_path, schema_overrides={"one_hot_encoding": pl.String})
    
    # Extract pixel features except for the last 2 columns (letter, one_hot_encoding)
    X = data[:, 1:-2].to_numpy()
    # Here we convert the one-hot-encoding string into an numpy array
    y_one_hot = np.array([list(map(int, list(row))) for row in data["one_hot_encoding"].to_numpy()])
    
    return X, y_one_hot


# Accuracy metric
def accuracy(y_pred, y_true):
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_true, axis=1)
    return np.mean(y_pred_classes == y_true_classes)


# Modify train_neural_network to store loss values
def train_neural_network(model, X, y, learning_rate, epochs, acc_limit=0.5):
    loss_values = []  # Store loss values to plot later
    for epoch in range(epochs):
        # Forward propagation
        y_pred = model.forward(X)
        
        # Compute loss
        loss = model.compute_loss(y_pred, y)
        loss_values.append(loss)  # Save loss
        
        # Backward propagation
        model.backward(X, y, learning_rate)
        
        # Compute accuracy
        acc = accuracy(y_pred, y)
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")
        
        if acc > acc_limit:
            print("Reached target accuracy. Stopping training.")
            break
    
    # Plotting the loss function over epochs
    plt.plot(range(len(loss_values)), loss_values)
    plt.title("Loss Function Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    # Visualize Softmax output for the entire dataset
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        y_pred.T,  # Transpose to show classes on y-axis
        cmap="viridis",  # Color map
        cbar=True,
        xticklabels=100,  # Show every 50 samples on x-axis
        yticklabels=[chr(i + 77) for i in range(y_pred.shape[1])],   # Show all class indices on y-axis
    )
    plt.title("Softmax Activation Output for Entire Dataset")
    plt.xlabel("Sample Index")
    plt.ylabel("Class")
    plt.show()
    # # Plot the Softmax output for a single sample
    # sample_index = 0  # Choose the first sample to visualize
    # class_probabilities = y_pred[sample_index]  # Probabilities for each class
    
    # plt.figure(figsize=(8, 6))
    # plt.bar(range(len(class_probabilities)), class_probabilities)
    # plt.title(f"Softmax Activation Output (Sample {sample_index})")
    # plt.xlabel("Class Index")
    # plt.ylabel("Probability")
    # plt.grid(True)
    # plt.show()

def from_class_to_letter(_class: int):
    pass


def test_neural_network(model: NeuralNetwork, X_test, y_test):
    """
    Test the neural network's performance on test data.
    
    Args:
        model (NeuralNetwork): Trained neural network model.
        X_test (np.ndarray): Test input data.
        y_test (np.ndarray): Test labels (one-hot encoded).
    
    Returns:
        None
    """
    # Forward propagation
    y_pred = model.forward(X_test)
    
    # Compute accuracy
    acc = accuracy(y_pred, y_test)
    
    print(f"Test Accuracy: {acc:.4f}")


    # For multi-class, one-vs-rest approach
    n_classes = y_test.shape[1]  # Number of classes

    # Store the false positive rate, true positive rate, and thresholds for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Calculate the ROC curve and AUC for each class
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot the ROC curves for each class
    plt.figure(figsize=(10, 8))

    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], color='C' + str(i), lw=2, label=f'Class {chr(i + 77)} (AUC = {roc_auc[i]:.2f})')

    # Plot the diagonal (chance) line
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-Class Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

    
    # Optionally, return predictions and true values for further analysis
    return y_pred, y_test


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