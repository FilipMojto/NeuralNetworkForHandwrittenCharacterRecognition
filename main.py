import numpy as np
import polars as pl

from neuralnetwork import NeuralNetwork


# Load and preprocess the dataset
def load_data(file_path):
    data = pl.read_csv(file_path, dtypes={"one_hot_encoding": pl.String})
    
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


# Training the neural network
def train_neural_network(model: NeuralNetwork, X, y, learning_rate: float, epochs: int, acc_limit: float = 0.5):
    for epoch in range(epochs):
        # Forward propagation
        y_pred = model.forward(X)
        
        # Compute loss
        loss = model.compute_loss(y_pred, y)
        
        # Backward propagation
        model.backward(X, y, learning_rate)
        
        # Compute accuracy
        acc = accuracy(y_pred, y)
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")
        
        # Stop training if accuracy exceeds 50%
        if acc > acc_limit:
            print("Reached target accuracy. Stopping training.")
            break

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
    
    # Optionally, return predictions and true values for further analysis
    return y_pred, y_test




def load_df(path: str):
    dataset = pl.read_csv(path, dtypes={"one_hot_encoding": pl.String})
     
    # Extract pixel features except for the last 2 columns (letter, one_hot_encoding)
    input_X = dataset[:, 1:-2].to_numpy()
    output_y = np.array([list(map(int, list(row))) for row in dataset["one_hot_encoding"].to_numpy()])

    return input_X, output_y


if __name__ == "__main__":
    # Load the dataset
    TRAIN_DF_PATH = "./data/preprocessed-train.csv"
    TEST_DF_PATH = './data/preprocessed-test.csv'
    ACCURACY_LIMIT = 0.7

    # Load training data
    input_X, output_y = load_df(path=TRAIN_DF_PATH)
    # input_X = input_X / 1.0  # Normalize features to range [0, 1]
    
    # Load testing data
    test_X, test_y = load_df(path=TEST_DF_PATH)
    # test_X = test_X / 1.0  # Normalize test features
    
    # Initialize the model
    input_size = 784  # Number of pixels in the input layer
    hidden_size = 60  # Number of neurons in the hidden layer
    output_size = output_y.shape[1]  # Number of unique classes
    model = NeuralNetwork(input_size, hidden_size, output_size)
    
    # Train the model
    learning_rate = 0.01
    epochs = 1000  # Set a high limit for epochs
    train_neural_network(model, input_X, output_y, learning_rate, epochs, acc_limit=ACCURACY_LIMIT)
    
    # Test the model
    test_neural_network(model, test_X, test_y)