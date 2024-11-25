import numpy as np
import numpy.typing as npt


from model import NeuralNetwork
from plotting import plot_softmax, plot_ROC

# Accuracy metric
def accuracy(y_pred: npt.NDArray, y_true: npt.NDArray):
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_true, axis=1)
    return np.mean(y_pred_classes == y_true_classes)


def test_neural_network(model: NeuralNetwork, X_test: npt.NDArray, y_test: npt.NDArray,
                        plt_softmax = False, plt_ROC = False):
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

    if plt_softmax:
        plot_softmax(y_pred=y_pred)

    if plt_ROC:
        plot_ROC(y_true=y_test, y_pred=y_pred)
       

    # Optionally, return predictions and true values for further analysis
    return y_pred, y_test
