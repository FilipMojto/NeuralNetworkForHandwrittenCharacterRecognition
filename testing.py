import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from polars import read_csv, String as pl_String


from model import NeuralNetwork


# Accuracy metric
def accuracy(y_pred, y_true):
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_true, axis=1)
    return np.mean(y_pred_classes == y_true_classes)


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
    dataset = read_csv(path, schema_overrides={"one_hot_encoding": pl_String})
     
    # Extract pixel features except for the last 2 columns (letter, one_hot_encoding)
    input_X = dataset[:, 1:-2].to_numpy()
    output_y = np.array([list(map(int, list(row))) for row in dataset["one_hot_encoding"].to_numpy()])

    return input_X, output_y
