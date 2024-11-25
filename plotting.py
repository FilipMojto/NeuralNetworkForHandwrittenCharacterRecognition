import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

import numpy.typing as npt
from typing import Iterable

def plot_softmax(y_pred: npt.NDArray):

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


def plot_ROC(y_true, y_pred):
    n_classes = y_true.shape[1]  # Number of classes


    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Calculate the ROC curve and AUC for each class
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
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


def plot_loss(values: Iterable):
    # Plotting the loss function over epochs
    plt.plot(range(len(values)), values)
    plt.title("Loss Function Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()