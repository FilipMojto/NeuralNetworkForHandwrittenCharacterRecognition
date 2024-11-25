import matplotlib.pyplot as plt
import seaborn as sns

from model import NeuralNetwork
from testing import accuracy


# Modify train_neural_network to store loss values
def train_neural_network(model: NeuralNetwork, X, y, learning_rate: float, epochs: int, acc_limit=0.5):
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