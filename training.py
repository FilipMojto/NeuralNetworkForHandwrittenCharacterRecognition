import numpy.typing as npt

from model import NeuralNetwork
from testing import accuracy
from plotting import plot_softmax, plot_loss


# Modify train_neural_network to store loss values
def train_neural_network(model: NeuralNetwork, X: npt.NDArray, y: npt.NDArray, learning_rate: float, 
                        epochs: int, acc_limit=0.5, plt_loss = False, plt_softmax = False):
    loss_values = []  # Store loss values to plot later

    for epoch in range(epochs):
        # Forward propagation
        y_pred = model.forward(X)
        
        # Compute loss
        # porovna sa predikovany vystup y so skutocnym vystupom pomocou stratovej funkcie
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
    
    if plt_loss:
        # Plotting the loss function over epochs
        plot_loss(values=loss_values)

    if plt_softmax:
        plot_softmax(y_pred=y_pred)
