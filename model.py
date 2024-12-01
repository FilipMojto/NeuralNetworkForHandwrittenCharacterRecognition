import numpy as np
import numpy.typing as npt
import sympy as sp
import pickle

class NeuralNetwork:

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        # Initialize weights and biases
        self.params = {
            "w1": np.random.randn(input_size, hidden_size) * np.sqrt(1 / input_size),  # Xavier initialization
            "b1": np.zeros((1, hidden_size)),
            "w2": np.random.randn(hidden_size, output_size) * np.sqrt(1 / hidden_size),
            "b2": np.zeros((1, output_size))
        }

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.z1 = None
        self.a1 = None

        self.z2 = None
        self.a2 = None

    def forward(self, X: npt.NDArray):
        # Forward pass
        # self.z1 = np.dot(X, self.params["w1"]) + self.params["b1"]
        self.z1 = np.matmul(X, self.params["w1"]) + self.params["b1"]
        self.a1 = self.sigmoid(self.z1)

        # self.z2 = np.dot(self.a1, self.params["w2"]) + self.params["b2"]
        self.z2 = np.matmul(self.a1, self.params["w2"]) + self.params["b2"]
        self.a2 = self.softmax(self.z2)
        
        return self.a2

    @staticmethod
    def sigmoid(z: npt.NDArray):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def softmax(z: npt.NDArray):
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def backward(self, X: npt.NDArray, y: npt.NDArray, learning_rate: float):
        no_of_samples = y.shape[0]  # Number of examples

        ## Output layer gradients

        # z2_delta = [samples x output]
        z2_delta = self.a2 - y  # ∂L/∂z2 (derivative of loss w.r.t. z2)

        # calculate average of gradients
        # w2_delta = [hidden x output]
        # b2_delta = [1 x output]
        w2_delta = (1 / no_of_samples) * np.dot(self.a1.T, z2_delta)  # ∂L/∂w2
        b2_delta = (1 / no_of_samples) * np.sum(z2_delta, axis=0, keepdims=True)  # ∂L/∂b2

        ## Hidden layer gradients

        # This part calculates how much each hidden layer activation (a1) contributed to the loss by accounting
        # for the error signal from the next layer (z2_detla)
        # z1_delta = [samples x hidden]
        z1_delta = np.dot(z2_delta, self.params["w2"].T) * (self.a1 * (1 - self.a1))  # ∂L/∂z1 (using sigmoid derivative)
        # calculate average of gradients
        # w1_delta = [input x output]
        # b1_delta = [1 x output]
        w1_delta = (1 / no_of_samples) * np.dot(X.T, z1_delta)  # ∂L/∂w1
        b1_delta = (1 / no_of_samples) * np.sum(z1_delta, axis=0, keepdims=True)  # ∂L/∂b1

        # Update weights and biases
        self.params["w1"] -= learning_rate * w1_delta 
        self.params["b1"] -= learning_rate * b1_delta
        self.params["w2"] -= learning_rate * w2_delta
        self.params["b2"] -= learning_rate * b2_delta

    @staticmethod
    def compute_loss(y_pred: npt.NDArray, y_true: npt.NDArray):
        # Cross-entropy loss
        m = y_true.shape[0]
        loss = -np.sum(y_true * np.log(y_pred + 1e-9)) / m
        return loss
    

    def save_model(self, file_path: str):
        """Save the neural network parameters and activations to a file."""
        state = {
            "params": self.params,
            "z1": self.z1,
            "a1": self.a1,
            "z2": self.z2,
            "a2": self.a2
        }
        with open(file_path, 'wb') as f:
            pickle.dump(state, f)
        print(f"Model saved to {file_path}.")

    @staticmethod
    def load_model(file_path: str) -> "NeuralNetwork":
        """Load the neural network parameters and activations from a file."""
        with open(file_path, 'rb') as f:
            state = pickle.load(f)

        # Create a new network instance
        input_size = state["params"]["w1"].shape[0]
        hidden_size = state["params"]["w1"].shape[1]
        output_size = state["params"]["w2"].shape[1]
        network = NeuralNetwork(input_size, hidden_size, output_size)

        # Restore parameters and activations
        network.params = state["params"]
        network.z1 = state["z1"]
        network.a1 = state["a1"]
        network.z2 = state["z2"]
        network.a2 = state["a2"]

        print(f"Model loaded from {file_path}.")
        return network