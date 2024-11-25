import numpy as np


class NeuralNetwork:
    
    def __init__(self, input_size, hidden_size, output_size):
        # Inicializácia váh a biasov pre každú vrstvu
        self.params = {
            "w1": np.random.randn(input_size, hidden_size) * np.sqrt(1 / input_size),  # Xavier initialization
            "b1": np.zeros((1, hidden_size)),  # Biasy inicializované na nuly
            "w2": np.random.randn(hidden_size, output_size) * np.sqrt(1 / hidden_size),  # Xavier initialization
            "b2": np.zeros((1, output_size))  # Biasy inicializované na nuly
        }

    def forward(self, X):
        # Dopredné šírenie
        # here we multiplicate the matrices and add bias to the outcome
        self.z1 = np.dot(X, self.params["w1"]) + self.params["b1"]  # Vstup do skrytej vrstvy
        self.a1 = self.sigmoid(self.z1)  # Aktivácia skrytej vrstvy

        self.z2 = np.dot(self.a1, self.params["w2"]) + self.params["b2"]  # Vstup do výstupnej vrstvy
        self.a2 = self.softmax(self.z2)  # Aktivácia výstupnej vrstvy
        
        return self.a2

    @staticmethod
    def sigmoid(z):
        # Sigmoidná aktivačná funkcia
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def softmax(z):
        # Softmax funkcia
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Stabilizácia exponenciálnej funkcie
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def backward(self, X, y, learning_rate):
        # Spätné šírenie
        m = X.shape[0]
        
        # Výpočet gradientov
        dz2 = self.a2 - y  # Chyba výstupnej vrstvy
        dw2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        dz1 = np.dot(dz2, self.params["w2"].T) * self.a1 * (1 - self.a1)  # Chyba skrytej vrstvy
        dw1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # Aktualizácia váh a biasov
        self.params["w1"] -= learning_rate * dw1
        self.params["b1"] -= learning_rate * db1
        self.params["w2"] -= learning_rate * dw2
        self.params["b2"] -= learning_rate * db2

    @staticmethod
    def compute_loss(y_pred, y_true):
        # Cross-entropy loss
        m = y_true.shape[0]
        loss = -np.sum(y_true * np.log(y_pred + 1e-9)) / m
        return loss