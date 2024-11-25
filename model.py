import numpy as np
import numpy.typing as npt

class NeuralNetwork:
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        # Inicializácia váh a biasov pre každú vrstvu
        self.params = {
            "w1": np.random.randn(input_size, hidden_size) * np.sqrt(1 / input_size),  # Xavier initialization
            "b1": np.zeros((1, hidden_size)),  # Biasy inicializované na nuly
            "w2": np.random.randn(hidden_size, output_size) * np.sqrt(1 / hidden_size),  # Xavier initialization
            "b2": np.zeros((1, output_size))  # Biasy inicializované na nuly
        }

    def forward(self, X: npt.NDArray):
        # Dopredné šírenie
        # here we multiplicate the matrices and add bias to the outcome
        # vstupy x1, x2, ..., xn sa prenesu do siete
        # pre kazdy neuron sa vypocita vazeny sucet vstupov z = wx + b
        self.z1 = np.dot(X, self.params["w1"]) + self.params["b1"]  # Vstup do skrytej vrstvy
        # aktivacna funkcia sa pouzije na linearnu kombinaciu a
        self.a1 = self.sigmoid(self.z1)  # Aktivácia skrytej vrstvy

        # opakuje sa lin. kombinacia a aktivacia akt. funkcie na vsetky vrstvy
        self.z2 = np.dot(self.a1, self.params["w2"]) + self.params["b2"]  # Vstup do výstupnej vrstvy
        # na vystupnej vrstve sa ziskaju predikcie y
        self.a2 = self.softmax(self.z2)  # Aktivácia výstupnej vrstvy
        
        return self.a2

    @staticmethod
    def sigmoid(z: npt.NDArray):
        # Sigmoidná aktivačná funkcia
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def softmax(z: npt.NDArray):
        # Softmax funkcia
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Stabilizácia exponenciálnej funkcie
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def backward(self, X: npt.NDArray, y: npt.NDArray, learning_rate: float):
        # Spätné šírenie
        m = X.shape[0]
        
        # Výpočet gradientu chyby na vystupnej vrstve
        dz2 = self.a2 - y  # Chyba výstupnej vrstvy
        dw2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        # chyba sa prenasa opat cez vrstvy
        dz1 = np.dot(dz2, self.params["w2"].T) * self.a1 * (1 - self.a1)  # Chyba skrytej vrstvy
        dw1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # Aktualizácia váh a biasov podla gradientu a parametru learning rate
        self.params["w1"] -= learning_rate * dw1
        self.params["b1"] -= learning_rate * db1
        self.params["w2"] -= learning_rate * dw2
        self.params["b2"] -= learning_rate * db2

    @staticmethod
    def compute_loss(y_pred: npt.NDArray, y_true: npt.NDArray):
        # Cross-entropy loss
        m = y_true.shape[0]
        # 1e-9 is a safeguard against numerical instability (for y_pred == 0, log is undefined)
        loss = -np.sum(y_true * np.log(y_pred + 1e-9)) / m
        return loss