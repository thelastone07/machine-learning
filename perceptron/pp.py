import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def  sig_d(x):
    s = sigmoid(x)
    return s* (1 - s)

def dL_dZ(y, y_hat):
    return y_hat - y

def binary_cross_entropy(y, y_hat):
    e = 1e-8
    return -np.mean( y * np.log(y_hat + e) + (1- y) * np.log(y_hat + e))

class Perceptron:
    def __init__(self, layer_dims):
        self.L = len(layer_dims) - 1
        self.params = {}
        self.cache = {} 
        np.random.seed(42)
        for l in range(1, len(layer_dims)):
            self.params[f"W{l}"] = np.random.randn(layer_dims[l],layer_dims[l-1])
            self.params[f"b{l}"] = np.zeros((layer_dims[l],1))

        
    def forward(self, X):
        A = X
        self.cache['A0'] = X 
        for l in range(1, self.L + 1):
            w = self.params[f"W{l}"]
            b = self.params[f"b{l}"]
            z = w @ A + b 
            A = sigmoid(z)
            self.cache[f"Z{l}"] = z 
            self.cache[f"A{l}"] = A
        return A
    
    def backward(self,Y):
        m = Y.shape[1]
        grads = {}
        dZ = dL_dZ(Y, self.cache[f"A{self.L}"])

        for l in reversed(range(1, self.L+1)):
            A_prev = self.cache[f"A{l-1}"]
            w = self.params[f"W{l}"]

            grads[f"dw{l}"] = (1/m) * dZ @ A_prev.T
            grads[f"db{l}"] = (1/m) * np.sum(dZ, axis = 1, keepdims=True)

            if l > 1:
                dA_prev = w.T @ dZ 
                dZ = dA_prev * sig_d(self.cache[f"Z{l-1}"])
            
        return grads
    
    def update_params(self, grads, lr):
        for l in range(1, self.L+1):
            self.params[f"W{l}"] -= lr* grads[f"dw{l}"]
            self.params[f"b{l}"] -= lr* grads[f"db{l}"]

    def train(self, X, y, epochs=1000, lr = 0.1, print_loss=True):
        for epoch in range(epochs):
            y_hat =  self.forward(X)
            loss = binary_cross_entropy(y, y_hat)
            grads = self.backward(y)
            self.update_params(grads,lr)
            if print_loss and epoch % 100 == 0:
                print(f"epoch {epoch} : Loss = {loss:.4f}")
    
    def predict(self, X):
        y = self.forward(X)
        return y > 0.5

