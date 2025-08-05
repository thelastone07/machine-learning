import numpy as np

class Dense:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim 
        self.output_dim = output_dim 

        self.W = np.random.randn(input_dim, output_dim) * 0.01
        self.b = np.zeros((1,output_dim))

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b
    
    def backward(self, dout):
        self.dW = self.x.T @ dout 
        self.db = np.sum(dout, axis=0, keepdims=True)
        dx = dout @ self.W.T 
        return dx 
    
    def get_params_and_grads(self):
        return [(self.W, self.dW), (self.b, self.db)]

