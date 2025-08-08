import numpy as np 

class ReLU:
    def forward(self, x):
        self.mask = x > 0 
        return x * self.mask 
    
    def backward(self, dout): 
        return dout * self.mask 
    
class Sigmoid: 
    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out
    
    def backward(self, dout):
        dx = dout * self.out * (1 - self.out)
        return dx
