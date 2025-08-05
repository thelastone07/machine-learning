import numpy as np

class SGD:
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters 
        self.lr = lr 

    def step(self):
        for param, grad in self.parameters : 
            param -= self.lr * grad 