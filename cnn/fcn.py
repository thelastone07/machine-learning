import numpy as np 

class FC:
    def forward(self, X):
       self.x_shape = X.shape 
       return X.reshape(X.shape[0],-1)
    
    def backward(self,dout):
        return dout.reshape(self.x_shape)