import numpy as np

class PrimalSVM:
    '''
    hinge loss + L2 regularization
    '''
    def __init__(self, lr=0.0001, C=1.0, epochs=1000):
        self.lr = lr
        self.C = C
        self.epochs = epochs 
        self.w = None 
        self.b = 0
    
    def fit(self, X, y):
        n, d = X.shape
        assert n == y.shape[0]
        self.w = np.zeros(d)
        self.b = 0 

        for epoch in range(self.epochs):
            cond = y * (X @ self.w + self.b)
            
            indicator = (cond < 1).astype(float)

            dw = self.w - self.C * ((indicator * y)[:, np.newaxis] * X).sum(axis=0)
            db = -self.C * np.sum(indicator * y)

            self.w -= self.lr * dw 
            self.b -= self.lr * db

    def predict(self,X):
        return np.sign(X @ self.w + self.b)
    
    def score(self, X, y):
        return np.mean(self.predict(X) == y)
    
class SVM:
    def __init__(self, lr=0.0001, C=1.0, epochs=1000):
        self.lr = lr
        self.C = C
        self.epochs = epochs
        self.w = None
        self.b = 0
            