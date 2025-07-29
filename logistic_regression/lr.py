import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def log_reg(X,y,epochs=100, alpha = 0.01, lmbda = 0.01 ):
    '''
    X -> (m,n)
    y -> (m,1)
    y E [0,1]
    '''
    m, n = X.shape
    assert m == y.shape[0]
    w = np.random.rand(n,1)
    b = np.random.rand(m,1)

    for _ in range(epochs):
        z = X.dot(w) + b 
        y_pred = sigmoid(z)
        loss = - 1/m * (y.T.dot(np.log(y_pred)) + (1 - y).T.dot(np.log(1-y_pred)))
        loss_reg = loss + lmbda/(2*m)* (w.T.dot(w))

        dw = 1 / m * X.T.dot(y_pred - y) + lmbda/m * w
        db = 1 / m * (y_pred - y)

        w -= alpha * dw 
        db -= alpha * db 

    y_pred = (sigmoid(X.dot(w) + b) > 0.5).astype(int)
    loss = np.mean(y_pred - y)

    print(f"Loss in logistic regression {loss:.4f}")


def softmax(z):
    z_shifted = z - np.max(z, axis=1,keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis = 1, keepdims = True)


def log_reg_multi(X, y, epochs=100, alpha = 0.01, lmbda = 0.01):
    '''
    x -> (n,d)
    y -> (n,k) # one-hot
    w -> (k,d)
    b -> (k,1)
    z -> (n,k)
    '''

    assert X.shape[0] == y.shape[0]

    n , d = X.shape
    _ , k = y.shape

    w = np.random.rand(k,d)
    b = np.random.rand(1,k) #explicity changing 

    for epoch in range(epochs):
        z = X.dot(w.T) + b
        y_pred = softmax(z)
        '''
        ensure that there is no log(0)
        '''
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        L = - 1/n * np.mean(np.sum(y * np.log(y_pred), axis=1))

        dw = (y_pred - y).T @ X 
        db = np.mean(y_pred-y, 0, keepdims = True)

        w -= alpha * dw 
        b -= alpha * db 

        if epoch == epochs -1 :
            print(f"Loss in log_reg multi {L:.4f}")

    
