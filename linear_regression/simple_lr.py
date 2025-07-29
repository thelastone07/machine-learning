import numpy as np
import random

def simple_linear_regression(X, y):
    '''
    both have 1d
    y = aX + b
    '''
    x_mean = np.mean(X)
    y_mean = np.mean(y)
    
    X = X - x_mean 
    y = y - y_mean 

    a = np.mean(X*y) / np.mean(X**2)

    b = y_mean - a * x_mean

    loss = np.mean(X*a - b)
    print(f"Loss in simple linear regression {loss:.4f}")

    

def linear_regression_normal_eqn(X,y):
    '''
    multiple dimension, but solving using normal equationo
    y -> m * 1
    x -> m * n 
    w -> (n+1) * 1
    theta -> (n+1)*1
    '''
    assert X.shape[0] == y.shape[0]

    b = np.ones((X.shape[0],1))
    X = np.concatenate((X,b),axis = 1)

    XTX = X.T @ X 
    try :
        theta = np.linalg.inv(XTX) @ X.T @ y
    except np.linalg.LinAlgError:
        theta = np.linalg.pinv(XTX) @ X.T @ y
    
    loss = np.mean(X.dot(theta) - y)
    print(f"Loss in  linear regression normal eqn {loss:.4f}")

def linear_regression(X,y,alpha = 0.01, epochs = 1000):
    '''
    let's try multiple weight initalization
    y -> m * 1
    X -> m * n
    theta0 -> m * 1
    theta1 -> n * 1
    
    '''
    assert X.shape[0] == y.shape[0]
    m, n = X.shape
    #zero
    theta0 = np.zeros((m,1))
    theta1 = np.zeros((n,1))
    #random 
    theta0_random =  np.random.rand(m,1)
    theta1_random = np.random.rand(n,1)

    #uniform
    theta1_u = (theta1_random - np.min(theta1_random)) / (np.max(theta1_random) - np.min(theta1_random))
    
    #xavier
    '''
    Var[y] = var[x]
    so var[w] = 1/n
    '''
    ll_x = - np.sqrt(2)/np.sqrt(n) 
    ul_x = np.sqrt(2)/np.sqrt(n)
    theta1_x = theta1_u * (ul_x - ll_x) + ll_x

    for epoch in range(epochs):
        y_pred = X.dot(theta1) + theta0
        dw = (2/m) * X.T.dot(y_pred - y)
        db = (2/m) * np.sum(y_pred - y)

        theta1 -= alpha * dw 
        theta0 -= alpha * db
    
    loss = np.mean(X.dot(theta1) + theta0 - y)
    print(f"Loss using linear regression {loss:.4f}")


        


     
    


    
    
    