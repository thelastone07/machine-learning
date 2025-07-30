from linear_regression import simple_lr
from logistic_regression import lr
from distance_based import knn, k_means
import numpy as np


def run_linear_regression():
    X = np.array([1,2,3])
    y = np.array([5,7,9])
    simple_lr.simple_linear_regression(X,y) 

    y = np.random.rand(10,1)
    X = np.random.rand(10,3)
    simple_lr.linear_regression_normal_eqn(X,y)

    simple_lr.linear_regression(X,y)

def run_logistic_regression():
    X = np.random.rand(10,4)
    y = np.random.rand(10,1)
    y = (y > 0.5).astype(int)
    lr.log_reg(X,y)

    X = np.random.rand(10,4)
    y_random = np.random.randint(0,4,10)
    y = np.eye(4)[y_random]

    lr.log_reg_multi(X,y)

def run_knn():
    X_train = np.random.rand(100,3)
    y_train = np.random.randint(0,5,size=100)
    X_test = np.random.rand(10,3)
    knn.knn(X_train, y_train, X_test, 5)

def run_kmeans():
    X = np.random.rand(10,4)
    k = 3
    k_means.kmeans(X,k)




