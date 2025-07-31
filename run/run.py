from linear_regression import simple_lr
from logistic_regression import lr
from distance_based import knn, k_means
from SVM import svm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


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

def run_svm1():
    model = svm.PrimalSVM(0.001,10,1000)
    X, y = make_blobs(n_samples = 100, centers = 2, random_state=42)
    y = 2*y - 1 
    model.fit(X,y)
    print(f"Training accurcy for svm1: {model.score(X,y)}")

    def plot_boundary(model, X, y):
        plt.scatter(X[:,0], X[:,1], c=y, cmap='bwr', s=30)
        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        xx, yy = np.meshgrid(np.linspace(*xlim,100), np.linspace(*ylim,100))
        grid = np.c_[xx.ravel(),yy.ravel()]
        z = model.predict(grid).reshape(xx.shape)

        plt.contourf(xx, yy, z, cmap='bwr', alpha = 0.2)
        plt.title("SVM decision boundary (Primal)")
        plt.show()
    plot_boundary(model, X, y)
