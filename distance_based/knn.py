import numpy as np
from scipy.stats import mode

def knn(X_train,y_train,X_test,k):
    """
    guesses the point based on K closest points
    X_train -> (n,d)
    y_train -> (n,1)
    X_train -> (m,1)
    dists -> (m,n)
    """
    assert X_train.shape[0] == y_train.shape[0]
    assert X_train.shape[1] == X_test.shape[1]
    X_train_sq = np.sum(X_train**2, axis = 1)
    X_test_sq = np.sum(X_test**2, axis = 1)

    cross_term = X_test @ X_train.T

    dists = np.sqrt(X_test_sq.reshape(-1,1) + X_train_sq.reshape(1,-1) -2*cross_term)

    idx = np.argsort(dists, axis=1)[:,:k]
    labels = y_train[idx]

    y_pred, _ = mode(labels, axis = 1)

    y_pred = y_pred.ravel()
    print(f"Predctions of knn {y_pred}")
