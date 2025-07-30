import numpy as np

def kmeans(X,k, epochs = 100):
    '''
    given - break it into k-clusters
    X -> (n,d)
    '''
    n, d = X.shape
    cds = X[np.random.randint(0,n-1,size=k)] # (k,d)
    c = np.zeros((n,1)) # (n,1)
    for _ in range(epochs):
        x2 = np.sum(X**2, axis = 1) # (n,1)
        c2 = np.sum(cds**2, axis = 1) #(k,1)
        cross = X @ cds.T # (n,k)
        diff = np.sqrt(x2[:,np.newaxis] + c2[np.newaxis,:] - 2*cross) #(n,k)

        idx = np.argsort(diff, axis = 1)[:,0] #(n,1)
        c = idx
        for i in range(k):
            mask = (c == i)
            X_ = X[mask]
            if len(X_) > 0:
                cds[i] = X_.mean(axis=0)

    c = c.ravel()
    print(f"classes after kmeans {c}")
