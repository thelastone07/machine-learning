from linear_regression import simple_lr
from logistic_regression import lr
from distance_based import knn, k_means
from SVM import svm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from perceptron import pp
from dl.dense.dense import Dense
from dl.activation.activation import ReLU, Sigmoid
from dl.loss.loss import CrossEntropyLoss, MSELoss
from dl.optimizers.SGD import SGD
from cnn.conv import Conv2D
from cnn.pool import MaxPool
from cnn.fcn import FC
 


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

def run_perceptron():
    model = pp.Perceptron([2,4,4,1])

    X = np.array([[0,0,1,1],
                  [0,1,0,1]])
    y = np.array([[0,1,1,0]])

    model.train(X,y, epochs=3000, lr = 1)
    pred = model.predict(X)
    print("Preditcions using preceptron",pred.astype(int))

def run_dl():
    np.random.seed(0)

    # Dummy data: 5 samples, 4 features, 3 classes
    x = np.random.randn(5, 4)
    y_true = np.array([0, 2, 1, 2, 0])  # class labels

    # Create model: Dense1 -> ReLU -> Dense2
    dense1 = Dense(4, 10)
    relu = ReLU()
    dense2 = Dense(10, 3)

    # Loss function
    loss_fn = CrossEntropyLoss()

    # Optimizer
    params = dense1.get_params_and_grads() + dense2.get_params_and_grads()
    optimizer = SGD(parameters=params, lr=0.1)

    # Training loop
    epochs = 100
    for epoch in range(1, epochs + 1):
        # === Forward ===
        out1 = dense1.forward(x)
        out2 = relu.forward(out1)
        logits = dense2.forward(out2)
        loss = loss_fn.forward(logits, y_true)

        # === Backward ===
        dlogits = loss_fn.backward()
        dout2 = dense2.backward(dlogits)
        dout1 = relu.backward(dout2)
        _ = dense1.backward(dout1)

        # === Optimizer step ===
        optimizer.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Loss: {loss:.4f}")

    # === Predict on training data ===
    out1 = dense1.forward(x)
    out2 = relu.forward(out1)
    logits = dense2.forward(out2)
    preds = np.argmax(logits, axis=1)

    # === Accuracy ===
    accuracy = np.mean(preds == y_true)
    print("\nPredictions for dl:", preds)
    print("Ground Truth:", y_true)
    print(f"Accuracy: {accuracy * 100:.2f}%")

def run_cnn():
    X = np.random.randn(4, 3, 32, 32)
    y_true = np.random.randn(4, 10)

    # Layers
    conv = Conv2D(c=3, h=32, w=32, kernels=8, size=(3, 3), stride=1, n=4)
    pool = MaxPool()
    flatten = FC()
    dense1 = Dense(8 * 15 * 15, 64)
    activation1 = Sigmoid()
    dense2 = Dense(64, 10)
    loss_fn = MSELoss()

    # Optimizer using get_params_and_grads
    params = (
        conv.get_params_and_grads() +
        dense1.get_params_and_grads() +
        dense2.get_params_and_grads()
    )
    optimizer = SGD(params, lr=0.01)

    # Training Step
    for epoch in range(50):
        # Forward pass
        out = conv.forward(X)  
        out = pool.forward(out, kh=2, kw=2, stride=2) 
        out = flatten.forward(out)           
        out = dense1.forward(out) 
        out = activation1.forward(out)       
        out = dense2.forward(out)        
        loss = loss_fn.forward(out, y_true)


        print(f"Epoch {epoch+1}: Loss = {loss:.4f}")

        # Backward pass
        dout = loss_fn.backward()
        dout = dense2.backward(dout)
        dout = activation1.backward(dout)
        dout = dense1.backward(dout)
        dout = flatten.backward(dout)
        dout = pool.backward(dout)
        dout = conv.backward(dout)

        # Update weights
        optimizer.step()

    # Prediction after training
    out = conv.forward(X)
    out = pool.forward(out, kh=2, kw=2, stride=2)
    out = flatten.forward(out)
    out = dense1.forward(out)
    out = activation1.forward(out)
    out = dense2.forward(out)

    # Print prediction
    print("Predictions (logits):")
    print(out)

    # If classification, print class predictions
    predicted_classes = np.argmax(out, axis=1)
    print("Predicted classes:")
    print(predicted_classes)




    
