from linear_regression import simple_lr
from logistic_regression import lr
import numpy as np

def main():
    # X = np.array([1,2,3])
    # y = np.array([5,7,9])
    # simple_lr.simple_linear_regression(X,y) 

    # y = np.random.rand(10,1)
    # X = np.random.rand(10,3)
    # simple_lr.linear_regression_normal_eqn(X,y)

    # simple_lr.linear_regression(X,y)

    X = np.random.rand(10,4)
    y = np.random.rand(10,1)
    y = (y > 0.5).astype(int)
    lr.log_reg(X,y)

    X = np.random.rand(10,4)
    y_random = np.random.randint(0,4,10)
    y = np.eye(4)[y_random]

    lr.log_reg_multi(X,y)


if __name__ == "__main__":
    main()