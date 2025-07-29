from linear_regression import simple_lr
import numpy as np

def main():
    X = np.array([1,2,3])
    y = np.array([5,7,9])
    simple_lr.simple_linear_regression(X,y) 

    y = np.random.rand(10,1)
    X = np.random.rand(10,3)
    simple_lr.linear_regression_normal_eqn(X,y)

    simple_lr.linear_regression(X,y)



if __name__ == "__main__":
    main()