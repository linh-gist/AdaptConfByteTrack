import numpy as np


def descent(X, y, learning_rate=0.001, iters=100):
    w = np.zeros((X.shape[1], 1))
    for i in range(iters):
        grad_vec = -(X.T).dot(y - X.dot(w))
        w = w - learning_rate * grad_vec
    return w


X = np.loadtxt("MOT16-13_f250.txt")
print(X)
# grad_vec = -(X.T).dot(y - X.dot(w))
