from sklearn.datasets import load_boston
boston = load_boston()

from matplotlib import pyplot as plt
plt.plot(boston.data[:,5], boston.target, 'o', label='Original data', markersize=10, color='b')

import numpy as np

x = boston.data
y = boston.target
m = len(y)
theta = np.zeros(len(x[0, :]))
alpha = 0.1
num_iters = 400

# x*theta
def compute_cost(x, y, theta):
    j = 0
    m = len(y)
    for i in range(m):
        h = np.dot(x[i,:], theta)
        j += (h - y[i])**2;

    return (j/(2*m))

def gradient_descent(x, y, theta, alpha, num_iters):
    m = len(y)
    n = len(theta)
    j_history = np.zeros(num_iters)
    tmp = np.zeros(n)
    for iter in range(num_iters):
        for j in range(n):
            s = 0
            for i in range(m):
                print("xi{}".format(x[i,:]))
                print(theta)
                h = np.dot(x[i,:], theta)
                s += (h - y[i])*x[i][j]
            tmp[j] = theta[j] - (alpha * 1.0/m) * s
        theta = tmp

        # Saving cost history in every iteration
        j_history[iter] = compute_cost(x, y, theta)
        if iter > 0:
            diff = j_history[iter - 1] - j_history[iter]
            if(diff <= 0):
                print("Diff = {} not decreasing\n".format(diff));
                return 1
    return theta

print(compute_cost(x,y,theta))
print("N={}, M={}".format(len(theta),m))
res = gradient_descent(x, y, theta, alpha, num_iters)
print(res)
