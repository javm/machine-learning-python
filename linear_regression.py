from sklearn.datasets import load_boston
boston = load_boston()

from matplotlib import pyplot as plt
plt.plot(boston.data[:,5], boston.target, 'o', label='Original data', markersize=10, color='b')


import numpy as np

x = boston.data
y = boston.target
m = len(y)
theta = np.zeros(len(x[0, :]))

# x*theta
def compute_cost(x, y, theta):
    j = 0
    m = len(y)
    for i in range(m):
        h = np.dot(x[i,:], theta)
        j += (h - y[i])**2;

    return j/(2*m)

print(compute_cost(x,y,theta))
