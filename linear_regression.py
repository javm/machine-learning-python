from sklearn.datasets import load_boston
boston = load_boston()

from matplotlib import pyplot as plt
plt.plot(boston.data[:,5], boston.target, 'o', label='Original data', markersize=10, color='b')

import numpy as np

y = boston.target
data = boston.data
m = len(y)

x = np.ones((m, len(data[0, :])+1))
for i in range(len(y)):
    x[i, :] = np.insert(data[i, :], 0, 1)
    #print(x[i, :])



theta = np.zeros(len(x[0, :]))
alpha = 0.00001
num_iters = 20000

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
                #print("xi{}".format(x[i,:]))
                #print(theta)
                h = np.dot(x[i,:], theta)
                s += (h - y[i])*x[i][j]
            tmp[j] = theta[j] - (alpha * 1.0/m) * s
        theta = tmp

        # Saving cost history in every iteration
        j_history[iter] = compute_cost(x, y, theta)
        print("Iteration: {}, cost: {}".format(iter, j_history[iter]))
        if iter > 0:
            diff = j_history[iter - 1] - j_history[iter]
            if(diff <= 0):
                print("Diff = {} not decreasing\n".format(diff));
                return 1
    return theta

print("N={}, M={}".format(len(theta),m))
res = gradient_descent(x, y, theta, alpha, num_iters)
print(res)

'''
Iteration: 19998, cost: 21.015173242118593
Iteration: 19999, cost: 21.015042196545636
[ 0.08525868 -0.08482517  0.10372558 -0.03622697  0.06152549  0.0364464
  0.91197831  0.10822184  0.0081737   0.0888527  -0.00250904  0.47756314
  0.02797547 -0.80430518]
'''
