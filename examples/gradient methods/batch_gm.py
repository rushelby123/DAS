import matplotlib.pyplot as plt
import numpy as np


# Create an example of the batch gradient method,
# consdering a collection of N quadratic cost functions 
# and a decision variable z in R^d

# N cost functions and d is the dimension of the decision variable
N = 10
d = 2

# Quadratic cost functions
# Q = [Q0, Q1, ...Q_N-1] collection of d x d matrices
Q = []
# r = [r0, r1, ...r_N-1] collection of d x 1 vectors
r = []


# ell_i (z) = 0.5* z'*Qi*z+ ri'*z
for i in range(N):
    Qi = np.random.uniform(size=(d, d)) # random matrix for quadratic cost
    Qi = 0.5 * (Qi + Qi.T) # make it symmetric
    Q += [Qi] # add to the list of cost functions

    ri = np.random.uniform(size=(d)) # random affine vector 
    r += [ri] 

# Initiliazation 
max_iters = 1000
stepsize = 1e-4
z = np.zeros(shape=(max_iters, d))
zinit = np.random.normal(size=(d))
z[0] = zinit
cost = np.zeros(shape=(max_iters))

# Batch gradient method
for k in range(max_iters - 1):
    print(f"Iteration {k:d}")

    batch_gradient = np.zeros(shape=(d))
    for i in range(N):
        # Compute the gradient of the cost function i with respect to z
        batch_gradient += Q[i] @ z[k] + r[i]

        # Compute the cost function i at z
        cost[k] += 0.5 * z[k].T @ Q[i] @ z[k] + r[i].T @ z[k]

    # Compute the "discent" direction
    direction = -batch_gradient

    # Update the decision variable
    z[k + 1] = z[k] + stepsize * direction

plt.plot(np.arange(max_iters), z)
plt.show()
