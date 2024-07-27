import matplotlib.pyplot as plt
import numpy as np

# Create a simple example of the incremental gradient method,
# considering a collection of N quadratic cost functions
# and a decision variable z in R^d
N = 3
d = 2

# N random quadratic cost functions
Q = []
r = []
for i in range(N):
    Qi = np.random.uniform(size=(d, d))
    Qi = 0.5 * (Qi + Qi.T)
    Q += [Qi]
    ri = np.random.uniform(size=(d))
    r += [ri]

# Initialization
max_iters = 100
stepsize = 1e-4
z = np.zeros(shape=(max_iters, d))
zinit = np.random.normal(size=(d))
z[0] = zinit
cost = np.zeros(shape=(max_iters))

# Define a function to compute the direction of the incremental gradient method
# where z is the current decision variable
# Q is the collection of cost function matrices associated with each cost function
# r is the collection of affine vectors associated with each cost function
# idx is the index of the cost functions to be considered in the computation
def compute_direction(z, Q, r, idx):
    d = len(z)
    direction = np.zeros(shape=(d))
    # n = len(Q)
    for i in idx:
        direction += Q[i] @ z + r[i]
    return direction

# Incremental gradient method
batch_size = 2 # batch size, must be less than N
for k in range(max_iters - 1):
    print(f"Iteration {k:d}")

    # Define a policy to select which cost functions to consider
    # ik = [k % N]  # incremental (cyclic order)
    # ik = [np.random.random_integers(low=0, high=N - 1)]  # incremental (random order)
    ik = np.random.choice(range(0, N - 1), batch_size, replace=False)  # batch (random order) replace = False to avoid repetitions

    print(f"Cost functions indeces: {ik}")
    batch_gradient = compute_direction(z[k], Q, r, ik)
    direction = -batch_gradient

    z[k + 1] = z[k] + stepsize * direction

    for i in range(N):
        cost[k] += 0.5 * z[k].T @ Q[i] @ z[k] + r[i].T @ z[k]

plt.plot(np.arange(max_iters), z)
plt.show()
