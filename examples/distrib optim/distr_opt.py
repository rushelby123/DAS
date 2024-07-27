#This code provide an example of distributed optimization on a network. 
#The code generates a random network and then performs distributed 
#optimization on the network. 
#The two type of distributed optimization algorithm we are going to see are 
#distributed gradient descent method and distributed tracking method

import matplotlib.pyplot as plt
import numpy as np

# number of agents
NN = 10

# generate a random network
I_NN = np.eye(NN)
while 1:
    #generate a random network
    Adj = np.random.binomial(n=1, p=0.3, size=(NN, NN))
    #keep just the symmetric part of the matrix
    Adj = np.logical_or(Adj, Adj.T)
    #remove the self loops
    Adj = np.logical_and(Adj, np.logical_not(I_NN)).astype(int)

    # check if the network is connected
    test = np.linalg.matrix_power(I_NN + Adj, NN)
    if np.all(test > 0):
        break

#Compute the set of weights for the network, using Metropolis-Hastings weights methood set of slides 3 (avaraging)
AA = np.zeros(shape=(NN, NN))
for ii in range(NN):
    #get the neighbors of the agent ii
    N_ii = np.nonzero(Adj[ii])[0]
    #compute the degree of the agent ii
    deg_ii = len(N_ii)
    #compute the weights of the agent ii
    for jj in N_ii:
        deg_jj = len(np.nonzero(Adj[jj])[0])
        AA[ii, jj] = 1 / (1 + max([deg_ii, deg_jj]))
AA += I_NN - np.diag(np.sum(AA, axis=0))

if 0:
    #since the matrix AA is doubly stochastic matrix, 
    #the sum of the rows and columns should be 1
    print(np.sum(AA, axis=0))
    print(np.sum(AA, axis=1))

# define the quadratic function for our distributed optimization problem
def quadratic_fn(z, q, r):
    # returns the value and the gradient of the function
    return 0.5 * q * z**2 + r * z, q * z + r

# define matrices Q and R as 
Q = np.random.uniform(size=(NN))
R = np.random.uniform(size=(NN))

#GRADIENT DESCENT METHOD
MAXITERS = 1000
# dd = 3

ZZ = np.zeros((MAXITERS, NN))
cost = np.zeros((MAXITERS))
# step size
alpha = 1e-2

for kk in range(MAXITERS - 1):
    print(f"iter {kk}")

    for ii in range(NN):
        N_ii = np.nonzero(Adj[ii])[0]

        ZZ[kk + 1, ii] += AA[ii, ii] * ZZ[kk, ii]
        for jj in N_ii:
            ZZ[kk + 1, ii] += AA[ii, jj] * ZZ[kk, jj]

        _, grad_ell_ii = quadratic_fn(ZZ[kk + 1, ii], Q[ii], R[ii])

        # update the value of the agent using diminishing step size! 
        # diminishing step size guarantees convergency of the distributed
        # algorithm, if you want to check remove "/ (kk + 1)"
        ZZ[kk + 1, ii] -= alpha / (kk + 1) * grad_ell_ii

        ell_ii, _ = quadratic_fn(ZZ[kk, ii], Q[ii], R[ii])
        cost[kk] += ell_ii

#plot the evolution of the agents at each iteration and the cost
fig, ax = plt.subplots()
ax.plot(np.arange(MAXITERS), ZZ)

fig, ax = plt.subplots()
ax.plot(np.arange(MAXITERS - 1), cost[:-1])

plt.show()
