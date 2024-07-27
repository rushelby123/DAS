#This code provide an example of distributed optimization on a network. 
#The code generates a random network and then performs distributed 
#optimization on the network. 
#The two type of distributed optimization algorithm we are going to see are 
#distributed gradient descent method and distributed tracking method

import matplotlib.pyplot as plt
import numpy as np

# same as before
np.random.seed(0)
NN = 10

I_NN = np.eye(NN)
while 1:
    Adj = np.random.binomial(n=1, p=0.3, size=(NN, NN))
    Adj = np.logical_or(Adj, Adj.T)
    Adj = np.logical_and(Adj, np.logical_not(I_NN)).astype(int)

    test = np.linalg.matrix_power(I_NN + Adj, NN)
    if np.all(test > 0):
        break


AA = np.zeros(shape=(NN, NN))

for ii in range(NN):
    N_ii = np.nonzero(Adj[ii])[0]
    deg_ii = len(N_ii)
    for jj in N_ii:
        deg_jj = len(np.nonzero(Adj[jj])[0])
        AA[ii, jj] = 1 / (1 + max([deg_ii, deg_jj]))
AA += I_NN - np.diag(np.sum(AA, axis=0))

if 0:
    print(np.sum(AA, axis=0))
    print(np.sum(AA, axis=1))


def quadratic_fn(z, q, r):
    return 0.5 * q * z * z + r * z, q * z + r

#DISTRIUBTED GRADIENT TRACKING METHOD

Q = np.random.uniform(size=(NN))
R = np.random.uniform(size=(NN))

MAXITERS = 1000
# dd = 3
ZZ = np.zeros((MAXITERS, NN)) #ZZ[kk,ii,:]
cost = np.zeros((MAXITERS))

ZZ_gt = np.zeros((MAXITERS, NN))
SS_gt = np.zeros((MAXITERS, NN))
for ii in range(NN):
    _, SS_gt[0, ii] = quadratic_fn(ZZ_gt[0, ii], Q[ii], R[ii])

cost_gt = np.zeros((MAXITERS))

alpha = 1e-2

for kk in range(MAXITERS - 1):
    print(f"iter {kk}")

    # Distributed gradient
    for ii in range(NN):
        N_ii = np.nonzero(Adj[ii])[0]

        ZZ[kk + 1, ii] += AA[ii, ii] * ZZ[kk, ii]
        for jj in N_ii:
            ZZ[kk + 1, ii] += AA[ii, jj] * ZZ[kk, jj]

        _, grad_ell_ii = quadratic_fn(ZZ[kk + 1, ii], Q[ii], R[ii])

        ZZ[kk + 1, ii] -= alpha / (kk + 1) * grad_ell_ii

        ell_ii, _ = quadratic_fn(ZZ[kk, ii], Q[ii], R[ii])
        cost[kk] += ell_ii

    # gradient tracking
    for ii in range(NN):
        N_ii = np.nonzero(Adj[ii])[0]

        ZZ_gt[kk + 1, ii] += AA[ii, ii] * ZZ_gt[kk, ii]
        SS_gt[kk + 1, ii] += AA[ii, ii] * SS_gt[kk, ii]

        for jj in N_ii:
            ZZ_gt[kk + 1, ii] += AA[ii, jj] * ZZ_gt[kk, jj]
            SS_gt[kk + 1, ii] += AA[ii, jj] * SS_gt[kk, jj]

        ZZ_gt[kk + 1, ii] -= alpha * SS_gt[kk, ii]

        # print(Q[ii])
        _, grad_ell_ii_new = quadratic_fn(ZZ_gt[kk + 1, ii], Q[ii], R[ii])
        _, grad_ell_ii_old = quadratic_fn(ZZ_gt[kk, ii], Q[ii], R[ii])
        SS_gt[kk + 1, ii] += grad_ell_ii_new - grad_ell_ii_old

        ell_ii_gt, _ = quadratic_fn(ZZ_gt[kk, ii], Q[ii], R[ii])
        cost_gt[kk] += ell_ii_gt

if 0:
    fig, ax = plt.subplots()
    ax.plot(np.arange(MAXITERS), ZZ)
    ax.grid()

    fig, ax = plt.subplots()
    ax.plot(np.arange(MAXITERS), ZZ_gt)
    ax.grid()


ZZ_opt = -np.sum(R) / np.sum(Q)
opt_cost = 0.5 * np.sum(Q) * ZZ_opt**2 + np.sum(R) * ZZ_opt
# print(opt_cost)
# print(cost[-2])
# print(cost_gt[-2])

fig, ax = plt.subplots(sharex='all')
ax.semilogy(np.arange(MAXITERS - 1), np.abs(cost[:-1] - opt_cost),label="cost" )
ax.semilogy(np.arange(MAXITERS - 1), np.abs(cost_gt[:-1] - opt_cost),label="gradient tracking" )
plt.legend()
ax.grid()

plt.show()

# plt.plot(xx[0,:].T, xx[1,:].T, label="Real Path", color="C1")

# if x_des is not None:

#     plt.plot(x_des[0,:].T, x_des[1,:].T, label="Desired path", color="C2", linestyle="dashed")

#     plt.legend()

# plt.grid()

# plt.title(Title+ ": Position")

# plt.axis('equal')

# plt.xlabel("x(m)")

# plt.ylabel("y(m)")

# plt.show()