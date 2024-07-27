import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from functions import animation as my_animate


NN = 15
I_NN = np.eye(NN)

p_edge = 0.7
while True:
    graph = nx.binomial_graph(NN, p_edge)
    Adj = nx.adjacency_matrix(graph).toarray()

    test = np.linalg.matrix_power(I_NN + Adj, NN)

    if np.all(test > 0):
        print("The graph is connected")
        break
    else:
        print("The graph is connected")

deg_vect = np.sum(Adj, axis=0)
deg_matrix = np.diag(deg_vect)
Laplacian = deg_matrix - Adj

# print(Laplacian)
n_x = 2

n_le = 5
n_fo = NN - n_le

L_f = Laplacian[0:n_fo, 0:n_fo]
L_fl = Laplacian[0:n_fo, n_fo:]


LL = np.block(
    [
        [L_f, L_fl],
        [np.zeros((n_le, NN))],
    ]
)

I_nx = np.eye(n_x)
LL_kron = np.kron(LL, I_nx)
BB_kron = np.zeros((n_x * NN, n_x * n_le))
BB_kron[n_x * n_fo :, :] = np.eye((n_x * n_le))

k_i = 5
K_I = np.block(
    [
        [-k_i * np.eye(n_x * n_fo)],
        [np.zeros((n_x * n_le, n_x * n_fo))],
    ]
)

LL_ext = np.block(
    [
        [LL_kron, K_I],
        [np.kron(L_f, I_nx), np.kron(L_fl, I_nx), np.zeros((n_x * n_fo, n_x * n_fo))],
    ]
)

BB_ext = np.block(
    [
        [BB_kron],
        [np.zeros((n_x * n_fo, n_x * n_le))],
    ]
)

CC_ext = np.block(
    [
        [
            np.eye(n_x * NN),
            np.zeros((n_x * NN, n_x * n_fo)),
        ]
    ]
)

A = -LL_ext
B = BB_ext
C = CC_ext
D = 0

XX_0 = np.random.rand(n_x * (NN + n_fo))

Tmax = 60.0
dt = 0.01
horizon = np.arange(0.0, Tmax, dt)

v_le = 1

UU = v_le * np.ones((n_x * n_le, len(horizon)))

# sys = control.StateSpace(A, B, C, D)
# (hor, XX) = control.forced_response(sys, X0=XX_0, U=UU, T=horizon)
XX = np.zeros((n_x * (NN + n_fo), len(horizon)))
XX[:, 0] = XX_0
for k in range(len(horizon) - 1):
    dX = A @ XX[:, k] + B @ UU[:, k]
    XX[:, k + 1] = XX[:, k] + dX * dt


if 0:
    plt.plot(horizon, XX.T)

if 1:
    plt.figure("Animation")
    my_animate(XX, NN, n_x, n_le, horizon, dt=10)

plt.show()
