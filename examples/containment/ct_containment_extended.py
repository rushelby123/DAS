import control
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from functions import plot_trajectory as my_plot
from functions import animation as my_animate

# EXAMPLE OF THE CONTAINMENT IN CONTINUOUS TIME
# LOOK AT THIS EXAMPLE AFTER ct_containment.py
# the difference from the previous case is that in this example we are going to extend
# the result of the previous example using the Kronecker product

NN = 4
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
n_x = 2 # number of states of each agent

n_le = 3
n_fo = NN - n_le

L_f = Laplacian[0:n_fo, 0:n_fo]
L_fl = Laplacian[0:n_fo, n_fo:]


LL = np.block(
    [
        [L_f, L_fl],
        [np.zeros((n_le, NN))],
    ]
)

# I_nx is the identity matrix of size of the state of each agent
I_nx = np.eye(n_x)
# LL_kron is the Kronecker product of LL and I_nx
LL_kron = np.kron(LL, I_nx)

# the kronacker product is used


A = -LL_kron
B = np.zeros((n_x * NN, n_x * n_le))
B[n_x * n_fo :, :] = np.eye((n_x * n_le)) # this matrix is used to control the leaders
C = np.eye(n_x * NN)
D = np.zeros((n_x * NN, n_x * n_le))

sys = control.StateSpace(A, B, C, D)

Tmax = 20.0
dt = 0.01
horizon = np.arange(0.0, Tmax, dt)
XX_0 = np.random.rand(n_x * NN, 1)

# here we can set the velocity of the leaders
#     | x_velocity |  
# v = |            |
#     | y_velocity |

v_le = 1

UU = v_le * np.ones((n_x * n_le, len(horizon))) 

(hor, XX) = control.forced_response(sys, X0=XX_0, U=UU, T=horizon)

if 1:
    plt.plot(hor, XX.T)

if 1:
    plt.figure("Animation")
    my_animate(XX, NN, n_x, n_le, horizon, dt=10)

plt.show()
