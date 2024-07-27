import control
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from functions import plot_trajectory as my_plot
from functions import animation as my_animate

# EXAMPLE OF THE CONTAINMENT IN CONTINUOUS TIME


# NN is the number of agents and I_NN is the identity matrix of size NN
NN = 4 # number of followers and leaders
I_NN = np.eye(NN)

# Generate a random connected graph, note that the graph is undirected
p_edge = 0.7 # probability of edge existence
while True:
    graph = nx.binomial_graph(NN, p_edge) # generate a random graph
    Adj = nx.adjacency_matrix(graph).toarray() # adjacency matrix of the graph

    test = np.linalg.matrix_power(I_NN + Adj, NN) # check if the graph is connected
    #nx.draw(graph, with_labels=True, font_weight="bold")

    if np.all(test > 0):
        print("The graph is connected")
        break
    else:
        print("The graph is connected")

# check slide 2 of part4 for the definition of the Laplacian matrix
deg_vect = np.sum(Adj, axis=0) # out-degree vector of the graph
deg_matrix = np.diag(deg_vect) # degree matrix of the graph
Laplacian = deg_matrix - Adj # Laplacian matrix of the graph

#print(Laplacian)

# number of followers and leaders
n_le = 2
n_fo = NN - n_le

# The laplacian matrix can be partitioned as follows
L_f = Laplacian[0:n_fo, 0:n_fo]
L_fl = Laplacian[0:n_fo, n_fo:]

# The Laplacian matrix of the entire system is
LL = np.block(
    [
        [L_f, L_fl],
        [np.zeros((n_le, NN))],# in this case the leaders are stationary
    ]
)
# print(LL)

# The state-space representation of the system is
# x_dot = Ax + Bu = -LLx 
A = -LL 
B = np.zeros((NN, n_le))
# y = Cx + Du = x
C = np.eye(NN)
D = np.zeros((NN, n_le))

sys = control.StateSpace(A, B, C, D)

# Comppute the forced response of the system
Tmax = 5.0 # simulation time
dt = 0.01 # time step
horizon = np.arange(0.0, Tmax, dt) # This function generates a time vector from 0 to Tmax with time step dt
XX_0 = np.random.rand(NN, 1) # initial condition of the system
UU = 0 * np.ones((n_le, len(horizon))) # input of the system (in this case the input is zero)

(hor, XX) = control.forced_response(sys, X0=XX_0, U=UU, T=horizon)

plt.plot(hor, XX.T)
plt.show()
