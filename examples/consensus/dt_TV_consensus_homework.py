#
# Discrete-time Time-Varing Average Consensus
# Ivano Notarnicola, Lorenzo Pichierri
# Bologna, 05/03/2024
#

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

#################################################################

max_iters = 100    # time horizon
NN = 10    # number of agents
p_ER = 0.1    # probability of edge creation

I_NN = np.identity(NN, dtype=int)
ONES = np.ones((NN,NN))
ZEROS = np.zeros((NN,NN))
XX = np.zeros((NN,max_iters))
XX_init = 10*np.random.rand(NN)
XX[:,0] = XX_init 

for kk in range (max_iters-1):
    #IMPORTANT NOTE:
    # Generate a new undirected graph at each step t, note that the graph is not connected at each t
    # Therefore in the plot you may see some agents that have stopped updating their state and 
    # its value is constant from a certain iterval of time
    G = nx.binomial_graph(NN,p_ER)
    Adj = nx.adjacency_matrix(G)
    Adj = Adj.toarray()

    # Generate the weighted adjacency matrix
    AA = 1.5*I_NN + 0.5*Adj

    ## Row-stochasticity
    # while any(abs(np.sum(AA,axis=1)-1) > 1e-10):
    #     AA = AA/(AA@ONES) # -> Row-stochasticity
    #     AA = np.maximum(AA,ZEROS)

    ## Doubly-stochasticity
    while any(abs(np.sum(AA,axis=1)-1) > 1e-10) or any(abs(np.sum(AA,axis=0)-1) > 1e-10):
        AA = AA/(AA@ONES) # -> Row-stochasticity
        AA = AA/(ONES@AA) # -> Col-stochasticity
        AA = np.maximum(AA,ZEROS)

    ################################################
    ## CONSENSUS Algorithm
    
    for ii in range (NN):
            
        # Find the set of neighbors
        Nii = list(G.neighbors(ii)) 

        # Update the state with its own contribution 
        XX[ii,kk+1] = AA[ii,ii]*XX[ii,kk]

        # Update the state with neighbors' contribution
        for jj in Nii:
            XX[ii,kk+1] += AA[ii,jj]*XX[jj,kk]


################################################
# Drawings

avg = np.mean(XX_init, axis=0)
print(avg)

plt.figure()

# Local Estimates
label = []
for ii in range(NN):
    plt.plot(np.arange(max_iters), XX[ii])
    label.append(f'$x_{ii}$')
plt.legend(label, loc='lower right')

# Average Constant Line
plt.plot(np.arange(max_iters), np.repeat(avg,max_iters), '--', linewidth=3, color='gold')

plt.xlim(0,max_iters)
plt.xlabel("iterations $k$")
plt.ylabel("$x_i^{k+1}$")
plt.title("Evolution of the local estimates")

plt.grid()
plt.show()
