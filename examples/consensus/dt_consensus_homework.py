#
# Discrete-time Average Consensus
# Ivano Notarnicola, Lorenzo Pichierri
# Bologna, 05/03/2024
#

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

max_iters = 200    # time horizon
NN = 10    # number of agents

I_NN = np.identity(NN, dtype=int)
ONES = np.ones((NN,NN))
ZEROS = np.zeros((NN,NN))

# IMPORTANT NOTE:
# The topology of the graph is important for the convergence of the consensus algorithm
# defining the radius of the graph as the maximum distance between any two nodes in the worst case
# the smaller the radius, the faster the convergence!!!!!!!!
# EXAMPLE:
# in path graph the radius is N-1 and in 100 iterration we don't reach consensus
# in star graph the radius is 2 and in 100 iterration we reach consensus, the center of the star is the faster to converge

# Generate a connected undirected graph
G = nx.path_graph(NN)
# G = nx.cycle_graph(NN)
# G = nx.star_graph(NN-1)
# G = nx.binomial_graph(NN,0.5)

# Generate the adjacency matrix
Adj = nx.adjacency_matrix(G)
Adj = Adj.toarray()

# Generate a weighted adjacency matrix
AA = 1.5*I_NN + 0.5*Adj

# Make the weighted adjacency matrix row/doubly-stochastic
# IMPORTANT NOTE:
# The weighted adjacency matrix must be row-stochastic for the consensus algorithm to converge

# Row-stochasticity
while any(abs(np.sum(AA,axis=1)-1) > 1e-10):
    AA = AA/(AA@ONES) # -> Row-stochasticity
    AA = np.maximum(AA,ZEROS)

# IMPORTANT NOTE:
# if the graph is doubly stochastic, the agents converge to avarage consensus 

## Doubly-stochasticity
# while any(abs(np.sum(AA,axis=1)-1) > 1e-10) or any(abs(np.sum(AA,axis=0)-1) > 1e-10):
#     AA = AA/(AA@ONES) # -> Row-stochasticity
#     AA = AA/(ONES@AA) # -> Col-stochasticity
#     AA = np.maximum(AA,ZEROS)


################################################
## CONSENSUS Algorithm
 
XX = np.zeros((NN,max_iters))
XX_init = 10*np.random.rand(NN)
XX[:,0] = XX_init 

for kk in range (max_iters-1):

	for ii in range (NN):

		# Find the set of neighbors
		Nii = list(G.neighbors(ii)) 
		#alternitavely, you can use the adjacency matrix
		#Nii = np.nonzero(Adj[ii,:])[0] #the [0] is to extract the list of row from the tuple, 1 is the column

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
