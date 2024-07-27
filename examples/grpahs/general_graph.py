import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

#draw a path graph with 10 agents

N=10
pathG = nx.path_graph(N)
#pathG = nx.cycle_Graph(N)
#pathG = nx.star_graph(N)
pathG = nx.binomial_graph(N, p=0.5)#p probability to have an edge between two nodes

#show the graph
nx.draw(pathG, with_labels=True)
plt.show()

#show the adjacency matrix
Adj = nx.adjacency_matrix(pathG).toarray()
print(Adj)

#check if the graph is connected
#print(f"The graph is connected: {nx.is_connected(pathG)}")
print(np.linalg.matrix_power(Adj, N))#if the graph is connected, the matrix power of the adjacency matrix should have all non-zero elements

