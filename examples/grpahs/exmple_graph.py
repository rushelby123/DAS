import matplotlib.pyplot as plt
import networkx as nx

#create a graph
G = nx.Graph()

G.add_node(1)
G.add_nodes_from([2, 3])
G.add_edge(1, 2)
G.add_edges_from([(1, 3), (2, 3)])

nx.draw(G, with_labels=True, font_weight="bold")

plt.show()

#show the adjacency matrix
Adj = nx.adjacency_matrix(G).toarray()
print(Adj)

#compute the neighbors of a node
i=int(1)
Ni = list(nx.neighbors(G, i))
print(f"The neighbors of node {i} are {Ni}")

if 1:
    nx.draw_circular(G, with_labels=True)
    plt.show()


