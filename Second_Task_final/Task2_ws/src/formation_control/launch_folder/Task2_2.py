from launch import LaunchDescription
from launch_ros.actions import Node
import networkx as nx
import numpy as np
from launch.actions import ExecuteProcess

# seed randomness
np.random.seed(0)

#CONSTANTS
MAXITERS = 100
NN = 10
DIM = 2

#ALGORITHM PARAMETERS 
gamma = 10
alpha = 1e-2

#VISUALIZATION PARAMETERS
map_dimension = 5

#ALGORITHM VARIABLES
ZZ_init = np.random.randint(-map_dimension,map_dimension,(NN*DIM)) # Initial decision variable
#ZZ_init = np.array([-5,-5,5,5,-5,5,5,-5])
RR_init = np.random.randint(-map_dimension,map_dimension,(NN*DIM)) # Initial adversary position
print(f"ZZ_init:{ZZ_init}")

#CREATE A GRAPH
I_NN = np.eye(NN)
while 1:
    G = nx.binomial_graph(NN, p=0.5)
    if nx.is_connected(G):
        break
Adj = nx.adjacency_matrix(G).toarray()

# Compute the set of weights for the network, using Metropolis-Hastings weights methood set of slides 3 (avaraging)
AA = np.zeros(shape=(NN, NN))
for ii in range(NN):
    # get the neighbors of the agent ii
    N_ii = np.nonzero(Adj[ii])[0]
    # compute the degree of the agent ii
    deg_ii = len(N_ii)
    # compute the weights of the agent ii
    for jj in N_ii:
        deg_jj = len(np.nonzero(Adj[jj])[0])
        AA[ii, jj] = 1 / (1 + max([deg_ii, deg_jj]))
AA += I_NN - np.diag(np.sum(AA, axis=0))
print(f"AA:{AA}")
# GENERATE NN NODES WITH DIFFERENT PARAMETERS
def generate_launch_description():
    node_list = []

    for ii in range(NN):
        N_ii = list(G.neighbors(ii))
        lower_index = ii*DIM
        upper_index = (ii+1)*DIM
        node_list.append(
            Node(
                package="formation_control",
                namespace=f"agent_{ii}",
                executable="generic_agent",
                parameters=[
                    {            
                        "AA": AA[ii,:].tolist(),     
                        "NN": NN,
                        "DIM": DIM,
                        "MAXITERS": MAXITERS,          
                        "gamma": gamma,
                        "alpha": alpha,
                        "id": ii,
                        "Nii": N_ii,
                        "ZZinit": ZZ_init[lower_index:upper_index].tolist(),
                        "RRinit": RR_init[lower_index:upper_index].tolist(),
                    }
                ],
                output="screen",
                prefix=f'xterm -title "agent_{ii}" -hold -e',
            )
        )

    node_list.append(
        ExecuteProcess(
                            cmd=['rviz2'],
                            output='screen',
                    ),
    )

    node_list.append(
        Node(
                package="formation_control",
                namespace=f"plotter",
                executable="plotter",
                parameters=[
                    {             
                        "NN": NN,
                        "DIM": DIM,
                        "MAXITERS": MAXITERS,    
                        "RRinit": RR_init.tolist(),
                        "gamma": gamma,     
                    }
                ],
                output="screen",
                prefix=f'xterm -title "plotter" -hold -e',
            ),
    )

    return LaunchDescription(node_list)
