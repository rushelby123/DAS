from launch import LaunchDescription
from launch_ros.actions import Node
import networkx as nx
import numpy as np
from launch.actions import ExecuteProcess

# seed randomness
np.random.seed(2)

#CONSTANTS
MAXITERS = 400
NN = 10
DIM = 2

#ALGORITHM PARAMETERS 
gamma = 20
alpha = 5e-3

#VISUALIZATION PARAMETERS
map_dimension = 5

#BARRIER FUNCTION PARAMETERS x^2*a - y^2*b - 1 <= 0
a = 5.0
b = 1.0
epsilon = 2

#ALGORITHM VARIABLES
x_ZZ_init = np.random.randint(-10,10,(NN))/10.0 # Initial decision variable [x1, x2, x3]
y_ZZ_init = np.random.randint(30,50,(NN))/10.0 # Initial decision variable  [y1, y2, y3]
x_RR_init = np.random.randint(-10,10,(NN))/10.0 # Initial adversary position
y_RR_init = np.random.randint(-50,-30,(NN))/10.0 # Initial adversary position
#[x1,y1,x2,y2,x3,y3]
ZZ_init = np.zeros((NN*DIM))
RR_init = np.zeros((NN*DIM))
for ii in range(NN*DIM):
    ZZ_init[ii] = x_ZZ_init[ii//2] if ii%2 == 0 else y_ZZ_init[ii//2]
    RR_init[ii] = x_RR_init[ii//2] if ii%2 == 0 else y_RR_init[ii//2]
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
                executable="agent_task2_3",
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
                        "a": a,
                        "b": b,
                        "epsilon": epsilon,
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
                executable="plotter_task2_3",
                parameters=[
                    {             
                        "NN": NN,
                        "DIM": DIM,
                        "MAXITERS": MAXITERS,    
                        "RRinit": RR_init.tolist(),
                        "gamma": gamma,   
                        "a": a,
                        "b": b,  
                        "epsilon": epsilon,
                    }
                ],
                output="screen",
                prefix=f'xterm -title "plotter" -hold -e',
            ),
    )

    return LaunchDescription(node_list)
