import numpy as np
import matplotlib.pyplot as plt
import DASlibrary as alg
import networkx as nx

# Task 1.2 [Centralized Classification] Centralized Gradient method to minimize a Logistic Regression function

MM = 100 # DATA SET SIZE 
NN = 5 # Number of agents
MAXITERS = 500 # maximum number of iterations
alpha = 1e-2
termination_condition = 1e-1
type_of_graph = 1 #STAR = 1 BINOMIAL = 2 CYCLE = 3 PATH = 4

# generate a network
I_NN = np.eye(NN)
while 1:
    graph = alg.Graph(type_of_graph,NN) 
    if graph.is_connected:
        break
Adj = graph.Adj

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

# ellipse parameters
a = 2
b = 3
c = 0
d = 0
e = -1

# Generate random values in the x-y plane
x0 = -c / (2 * a)
y0 = -d / (2 * b)
num = e - a * x0**2 - b * y0**2
semi_major_axis = np.sqrt(-num / a)
semi_minor_axis = np.sqrt(-num / b)
Space = 1.5*max(semi_major_axis, semi_minor_axis)
x = np.random.uniform(-Space+x0, Space+x0,MM*NN)
y = np.random.uniform(-Space+y0, Space+y0,MM*NN)

# Create the dataset
data = []
for i in range(MM):
    agent_index = np.random.randint(0,NN)
    if (a*x[i]**2 + b*y[i]**2 + c*x[i] + d*y[i] + e <= 0):
        data.append([x[i], y[i],  1, agent_index]) 
    else: 
        data.append([x[i], y[i], -1, agent_index])
        pass

# Create the optimization problem
Z_opt = np.array([a,b,c,d,e])
opt_cost = 0 
ZZ = np.zeros((MAXITERS, len(Z_opt)))

# Define the logistic function
def logistic(ZZ):
    ''''''
    agent_index = ZZ[-1]
    ZZ = ZZ[:-1]
    cost = 0
    gradient_of_cost = np.zeros(ZZ.shape)
    w = np.array([ZZ[0],ZZ[1],ZZ[2],ZZ[3]]) #x y, x**2 and y**2 squared
    bb = ZZ[4] #bias
    for ii in range (MM):
        if(agent_index==data[ii][-1]):
            x = data[ii][0]
            y = data[ii][1]
            pp = data[ii][2]
            phi = np.array([x**2,y**2,x,y])
            cost += np.log(1 + np.exp(-pp * (w.T @ phi + bb)))
            gradient_of_cost[:4] += (-pp * phi) * (1/(1 + np.exp(-pp * (w.T @ phi + bb))))*np.exp(-pp * (w.T @ phi + bb))
            gradient_of_cost[4] += -pp * (1/(1 + np.exp(-pp * (w.T @ phi + bb))))*np.exp(-pp * (w.T @ phi + bb))
    return cost,gradient_of_cost

# Gradient tracking method
Z_init =  np.random.uniform(-Space, Space, (NN,len(Z_opt)))
ZZ,SS_gt,cost_gt=alg.gradient_tracking_different_costs(logistic, Z_init, alpha, termination_condition, MAXITERS, AA, Adj)
final_iter = ZZ.shape[0]

# compute the norm of the estimated gradients
norm_grad = np.zeros((final_iter, NN))
for ii in range(NN):
    for kk in range(final_iter - 1):
        norm_grad[kk, ii] = np.linalg.norm(SS_gt[kk,ii,:])

# compute the error between z optimal and the computed z from gradient tracking
error = np.zeros(final_iter)
for ii in range(NN):
    z_gt = e/ ZZ[-1][ii][-1] * ZZ #normalization
    for kk in range(final_iter - 1):
        error[kk] += np.linalg.norm(z_gt[kk,ii,:] - Z_opt[:])

# compute the mean of the norm of the estimated gradients
SS_distance_mean = np.zeros(final_iter)
SS_mean_norm = np.zeros(final_iter)
for kk in range(final_iter - 1):
    SS_mean_norm[kk] = np.mean([norm_grad[kk,ii] for ii in range(NN)])
    for ii in range(NN):
        SS_distance_mean[kk] += abs(norm_grad[kk,ii]-SS_mean_norm[kk])

# compute the real gradient 
real_grad = np.zeros((final_iter, len(Z_opt)))
real_grad_norm = np.zeros(final_iter)
for kk in range(final_iter - 1):
    for ii in range(NN):
        _,grad = logistic(np.append(ZZ[kk,ii,:],ii))
        real_grad[kk,:] += grad
    real_grad_norm[kk] = np.linalg.norm(real_grad[kk,:])/NN

#compute the distance between the mean norm of the esimate and the global gradient 
mean_glob_distance = np.zeros(final_iter)
for kk in range(final_iter - 1):
    mean_glob_distance[kk] = abs(SS_mean_norm[kk] - real_grad_norm[kk])

# print datas
for i in range(NN):
    count=0
    for j in range(MM):
        if data[j][3] == i:
            count+=1
    print(f'the number of points assigned to agent {i} is {count}')
print(f"The last iteration was {ZZ.shape[0]}-th iteration")
print(f"The optimal solution is {Z_opt}")
for ii in range(NN):
    z_gt = e*ZZ[-1][ii] / ZZ[-1][ii][-1]
    print(f'agent {i} normalized estimated is {z_gt}')

# plot the graph
nx.draw(graph.pathG, with_labels=True)
plt.show()

# plot the gradients norm and the costs
fig, ax = plt.subplots(2)
ax[0].set_title(f'norm of gradient')
ax[0].semilogy(np.arange(final_iter - 1), norm_grad[:final_iter-1, 0], label= 'Norm of gradient')
ax[0].grid()
ax[0].set_ylabel(r'$|\sum_{i=1}^{N} \nabla \ell(z_{i}^{k})|$',fontsize=12)
ax[1].set_title('logistic regression cost')
ax[1].plot(np.arange(final_iter - 1), cost_gt[:final_iter-1])
ax[1].grid()
ax[1].set_ylabel(r'$\sum_{i=1}^{N} \ell(z_{i}^{k})$',fontsize=12)
ax[1].set_xlabel(r'iterations $k$')
plt.subplots_adjust(hspace=0.2) 
plt.legend()
plt.show()

# plot the dataset
markers = ['o', 's', '*', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 'p', 'P', 'h', 'H', '+', 'x', 'X', 'D', 'd', '|', '_']
for i in range(MM):
    if data[i][2] == 1:
        plt.scatter(data[i][0], data[i][1], color='blue',alpha=0.5, linewidths=0.1,marker=markers[data[i][-1] % len(markers)])
    else:
        plt.scatter(data[i][0], data[i][1], color='red',alpha=0.5, linewidths=0.1,marker=markers[data[i][-1] % len(markers)])

# plot the ellipses

final_iter = ZZ.shape[0]
ten_perc_index = int(final_iter/10)
fifty_perc_index = int(final_iter/2)
ninety_perc_index = int(final_iter*9/10)
alg.plot_ellipse(a,b,c,d,e, 'Optimal Solution')
a1,b1,c1,d1,e1 = ZZ[-1][0]
for i in range(NN):
    alg.plot_ellipse(ZZ[ten_perc_index][i,0],ZZ[ten_perc_index][i,1],ZZ[ten_perc_index][i,2],ZZ[ten_perc_index][i,3],ZZ[ten_perc_index][i,4], label=f'Agent {i} estimated')
plt.axis('equal')
plt.legend()
plt.title('10% of the iterations')
plt.grid()
plt.show()

# plot the dataset
markers = ['o', 's', '*', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 'p', 'P', 'h', 'H', '+', 'x', 'X', 'D', 'd', '|', '_']
for i in range(MM):
    if data[i][2] == 1:
        plt.scatter(data[i][0], data[i][1], color='blue',alpha=0.5, linewidths=0.1,marker=markers[data[i][-1] % len(markers)])
    else:
        plt.scatter(data[i][0], data[i][1], color='red',alpha=0.5, linewidths=0.1,marker=markers[data[i][-1] % len(markers)])

# plot the ellipse
alg.plot_ellipse(a,b,c,d,e, 'Optimal Solution')
a1,b1,c1,d1,e1 = ZZ[-1][0]
for i in range(NN):
    alg.plot_ellipse(ZZ[fifty_perc_index][i,0],ZZ[fifty_perc_index][i,1],ZZ[fifty_perc_index][i,2],ZZ[fifty_perc_index][i,3],ZZ[fifty_perc_index][i,4], label=f'Agent {i} estimated')
plt.axis('equal')
plt.legend()
plt.title('50% of the iterations')
plt.grid()
plt.show()

# plot the dataset
markers = ['o', 's', '*', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 'p', 'P', 'h', 'H', '+', 'x', 'X', 'D', 'd', '|', '_']
for i in range(MM):
    if data[i][2] == 1:
        plt.scatter(data[i][0], data[i][1], color='blue',alpha=0.5, linewidths=0.1,marker=markers[data[i][-1] % len(markers)])
    else:
        plt.scatter(data[i][0], data[i][1], color='red',alpha=0.5, linewidths=0.1,marker=markers[data[i][-1] % len(markers)])

# plot the ellipse
alg.plot_ellipse(a,b,c,d,e, 'Optimal Solution')
a1,b1,c1,d1,e1 = ZZ[-1][0]
for i in range(NN):
    alg.plot_ellipse(ZZ[ninety_perc_index][i,0],ZZ[ninety_perc_index][i,1],ZZ[ninety_perc_index][i,2],ZZ[ninety_perc_index][i,3],ZZ[ninety_perc_index][i,4], label=f'Agent {i} estimated')
plt.axis('equal')
plt.legend()
plt.title('90% of the iterations')
plt.grid()
plt.show()

# plot the dataset
markers = ['o', 's', '*', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 'p', 'P', 'h', 'H', '+', 'x', 'X', 'D', 'd', '|', '_']
for i in range(MM):
    if data[i][2] == 1:
        plt.scatter(data[i][0], data[i][1], color='blue',alpha=0.5, linewidths=0.1,marker=markers[data[i][-1] % len(markers)])
    else:
        plt.scatter(data[i][0], data[i][1], color='red',alpha=0.5, linewidths=0.1,marker=markers[data[i][-1] % len(markers)])

# plot the ellipse
alg.plot_ellipse(a,b,c,d,e, 'Optimal Solution')
a1,b1,c1,d1,e1 = ZZ[-1][0]
for i in range(NN):
    alg.plot_ellipse(ZZ[-1][i,0],ZZ[-1][i,1],ZZ[-1][i,2],ZZ[-1][i,3],ZZ[-1][i,4], label=f'Agent {i} estimated solution')
plt.axis('equal')
plt.legend()
plt.grid()
plt.title('100% of the iterations')
plt.show()

#plot the evolution of the agents
# dimZ=ZZ.shape[2]
# fig, ax = plt.subplots(dimZ)
# for d in range (dimZ):
#     ax[d].set_title(f'evolution of z[{d}] along the iterations')
#     for i in range (NN):
#         z_gt = e/ ZZ[-1][ii][-1] * ZZ #normalization
#         ax[d].plot(np.arange(final_iter), z_gt[:final_iter, i,d], label= f'Agent {i}', color = 'C' + str(i))
#     ax[d].axhline(y = Z_opt[d], linestyle='--', label = 'optimal solution', color = 'r')
#     ax[d].grid()
#     #ax[d].legend()
# plt.tight_layout()
# plt.legend()
# plt.show()

#plot the evolution of the estimated gradient and real one
# fig, ax = plt.subplots(3,sharex=True,figsize=(10, 12))
# ax[0].set_title(f'evolution of the norm of the estimated gradient SS_i along the iterations for each agent')
# for i in range (NN):
#     ax[0].semilogy(np.arange(final_iter - 1), norm_grad[:final_iter-1, i], label=r'Agent estimate $|s_{{{}}}^{{k}}|$'.format(i), color = 'C' + str(i))
# ax[0].semilogy(np.arange(final_iter - 1), SS_mean_norm[:final_iter-1], label=r'Mean gradient estimate $|\nabla \tilde{\ell}^{k}|$', color = 'red', linestyle = '-.')
# ax[0].semilogy(np.arange(final_iter - 1), real_grad_norm[:final_iter-1], label= r'Global gradient $|\nabla \ell^{*k}|$', color = 'green', linestyle = ':')
# ax[0].grid()
# ax[0].legend(fontsize=8, loc='upper right')
# ax[1].set_title('overall distance between the estimates of the gradient and the mean')
# ax[1].semilogy(np.arange(final_iter - 1), SS_distance_mean[:final_iter-1])
# ax[1].set_ylabel(r'$\sum_{i=1}^{N} \left| |\nabla \ell(z_{i}^{k})| - |\nabla \tilde{\ell}^{k}| \right|$', fontsize=12)
# ax[1].grid()
# ax[2].set_title('distance between the mean of the estimates and the global gradient')
# ax[2].semilogy(np.arange(final_iter - 1), mean_glob_distance[:final_iter-1])
# ax[2].set_ylabel(r'$||\nabla \tilde{\ell}^{k}| - |\nabla \ell^{*k}||$', fontsize=12)
# ax[2].grid()
# plt.tight_layout()
# plt.legend()
# plt.subplots_adjust(hspace=0.2) 
# plt.show()

#plot the estimation error
fig, ax = plt.subplots(2)
ax[0].semilogy(np.arange(final_iter - 1), np.abs(cost_gt[:final_iter - 1] - opt_cost) )
ax[0].grid()
ax[0].set_title('Error between the optimal cost and the computed cost')
ax[1].semilogy(np.arange(final_iter - 1), error[:final_iter - 1]  )
ax[1].grid()
ax[1].set_title('Error between the optimal z and the computed z')
plt.tight_layout()
plt.show()




