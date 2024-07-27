# Task 1 [Distributed Classification via Logistic Regression]
# Task 1.1 [Distibuted Optimization] Gradient Tracking with various graph patterns and Metropolis-Hasting weights
#This code provide an example of distributed optimization on a network. 
#The code generates a random network and then performs distributed 
#optimization on the network. 
#The two type of distributed optimization algorithm we are going to see are 
#distributed gradient descent method and distributed gradient tracking method

import matplotlib.pyplot as plt
import numpy as np
import DASlibrary as alg
import networkx as nx
np.random.seed(1)

MAXITERS = 100000 # number of iterations
NN = 5 # number of agents
dd = 2 # dimension of the states of the agents
alpha = 1e-3 # constant step size
termination_condition = 1e-4 # termination condition
type_of_graph = 2 #STAR = 1 BINOMIAL = 2 CYCLE = 3 PATH = 4

# generate a random network
I_NN = np.eye(NN)
while 1:
    graph = alg.Graph(type_of_graph,NN) 
    if graph.is_connected():
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

#DISTRIUBTED GRADIENT TRACKING METHOD (may we need to create a class or a function for this? what if we add a terminal condition?)
# Q is symmetric and positive definite matrix!! why is this important?
Q = np.zeros((dd,dd,NN))
R = np.zeros((dd,NN))
QQopt = np.zeros((dd,dd))
RRopt = np.zeros(dd)
for ii in range (NN):
    # R[:,ii] = np.random.uniform(low=0,high=1,size=(dd))
    # Q[:,:,ii] =np.random.uniform(low=0,high=1,size=(dd,dd))
    # Q[:,:,ii] = Q[:,:,ii] @ Q[:,:,ii].T
    Q[:,:,ii] = np.eye(dd)
    RRopt += R[:,ii]
    QQopt += Q[:,:,ii]
ZZ_init = np.zeros((NN,dd))

# QUADRATIC FUNCTION DEFINITION we will use this function as a test for our algorithm
def quadratic_fn(z):
    agent_index = int(z[-1])
    z = z[:-1]
    QQ=Q[:,:,agent_index]
    RR=R[:,agent_index]
    return 0.5 * z.T @ QQ @ z + RR.T @ z, QQ @ z + RR

# Initilize the state for each agent
for ii in range(NN):
    ZZ_init[ii,:] = np.random.uniform(low=-10,high=10,size=(dd))
      
# Gradient tracking method
ZZ_gt,SS_gt,cost_gt=alg.gradient_tracking_different_costs(quadratic_fn, ZZ_init, alpha, termination_condition, MAXITERS, AA, Adj)
final_iter = ZZ_gt.shape[0]
print(f"The last iteration was {final_iter}-th iteration")

# compute the optimal solution using closed form solution
ZZ_opt = - np.linalg.inv(QQopt) @ RRopt
ZZ_opt_ext = np.append(ZZ_opt,NN)
opt_cost = 0    
for ii in range(NN):
    opt_cost += quadratic_fn(np.append(ZZ_opt,ii))[0]

# compute the error between z optimal and the computed z from gradient tracking
error = np.zeros(final_iter)
for ii in range(NN):
    for kk in range(final_iter - 1):
        error[kk] += np.linalg.norm(ZZ_gt[kk,ii,:] - ZZ_opt[:])

# compute the norm of the estimated gradient
norm_grad = np.zeros((final_iter, NN))
for ii in range(NN):
    for kk in range(final_iter - 1):
        norm_grad[kk, ii] = np.linalg.norm(SS_gt[kk,ii,:])

# compute the global gradient 
gradient = np.zeros((final_iter, dd))
norm_gradient_global = np.zeros((final_iter))
ZZ_gt_ext = np.zeros((final_iter, NN, dd + 1))
for kk in range(final_iter - 1):
    for ii in range(NN):
        ZZ_gt_ext[kk,ii,:] = np.append(ZZ_gt[kk,ii,:],ii)
        gradient[kk,:] += quadratic_fn(ZZ_gt_ext[kk, ii, :])[1]
    norm_gradient_global[kk] = np.linalg.norm(gradient[kk,:])/NN

# compute the mean of the norm of the estimated gradients
SS_distance_mean = np.zeros(final_iter)
SS_mean_norm = np.zeros(final_iter)
for kk in range(final_iter - 1):
    SS_mean_norm[kk] = np.mean([norm_grad[kk,ii] for ii in range(NN)])
    for ii in range(NN):
        SS_distance_mean[kk] += abs(norm_grad[kk,ii]-SS_mean_norm[kk])

#compute the distance between the mean norm of the esimate and the global gradient 
mean_glob_distance = np.zeros(final_iter)
for kk in range(final_iter - 1):
    mean_glob_distance[kk] = abs(SS_mean_norm[kk] - norm_gradient_global[kk])

# show the graph
nx.draw(graph.pathG, with_labels=True)
print(graph.Adj)
plt.show()

# plots
fig, ax = plt.subplots(3, sharex=True, figsize=(10, 20)) 
ax[0].semilogy(np.arange(final_iter - 1), norm_gradient_global[:final_iter - 1]  )
ax[0].grid()
ax[0].set_title('Global gradient norm')
ax[0].set_ylabel(r'$|\sum_{i=1}^{N} \nabla \ell(z_{i}^{k})|$',fontsize=12)
ax[1].semilogy(np.arange(final_iter - 1), abs(cost_gt[:final_iter - 1] - opt_cost) )
# ax[1].plot(np.arange(final_iter - 1), cost_gt[:final_iter - 1])
# ax[1].axhline(y=opt_cost, linestyle='--', color='r')
ax[1].grid()
ax[1].set_title('Error between optimal cost and computed cost')
ax[1].set_ylabel(r'$|\ell(z^{k}) - \ell(z^{*})|$',fontsize=12)
ax[2].semilogy(np.arange(final_iter - 1), error[:final_iter - 1])
ax[2].grid()
ax[2].set_title('Error between optimal solution and computed solution')
ax[2].set_ylabel(r'$\sum_{i=1}^{N}|| z^{k}_{i} - z^{*}||$',fontsize=12)
ax[2].set_xlabel(r'iterations $k$')
plt.subplots_adjust(hspace=0.2) 
plt.show()

print ("matrix Q")
print (Q)
print ("affine term R")
print (R)
print ("adjacency matrix matrix")
print (Adj)
print ("weights matrix, non negative and symmetric matrix")
print (AA)
print ("check if the matrix AA is doubly stochastic, you should see 1s")
print(np.sum(AA, axis=0))
print(np.sum(AA, axis=1))
#plot the evolution of the agents
fig, ax = plt.subplots(dd)
if dd == 1:
    ax = [ax]
for d in range (dd):
    ax[d].set_title(f'evolution of z[{d}] along the iterations')
    for i in range (NN):
        ax[d].plot(np.arange(final_iter), ZZ_gt[:final_iter, i,d], label= f'Agent {i}', color = 'C' + str(i))
    ax[d].axhline(y = ZZ_opt[d], linestyle='--', label = 'optimal solution', color = 'r')
    ax[d].grid()
plt.tight_layout()
plt.legend()
plt.show()
#plot the evolution of the estimated gradient and real one
fig, ax = plt.subplots(3,sharex=True,figsize=(10, 12))
ax[0].set_title(f'Norm of the estimated gradient')
for i in range (NN):
    ax[0].loglog(np.arange(final_iter - 1), norm_grad[:final_iter-1, i], label=f'agent {i} $|s_{{{i}}}^{{k}}|$', color = 'C' + str(i),alpha = 0.4)
ax[0].loglog(np.arange(final_iter - 1), SS_mean_norm[:final_iter-1], label=r'Mean gradient', color = 'red', linestyle = '-.')
ax[0].loglog(np.arange(final_iter - 1), norm_gradient_global[:final_iter-1], label= r'Global gradient', color = 'green', linestyle = ':')
ax[0].grid()
ax[0].legend(fontsize=8, loc='upper right')
ax[1].set_title('Difference between the local estimates and the mean gradient')
ax[1].loglog(np.arange(final_iter - 1), SS_distance_mean[:final_iter-1])
#ax[1].set_ylabel(r'$\sum_{i=1}^{N} \left| |\nabla \ell(z_{i}^{k})| - |\nabla \tilde{\ell}^{k}| \right|$', fontsize=12)
ax[1].grid()
ax[2].set_title('Difference between the mean gradient and the global gradient')
ax[2].loglog(np.arange(final_iter - 1), mean_glob_distance[:final_iter-1])
#ax[2].set_ylabel(r'$||\nabla \tilde{\ell}^{k}| - |\nabla \ell^{*k}||$', fontsize=12)
ax[2].grid()
ax[2].set_xlabel(r'iterations $k$')
plt.legend()
plt.subplots_adjust(hspace=0.2) 
plt.show()


