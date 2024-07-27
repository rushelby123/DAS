import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class Graph:
    """    
    Number of agents:
        NN
    type of graph:
        STAR = 1
        BINOMIAL = 2
        CYCLE = 3
        PATH = 4
    """
    def __init__(self, type, Nagents):
        self.NN = Nagents
        self.type = type
        match self.type:
            case 1:
                self.pathG = nx.star_graph(self.NN-1)
            case 2:
                self.pathG =  nx.binomial_graph(self.NN, p=0.5)
            case 3:
                self.pathG =  nx.cycle_graph(self.NN)
            case 4:
                self.pathG =  nx.path_graph(self.NN)
            case _:
                self.pathG = "Invalid input"
        self.Adj = nx.adjacency_matrix(self.pathG).toarray()

    def is_connected(self):
        """check if graph is connected"""
        return nx.is_connected(self.pathG)

def gradient_tracking(function, initial_conditions, alpha, termination_condition, MAXITERS, AA, Adj):
    (NN, dd) = initial_conditions.shape
    ZZ_gt = np.zeros((MAXITERS, NN, dd))
    SS_gt = np.zeros((MAXITERS, NN, dd))
    cost_gt = np.zeros((MAXITERS))

    # Initialization of the algorithm
    for ii in range(NN):
        # Initialize the initial point for each agent
        ZZ_gt[0, ii, :] = initial_conditions[ii]
        # Initialize the gradient for each agent
        _, SS_gt[0, ii, :] = function(ZZ_gt[0, ii, :])

    # Gradient tracking method
    final_iter = MAXITERS
    for kk in range(MAXITERS - 1):
        if kk % 100 == 0:
            print(f"iteration {kk}")
        for ii in range(NN):
            N_ii = np.nonzero(Adj[ii])[0]
            ZZ_gt[kk + 1, ii, :] += AA[ii, ii] * ZZ_gt[kk, ii, :]
            SS_gt[kk + 1, ii, :] += AA[ii, ii] * SS_gt[kk, ii, :]
            for jj in N_ii:
                ZZ_gt[kk + 1, ii, :] += AA[ii, jj] * ZZ_gt[kk, jj, :]
                SS_gt[kk + 1, ii, :] += AA[ii, jj] * SS_gt[kk, jj, :]
            ZZ_gt[kk + 1, ii, :] -= alpha * SS_gt[kk, ii, :]  # / (1 + kk ) #for diminishing step size
            _, grad_ell_ii_new = function(ZZ_gt[kk + 1, ii, :])
            ell_ii_gt, grad_ell_ii_old = function(ZZ_gt[kk, ii, :])
            SS_gt[kk + 1, ii, :] += grad_ell_ii_new - grad_ell_ii_old
            cost_gt[kk] += ell_ii_gt
        # Termination condition (if the norm of the estimated gradient for each agent is less than a threshold)
        if all( np.linalg.norm(SS_gt[kk + 1, i, :]) < termination_condition for i in range(NN)):
            final_iter = kk
            break

    return ZZ_gt[:final_iter, :, :], SS_gt[:final_iter, :, :], cost_gt[:final_iter]

def gradient_tracking_different_costs(function, initial_conditions, alpha, termination_condition, MAXITERS, AA, Adj):
    (NN, dd) = initial_conditions.shape
    ZZ_gt = np.zeros((MAXITERS, NN, dd))
    SS_gt = np.zeros((MAXITERS, NN, dd))
    cost_gt = np.zeros((MAXITERS))

    # Initialization of the algorithm
    for ii in range(NN):
        # Initialize the initial point for each agent
        ZZ_gt[0, ii, :] = initial_conditions[ii]
        
        # Initialize the gradient for each agent
        ZZ_ext = np.append(ZZ_gt[0, ii, :],ii)
        _, SS_gt[0, ii, :] = function(ZZ_ext)

    # Gradient tracking method
    final_iter = MAXITERS
    for kk in range(MAXITERS - 1):
        if kk % 100 == 0:
            print(f"iteration {kk}")
        for ii in range(NN):
            N_ii = np.nonzero(Adj[ii])[0]
            ZZ_gt[kk + 1, ii, :] += AA[ii, ii] * ZZ_gt[kk, ii, :]
            SS_gt[kk + 1, ii, :] += AA[ii, ii] * SS_gt[kk, ii, :]
            for jj in N_ii:
                ZZ_gt[kk + 1, ii, :] += AA[ii, jj] * ZZ_gt[kk, jj, :]
                SS_gt[kk + 1, ii, :] += AA[ii, jj] * SS_gt[kk, jj, :]
            ZZ_gt[kk + 1, ii, :] -= alpha * SS_gt[kk, ii, :]  # / (1 + kk ) #for diminishing step size
            ZZ_ext_kplus1 = np.append(ZZ_gt[kk + 1, ii, :],ii)
            ZZ_ext_k = np.append(ZZ_gt[kk, ii, :],ii)
            _, grad_ell_ii_new = function(ZZ_ext_kplus1)
            ell_ii_gt, grad_ell_ii_old = function(ZZ_ext_k)
            SS_gt[kk + 1, ii, :] += grad_ell_ii_new - grad_ell_ii_old
            cost_gt[kk] += ell_ii_gt
        # Termination condition (if the norm of the estimated gradient for each agent is less than a threshold)
        if all( np.linalg.norm(SS_gt[kk + 1, i, :]) < termination_condition for i in range(NN)):
            final_iter = kk
            break

    return ZZ_gt[:final_iter, :, :], SS_gt[:final_iter, :, :], cost_gt[:final_iter]

def plot_ellipse(a, b, c, d, e, label=None ,alpha=1.0):
    # Calculate the coefficients for the standard form
    h = -c / (2 * a)
    k = -d / (2 * b)
    num = e - a * h**2 - b * k**2

    # Calculate the semi-major and semi-minor axes
    semi_major_axis = np.sqrt(-num / a)
    semi_minor_axis = np.sqrt(-num / b)

    # Generate angles from 0 to 2*pi
    theta = np.linspace(0, 2*np.pi, 100)

    # Parametric equations of ellipse
    x = h + semi_major_axis * np.cos(theta)
    y = k + semi_minor_axis * np.sin(theta)

    # Plot the ellipse
    plt.plot(x, y,label=label,alpha=alpha)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    #plt.axis('equal')
    plt.grid(True)



def real_gradient(function, ZZ):
    '''compute the gradient as for centralized optimization'''
    (MAXITERS, NN, dd) = ZZ.shape
    gradient = np.zeros((MAXITERS, dd))
    norm_gradient = np.zeros((MAXITERS))

    for kk in range(MAXITERS - 1):
        for ii in range(NN):
            gradient[kk,:] += function(ZZ[kk, ii, :])[1]
        norm_gradient[kk] = np.linalg.norm(gradient[kk,:])/NN
    return gradient, norm_gradient