import numpy as np
import matplotlib.pyplot as plt
import DASlibrary as alg

np.random.seed(0)
# Task 1.2 [Centralized Classification] Centralized Gradient method to minimize a Logistic Regression function

MM = 50 # DATA SET SIZE
MAXITERS = 100000 # maximum number of iterations
alpha = 1e-2 # step size
termination_condition = 1e-1

# parabola parameters
aa = 1
bb = 0
cc = 1

# param
Space = 10

# Generate random values in the x-y plane
x_vertex = - bb/(2*aa)
y_vertex = cc - bb**2/(4*aa)
x = np.random.uniform(-Space+x_vertex, Space+x_vertex,MM)
y = np.random.uniform(-Space+y_vertex, Space+y_vertex,MM)

# Create the dataset
data = []
for i in range(MM):
    phi = np.array([x[i], y[i], x[i]**2])
    w = np.array([bb, -1, aa])
    b = cc
    if w.T @ phi + b >= 0:
    #if y[i] - m * x[i] - c >= 0: 
        data.append([x[i], y[i], 1]) 
    else: 
        data.append([x[i], y[i], -1])
        pass

# Create the optimization problem
        #x   y   x**2  bias
Z_opt = [bb, -1, aa, cc]
ZZ = np.zeros((MAXITERS, len(Z_opt)))

# Define the logistic function
def logistic(ZZ):
    cost = 0
    Z_dim = np.shape(ZZ) [0]
    gradient_of_cost = np.zeros(Z_dim)
    w = np.array([ZZ[0], ZZ[1], ZZ[2]]) #x, y, x**2
    bb = ZZ[3] #bias
    for ii in range (MM):
        x = data[ii][0]
        y = data[ii][1]
        pp = data[ii][2]
        phi = np.array([x, y, x**2])
        cost += np.log(1 + np.exp(-pp * (w.T @ phi + bb)))
        gradient_of_cost[:Z_dim-1] += (-pp * phi) * (1/(1 + np.exp(-pp * (w.T @ phi + bb))))*np.exp(-pp * (w.T @ phi + bb))
        gradient_of_cost[Z_dim-1] += -pp * (1/(1 + np.exp(-pp * (w.T @ phi + bb))))*np.exp(-pp * (w.T @ phi + bb))
    return cost,gradient_of_cost

# Gradient tracking method
Z_init =  np.random.uniform(-Space, Space, (1,len(Z_opt))) # 1 because there is only one agent
AA = np.ones((1, 1))
Adj = np.zeros((1, 1))
ZZ,SS_gt,cost_gt=alg.gradient_tracking(logistic, Z_init, alpha, termination_condition, MAXITERS, AA, Adj)
final_iter = ZZ.shape[0]

# compute the norm of the gradient
NN=1 # Centralized setting
norm_grad = np.zeros((final_iter, NN))
for ii in range(NN):
    for kk in range(final_iter - 1):
        norm_grad[kk, ii] = np.linalg.norm(SS_gt[kk,ii,:])

#plots
fig, ax = plt.subplots(2)
ax[0].set_title(f'norm of gradient')
ax[0].semilogy(np.arange(final_iter - 1), norm_grad[:final_iter-1, 0], label= 'Norm of gradient')
ax[0].grid()
ax[0].set_ylabel(r'$|\nabla \ell(z^{k})|$',fontsize=12)
ax[1].set_title('logistic regression cost')
ax[1].plot(np.arange(final_iter - 1), cost_gt[:final_iter-1])
ax[1].grid()
ax[1].set_ylabel(r'$\ell(z^{k})$',fontsize=12)
ax[1].set_xlabel(r'iterations $k$')
plt.subplots_adjust(hspace=0.2) 
plt.tight_layout()
plt.legend()
plt.show()

# plot the data and the estimated line
for i in range(MM):
    if data[i][2] == 1:
        plt.scatter(data[i][0], data[i][1], color='blue',alpha=0.5, linewidths=0.1)
    else:
        plt.scatter(data[i][0], data[i][1], color='red',alpha=0.5, linewidths=0.1)
x_opt = np.linspace(-Space + x_vertex, Space + x_vertex, 10000)
y_opt = aa * x_opt**2 + bb * x_opt + cc
z_gt = ZZ[-1,:]
z_gt = -z_gt/z_gt[0,1]
x_gt = np.linspace(-Space + x_vertex, Space + x_vertex, 10000)
y_gt = z_gt[0, 2] * x_gt**2 + z_gt[0, 0] * x_gt + z_gt[0,3]
plt.plot(x_opt, y_opt, 'green',label='Optimal line')
plt.plot(x_gt, y_gt, 'black',label='Estimated line')
plt.axis('equal')
plt.legend()
plt.grid()
plt.show()

print(f"The last iteration was {ZZ.shape[0]}-th iteration")
print(f"The optimal solution is {Z_opt}")
print(f"The computed solution is {ZZ[-1]}")
print(f'The normalized computed solution is {z_gt}')
