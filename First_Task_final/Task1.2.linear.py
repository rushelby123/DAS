import numpy as np
import matplotlib.pyplot as plt
import DASlibrary as alg

# Task 1.2 [Centralized Classification] Centralized Gradient method to minimize a Logistic Regression function

MM = 50 # DATA SET SIZE
MAXITERS = 100000 # maximum number of iterations
alpha = 1e-2
termination_condition = 1e-1

# line parameters
c = 2
m = 4

# param
Space = 5

# Generate random values in the x-y plane
x = np.random.uniform(-Space, Space,MM)
y = np.random.uniform(-Space+c, Space+c,MM)

# Create the dataset
data = []
for i in range(MM):
    phi = np.array([x[i], y[i]])
    w = np.array([-m, 1])
    b = -c
    if w.T @ phi + b >= 0:
    #if y[i] - m * x[i] - c >= 0: 
        data.append([x[i], y[i], 1]) 
    else: 
        data.append([x[i], y[i], -1])
        pass

# Create the optimization problem
        #x   y  bias
Z_opt = [-m, 1, -c]
ZZ = np.zeros((MAXITERS, len(Z_opt)))

# Define the logistic function
def logistic(ZZ):
    cost = 0
    gradient_of_cost = np.zeros(ZZ.shape)
    w = np.array([ZZ[0],ZZ[1]]) #x and y
    bb = ZZ[2] #bias
    for ii in range (MM):
        x = data[ii][0]
        y = data[ii][1]
        pp = data[ii][2]
        phi = np.array([x,y])
        cost += np.log(1 + np.exp(-pp * (w.T @ phi + bb)))
        gradient_of_cost[:2] += (-pp * phi) * (1/(1 + np.exp(-pp * (w.T @ phi + bb))))*np.exp(-pp * (w.T @ phi + bb))
        gradient_of_cost[2] += -pp * (1/(1 + np.exp(-pp * (w.T @ phi + bb))))*np.exp(-pp * (w.T @ phi + bb))
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
x_opt = np.linspace(-Space , Space, 10000)
y_opt = m * x_opt + c
z_gt = ZZ[-1,:]
z_gt = z_gt/z_gt[0,1]
x_gt = np.linspace(-Space , Space, 10000)
y_gt = -z_gt[0, 0] * x_gt - z_gt[0, 2]
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
