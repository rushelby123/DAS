import numpy as np
import matplotlib.pyplot as plt
import DASlibrary as alg

# Task 1.2 [Centralized Classification] Centralized Gradient method to minimize a Logistic Regression function

MM = 200 # DATA SET SIZE
MAXITERS = 5000 # maximum number of iterations
ADD_OUTLIERS = False
alpha = 1e-2 # step size

if ADD_OUTLIERS:
    PROBABILIY_OUTLIER = 0.1
    termination_condition = 1e-6
else:
    termination_condition = 1e-1


# ellipse parameters
a = 2
b = 3
c = 0
d = -3
e = -1

# param

x0 = -c / (2 * a)
y0 = -d / (2 * b)
num = e - a * x0**2 - b * y0**2
semi_major_axis = np.sqrt(-num / a)
semi_minor_axis = np.sqrt(-num / b)

Space = 1.5*max(semi_major_axis, semi_minor_axis)

# Generate random values in the x-y plane
x = np.random.uniform(-Space+x0, Space+x0,MM)
y = np.random.uniform(-Space+y0, Space+y0,MM)

#inside_ellipse = (a*x**2 + b*y**2 + c*x + d*y + e <= 0)

# Create the dataset
data = []
for i in range(MM):
    if ADD_OUTLIERS:
        if (a*x[i]**2 + b*y[i]**2 + c*x[i] + d*y[i] + e <= 0):
            if np.random.uniform(0,1) < PROBABILIY_OUTLIER:
                data.append([x[i], y[i], -1])
            else:
                data.append([x[i], y[i], 1]) 
        else: 
            if np.random.uniform(0,1) < PROBABILIY_OUTLIER:
                data.append([x[i], y[i], 1])
            else:
                data.append([x[i], y[i], -1]) 
            pass
    
    else:
        if (a*x[i]**2 + b*y[i]**2 + c*x[i] + d*y[i] + e <= 0):
            data.append([x[i], y[i], 1]) 
        else: 
            data.append([x[i], y[i], -1])
            pass

'''
phi = np.array([x[i], y[i]])
w = np.array([a,b,c,d])
bias = e
'''

# Create the optimization problem
Z_opt = [a,b,c,d,e]
ZZ = np.zeros((MAXITERS, len(Z_opt)))

# Define the logistic function
def logistic(ZZ):
    cost = 0
    gradient_of_cost = np.zeros(ZZ.shape)
    w = np.array([ZZ[0],ZZ[1],ZZ[2],ZZ[3]]) #x and y + x and y squared
    bb = ZZ[4] #bias
    for ii in range (MM):
        x = data[ii][0]
        y = data[ii][1]
        pp = data[ii][2]
        phi = np.array([x**2,y**2,x,y])
        cost += np.log(1 + np.exp(-pp * (w.T @ phi + bb)))
        gradient_of_cost[:4] += (-pp * phi) * (1/(1 + np.exp(-pp * (w.T @ phi + bb))))*np.exp(-pp * (w.T @ phi + bb))
        gradient_of_cost[4] += -pp * (1/(1 + np.exp(-pp * (w.T @ phi + bb))))*np.exp(-pp * (w.T @ phi + bb))
    return cost,gradient_of_cost

'''
for i in range(MM):
    if data[i][2] == 1:
        plt.scatter(data[i][0], data[i][1], color='blue',alpha=0.5, linewidths=0.1)
    else:
        plt.scatter(data[i][0], data[i][1], color='red',alpha=0.5, linewidths=0.1)
plt_el(a,b,c,d,e)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Random Data')
plt.show()
'''


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

# compute the missclassified points
missclassified = 0
w = ZZ[-1,0,:-1] #x and y + x and y squared
bb = ZZ[-1,0,ZZ.shape[2]-1] #bias
for ii in range(MM):
    x = data[ii][0]
    y = data[ii][1]
    pp = data[ii][2]
    phi = np.array([x**2,y**2,x,y])
    if (-pp * (w.T @ phi + bb) > 0):
        missclassified += 1

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

# plot the ellipse
alg.plot_ellipse(a,b,c,d,e,  fr'Optimal solution')# ${a}^2+{b}^2+{c}*x+{d}*y+{e}=0$')
a1,b1,c1,d1,e1 = a*ZZ[-1][0] / ZZ[-1][0][0]
alg.plot_ellipse(a1,b1,c1,d1,e1, fr'Estimated Solution')# ${a1:.1f}^2+{b1:.1f}^2+{c1:.1f}*x+{d1:.1f}*y+{e1:.1f}=0$')
plt.legend()
plt.grid()
plt.show()

# Normalize the computed solution
z_gt = a*ZZ[-1][0] / ZZ[-1][0][0]

print(f"The last iteration was {ZZ.shape[0]}-th iteration")
print(f"The optimal solution is {Z_opt}")
print(f"The computed solution is {ZZ[-1]}")
print(f'The normalized computed solution is {z_gt}')
print(f'Missclassified points: {missclassified} out of {MM} points')
print(f'Percentage of missclassified points: {missclassified/MM*100}%')
print(f'Eficency of the algorithm: {1-missclassified/MM}')