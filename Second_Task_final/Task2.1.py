import numpy as np
import matplotlib.pyplot as plt

# seed randomness
np.random.seed(1)

#CONSTANTS
MOVING_INTRUDERS = False
RANDOM_MOVING = True
MAXITERS = 2000
NN = 4
DIM = 2
 
#ALGORITHM PARAMETERS
gamma = 10
alpha = 1e-3 # stepsize

#VISUALIZATION PARAMETERS
dt = 3 # time step
map_dimension = 5
markers = ['o', 's', '*', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 'p', 'P', 'h', 'H', '+', 'x', 'X', 'D', 'd', '|', '_']
movment_vel = 3e-3

#ALGORITHM VARIABLES
FF = np.zeros((MAXITERS)) #Cost vector f_k = sum(f_ik) -> F = [f_1, ..., f_maxiters].T
ZZ = np.zeros((NN*DIM,MAXITERS)) # Decision vector Z_k = [z_1, ..., z_N] ->  ZZ = [X_1, ..., X_maxiters]
RR = np.zeros((NN*DIM,MAXITERS)) # Adversary position
SS = np.zeros((NN*DIM,MAXITERS)) # Estimate of the aggregative variable sigma(x)
VV = np.zeros((NN*DIM,MAXITERS)) # Estimate of sum(nabla_2)
ZZ_init = np.random.randint(-map_dimension,map_dimension,(NN*DIM)) # Initial decision variable
RR_init = np.random.randint(-map_dimension,map_dimension,(NN*DIM)) # Initial adversary position

#INTRUDERS POSITION INIT
if not MOVING_INTRUDERS:
	RR[:] = RR_init[:, None]
else:
	if RANDOM_MOVING:
		DELTA = 1e-2
		RR[:] = RR_init[:, None]
		for kk in range(0,MAXITERS-1):
			RR[:,kk+1] = RR[:,kk]+DELTA*np.random.uniform(low=-1.0, high=1.0, size=NN*DIM)

	else:
		radius = map_dimension
		angle = 2*np.pi/NN
		for ii in range(NN):
			index = ii*DIM
			RR_init[index:(index+DIM)] = np.array([radius*np.cos(angle*ii),radius*np.sin(angle*ii)])
		RR[:] = RR_init[:,None]
		ind_move = int(0*MAXITERS)
		iter=0
		for kk in range(ind_move,MAXITERS):
			for ii in range(NN):
				index = ii*DIM
				psi = movment_vel*iter
				iter+=1
				RR[index:(index+DIM),kk] = np.array([radius*np.cos(angle*ii+psi),radius*np.sin(angle*ii+psi)])

#AGGREGATIVE FUNCTION DEFINITION
def phi_i(z_i):
  return z_i, np.eye(z_i.shape[0]) #Gradient of phi_i

#COST FUNCTION DEFINITION
def cost_function(z_i, sigma, r_i):
  """
  Estimated variables:
    z_i  = decision variable of agent i
	sigma = barycenter of the agents 
	r_i = target position
     
  Cost function for agent i:
    f_i = gamma*|z_i - r_i|^2 + |sigma(x) - b|^2
    b = target position
    r_i = target position
    sigma = barycenter of the agents
  """
  f_i = gamma*(z_i - r_i)@(z_i - r_i) + (sigma-z_i)@(sigma-z_i)
  df_i_dzi = 2*gamma*(z_i - r_i) - 2*(sigma-z_i)
  df_i_dsigma = 2*(sigma-z_i)
  return f_i, df_i_dzi, df_i_dsigma

# Generate Network Binomial Graph
I_NN = np.identity(NN, dtype=int)
p_ER = 0.3
while 1:
	Adj = np.random.binomial(1, p_ER, (NN,NN))
	Adj = np.logical_or(Adj,Adj.T)
	Adj = np.multiply(Adj,np.logical_not(I_NN)).astype(int)
	test = np.linalg.matrix_power((I_NN+Adj),NN)
	if np.all(test>0):
		break 

# Compute weighted adjacency matrix
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

# Initialization of the algorithm
ZZ[:,0] = ZZ_init
SS[:,0], _ = phi_i(ZZ_init) 

# Compute initial cost
for ii in range(NN):
	lower_index = ii*DIM
	upper_index = (ii+1)*DIM
	z_i = ZZ[lower_index:upper_index,0] # z = [x1,y1,x2,y2, x3, y3, ...] z_0 = [x1,y1] z_1 = [x2,y2] z_2 = [x3,y3] ...
	sigma_i = SS[lower_index:upper_index,0]
	r_i = RR[lower_index:upper_index,0]
	f_i_,_,df_i_ = cost_function(z_i, sigma_i, r_i)
	FF[0] += f_i_
	VV[lower_index:upper_index,0] = df_i_
     
# Aggregative Tracking distributed optimization algorithm
for kk in range (MAXITERS-1):
	if (kk % 50) == 0:
		print("Iteration {:3d}".format(kk), end="\n")

	for ii in range (NN):
		#accordingly to the neighbors of each agent ii
		Nii = np.nonzero(Adj[ii])[0]
		
		#compute the variables for agent i
		lower_index = ii*DIM
		upper_index = (ii+1)*DIM
		z_i = ZZ[lower_index:upper_index,kk]
		sigma_i = SS[lower_index:upper_index,kk]
		r_i = RR[lower_index:upper_index,kk]
		v_i = VV[lower_index:upper_index,kk]
		a_ii = AA[ii,ii]

		#compute the cost function for agent i
		_, dfi_dxi_k, df_dsigma_k = cost_function(z_i, sigma_i, r_i)

		# update phi(z) at iteration k
		phi_k, dphi_k = phi_i(z_i)

		# update of the decision variable at itereation k+1
		ZZ[lower_index:upper_index,kk+1] = z_i - alpha*(dfi_dxi_k + dphi_k@v_i)

		# update of the tracker of sigma(z) at iteration k+1
		phi_kp, _ = phi_i(ZZ[lower_index:upper_index,kk+1])

		# update of the tracker of sigma(z)
		SS[lower_index:upper_index,kk+1] = a_ii*sigma_i + phi_kp - phi_k

		for jj in Nii:
			a_ij=AA[ii,jj]
			ind = jj*DIM
			SS[lower_index:upper_index, kk+1] += a_ij*SS[ind:(ind+DIM),kk]

		# update of tracker of sum(nabla_2)
		f_i, _, df_dsigma_kp = cost_function(ZZ[lower_index:upper_index,kk+1], SS[lower_index:upper_index,kk+1], r_i) 

		VV[lower_index:upper_index, kk+1] = a_ii*v_i + df_dsigma_kp - df_dsigma_k
		for jj in Nii:
			a_ij=AA[ii,jj]
			ind = jj*DIM
			VV[lower_index:upper_index,kk+1] += a_ij*VV[ind:(ind+DIM),kk]

	# store the cost
	FF[kk+1] = f_i

# compute the norm of the gradient
grad_norm = np.zeros(MAXITERS-1)
for kk in range(MAXITERS-1):
	_, dfi_dxi_k, _ = cost_function(ZZ[:,kk],SS[:,kk],RR[:,kk])
	grad_norm[kk] += np.linalg.norm(dfi_dxi_k)

#Figure 1 : Evolution of the cost function
if 1:
	fig, ax = plt.subplots(2)
	ax[0].semilogy(np.arange(MAXITERS-1), np.abs(FF[:MAXITERS-1]))
	ax[0].grid() 
	ax[0].set_title('Cost function')
	ax[0].set_ylabel(r'$\sum_{i=1}^{N} \ell(z_{i}^{k})$',fontsize=12)
	ax[1].semilogy(np.arange(MAXITERS-1), grad_norm[:MAXITERS-1])
	ax[1].set_title('Global norm of the gradient of the cost function')
	ax[1].set_ylabel(r'$\sum_{i=1}^{N} |\nabla \ell(z_{i}^{k})|$',fontsize=12)
	ax[1].grid()
	ax[1].set_xlabel(r'iterations $k$')
	plt.subplots_adjust(hspace=0.2) 
	plt.legend()
	plt.show()

###############################################################################
# Figure 2 : Animation
if 1 and DIM == 2: # animation 
	plt.figure()

	for kk in range(0,MAXITERS,dt):
		if (kk % 10) == 0:
			print("Iteration {:3d}".format(kk), end="\n")

		for ii in range(NN):
			lower_index = ii*DIM
			
			# trajectory of agent i 
			plt.plot(ZZ[lower_index,:].T,ZZ[lower_index+1,:].T, linestyle='--', color = 'tab:blue',alpha=0.3)

			# initial position of agent i
			plt.plot(ZZ[lower_index,0],ZZ[lower_index+1,0], marker=markers[ii], markersize=10, color = 'tab:blue',alpha=0.3)

			# position of adversary i
			plt.plot(RR[lower_index,kk], RR[lower_index+1,kk], marker=markers[ii], markersize=10, color = 'tab:red')

			# position of agent i at time t 
			plt.plot(ZZ[lower_index,kk],ZZ[lower_index+1,kk], marker=markers[ii], markersize=10, color = 'tab:blue')

			# estimate of agent i of the centroid at time t
			plt.plot(SS[lower_index,kk],SS[lower_index+1,kk], marker='.', markersize=5, color = 'tab:red')

		# plot settings
		axes_lim = (np.min(ZZ)-1,np.max(ZZ)+1)
		plt.xlim(axes_lim); plt.ylim(axes_lim)
		plt.axis('equal')     
		plt.show(block=False)
		plt.pause(0.05)
		if kk < MAXITERS - dt - 1:
			plt.clf()

print("The end")