from time import sleep
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray as MsgFloat 
import numpy as np
import matplotlib.pyplot as plt 

# GLOBAL VARIABLES
timer_period = 2 # seconds

#AGGREGATIVE FUNCTION DEFINITION
def phi_i(z_i):
  return z_i, np.eye(z_i.shape[0]) #Gradient of phi_i

#COST FUNCTION DEFINITION
def cost_function(z_i, sigma, r_i, gamma):
  """
  Estimated variables:
    z_i  = decision variable of agent i
	sigma = barycenter of the agents 
	r_i = target position
     
  Cost function for agent i:
    f_i = gamma*|z_i - r_i|^2 + |sigma(x) - b|^2
    r_i = target position
    sigma = barycenter of the agents
  """

  f_i = gamma*(z_i - r_i)@(z_i - r_i) + (sigma-z_i)@(sigma-z_i)
  df_i_dzi = 2*gamma*(z_i - r_i) - 2*(sigma-z_i)
  df_i_dsigma = 2*(sigma-z_i)
  return f_i, df_i_dzi, df_i_dsigma

class Agent(Node):
    def __init__(self):
        super().__init__(
            "agent",
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )

        # GET PARAMETERS FROM LAUNCH FILE
        self.AA = np.array(self.get_parameter("AA").value)
        self.DIM = self.get_parameter("DIM").value
        self.NN = self.get_parameter("NN").value
        self.MAXITERS = self.get_parameter("MAXITERS").value
        self.gamma = self.get_parameter("gamma").value
        self.r_i = self.get_parameter("RRinit").value 
        
        # VARIABLES INITIALIZATION
        self.sync_barrier= np.zeros((self.NN))
        self.ZZ = np.zeros((self.DIM*self.NN,self.MAXITERS))
        self.SS = np.zeros((self.DIM*self.NN,self.MAXITERS))
        self.VV = np.zeros((self.DIM*self.NN,self.MAXITERS))
        self.RR = np.zeros((self.DIM*self.NN,self.MAXITERS))
        self.RR[:,:] = np.array(self.r_i)[:,None]
        self.FF = np.zeros((self.MAXITERS))


        # SUBSCRIBE TO ALL TOPICS 
        for ii in range(self.NN):
            self.create_subscription(MsgFloat, f"/topic_{ii}", self.listener_callback, 10)  

        # CREATE A TIMER
        self.create_timer(timer_period, self.timer_callback)

    def listener_callback(self, msg):
        # the message structure is defined as follow
        # [agent_id,iteration,z_i[0],z_i[1],...,z_i[DIM-1],s_i[0],s_i[1],...,s_i[DIM-1],v_i[0],v_i[1],...,v_i[DIM-1]]
        # READ THE MESSAGE
        agent_id = int(msg.data[0])
        iteration = int(msg.data[1])
        z_j = np.array(list(msg.data[2:self.DIM+2]))
        s_j = np.array(list(msg.data[self.DIM+2:2*self.DIM+2]))
        v_j = np.array(list(msg.data[self.DIM*2+2:3*self.DIM+2]))
        # STORE THE RECEIVED DATA
        j_index = agent_id*self.DIM
        self.ZZ[j_index:j_index+self.DIM,iteration] = z_j
        self.SS[j_index:j_index+self.DIM,iteration] = s_j
        self.VV[j_index:j_index+self.DIM,iteration] = v_j
        # UPDATE THE RECEIVED DATA
        self.sync_barrier[agent_id] += 1
        return None

    def timer_callback(self):
        print(f"Received data: {self.sync_barrier}")
        # WAIT OTHER AGENTS TO PUBLISH DATA
        all_received = all(self.sync_barrier[agent_id] >= self.MAXITERS-1 for agent_id in range(self.NN))
        if all_received:

            print(self.SS)
            # compute the norm of the gradient
            grad_norm = np.zeros(self.MAXITERS-1)
            grad_zz = np.zeros((self.DIM,self.MAXITERS-1))
            grad_ss = np.zeros((self.DIM,self.MAXITERS-1))
            for kk in range(self.MAXITERS-1):
                for ii in range(self.NN):
                    lower_index = ii*self.DIM
                    upper_index = (ii+1)*self.DIM
                    f_i, grad_z, grad_s = cost_function(self.ZZ[lower_index:upper_index,kk],self.SS[lower_index:upper_index,kk],self.RR[lower_index:upper_index,kk],self.gamma)
                    self.FF[kk] += f_i
                    grad_zz[:,kk] += grad_z
                    grad_ss[:,kk] += grad_s
                grad_norm[kk] = np.linalg.norm(np.vstack((grad_zz[:,kk],grad_ss[:,kk])))
        
            # PLOT THE RESULTS
            fig, ax = plt.subplots(2)
            ax[0].semilogy(np.arange(self.MAXITERS-1), np.abs(self.FF[:self.MAXITERS-1]))
            ax[0].grid() 
            ax[0].set_title('Cost function')
            ax[0].set_ylabel(r'$\sum_{i=1}^{N} \ell(z_{i}^{k})$',fontsize=12)
            ax[1].semilogy(np.arange(self.MAXITERS-1), grad_norm[:self.MAXITERS-1])
            ax[1].set_title('Global norm of the gradient of the cost function')
            ax[1].set_ylabel(r'$|\sum_{i=1}^{N} \nabla \ell(z_{i}^{k})|$',fontsize=12)
            ax[1].grid()
            ax[1].set_xlabel(r'iterations $k$')
            plt.subplots_adjust(hspace=0.2) 
            plt.legend()
            plt.show()

def main():
    rclpy.init()

    agent = Agent()
    rclpy.spin(agent)
    agent.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
