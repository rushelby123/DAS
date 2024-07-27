import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray as MsgFloat 
import numpy as np
import matplotlib.pyplot as plt 
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

# GLOBAL VARIABLES
timer_period = 3 # seconds

#AGGREGATIVE FUNCTION DEFINITION
def phi_i(z_i):
  return z_i, np.eye(z_i.shape[0]) #Gradient of phi_i

#COST FUNCTION DEFINITION
def cost_function(z_i, sigma, r_i, gamma,a,b,epsilon):
  """
  Estimated variables:
    z_i  = decision variable of agent i
	sigma = barycenter of the agents 
	r_i = target position
    gamma = weight of the target position
    a, b weights of the barrier function
    g(z_i) = a*z_i[0]^2 - b*z_i[1]^2 - 1 <= 0
     
     
  Cost function for agent i:
    f_i = gamma*|z_i - r_i|^2 + |sigma(x) - b|^2
    r_i = target position
    sigma = barycenter of the agents
    barrier = -log(-g(z_i)) = -log(-a*z_i[0]^2 + b*z_i[1]^2 + 1)
  """

  f_i = gamma*(z_i - r_i)@(z_i - r_i) + (sigma-z_i)@(sigma-z_i) #- epsilon*np.log(-a*z_i[0]**2 + b*z_i[1]**2 + 1)
  df_i_dzi = 2*gamma*(z_i - r_i) - 2*(sigma-z_i) - epsilon * (np.array([-2*a*z_i[0]/(-a*z_i[0]**2 + b*z_i[1]**2 + 1), 2*b*z_i[1]/(-a*z_i[0]**2 + b*z_i[1]**2 + 1)]))
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
        self.a = self.get_parameter("a").value
        self.b = self.get_parameter("b").value
        self.epsilon = self.get_parameter("epsilon").value
        
        # VARIABLES INITIALIZATION
        self.sync_barrier= np.zeros((self.NN))
        self.ZZ = np.zeros((self.DIM*self.NN,self.MAXITERS))
        self.SS = np.zeros((self.DIM*self.NN,self.MAXITERS))
        self.VV = np.zeros((self.DIM*self.NN,self.MAXITERS))
        self.RR = np.zeros((self.DIM*self.NN,self.MAXITERS))
        self.RR[:,:] = np.array(self.r_i)[:,None]
        self.FF = np.zeros((self.MAXITERS))
        self.DO_ONCE = True

        # SUBSCRIBE TO ALL TOPICS 
        for ii in range(self.NN):
            self.create_subscription(MsgFloat, f"/topic_{ii}", self.listener_callback, 100)  

        # CREATE A TIMER
        self.create_timer(timer_period, self.timer_callback)

        self.publisher_ = self.create_publisher(Marker, '/visualization_marker', self.MAXITERS)

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
            # compute the norm of the gradient
            grad_norm = np.zeros(self.MAXITERS-1)
            grad_zz = np.zeros((self.DIM,self.MAXITERS-1))
            grad_ss = np.zeros((self.DIM,self.MAXITERS-1))
            for kk in range(self.MAXITERS-1):
                for ii in range(self.NN):
                    lower_index = ii*self.DIM
                    upper_index = (ii+1)*self.DIM
                    f_i, grad_z, grad_s = cost_function(self.ZZ[lower_index:upper_index,kk],self.SS[lower_index:upper_index,kk],self.RR[lower_index:upper_index,kk],self.gamma,self.a,self.b,self.epsilon)
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

        if self.DO_ONCE:
            self.DO_ONCE = False
            # PUBLISH THE MARKERS OF THE BARRIER FUNCTION
            marker = Marker()
            marker.header.frame_id = "map"
            marker.type = Marker.POINTS
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.05  # Point size
            marker.scale.y = 0.05
            marker.scale.z = 0.05
            marker.color.a = 1.0  # Don't forget to set the alpha!
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            # Generate points that satisfy 5x^2 - y^2 - 1 = 0
            resolution = 100
            for x in range(int(resolution*(1/self.a)**0.5+1), 400,1):  
                x /= resolution
                y = ((self.a*x**2 - 1)/self.b)**0.5 # y = (5x^2 - 1)^0.5
                x = float(x)
                try:
                    y = float(y)
                    marker.points.append(Point(x=x, y=y, z=0.0))
                    marker.points.append(Point(x=-x, y=y, z=0.0))
                    if y != 0:  # Add the symmetric point
                        marker.points.append(Point(x=x, y=-y, z=0.0))
                        marker.points.append(Point(x=-x, y=-y, z=0.0))
                except ValueError:
                    print(f"Cannot convert {y} to float.")
                except TypeError:
                    print(f"Type error: {y} is not a number.")
            self.publisher_.publish(marker)


def main():
    rclpy.init()

    agent = Agent()
    rclpy.spin(agent)
    agent.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
