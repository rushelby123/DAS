from time import sleep
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray as MsgFloat 
from visualization_msgs.msg import Marker
import numpy as np

# GLOBAL VARIABLES
timer_period = 0.1 # seconds DON'T CHANGE THIS VALUE

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
        self.AA_i = np.array(self.get_parameter("AA").value)
        self.DIM = self.get_parameter("DIM").value
        self.NN = self.get_parameter("NN").value
        self.MAXITERS = self.get_parameter("MAXITERS").value
        self.gamma = self.get_parameter("gamma").value
        self.alpha = self.get_parameter("alpha").value
        self.agent_id = self.get_parameter("id").value
        self.neighbors = self.get_parameter("Nii").value
        self.z_i = self.get_parameter("ZZinit").value 
        self.r_i = self.get_parameter("RRinit").value 

        # PRINT IDENTIFICATION
        print(f"I am agent: {self.agent_id}")
        print(f"My neighbours are {self.neighbors}")
        print(f"AA_i:{self.AA_i}")

        # VARIABLES INITIALIZATION
        self.t = 0
        self.sync_barrier= -np.ones((self.NN))
        self.ZZ = np.zeros((self.DIM*self.NN,self.MAXITERS))
        self.RR = np.zeros((self.DIM*self.NN,self.MAXITERS))
        self.SS = np.zeros((self.DIM*self.NN,self.MAXITERS))
        self.VV = np.zeros((self.DIM*self.NN,self.MAXITERS))
        self.FF = np.zeros((self.MAXITERS,1))
        self.lower_index = self.agent_id*self.DIM
        self.upper_index = (self.agent_id+1)*self.DIM
        self.ZZ[self.lower_index:self.upper_index,0] = np.array(self.z_i)
        self.RR[self.lower_index:self.upper_index,:] = np.array(self.r_i)[:,None]
        self.SS[self.lower_index:self.upper_index,0], _ = phi_i(self.ZZ[self.lower_index:self.upper_index,0])
        f_i,_,df_i = cost_function(self.ZZ[self.lower_index:self.upper_index,0], self.SS[self.lower_index:self.upper_index,0], self.RR[self.lower_index:self.upper_index,0], self.gamma)
        self.VV[self.lower_index:self.upper_index,0] = df_i
        self.FF[0] = f_i
        self.do_init = True

        # CREATE A TOPIC TO PUBLISH DATA
        self.publisher = self.create_publisher(MsgFloat, f"/topic_{self.agent_id}", 10)

        # SUBSCRIBE TO TOPICS WITH ITS NEIGHBORS
        for j in self.neighbors:
            self.create_subscription(MsgFloat, f"/topic_{j}", self.listener_callback, 10)  

        # TIMER CALLBACK DEFINITION
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # CREATE A PUBLISHER TO VISUALIZE THE DATA IN RVIZ
        self.publisher_viz = self.create_publisher(Marker, f"/topic_viz_", 10)
        self.publisher_viz_target = self.create_publisher(Marker, f"/topic_viz_target", 10)
        self.publisher_barycenter = self.create_publisher(Marker, f"/topic_barycenter", 10)

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
        self.sync_barrier[agent_id] = iteration
        #print(f"self.sync_barrier: {self.sync_barrier}, iteration of the agent: {self.t}")
        return None

    def timer_callback(self):

        # CREATE A MESSAGE TO PUBLISH
        msg = MsgFloat()
        self.sync_barrier[self.agent_id] = self.t 
        if self.do_init:
            #print(f"init self.sync_barrier: {self.sync_barrier}")
            self.do_init = False
            # PUBLISH THE DATA
            data_list = [float(self.agent_id), float(self.t)]
            data_list += list(self.ZZ[self.lower_index:self.upper_index,self.t])
            data_list += list(self.SS[self.lower_index:self.upper_index,self.t])
            data_list += list(self.VV[self.lower_index:self.upper_index,self.t])
            msg.data = data_list
            self.publisher.publish(msg) 
        else:
            #print(f"I AM READY, self.sync_barrier: {self.sync_barrier}, iteration of the agent: {self.t}")
            # WAIT OTHER AGENTS TO PUBLISH DATA
            all_received = all(self.t <= self.sync_barrier[j] and not - 1 == self.sync_barrier[j] for j in self.neighbors)
            if all_received and self.t < self.MAXITERS-1:

                z_i = self.ZZ[self.lower_index:self.upper_index,self.t]
                sigma_i = self.SS[self.lower_index:self.upper_index,self.t]
                r_i = self.RR[self.lower_index:self.upper_index,self.t]
                v_i = self.VV[self.lower_index:self.upper_index,self.t]
                a_ii = self.AA_i[self.agent_id]

                #print(f"z_i{self.agent_id}[:,{self.t}] : {z_i}, sigma_i{self.agent_id}[:,{self.t}] : {sigma_i}, r_i{self.agent_id}[:,{self.t}] : {r_i}, v_i{self.agent_id}[:,{self.t}] : {v_i}")

                #compute the cost function for agent i
                _, dfi_dxi_k, df_dsigma_k = cost_function(z_i, sigma_i, r_i,self.gamma)

                # update phi(z) at iteration k
                phi_k, dphi_k = phi_i(z_i)

                # update of the decision variable at itereation k+1
                self.ZZ[self.lower_index:self.upper_index,self.t+1] = z_i - self.alpha*(dfi_dxi_k + dphi_k@v_i)

                # update of the tracker of sigma(z) at iteration k+1
                phi_kp, _ = phi_i(self.ZZ[self.lower_index:self.upper_index,self.t+1])

                # update of the tracker of sigma(z)
                self.SS[self.lower_index:self.upper_index,self.t+1] = a_ii * sigma_i + phi_kp - phi_k
                
                for jj in self.neighbors:
                    a_ij = self.AA_i[jj] 
                    ind = jj*self.DIM
                    self.SS[self.lower_index:self.upper_index, self.t+1] += a_ij * self.SS[ind:(ind+self.DIM),self.t]
                print(f"SS{self.agent_id}[:,{self.t}] : {self.SS[self.lower_index:self.upper_index, self.t+1]} ")
                
                # update of tracker of sum(nabla_2)
                f_i, _, df_dsigma_kp = cost_function(self.ZZ[self.lower_index:self.upper_index,self.t+1], self.SS[self.lower_index:self.upper_index,self.t+1], r_i,self.gamma) 

                self.VV[self.lower_index:self.upper_index, self.t+1] = a_ii*v_i + df_dsigma_kp - df_dsigma_k
                for jj in self.neighbors:
                    a_ij=self.AA_i[jj]
                    ind = jj*self.DIM
                    self.VV[self.lower_index:self.upper_index,self.t+1] += a_ij*self.VV[ind:(ind+self.DIM),self.t]

                # store the cost
                self.FF[self.t+1] = f_i       

                # PUBLISH THE DATA
                data_list = [float(self.agent_id), float(self.t+1)]
                data_list += list(self.ZZ[self.lower_index:self.upper_index,self.t+1])
                data_list += list(self.SS[self.lower_index:self.upper_index,self.t+1])
                data_list += list(self.VV[self.lower_index:self.upper_index,self.t+1])
                msg.data = data_list
                self.publisher.publish(msg)   
                
                self.t += 1

                # PUBLISH THE DATA IN RVIZ
                #THE BARYCENTER IS REPRESENTED BY A SPHERE
                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.id = self.agent_id
                marker.pose.position.x = self.SS[self.lower_index,self.t]
                marker.pose.position.y = self.SS[self.lower_index+1,self.t]
                marker.pose.position.z = 0.0
                marker.scale.x = 0.1
                marker.scale.y = 0.1
                marker.scale.z = 0.1
                marker.color.a = 0.5
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                marker.type = Marker.SPHERE
                self.publisher_barycenter.publish(marker)
                # THE AGENT IS REPRESENTED BY A SPHERE
                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.id = self.agent_id
                marker.pose.position.x = self.ZZ[self.lower_index,self.t]
                marker.pose.position.y = self.ZZ[self.lower_index+1,self.t]
                marker.pose.position.z = 0.0
                marker.scale.x = 0.2
                marker.scale.y = 0.2
                marker.scale.z = 0.2
                marker.color.a = 1.0
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.type = Marker.SPHERE
                self.publisher_viz.publish(marker)
                # THE TARGET IS REPRESENTED BY A CUBE
                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.id = self.agent_id
                marker.pose.position.x = self.RR[self.lower_index,self.t]
                marker.pose.position.y = self.RR[self.lower_index+1,self.t]
                marker.pose.position.z = 0.0
                marker.scale.x = 0.5
                marker.scale.y = 0.5
                marker.scale.z = 0.5
                marker.color.a = 0.5
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                marker.type = Marker.CUBE
                self.publisher_viz_target.publish(marker)

def main():
    rclpy.init()

    agent = Agent()
    sleep(5)
    rclpy.spin(agent)
    agent.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
