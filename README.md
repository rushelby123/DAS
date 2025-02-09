Read the pdf file for a detailed project description.

### **Distributed Autonomous Systems Project (2023-24)**  
**Course Project: Distributed Classification and Multi-Robot Control**  

This project, developed as part of the *Distributed Autonomous Systems* course, focuses on two main tasks: **distributed classification via logistic regression** and **multi-robot control using aggregative optimization**. The project demonstrates advanced skills in distributed optimization, machine learning, and robotics, with implementations in Python and ROS 2.  

---

#### **Task 1: Distributed Classification via Logistic Regression**  
**Objective**: Design and implement a distributed optimization algorithm to solve a nonlinear classification problem using logistic regression.  

1. **Distributed Optimization**:  
   - Implemented the **Gradient Tracking algorithm** to solve a consensus optimization problem of the form:  
     \[
     \min_z \sum_{i=1}^N \ell_i(z), \quad z \in \mathbb{R}^d
     \]  
     where \(\ell_i\) is a quadratic function.  
   - Tested the algorithm on various graph topologies (e.g., cycle, path, star) with weights determined by the **Metropolis-Hastings method**.  
   - Visualized the convergence of the cost function and gradient norms across iterations.  

2. **Centralized Classification**:  
   - Generated a dataset of \(M\) points in \(\mathbb{R}^d\) with binary labels.  
   - Implemented a **centralized gradient method** to minimize the logistic regression cost function:  
     \[
     \min_{w,b} \sum_{m=1}^M \log\left(1 + \exp(-p^m(w^\top \phi(D^m) + b))\right)
     \]  
     where \(\phi\) is a nonlinear mapping (e.g., quadratic for elliptical separation).  
   - Evaluated the algorithm on different dataset patterns and visualized convergence.  

3. **Distributed Classification**:  
   - Split the dataset into \(N\) subsets and extended the Gradient Tracking algorithm for distributed classification.  
   - Tested the algorithm on various dataset sizes and patterns, demonstrating convergence to a stationary point.  
   - Computed the percentage of misclassified points to evaluate solution quality.  

---

#### **Task 2: Aggregative Optimization for Multi-Robot Systems**  
**Objective**: Develop a distributed control algorithm for a team of robots to maintain formation while moving toward private targets.  

1. **Problem Setup**:  
   - Formalized the problem as an **aggregative optimization problem**:  
     \[
     \min_z \sum_{i=1}^N \ell_i(z_i, \sigma(z)), \quad \sigma(z) = \frac{1}{N} \sum_{i=1}^N \phi_i(z_i)
     \]  
     where \(\ell_i\) is a cost function balancing formation tightness and target proximity.  
   - Implemented the **Aggregative Tracking algorithm** in Python, enabling robots to:  
     - Maintain a tight formation.  
     - Move toward private targets.  
   - Conducted simulations with varying cost function parameters, target locations, and moving targets.  

2. **ROS 2 Implementation**:  
   - Created a ROS 2 package in Python to implement the Aggregative Tracking algorithm.  
   - Simulated robot behavior using RViz, demonstrating real-time distributed control.  

3. **Corridor Navigation**:  
   - Designed a new cost function to enable the robot team to navigate through a corridor without colliding with walls.  
   - Tested the implementation in ROS 2, showcasing the team's ability to reach targets while maintaining formation and avoiding obstacles.  

---

### **Key Skills Demonstrated**  
- **Distributed Optimization**: Gradient Tracking, consensus optimization, and aggregative optimization.  
- **Machine Learning**: Logistic regression, nonlinear classification, and dataset generation.  
- **Robotics**: Multi-robot control, formation maintenance, and ROS 2 implementation.  
- **Programming**: Python, MATLAB, ROS 2, and simulation tools (RViz).  
- **Data Visualization**: Convergence plots, cost function evolution, and animated robot behavior.  

---

### **Impact and Results**  
- Successfully implemented distributed algorithms for classification and multi-robot control, demonstrating scalability and robustness.  
- Achieved high classification accuracy and effective robot coordination in simulated environments.  
- Provided insights into the trade-offs between formation tightness and target proximity in multi-robot systems.  
