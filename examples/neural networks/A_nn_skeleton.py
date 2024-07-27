#
# 22 Apr 2024
#
# Ivano Notarnicola
#
# NN
#
import numpy as np
import matplotlib.pyplot as plt

###############################################################################
max_iters = 100


###############################################################################
# Activation Function
def sigmoid_fn(xi):
    return


# Derivative of Activation Function
def sigmoid_fn_derivative(xi):
    return


# Inference: xtp = f(xt,ut)
def inference_dynamics(xt, ut):
    """
    input:
              xt current state (size=d)
              ut current input (size=d*(d+1)), including bias)
    output:
              xtp next state
    """
    xtp = 0
    return xtp


# Forward Propagation
def forward_pass(uu, x0):
    """
    input:
              uu input trajectory: u[0],u[1],..., u[T-1]
              x0 initial condition
    output:
              xx state trajectory: x[1],x[2],..., x[T]
    """
    xx = 0
    return xx


# Adjoint dynamics:
def adjoint_dynamics(ltp, xt, ut):
    """
    input:
              llambda_tp current costate
              xt current state
              ut current input
    output:
              llambda_t next costate
              delta_ut loss gradient wrt u_t
    """
    lt = 0
    Delta_ut = 0
    return lt, Delta_ut


# Backward Propagation
def backward_pass(xx, uu, llambdaT):
    """
    input:
              xx state trajectory: x[1],x[2],..., x[T]
              uu input trajectory: u[0],u[1],..., u[T-1]
              llambdaT terminal condition
    output:
              llambda costate trajectory
              delta_u costate output, i.e., the loss gradient
    """
    llambda = 0
    Delta_u = 0
    return llambda, Delta_u


###############################################################################
# MAIN
###############################################################################

J = np.zeros(max_iters)  # Loss

# GO!
for k in range(max_iters - 1):
    if ((k + 2) % 50) == 0:
        print(f"Cost at iter {k+2:4d} is {J[k-1]:.4f}")

if 1:
    _, ax = plt.subplots()
    ax.plot(
        range(max_iters),
        J,
        marker=".",
        linestyle="--",
    )
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$J(u^k)$")
    ax.grid()

plt.show()
