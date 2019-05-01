import numpy as np
from scipy.integrate import odeint


# Function that returns dX/dt
def model(X, t):
    x1 = X[0]
    x2 = X[1]
    dx1 = x2
    dx2 = -3.5 * np.sin(x1) - x2
    dX = [dx1, dx2]
    return dX


# Function that solves the ODE and outputs the trajectory
def getPendulumTraj(IC, tEnd, num_steps):
    # time points
    t = np.linspace(0, tEnd, num=num_steps+1)
    # solve ODE
    X = odeint(model, IC, t)
    return X, t
