# (iii) Prediction of flux decline for a constant pressure run

# Import python packages and libraries
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# Define the parameters (replace with actual values)
n_val = 1.37       # Example value for n
beta = 0.15        # Example value for beta
tau = 0.2877       # Example value for tau
theta = 0.022      # Example value for theta
epsilon_so = 0.27  # Example value for epsilon_so
k_o = 1.89e-14     # Example value for k_o
DeltaP = 50000     # Example value for Delta P
mu = 0.00089       # Example value for mu
Rm = 1.37e11       # Example value for Rm
s = 0.00398

A = 1 - n_val - beta
Pa = (tau * np.exp((DeltaP / 1000) * theta)) * 1000
C = (2 * mu * s * A) / (k_o * Pa * (epsilon_so - s))

# Define the time interval and step
t_start = 100         # start time
t_end = 90 * 60       # end time in seconds (90 minutes)
dt = 100              # time step in seconds

t_values = np.arange(t_start, t_end + dt, dt)
y = np.zeros((len(t_values), 2))

# Loop through time steps
for i, t in enumerate(t_values):
    # Initial guesses for [DeltaPc, J]
    x0 = [100, 5e-5]
    
    # Define system of equations
    def F(x):
        return [
            x[0] - DeltaP + (mu * Rm * x[1]),
            x[1] + (((1 + (x[0] / Pa))**A - 1) /
                    np.sqrt(C * (((1 + (x[0] / Pa))**A) - 1) * t))
        ]
    
    # Solve using fsolve
    sol = fsolve(F, x0)
    y[i, 0] = sol[0]  # DeltaPc
    y[i, 1] = sol[1]  # J

# Extract results
DeltaPc_values = y[:, 0]
J_values = y[:, 1]

# Plot J vs. t
plt.figure()
plt.plot(t_values / 60, J_values * 1000 * 3600)  # LMH conversion
plt.xlabel('Time (minutes)')
plt.ylabel('Permeate flux (LMH)')
plt.title('Permeate flux vs. Time')
plt.grid(True)
plt.show()
