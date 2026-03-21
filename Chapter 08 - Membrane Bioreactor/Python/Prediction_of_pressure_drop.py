#  (iv) Prediction of total pressure drop increase in a constant flux run

# Import python packages and libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Define the parameters
n_val = 1.35
beta = 0.15
Pa = 1100
k0 = 1.89e-14
eps0 = 0.3
s = 0.00398
mu = 0.00089
J = 9.722e-6
Rm = 1.12e11

A = 1 - n_val - beta
C = (2 * mu * s * A) / (k0 * Pa * (eps0 - s))

# Define the time interval and step
t_start = 300          # seconds
t_end = 90 * 60        # 90 minutes in seconds
dt = 300               # step size in seconds
t_values = np.arange(t_start, t_end + dt, dt)

# Initialize cumulative P value
P_total = 0
y = np.zeros((len(t_values), 2))  # column 0 = P, column 1 = cumulative P_total

for i, t in enumerate(t_values):
    # Define the function for fsolve
    def F(P):
        return P - ((((J**2) * C * t) + 1)**(1 / A) - 1) * Pa - (mu * Rm * J)

    # Initial guess
    P0 = 100

    # Solve for P
    P_solution = fsolve(F, P0)[0]

    # Accumulate
    P_total += P_solution

    # Store results
    y[i, 0] = P_solution
    y[i, 1] = P_total

# Extract cumulative total pressures
TotalP_values = y[:, 1]

# Plot results
plt.figure()
plt.plot(t_values / 60, TotalP_values / 1000)  # Convert time to minutes, pressure to kPa
plt.xlabel("Time (minutes)")
plt.ylabel("Total pressure drop (kPa)")
plt.title("Total pressure drop vs. Time")
plt.grid(True)
plt.show()
