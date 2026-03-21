# (ii) Solving all the model equations in the mathematical model

# Import python packages and libraries
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# Parameters (replace with actual values from experimental data fitting)
n = 1.8213
beta = 0.1671
tau = 0.3660
theta = 0.0264
epsilon_so = 0.3010
k_o = 1.8739e-14
DeltaP = 50000
mu = 0.00089
Rm = 1.37e11
s = 0.00398
rhos = 1095.2
A = 1 - n - beta
Pa = (tau * np.exp((DeltaP / 1000) * theta)) * 1000
C = (2 * mu * s * A) / (k_o * Pa * (epsilon_so - s))
alpha_o = 1 / (epsilon_so * k_o * rhos)

# Time settings
t_start = 100
t_end = 90 * 60
dt = 100
t_values = np.arange(t_start, t_end + dt, dt)
num_steps = len(t_values)

# Preallocate
J_values = np.zeros(num_steps)
DeltaPc_values = np.zeros(num_steps)

# Solve for J and DeltaPc at each time step
x0 = [100, 5e-5]  # Initial guess

for idx, t in enumerate(t_values):
    def F(x):
        return [
            x[0] - DeltaP + (mu * Rm * x[1]),
            x[1] + (((1 + (x[0] / Pa)) ** A - 1) / np.sqrt(C * ((1 + (x[0] / Pa)) ** A - 1) * t))
        ]
    m, _, flag, _ = fsolve(F, x0, full_output=True)
    DeltaPc_values[idx] = m[0]
    J_values[idx] = m[1]
    x0 = m  # Use previous solution as next initial guess

# f, Ps, alpha, epsilon calculations
f_values = np.arange(0, 1.01, 0.01)
num_f = len(f_values)
Ps_values = np.zeros((num_f, num_steps))
alpha_values = np.zeros((num_f, num_steps))
epsilon_values = np.zeros((num_f, num_steps))

for j in range(num_steps):
    DeltaPc = DeltaPc_values[j]
    V = (1 + DeltaPc / Pa) ** A
    for i, f in enumerate(f_values):
        # Ps
        Ps0 = 40000
        def F1(Ps):
            return (Ps / Pa) - (1 + (1 - f) * (V - 1)) ** (1 / A) + 1
        Ps_sol = fsolve(F1, Ps0)[0]
        Ps_values[i, j] = Ps_sol

        # epsilon
        epsilon0 = 0.5
        def F2(epsilon):
            return epsilon - 1 + epsilon_so * ((V - 1) * (1 - f) + 1) ** (beta / A)
        epsilon_sol = fsolve(F2, epsilon0)[0]
        epsilon_values[i, j] = epsilon_sol

        # alpha
        alpha0 = 4e13
        def F3(alpha):
            return alpha - alpha_o * ((V - 1) * (1 - f) + 1) ** (n / A)
        alpha_sol = fsolve(F3, alpha0)[0]
        alpha_values[i, j] = alpha_sol

# Plot Permeate flux vs Time
plt.figure(1)
plt.plot(t_values / 60, J_values * 1000 * 3600)
plt.xlabel('Time (minutes)')
plt.ylabel('Permeate flux (LMH)')
plt.title('J vs Time')
plt.grid(True)

# Plot solid compressive pressure profile across the fouling layer
plt.figure(2)
for r in range(num_steps):
    plt.plot(f_values, Ps_values[:, r])
plt.xlabel('Relative cake thickness, x/L')
plt.ylabel('Solid compressive pressure (kPa)')
plt.title('Ps vs Time')
plt.grid(True)

# Plot porosity profile across the cake layer
plt.figure(3)
for r in range(num_steps):
    plt.plot(f_values, epsilon_values[:, r])
plt.xlabel('Relative cake thickness, x/L')
plt.ylabel('Porosity')
plt.title('Porosity vs Time')
plt.grid(True)

plt.show()