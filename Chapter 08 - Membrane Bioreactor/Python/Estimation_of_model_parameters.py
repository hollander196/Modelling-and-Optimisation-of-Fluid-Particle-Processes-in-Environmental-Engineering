# (i) Estimation of the model parameters from experimental data

# Import python packages and libraries
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from numba import njit, prange

def estimated_parameters():
    
    # Experimental data
    Exp_t = np.array([
        300, 600, 900, 1200, 1500, 1800, 2100, 2400,
        2700, 3000, 3300, 3600, 3900, 4200, 4500,
        4800, 5100, 5400
    ], dtype=np.float64)

    Exp_J1 = np.array([
        5.22222E-05, 3.67778E-05, 3.00741E-05, 2.58333E-05, 2.32444E-05,
        2.12222E-05, 1.96825E-05, 1.83333E-05, 1.73827E-05, 1.64889E-05,
        1.56566E-05, 1.4963E-05, 1.44103E-05, 1.38889E-05, 1.34074E-05,
        0.000013, 1.25621E-05, 1.22469E-05
    ], dtype=np.float64)

    Exp_DeltaPc = np.array([
        33481.93333, 35416.98438, 36238.53236, 36781.70458, 37091.31275,
        37352.03542, 37538.26589, 37713.58443, 37827.31111, 37938.66142,
        38044.58, 38132.84549, 38201.26429, 38265.72869, 38327.02956,
        38375.57557, 38432.78838, 38470.06491
    ], dtype=np.float64)

    # Experimental constants
    s = 0.00398
    mu = 0.00089
    DeltaP = 40000.0

    # Fitted model function
    @njit(fastmath=True)
    def model_funct(bestx, Exp_DeltaPc, Exp_t, s, mu, DeltaP):
        n, beta, tau, theta, k0, eps0 = bestx
        A = 1.0 - n - beta
        Pa = (tau * np.exp((DeltaP / 1000.0) * theta)) * 1000.0
        C = (2.0 * mu * s * A) / (k0 * Pa * (eps0 - s))
        term = (1.0 + (Exp_DeltaPc / Pa)) ** A - 1.0
        return -(term / np.sqrt(C * term * Exp_t))

    # Parallelized cost function
    @njit(fastmath=True, parallel=True)
    def para_function(bestx, Exp_t, Exp_J1, Exp_DeltaPc, s, mu, DeltaP):
        J_model = model_funct(bestx, Exp_DeltaPc, Exp_t, s, mu, DeltaP)
        err = 0.0
        for i in prange(Exp_t.shape[0]):
            diff = Exp_J1[i] - J_model[i]
            err += diff * diff
        return err

    # SciPy wrapper
    def cost_fn(bestx):
        return para_function(bestx, Exp_t, Exp_J1, Exp_DeltaPc, s, mu, DeltaP)

    # Initial guess
    initial_bestx = np.array([1.37, 0.15, 0.3877, 0.055, 1.89E-14, 0.3])

    # Optimization
    result = minimize(cost_fn, initial_bestx, method='Nelder-Mead',
                      options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12})

    est_params = result.x

    # Display results
    print("Estimated Parameters:")
    labels = ["n", "beta", "tau", "theta", "k0", "eps0"]
    for name, val in zip(labels, est_params):
        if name == "k0":
            print(f"{name} = {val:.4e}")
        else:
            print(f"{name} = {val:.4f}")

    # Plot results
    plt.figure(figsize=(7, 5))
    plt.plot(Exp_t / 60, Exp_J1, 'ro', label='Experimental Data')
    plt.plot(Exp_t / 60, model_funct(est_params, Exp_DeltaPc, Exp_t, s, mu, DeltaP),
             'b-', linewidth=2, label='Model')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Permeate flux (m/s)')
    plt.title('Data Fitting for J vs t')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    estimated_parameters()
