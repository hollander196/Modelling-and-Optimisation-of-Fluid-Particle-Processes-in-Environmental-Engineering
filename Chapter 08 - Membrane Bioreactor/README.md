# Chapter 8: Mathematical Modelling of the Effects of Compression of Fouling Layers as a Predictive Tool in Membrane Bioreactor Systems (MBR)

## Overview

This chapter presents mathematical modelling of the effects of compression of fouling layers as a predictive tool in membrane bioreactor (MBR) systems. The implementations demonstrate:

- Parameter estimation from experimental data
- Nonlinear optimisation technique using Nelder-Mead method
- Solving the complete set of mathematical model equations
- Prediction of permeate flux decline under constant pressure
- Prediction of total pressure drop increase under constant flux

## Contents

### Python Implementation Steps

Scripts Located in `Python/` directory:

- **Estimation_of_model_parameters.py** — `(i)` Estimates the six model parameters from experimental MBR data using the Nelder-Mead optimisation algorithm with Numba JIT compilation for performance; provides a graphical comparison of model fit vs experimental data.
- **Solving_mathematical_model_equations.py** — `(ii)` Solves the complete set of model equations to compute permeate flux (J), cake pressure drop (ΔPc), solid compressive pressure (Ps), local porosity (ε), and specific cake resistance (α) as a function of time and normalised cake depth.
- **Prediction_of_flux_decline.py** — `(iii)` Predicts permeate flux decline over time for a constant-pressure filtration run using the fitted model parameters.
- **Prediction_of_pressure_drop.py** — `(iv)` Predicts the total transmembrane pressure drop increase over time for a constant-flux filtration run using the fitted model parameters.
- **Jupyter_notebook.ipynb** — Interactive Jupyter notebook combining all four modelling steps (i)–(iv) with inline codes, results and figures; displays estimated parameters and plots directly in the notebook.
- **Jupyter_notebook.pdf** — PDF export of the executed Jupyter notebook for easy viewing and sharing without a Python environment.

### MATLAB Implementation Steps

Scripts Located in `MATLAB/` directory:

- **Estimation_of_model_parameters.m** — MATLAB equivalent of step `(i)`; estimates model parameters from experimental data using `fminsearch` (Nelder-Mead) optimization.
- **Solving_mathematical_model_equations.m** — MATLAB equivalent of step `(ii)`; solves the full set of model equations for flux, cake pressure drop, porosity, solid compressive pressure, and specific cake resistance profiles.
- **Prediction_of_flux_decline.m** — MATLAB equivalent of step `(iii)`; predicts permeate flux decline for a constant-pressure run.
- **Prediction_of_pressure_drop.m** — MATLAB equivalent of step `(iv)`; predicts total pressure drop increase for a constant-flux run.
- **MATLAB_Livescript.mlx** — MATLAB Live Script combining all four modelling steps (i)–(iv) with inline codes, results and figures; displays estimated parameters and plots directly in the Live Script.
- **MATLAB_Livescript.pdf** — PDF export of the executed MATLAB Live Script for easy viewing and sharing without a MATLAB environment.

## Key Concepts

### Model Parameters

The membrane fouling compression model includes six key parameters:

1. **n** — Compressibility coefficient
2. **β (beta)** — Cake layer compressibility parameter
3. **τ (tau)** — Reference compressive pressure scaling coefficient
4. **θ (theta)** — Pressure sensitivity parameter
5. **k₀** — Initial (uncompressed) cake permeability (m²)
6. **ε₀ (eps0)** — Initial (uncompressed) cake porosity

### Experimental Constants

- **s** — Membrane thickness (m)
- **μ (mu)** — Fluid viscosity (Pa·s)
- **ΔP** — Transmembrane pressure (Pa)
- **Rm** — Clean membrane resistance (m⁻¹)
- **ρs** — Particle density (kg/m³)

### Response Variables

- **J** — Permeate flux (m/s or LMH)
- **ΔPc** — Cake layer pressure drop (Pa)
- **Ps** — Solid compressive pressure profile (Pa)
- **ε** — Local cake porosity profile
- **α** — Local specific cake resistance (m/kg)
- **t** — Time (s)

## Usage

### Python

#### Step (i) — Parameter Estimation
```bash
cd Chapter 08 - Membrane Bioreactor/Python
python Estimation_of_model_parameters.py
```

The script will:
1. Load experimental flux and pressure drop data
2. Perform Nelder-Mead parameter optimisation (Numba-accelerated)
3. Display the six estimated parameters
4. Plot model fit vs experimental data

#### Step (ii) — Solve Model Equations
```bash
python Solving_mathematical_model_equations.py
```

Uses estimated parameters to compute and plot J, ΔPc, Ps, ε, and α profiles over time.

#### Step (iii) — Predict Flux Decline (Constant Pressure)
```bash
python Prediction_of_flux_decline.py
```

Plots permeate flux (LMH) vs time (minutes) for a constant-pressure run.

#### Step (iv) — Predict Pressure Drop (Constant Flux)
```bash
python Prediction_of_pressure_drop.py
```

Plots total pressure drop (kPa) vs time (minutes) for a constant-flux run.

#### Step (v) — Run Interactive Jupyter Notebook
```bash
jupyter notebook Jupyter_notebook.ipynb
```

Runs all four steps interactively with inline outputs and plots.

### MATLAB

#### Step (i) — Parameter Estimation
```matlab
run('Estimation_of_model_parameters.m')
```

The script will:
1. Load experimental flux and pressure drop data
2. Perform Nelder-Mead parameter optimisation (Numba-accelerated)
3. Display the six estimated parameters
4. Plot model fit vs experimental data

#### Step (ii) — Solve Model Equations
```matlab
run('Solving_mathematical_model_equations.m')
```

Uses estimated parameters to compute and plot J, ΔPc, Ps, ε, and α profiles over time.

#### Step (iii) — Predict Flux Decline (Constant Pressure)
```matlab
run('Prediction_of_flux_decline.m')
```

Plots permeate flux (LMH) vs time (minutes) for a constant-pressure run.

#### Step (iv) — Predict Pressure Drop (Constant Flux)
```matlab
run('Prediction_of_pressure_drop.m')
```

Plots total pressure drop (kPa) vs time (minutes) for a constant-flux run.

#### #### Step (v) — Run Interactive MATLAB Livescript
```matlab
run('MATLAB_livescript.mlx')
```

Runs all four steps interactively with inline outputs and plots.


## Python Dependencies (see requirements file)

- `numpy`
- `scipy`
- `matplotlib`
- `numba`

Install using:
```bash
pip install -r requirements.txt
```

### MATLAB Toolboxes
- Optimisation Toolbox (required for `fminsearch`)
- Global Optimisation Toolbox (optional, for advanced optimisation strategies)

## Model Description

The model describes the compression behaviour of the fouling (cake) layer formed on a membrane surface during MBR operation. It captures the complex relationship between:

- Transmembrane pressure and cake-layer compression
- Local porosity and permeability profiles through the cake depth
- Permeate flux decline over time
- Total pressure drop evolution over time

### Key Features

1. **Nonlinear Optimisation** — Nelder-Mead algorithm for robust parameter estimation from experimental data
2. **Performance Optimisation** — Numba JIT compilation and parallelization for fast computation (Python)
3. **Full Profile Computation** — Spatially resolved Ps, ε, and α profiles across the cake layer
4. **Dual-mode Prediction** — Separate prediction modules for constant-pressure and constant-flux operating modes
5. **Cross-platform** — Equivalent implementations in both Python and MATLAB

## Workflow

1. **Data Collection** — Gather experimental time-series data: permeate flux (J) and cake pressure drop (ΔPc)
2. **Parameter Estimation** — Run `Estimation_of_model_parameters` to fit the six model parameters
3. **Model Solution** — Run `Solving_mathematical_model_equations` to compute full spatial and temporal profiles
4. **Flux Prediction** — Run `Prediction_of_flux_decline` to predict flux decline under constant pressure
5. **Pressure Prediction** — Run `Prediction_of_pressure_drop` to predict pressure build-up under constant flux
6. **Validation** — Compare model outputs with independent experimental data

## Input Data Format

All scripts expect experimental data as arrays:
- `Exp_t` — Time points (s)
- `Exp_J1` — Measured permeate flux (m/s)
- `Exp_DeltaPc` — Measured cake layer pressure drop (Pa)

Sample data (18 time points from 300 s to 5400 s) is embedded directly in the scripts and as an excel file.

## Output

The scripts provide:
- Estimated parameter values (n, β, τ, θ, k₀, ε₀)
- Graphical comparison of model vs experimental flux and pressure drop
- Spatial profiles of Ps, ε, and α across the cake layer
- Permeate flux vs time (constant-pressure prediction)
- Total pressure drop vs time (constant-flux prediction)

## Notes

- Initial parameter guesses significantly affect optimisation convergence; multiple runs with different starting points are recommended.
- Numba JIT compilation introduces a one-time overhead on the first call but provides ~10–100× speedup thereafter.
- The `Jupyter_notebook.pdf` and `MATLAB_Livescript.pdf` provide pre-executed outputs and can be used for reference without running any code.

## Advanced Usage

For sensitivity analysis or uncertainty quantification:
1. Run multiple optimisations with different initial parameter guesses
2. Perform bootstrap resampling of experimental data
3. Calculate confidence intervals for the estimated parameters
4. Validate the model against independent test datasets

## References

See the main book "Modelling and Optimisation of Fluid-Particle Processes in Environmental Engineering" for:
- Detailed derivation of the mathematical model
- Physical interpretation of all parameters
- Validation studies
- Case studies from real MBR operations