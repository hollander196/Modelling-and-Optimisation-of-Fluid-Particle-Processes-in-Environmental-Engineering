# Chapter 5: Numerical Simulation and Design Optimisation of Hydrodynamics and Mixing Process in a Stirred Tank Reactor (STR)

- [Overview](#overview)
- [Contents](#contents)
- [Key Concepts](#key-concepts)
- [Usage](#usage)
- [Python Dependencies](#python-dependencies)
- [Model Description](#model-description)
- [Workflow](#workflow)
- [Input Data Format](#input-data-format)
- [Output](#output)
- [Notes](#notes)
- [References](#references)

## Overview

This chapter focuses on the numerical simulation and design optimisation of hydrodynamics and mixing processes in a Stirred Tank Reactor (STR). The implementations provided here demonstrate:

- Numerical Simulation of mixing and hydrodynamics in stirred tank
- Design of Experiments (DOE) using Central Composite Design (CCD)
- Response Surface Analysis of hydrodynamics and mixing parameters
- Multi-objective Optimisation of hydrodynamics and mixing parameters

## Contents

### Python Implementation Steps

Scripts Located in `Python/` directory:

- **Load_and_display_DoE_table.py** — `(i)` Load the DoE table with data points for factors and each response and display the DoE table.
- **Compute_mean_effects_and_interactions.py** — `(ii)`Compute the mean and interaction effects of the factors on the response and display the results in bar charts.
- **Create_response_surface_and_contour_plots.py** — `(iii)` Perform a Response Surface Analysis and create a 3D response surface and 2D contour plots for the significant intercations.
- **MOGA_optimisation_with_polynomial.py** — `(iv)` Fit a second degree polynomial function to the response surface and perform a MOGA optimisation for the desired responses.
- **Jupyter_notebook.ipynb** — Interactive Jupyter notebook combining all four modelling steps (i)–(iv) with inline results and figures; displays the computed parameters and plots directly in the notebook.
- **Jupyter_notebook.pdf** — PDF export of the executed Jupyter notebook for easy viewing and sharing without a Python environment.

### MATLAB Implementation Steps

Scipts Located in `MATLAB/` directory:

- **Load_and_display_DoE_table.m** — `(i)` Load the DoE table with data points for factors and each response and display the DoE table.
- **Compute_mean_effects_and_interactions.m** — `(ii)`Compute the mean and interaction effects of the factors on the response and display the results in bar charts.
- **Response_surface_and_contour_plots.m** — `(iii)` Perform a Response Surface Analysis and create a 3D response surface and 2D contour plots for the significant intercations.
- **MOGA_optimisation_with_polynomial.m** — `(iv)` Fit a second degree polynomial function to the response surface and perform a MOGA optimisation for the desired responses.
- **MATLAB_Livescript.mlx** — MATLAB Live Script combining all four modelling steps (i)–(iv) with inline results and figures; displays the computed parameters and plots directly in the Live Script.
- **MATLAB_Livescript.pdf** — PDF export of the executed MATLAB Live Script for easy viewing and sharing without a MATLAB environment.

## Key Concepts

### Design Variables

The stirred tank reactor model considers four key design factors:

1. **Baffle Thickness** (10-50 mm)
2. **Baffle Length** (50-200 mm)
3. **Blade Chord** (75-110 mm)
4. **Impeller RPM** (25-150 rpm)

### Response Variables 

The simulations and experiments typically measure:

- Mixing time (CFD Analysis)
- Power consumption (CFD Analysis)
- Safety factor (FEA Analysis)
- Flow patterns (CFD Analysis)
- Velocity distributions (CFD Analysis)

### CFD Simulation Data

The CFD data is stored in `Python/` and `MATLAB/`directory:

- Mixing_tank_single_response.xlsx (Contains all Predictors and the first Response e.g. "Average-k")
- Mixing_tank_multiple_responses.xlsx (Contains all Predictors and all Responses)

## Usage

### Python

#### Step (i) — Loading the CCD table from Excel File
```bash
cd Chapter 05 - Stirred Tank Reactor/Python
python Load_and_display_DoE_table.py
```

The script will:
1. Load the CCD DoE table generated with ANSYS DesignXplorer for all factors and each response 
2. Display the DoE table

#### Step (ii) — Compute Mean Effects and Interactions for Each Response.
```bash
python Compute_mean_effects_and_interactions.py
```

Compute the mean and interaction effects of the factors on the response, perform ANOVA to idenity significant terms.

#### Step (iii) — Create 3D Response Surface and Contour Plots for Two Most Significant Factors
```bash
python Response_surface_and_contour_plots.py
``

Perform a response surface analysis and create a 3D response surface and 2D contour plots for the significant intercations.

#### Step (iv) — Perform a MOGA optimisation using polynomial function for the desired responses
```matlab
python MOGA_optisation_with_polynomial.py
``

Fit a second degree polynomial function to the response surface and perform a MOGA optimisation for the desired responses.

#### Step (v) — Run Interactive Jupyter Notebook
```bash
jupyter notebook Jupyter_notebook.ipynb
```

Runs all four steps interactively with inline outputs and plots.

### MATLAB

#### Step (i) — Load the CCD table from Excel File
```matlab
run('Load_and_display_DoE_table.m')
```

The script will:
1. Load the CCD DoE table generated with ANSYS DesignXplorer for all factors and each response 
2. Display the DoE table

#### Step (ii) — Compute Mean Effects and Interactions for Each Response
```matlab
run('Compute_mean_effects_and_interactions.m')
``

Compute the mean and interaction effects of the factors on the response, perform ANOVA to idenity significant terms.

#### Step (iii) — Create 3D Response Surface and Contour Plots for Two Most Significant Factors
```matlab
run('Response_surface_and_contour_plots.m')
``

Perform a response surface Analysis and create a 3D response surface and 2D contour plots for the significant intercations.

#### Step (iv) — Perform a MOGA optimisation using polynomial function on the desired responses
```matlab
run('MOGA_optisation_with_polynomial.m')
``

Fit a second degree polynomial function to the response surface and perform a MOGA optimisation for the desired responses.

#### #### Step (vi) — Run Interactive MATLAB Livescript
```matlab
run('MATLAB_livescript.mlx')
```

Runs all four steps interactively with inline outputs and plots.


## Python Dependencies

- `numpy`
- `scipy`
- `pandas`
- `matplotlib`
- `statsmodels`
- `numba`
- `scikit-learn`
- `seaborn`
- `pyDOE3`
- `itertools`


Install using:
```bash
pip install -r requirements.txt

### MATLAB Toolboxes
- Statistics and Machine Learning Toolbox
- Global Optimisation Toolbox (for advanced optimisation strategies)

## Model Description

The model describes the numerical simulation and design optimisation of hydrodynamics and mixing processes in a Stirred Tank Reactor (STR). It captures the complex relationship between:

- Reactor geometry (baffle thickness, baffle length, blade chord) and mixing performance
- Impeller speed and flow characteristics (flow patterns, velocity distributions)
- Design variables and response variables (average k, power consumption, safety factor, mixing time)
- Multi-objective trade-offs between competing performance criteria

### Key Features

1. **Design of Experiments** — Central Composite Design (CCD) generated with ANSYS DesignXplorer for systematic exploration of the design space
2. **Statistical Analysis** — ANOVA and mean/interaction effect analysis to identify significant factors
3. **Response Surface Methodology** — 3D surface and 2D contour plots for visualising factor–response relationships
4. **Multi-Objective Optimisation** — MOGA with polynomial and kriging surrogate models for Pareto-optimal designs
5. **Cross-platform** — Equivalent implementations in both Python and MATLAB

## Workflow

1. **Design Phase** — Load the CCD DoE table using `Load_and_display_DoE_table`
2. **Effect Analysis** — Run `Compute_mean_effects_and_interactions` to compute mean and interaction effects and perform ANOVA
3. **Response Surface Analysis** — Run `Response_surface_and_contour_plots` to create 3D surface and 2D contour plots for significant interactions
4. **Optimisation** — Run `MOGA_optimisation_with_polynomial` or `MOGA_optimisation_with_kriging` to find Pareto-optimal operating parameters

## Input Data Format

The scripts expect CFD simulation data as Excel files:
- `Mixing_tank_single_response.xlsx` — Contains all predictors and the first response (e.g. "Average-k")
- `Mixing_tank_multiple_responses.xlsx` — Contains all predictors and all responses

Data files should be placed in the appropriate `Python/` or `MATLAB/` directory.

## Output

The scripts provide:
- DoE table display with all factors and responses
- Bar charts of mean and interaction effects for each response
- ANOVA results identifying statistically significant terms
- 3D response surface and 2D contour plots for the most significant factor interactions
- Pareto-optimal solutions from multi-objective optimisation

## Notes

- The Python and MATLAB scripts include commented sections for additional guidance.
- Adjust file paths in scripts if necessary to match your directory structure.
- For CFD simulations, you will need access to ANSYS Fluent or similar CFD software.
- The `Jupyter_notebook.pdf` and `MATLAB_Livescript.pdf` provide pre-executed outputs and can be used for reference without running any code.

## References

See the main book "Modelling and Optimisation of Fluid-Particle Processes in Environmental Engineering" for:
- Detailed numerical simulation strategies
- Generation of the Design of Experiments (DoE) table
- Analysis of the mixing characteristics
- Optimisation of the mixing performance
