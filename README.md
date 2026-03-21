# Modelling and Optimisation of Fluid-Particle Processes in Environmental Engineering

[![License](https://img.shields.io/badge/License-Custom-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)
[![MATLAB](https://img.shields.io/badge/MATLAB-R2018b%2B-orange.svg)](https://www.mathworks.com/)

## 📖 About

This repository contains the supplementary software codes and data for the book titled **"Modelling and Optimisation of Fluid-Particle Processes in Environmental Engineering"** published by Taylor & Francis. The repository provides detailed implementations in both Python and MATLAB to support the concepts and methodologies discussed in the book.

![image alt](https://github.com/hollander196/Modelling-and-Optimisation-of-Fluid-Particle-Processes-in-Environmental-Engineering/blob/64e54bd4cde4ab128440888931fc65a67df04084/Book%20cover.jpg)


## 📚 Book Description

The book covers advanced mathematical modeling and optimisation techniques for fluid-particle processes commonly encountered in environmental engineering applications. This repository specifically focuses on:

- **Chapter 5**: Numerical simulation and design optimisation of hydrodynamics and mixing process in a stirred tank reactor (STR)
- **Chapter 8**: Mathematical modelling of the effects compression of fouling layers as a predictive tool in membrane bioreactor systems

## 🗂️ Repository Structure

```
.
├── Chapter 05 - Stirred Tank Reactor/
│   ├── Python/              # Python implementations
│   ├── MATLAB/              # MATLAB implementations
│   └── README.md            # Chapter 5 specific documentation
├── Chapter 08 - Membrane Bioreactor/
│   ├── Python/              # Python implementations
│   ├── MATLAB/              # MATLAB implementations
│   └── README.md            # Chapter 8 specific documentation
├── Datasets                 # Experimental and CFD Datasets
├── Book cover.jpg
├── LICENSE
├── README.md
└── requirements.txt         # Python dependencies
```

## 🔧 Tools and Technologies

- **Excel** - Data storage and preliminary analysis
- **MATLAB** - Design of Experiments (DOE) and Response Surface optimisation, mathematical modelling, and optimisation
- **Python** - Design of Experiments (DOE), and Response Surface optimisation, mathematical modelling, and optimisation
- **Fluent** - Numerical simulation of mixing parameters

## 🚀 Getting Started

### Prerequisites

#### For Python
- Python 3.7 or higher
- pip package manager

#### For MATLAB
- MATLAB R2018b or higher
- Statistics and Machine Learning Toolbox
- Optimisation Toolbox

### Installation

#### Python Setup

1. Clone the repository:
```bash
git clone https://github.com/hollander196/Modelling-and-Optimisation-of-Fluid-Particle-Processes-in-Environmental-Engineering.git
cd Modelling-and-Optimisation-of-Fluid-Particle-Processes-in-Environmental-Engineering
```


2. Create a virtual environment (recommended):
```bash
# On Windows 
python -m venv venv
venv\Scripts\activate
```

```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. Install required Python packages:
```bash
pip install -r requirements.txt
```

#### MATLAB Setup

1. Clone the repository (same as above)
2. Open MATLAB and navigate to the repository directory
3. Add the relevant chapter folders to your MATLAB path

## 💻 Usage

### Chapter 5: Stirred Tank Reactor

#### Python Example
```python
# Navigate to Chapter 5 Python directory
cd Chapter 05 - Stirred Tank Reactor/Python

# Run Load and display DoE script
python Load_and_display_DoE.py

# Run Compute mean effects and interaction script
python Compute_mean_effects_and_interactions.py

# Run create response surface and contour script
python Create_response_surface_and_contour_plots.py

# Run MOGA optimisation with polynomial script
python MOGA_optimisation_with_polynomial.py

# Run the codes interactively in Jupyter Notebook on Jupyter server
jupyter notebook Jupyter_notebook.ipynb 

# Convert the notebook to HTML or PDF file for viewing or sharing (Optional)
jupyter nbconvert Jupyter_notebook.ipynb --to HTML
jupyter nbconvert Jupyter_notebook.ipynb --to PDF
```

#### MATLAB Example
```matlab
% Navigate to Chapter 5 MATLAB directory
cd Chapter 05 - Stirred Tank Reactor/MATLAB

% Run Load and display DoE script
run('Load_and_display_DoE.m')

% Run Compute mean effects and interaction script
run('Compute_mean_effects_and_interactions.m')

% Run create response surface and contour script
run('Create_response_surface_and_contour_plots.m')

% Run MOGA optimisation with polynomial script
run('MOGA_optimisation_with_polynomial.m')

% Run the codes interactively in MATLAB Livescript
run('MATLAB_livescript.mlx')

% Convert the livescript to HTML file for viewing or sharing (Optional)
export('MATLAB_livescript.mlx', 'MATLAB_livescript.HTML');
export('MATLAB_livescript.mlx', 'MATLAB_livescript.PDF');
```

### Chapter 8: Membrane Bioreactor

#### Python Example
```python
# Navigate to Chapter 8 Python directory
cd Chapter 08 - Membrane Bioreactor/Python

# Run the parameter estimation script
python Estimation_of_model_parameters.py

# Run the solving mathematical model equations script
python Solving_mathematical_model_equations.py

# Run prediction of flux decline script
python Prediction_of_flux_decline.py

# Run prediction of pressure drop script
python Prediction_of_pressure_drop.py

# Run the codes interactively in Jupyter Notebook on Jupyter server
jupyter notebook Jupyter_notebook.ipynb 

# Convert the notebook to HTML or PDF file for viewing or sharing (Optional)
jupyter nbconvert Jupyter_notebook.ipynb --to html
jupyter nbconvert Jupyter_notebook.ipynb --to PDF
```

#### MATLAB Example
```matlab
% Navigate to Chapter 8 MATLAB directory
cd Chapter 08 - Membrane Bioreactor/MATLAB

% Run the parameter estimation script
run('Estimation_of_model_parameters.m')

% Run the solving mathematical model equations script
run('Solving_mathematical_model_equations.m')

% Run prediction of flux decline script
run('Prediction_of_flux_decline.m')

% Run prediction of pressure drop script
run('Prediction_of_pressure_drop.m')

% Run the codes interactively in MATLAB Livescript
run('MATLAB_livescript.mlx')

% Convert the livescript to HTML file for viewing or sharing (Optional)
export('MATLAB_livescript.mlx', 'MATLAB_livescript.html');
export('MATLAB_livescript.mlx', 'MATLAB_livescript.PDF');
```

## 📊 Data Sources

The primary datasets for the mathematical modelling and optimisation described in Chapters 5 and 8 are included within the respective chapter directories. These datasets provide:

- Quantitative data from experimental measurements
- Quantitative data from numerical simulations
- Design of experiments (DOE) table
- Model validation data

## 🤝 Contributing

Contributions to improve the code implementations or documentation are welcome. Please feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## 📄 License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.

## 📧 Contact

For questions or issues related to the code implementations, please open an issue in the GitHub repository.

For questions about the book content, please contact the publisher: Taylor & Francis.

## 🙏 Acknowledgments

- Taylor & Francis for publishing the book
- Contributors to the open-source Python and MATLAB communities
- All researchers and practitioners who use these tools for environmental engineering applications

## 📚 Citation

If you use this code in your research, please cite the book:

```
[Benjamin Oyegbie, Suzanne Yala]. "Modelling and Optimisation of Fluid-Particle Processes in Environmental Engineering". Taylor & Francis, [2026]. DOI: https://doi.org/10.1201/9781003518426
```

---

**Note**: Some implementations may require commercial software license (e.g., ANSYS Fluent for CFD simulations, Mathworks MATLAB for Scientific Computations and Machine Learning). Please ensure you have appropriate licenses for such software.
