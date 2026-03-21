# A script that computes mean effects of individual factors and their pairwise interactions, perform ANOVA to identify significant effects, 
# and evaluates the goodness of fit for a linear model with main effects, interactions, and quadratic terms. 
# It also includes visualizations of mean effects and interactions.
# Assumptions:
# Excel file: 'Mixing_tank_1.xlsx' (in current folder) — adjust filename if needed.
# Desired factor names (human-friendly): 'Baffle Thickness', 'Baffle Length', 'Blade Chord', 'Impeller RPM'
# Response column name is 'Average k' (kept for compatibility).

# import necessary python packages and libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from statsmodels.formula.api import ols
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score

file_path = 'Mixing_tank_single_response.xlsx'  # Excel file located in the current directory
data = pd.read_excel(file_path) # Load the data from Excel

# Ensure column names have no leading/trailing whitespace and factor names match dataframe
data.columns = data.columns.str.strip()

# Define/clean factor and response names (adjust if needed)
factors = ['Baffle Thickness', 'Baffle Length', 'Blade Chord', 'Impeller RPM']   # corrected name (no trailing space)
response = 'Average k'

# Make sure the global factor_names (if present) is consistent with cleaned columns
try:
    factor_names = [fn.strip() for fn in factor_names]
except NameError:
    factor_names = factors.copy()

# Calculate mean effect of each factor
mean_effects = data[factor_names].mean()

# Plot mean effects as a bar chart
plt.figure(figsize=(8, 5))
sns.barplot(x=mean_effects.index, y=mean_effects.values, palette="viridis")
plt.title("Mean Effect of Each Factor")
plt.ylabel("Mean Value")
plt.xlabel("Factors")
plt.tight_layout()
plt.show()

# Calculate mean pairwise interaction effects for all pairs of factors
interaction_means = {}
for f1, f2 in combinations(factor_names, 2):
    interaction_term = data[f1] * data[f2]
    interaction_means[f"{f1}*{f2}"] = interaction_term.mean()

# Convert to Series for plotting
interaction_means_series = pd.Series(interaction_means)

# Plot mean interaction effects as a bar chart
plt.figure(figsize=(10, 6))
sns.barplot(x=interaction_means_series.index, y=interaction_means_series.values, palette="mako")
plt.title("Mean Interaction Effect of Factor Pairs")
plt.ylabel("Mean Interaction Value")
plt.xlabel("Factor Pairs")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Construct the formula for the linear model including main effects, 2-factor interactions, and quadratic terms
terms = factors.copy()

# Add 2-factor interactions
for combo in combinations(factors, 2):
    terms.append(f"{combo[0]}:{combo[1]}")

# Add quadratic terms
for f in factors:
    terms.append(f"I({f}**2)")

formula = f"{response} ~ " + " + ".join(terms)

# print("\n ANOVA completed and results printed on screen.")
print(formula)

# Fit the model and compute Type II ANOVA
# Drop missing values before fitting the model
data_clean = data.dropna()

# Basic validation and cleanup
if not isinstance(formula, str):
	raise TypeError("The variable 'formula' must be a string containing a valid statsmodels formula.")
data_clean.columns = data_clean.columns.str.strip()

# Many column names contain spaces; Patsy/statsmodels formulas require valid identifiers.
# Create a safe mapping (replace spaces and some other problematic chars) and update both the dataframe and the formula.
safe_map = {col: col.strip().replace(' ', '_').replace('-', '_') for col in data_clean.columns}
data_clean = data_clean.rename(columns=safe_map)

# Update the formula to use the safe column names
safe_formula = formula
for orig, safe in safe_map.items():
	# Replace occurrences of the original column name in the formula with the safe name.
	# Use simple replace because names may appear in interactions and I(...) terms.
	safe_formula = safe_formula.replace(orig, safe)

# Fit the model and compute Type II ANOVA (appropriate for balanced designs)
try:
	model = ols(safe_formula, data=data_clean).fit()
	anova_table = sm.stats.anova_lm(model, typ=2)  # Type II ANOVA
	print("Model fitted and ANOVA table computed successfully.")
except SyntaxError:
	print("SyntaxError encountered while parsing/executing the model fit. Check that the formula is valid after replacing spaces in column names.")
	raise
except Exception as e:
	print(f"Error fitting the model or computing ANOVA: {e}")
	raise

# Print the ANOVA table
print("\n ANOVA completed and results printed on screen.")
print(anova_table.round(4))

# Identify significant factors/interactions (p < 0.05)
alpha = 0.05
significant = anova_table[anova_table['PR(>F)'] < alpha]

print("\n========== SIGNIFICANT TERMS (p < 0.05) ==========")
if not significant.empty:
    print(significant.round(4))
else:
    print("No statistically significant effects found (p >= 0.05).")

# Print Model Summary and Goodness-of-Fit Statistics
# print("\n========== MODEL SUMMARY ==========")
print(model.summary())

print("\n========== GOODNESS OF FIT METRICS ==========")
print(f"R-squared       : {model.rsquared:.4f}")
print(f"Adj. R-squared  : {model.rsquared_adj:.4f}")
print(f"F-statistic     : {model.fvalue:.4f}")
print(f"p-value (Model) : {model.f_pvalue:.4e}")

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(model.model.endog, model.fittedvalues))
print(f"RMSE            : {rmse:.4f}")

# Repeat this sequence for the remaining responses in the Excel file 'Mixing_tank_multiple_responses.xlsx'
