# A Script that Creates Response Surface and Contour Plots from ANOVA Analysis using OLS regression on a dataset read from an Excel file.
# Print the model formular with Linear and Quadratic terms, and the two most significant predictors (main effects) with their p-values
# Assumptions:
# Excel file: 'Mixing_tank_1.xlsx' (in current folder) — adjust filename if necessary
# Desired factor names (human-friendly): 'Baffle Thickness', 'Baffle Length', 'Blade Chord', 'Impeller RPM'
# Response column name is 'Average k' (kept for compatibility)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import re
import sys
from pathlib import Path


xlsx_file = "Mixing_tank_single_response.xlsx"
sheet_name = 0  # 0 = first sheet, or use sheet name string

# Optional: set explicitly. If None, script auto-selects numeric columns.
response_var = None            # e.g., "Average k"
predictor_vars = None          # e.g., ["Baffle Thickness", "Baffle Length", "Blade Chord", "Impeller RPM"]

n_grid = 40

def make_safe_names(columns):
    """Create valid Python/statsmodels variable names and keep uniqueness."""
    used = set()
    mapping = {}
    for col in columns:
        safe = re.sub(r"\W|^(?=\d)", "_", str(col))
        if not safe:
            safe = "col"
        base = safe
        k = 1
        while safe in used:
            safe = f"{base}_{k}"
            k += 1
        used.add(safe)
        mapping[col] = safe
    return mapping

def load_excel_checked(file_name, sheet):
    """Load Excel file with clearer error messages."""
    path = Path(file_name)

    if not path.exists():
        raise FileNotFoundError(
            f"Excel file not found: '{path}'.\n"
            f"Tip: Use an absolute path or place the file in the current working directory:\n"
            f"  {Path.cwd()}"
        )

    try:
        return pd.read_excel(path, sheet_name=sheet)
    except ValueError as e:
        # Common for invalid sheet name/index
        raise ValueError(
            f"Could not read sheet '{sheet}' from '{path.name}'. "
            f"Check that the sheet name/index is correct."
        ) from e
    except Exception as e:
        raise RuntimeError(f"Failed to read Excel file '{path}': {e}") from e


def validate_columns(data_raw, response, predictors):
    """Validate configured response/predictor names and choose defaults."""
    all_cols = data_raw.columns.tolist()
    numeric_cols = data_raw.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 3:
        raise ValueError(
            "Need at least 3 numeric columns in the dataset "
            "(1 response + at least 2 predictors)."
        )

    # Response selection
    if response is None:
        local_response = numeric_cols[-1]
    else:
        if response not in all_cols:
            raise ValueError(
                f"Configured response_var '{response}' was not found.\n"
                f"Available columns: {all_cols}"
            )
        local_response = response

    # Predictor selection
    if predictors is None:
        local_predictors = [c for c in numeric_cols if c != local_response][:4]
    else:
        if not isinstance(predictors, (list, tuple)) or len(predictors) == 0:
            raise ValueError("predictor_vars must be a non-empty list/tuple of column names.")
        missing = [c for c in predictors if c not in all_cols]
        if missing:
            raise ValueError(
                f"Configured predictor column(s) not found: {missing}\n"
                f"Available columns: {all_cols}"
            )
        local_predictors = list(predictors)

    # Basic predictor checks
    local_predictors = [c for c in local_predictors if c != local_response]
    if len(local_predictors) < 2:
        raise ValueError("Need at least 2 predictors after validation.")

    # Numeric checks for modeling columns
    required_cols = [local_response] + local_predictors
    non_numeric = [c for c in required_cols if c not in numeric_cols]
    if non_numeric:
        raise ValueError(
            f"These required columns are not numeric: {non_numeric}. "
            f"Please choose numeric columns for response and predictors."
        )

    return local_response, local_predictors


def main():
    if not isinstance(n_grid, int) or n_grid < 2:
        raise ValueError("n_grid must be an integer >= 2.")

    # Load data with friendly path/sheet errors
    data_raw = load_excel_checked(xlsx_file, sheet_name)

    # Validate and choose columns
    local_response, local_predictors = validate_columns(data_raw, response_var, predictor_vars)

    # Keep required columns, drop missing rows
    required_cols = [local_response] + local_predictors
    data = data_raw[required_cols].dropna().copy()
    if data.empty:
        raise ValueError(
            "No data left after dropping rows with missing values in required columns: "
            f"{required_cols}"
        )

    # Rename to safe names for formula compatibility
    name_map = make_safe_names(data.columns)
    inv_map = {v: k for k, v in name_map.items()}
    data = data.rename(columns=name_map)

    response_safe = name_map[local_response]
    predictors_safe = [name_map[c] for c in local_predictors]

    # Build polynomial model:
    # response ~ main effects + pairwise interactions + squared terms
    main_terms = " + ".join(predictors_safe)
    interaction_terms = " + ".join(
        f"{a}:{b}" for i, a in enumerate(predictors_safe) for b in predictors_safe[i + 1:]
    )
    quadratic_terms = " + ".join(f"I({v}**2)" for v in predictors_safe)

    formula = f"{response_safe} ~ {main_terms}"
    if interaction_terms:
        formula += " + " + interaction_terms
    if quadratic_terms:
        formula += " + " + quadratic_terms

    mdl = smf.ols(formula=formula, data=data).fit()

    # Identify two most significant main-effect predictors
    pvals = mdl.pvalues
    main_effect_pvals = {
        v: pvals[v] for v in predictors_safe
        if v in pvals.index and pd.notna(pvals[v])
    }
    if len(main_effect_pvals) < 2:
        raise ValueError(
            "Could not find at least 2 valid main-effect predictors in the fitted model."
        )

    sig_predictors = sorted(main_effect_pvals, key=main_effect_pvals.get)[:2]
    p1, p2 = sig_predictors[0], sig_predictors[1]

    # Grid for the two significant predictors
    x1 = np.linspace(data[p1].min(), data[p1].max(), n_grid)
    x2 = np.linspace(data[p2].min(), data[p2].max(), n_grid)
    X1, X2 = np.meshgrid(x1, x2)

    # Prediction table: vary p1/p2, hold others at mean
    n_rows = X1.size
    predict_df = pd.DataFrame(index=np.arange(n_rows), columns=predictors_safe, dtype=float)

    for v in predictors_safe:
        if v == p1:
            predict_df[v] = X1.ravel()
        elif v == p2:
            predict_df[v] = X2.ravel()
        else:
            predict_df[v] = data[v].mean()

    y_hat = mdl.predict(predict_df).to_numpy()
    Yhat_grid = y_hat.reshape(n_grid, n_grid)

    # Plot: 3D surface + contour
    fig = plt.figure(figsize=(14, 6))
    fig.suptitle("3D Surface Plot and 2D Contour Plot", fontsize=14)

    # 3D surface
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    surf = ax1.plot_surface(X1, X2, Yhat_grid, cmap="viridis", edgecolor="none")
    ax1.set_xlabel(str(inv_map[p1]))
    ax1.set_ylabel(str(inv_map[p2]))
    ax1.set_zlabel(str(local_response))
    ax1.set_title("3D Surface Plot")
    ax1.view_init(elev=30, azim=135)
    fig.colorbar(surf, ax=ax1, shrink=0.6)

    # Contour
    ax2 = fig.add_subplot(1, 2, 2)
    c = ax2.contourf(X1, X2, Yhat_grid, levels=20, cmap="viridis")
    ax2.set_xlabel(str(inv_map[p1]))
    ax2.set_ylabel(str(inv_map[p2]))
    ax2.set_title("2D Contour Plot")
    fig.colorbar(c, ax=ax2)

    plt.tight_layout()
    plt.show()

    print("\nModel formula:")
    print(formula)
    print("\nMost significant predictors (main effects):")
    print(f"1) {inv_map[p1]} (p = {main_effect_pvals[p1]:.4g})")
    print(f"2) {inv_map[p2]} (p = {main_effect_pvals[p2]:.4g})")


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        print(f"\nError: {err}")
        sys.exit(1)

# Repeat this sequence for the remaining responses in the Excel file 'Mixing_tank_multiple_responses.xlsx'