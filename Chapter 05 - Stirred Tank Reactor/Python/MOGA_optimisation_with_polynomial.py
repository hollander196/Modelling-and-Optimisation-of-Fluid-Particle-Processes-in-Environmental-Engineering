# =============================================================================
#  A Script that Performs Multi-Objective Genetic Algorithm (NSGA-II-like)using Second Degree (Quadratic) Polynomial Regression
#  Excel file loaded from the same directory as this script
#  Excel file: first 4 cols = factors, last 4 = responses
#  Surrogate models (Polynomial Regression)
#  Objective: minimize Y3, while treating Y1 and Y4 as functional constraints subject to Y1 >= 0.03 and Y4 >= 2
# =============================================================================

# Load necessary python libraries
import numpy as np
import pandas as pd
import re
import random
from pathlib import Path
from sklearn.linear_model import LinearRegression


# Sanitise column names to valid Python identifiers
def make_valid_name(name: str) -> str:
    s = name.strip().replace(" ", "_")
    s = re.sub(r"[^0-9a-zA-Z_]", "_", s)
    if re.match(r"^\d", s):
        s = f"x_{s}"
    return s

# Fast non-dominated sorting (NSGA-II front construction)
def make_rsm_features(vars_):
    v = np.asarray(vars_, dtype=float).ravel()
    n = v.size
    linear = v
    quad = v ** 2
    inter = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            inter.append(v[i] * v[j])
    return np.concatenate([linear, quad, np.array(inter, dtype=float)]).reshape(1, -1)


def build_rsm_matrix(X):
    X = np.asarray(X, dtype=float)
    n = X.shape[1]
    linear = X
    quad = X ** 2
    inter_cols = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            inter_cols.append((X[:, i] * X[:, j]).reshape(-1, 1))
    inter = np.hstack(inter_cols) if inter_cols else np.empty((X.shape[0], 0))
    return np.hstack([linear, quad, inter])


def dominates(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return np.all(a <= b) and np.any(a < b)


def fast_non_dominated_sort(objs):
    N = objs.shape[0]
    S = [[] for _ in range(N)]
    n = np.zeros(N, dtype=int)
    fronts = []
    F = []

    for p in range(N):
        for q in range(N):
            if dominates(objs[p], objs[q]):
                S[p].append(q)
            elif dominates(objs[q], objs[p]):
                n[p] += 1
        if n[p] == 0:
            F.append(p)

    fronts.append(F)
    i = 0
    while len(fronts[i]) > 0:
        Q = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    Q.append(q)
        i += 1
        fronts.append(Q)

    if len(fronts[-1]) == 0:
        fronts.pop()
    return fronts


def crowding_distance(front, objs):
    l = len(front)
    if l == 0:
        return np.array([])
    dist = np.zeros(l, dtype=float)
    m = objs.shape[1]

    for k in range(m):
        vals = objs[front, k]
        idx = np.argsort(vals)
        sorted_vals = vals[idx]
        dist[idx[0]] = np.inf
        dist[idx[-1]] = np.inf
        fmin, fmax = sorted_vals[0], sorted_vals[-1]
        if fmax == fmin:
            continue
        for j in range(1, l - 1):
            dist[idx[j]] += (sorted_vals[j + 1] - sorted_vals[j - 1]) / (fmax - fmin)
    return dist


def tournament_selection(pop, objs):
    i1 = random.randint(0, len(pop) - 1)
    i2 = random.randint(0, len(pop) - 1)
    if dominates(objs[i1], objs[i2]):
        return pop[i1].copy()
    elif dominates(objs[i2], objs[i1]):
        return pop[i2].copy()
    else:
        lo, hi = min(i1, i2), max(i1, i2)
        return pop[random.randint(lo, hi)].copy()


# Simulated Binary Crossover (SBX)
def sbx_crossover(p1, p2, low, up, eta_c):
    c1 = p1.copy()
    c2 = p2.copy()
    for i in range(p1.size):
        if random.random() <= 0.5:
            if abs(p1[i] - p2[i]) > 1e-14:
                x1, x2 = min(p1[i], p2[i]), max(p1[i], p2[i])
                rand_beta = random.random()

                beta = 1 + 2 * (x1 - low[i]) / (x2 - x1)
                alpha = 2 - beta ** (-(eta_c + 1))
                if rand_beta <= 1 / alpha:
                    betaq = (rand_beta * alpha) ** (1 / (eta_c + 1))
                else:
                    betaq = (1 / (2 - rand_beta * alpha)) ** (1 / (eta_c + 1))
                c1[i] = 0.5 * ((x1 + x2) - betaq * (x2 - x1))

                beta = 1 + 2 * (up[i] - x2) / (x2 - x1)
                alpha = 2 - beta ** (-(eta_c + 1))
                if rand_beta <= 1 / alpha:
                    betaq = (rand_beta * alpha) ** (1 / (eta_c + 1))
                else:
                    betaq = (1 / (2 - rand_beta * alpha)) ** (1 / (eta_c + 1))
                c2[i] = 0.5 * ((x1 + x2) + betaq * (x2 - x1))

                c1[i] = min(max(c1[i], low[i]), up[i])
                c2[i] = min(max(c2[i], low[i]), up[i])
    return c1, c2

# Polynomial mutation
def polynomial_mutation(child, low, up, eta_m, mutprob):
    out = child.copy()
    for i in range(out.size):
        if random.random() < mutprob:
            x = out[i]
            delta1 = (x - low[i]) / (up[i] - low[i])
            delta2 = (up[i] - x) / (up[i] - low[i])
            rand_mut = random.random()
            mut_pow = 1 / (eta_m + 1)

            if rand_mut < 0.5:
                xy = 1 - delta1
                val = 2 * rand_mut + (1 - 2 * rand_mut) * (xy ** (eta_m + 1))
                deltaq = val ** mut_pow - 1
            else:
                xy = 1 - delta2
                val = 2 * (1 - rand_mut) + 2 * (rand_mut - 0.5) * (xy ** (eta_m + 1))
                deltaq = 1 - val ** mut_pow

            x = x + deltaq * (up[i] - low[i])
            out[i] = min(max(x, low[i]), up[i])
    return out


# Evaluate a candidate solution using trained surrogate models
# Obj1 = predicted Y3 (minimize)
# Obj2 = total constraint violation for Y1 and Y4 (minimize)
def evaluate_solution(x, models, y1_col, y3_col, y4_col, y1_lb, y4_lb):
    feat = make_rsm_features(x)
    preds = {name: float(model.predict(feat)[0]) for name, model in models.items()}
    y3 = preds[y3_col]
    viol = max(0.0, y1_lb - preds[y1_col]) + max(0.0, y4_lb - preds[y4_col])
    return np.array([y3, viol], dtype=float), preds


# Get script directory safely (works in script + notebook contexts)
def get_base_dir() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()


def main():
    # Reproducibility
    np.random.seed(42)
    random.seed(42)

    # Input file (script dir if .py, cwd if notebook)
    script_dir = get_base_dir()
    file_path = script_dir / "Mixing_tank_multiple_responses.xlsx"

    # Read and clean data
    data = pd.read_excel(file_path)
    data.columns = [make_valid_name(c) for c in data.columns]
    data = data.dropna().reset_index(drop=True)

    predictors = list(data.columns[:4])
    responses = list(data.columns[4:8])
    y1_col = responses[0]
    y3_col = responses[2]
    y4_col = responses[3]

    X = data[predictors].to_numpy(dtype=float)
    X_rsm = build_rsm_matrix(X)

    # Fit one Surrogate model per response and compute training metrics
    models = {}
    metrics = {}
    for r in responses:
        y = data[r].to_numpy(dtype=float)
        mdl = LinearRegression(fit_intercept=True)
        mdl.fit(X_rsm, y)
        y_pred = mdl.predict(X_rsm)
        mse = float(np.mean((y - y_pred) ** 2))
        r2 = float(1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))
        models[r] = mdl
        metrics[r] = {"MSE": mse, "R2": r2}

    metrics_flat = pd.DataFrame({
        "Response": list(metrics.keys()),
        "R2": [metrics[k]["R2"] for k in metrics.keys()],
        "MSE": [metrics[k]["MSE"] for k in metrics.keys()],
    })
    print("Surrogate model metrics (R2 and MSE):")
    print(metrics_flat)

    # GA parameters
    POP_SIZE = 100
    NGEN = 80
    CXPB = 0.9
    MUTPB = 0.2
    ETA_C = 20.0
    ETA_M = 20.0

    bounds = np.vstack([X.min(axis=0), X.max(axis=0)]).T
    low = bounds[:, 0]
    up = bounds[:, 1]

    y1_lb = 0.03
    y4_lb = 2.0

    # Initialize population
    pop = []
    objs = np.zeros((POP_SIZE, 2), dtype=float)
    preds = []
    for i in range(POP_SIZE):
        ind = low + (up - low) * np.random.rand(low.size)
        pop.append(ind)
        objs[i], p = evaluate_solution(ind, models, y1_col, y3_col, y4_col, y1_lb, y4_lb)
        preds.append(p)

    # Evolution loop
    for _ in range(NGEN):
        offspring = []
        while len(offspring) < POP_SIZE:
            p1 = tournament_selection(pop, objs)
            p2 = tournament_selection(pop, objs)

            if random.random() < CXPB:
                c1, c2 = sbx_crossover(p1, p2, low, up, ETA_C)
            else:
                c1, c2 = p1.copy(), p2.copy()

            c1 = polynomial_mutation(c1, low, up, ETA_M, MUTPB)
            c2 = polynomial_mutation(c2, low, up, ETA_M, MUTPB)
            offspring.append(c1)
            if len(offspring) < POP_SIZE:
                offspring.append(c2)

        off_objs = np.zeros((POP_SIZE, 2), dtype=float)
        off_preds = []
        for i in range(POP_SIZE):
            off_objs[i], p = evaluate_solution(offspring[i], models, y1_col, y3_col, y4_col, y1_lb, y4_lb)
            off_preds.append(p)

        combined = pop + offspring
        combined_objs = np.vstack([objs, off_objs])
        combined_preds = preds + off_preds

        fronts = fast_non_dominated_sort(combined_objs)

        new_pop, new_objs, new_preds = [], [], []
        for front in fronts:
            if len(new_pop) + len(front) > POP_SIZE:
                dist = crowding_distance(front, combined_objs)
                idx_sorted = np.argsort(-dist)
                for k in idx_sorted:
                    idx = front[k]
                    if len(new_pop) < POP_SIZE:
                        new_pop.append(combined[idx])
                        new_objs.append(combined_objs[idx])
                        new_preds.append(combined_preds[idx])
                    else:
                        break
                break
            else:
                for idx in front:
                    new_pop.append(combined[idx])
                    new_objs.append(combined_objs[idx])
                    new_preds.append(combined_preds[idx])

        pop = new_pop
        objs = np.array(new_objs, dtype=float)
        preds = new_preds

    # Pareto front
    fronts = fast_non_dominated_sort(objs)
    pf_idx = fronts[0]

    rows = []
    for i in pf_idx:
        row = {predictors[j]: pop[i][j] for j in range(len(predictors))}
        row["Y3_obj"] = objs[i, 0]
        row["total_violation"] = objs[i, 1]
        row.update(preds[i])
        rows.append(row)

    pareto_tbl = pd.DataFrame(rows)
    feasible = pareto_tbl[pareto_tbl["total_violation"] <= 1e-9].copy()

    if feasible.empty:
        feasible = pareto_tbl.nsmallest(min(10, len(pareto_tbl)), "total_violation").copy()

    best = feasible.nsmallest(1, "Y3_obj").copy()

    # Save outputs to Excel (same directory as this script)
    out_file = script_dir / "MOGA_Poly_Optimisation_results.xlsx"

    with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
        pareto_tbl.to_excel(writer, sheet_name="Pareto_front", index=False)
        feasible.to_excel(writer, sheet_name="Feasible", index=False)
        metrics_flat.to_excel(writer, sheet_name="Model_Metrics", index=False)

    # Print surrogate model statistics and the feasible solution
    print("\nBest feasible solution:")
    cols = predictors + responses + ["total_violation"]
    cols = [c for c in cols if c in best.columns]
    print(best[cols])
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()