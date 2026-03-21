% =============================================================================
%  A Script that Performs Multi-Objective Genetic Algorithm (NSGA-II-like)using Second Degree (Quadratic) Polynomial Regression
%  Excel file loaded from the same directory as this script
%  Excel file: first 4 cols = factors, last 4 = responses
%  Surrogate models (Polynomial Regression)
%  Objective function: minimize Y3, while treating Y1 and Y4 as functional constraints subject to Y1 >= 0.03 and Y4 >= 2
% =============================================================================

rng(42); % For reproducibility

% Load the CCD table with factors and all responses (same folder as script)
this_file = mfilename('fullpath');
if isempty(this_file)
    this_file = matlab.desktop.editor.getActiveFilename; % fallback for interactive/editor runs
end
script_folder = fileparts(this_file);
file_path = fullfile(script_folder, 'Mixing_tank_multiple_responses.xlsx');

if ~isfile(file_path)
    error('Excel file not found: %s', file_path);
end

% Read data
data = readtable(file_path);
data.Properties.VariableNames = matlab.lang.makeValidName(strrep(strtrim(data.Properties.VariableNames), ' ', '_'));
predictors = data.Properties.VariableNames(1:4);
responses = data.Properties.VariableNames(5:8);
Y1_col = responses{1};
Y3_col = responses{3};
Y4_col = responses{4};

% Remove rows with NaN
data = rmmissing(data);

% --- Build RSM features ---
X = data{:, predictors};
n = size(X,2);
linear = X;
quad = X.^2;
inter = [];
for i = 1:n-1
    for j = i+1:n
        inter = [inter, X(:,i).*X(:,j)];
    end
end
X_rsm = [linear, quad, inter];

% --- Fit surrogate models (Quadratic Polynomial Regression) ---
models = struct();
metrics = struct();
for i = 1:numel(responses)
    y = data{:, responses{i}};
    mdl = fitlm(X_rsm, y);
    y_pred = predict(mdl, X_rsm);
    models.(responses{i}) = mdl;
    metrics.(responses{i}).MSE = mean((y - y_pred).^2);
    metrics.(responses{i}).R2 = 1 - sum((y - y_pred).^2)/sum((y - mean(y)).^2);
end

disp('Surrogate model metrics:');
disp(struct2table(metrics))

% --- Prediction from variables ---
predict_from_vars = @(vars) structfun(@(mdl) predict(mdl, make_rsm_features(vars)), models, 'UniformOutput', false);

function feat = make_rsm_features(vars)
    n = numel(vars);
    linear = vars(:)';
    quad = (vars(:)').^2;
    inter = [];
    for i = 1:n-1
        for j = i+1:n
            inter = [inter, vars(i)*vars(j)];
        end
    end
    feat = [linear, quad, inter];
end

% --- GA parameters ---
POP_SIZE = 100;
NGEN = 80;
CXPB = 0.9;
MUTPB = 0.2;
ETA_C = 20.0;
ETA_M = 20.0;

bounds = [min(X); max(X)]';

Y1_lb = 0.03;
Y4_lb = 2.0;

% --- Evaluate solution ---
function [obj, preds] = evaluate_solution(x, models, Y1_col, Y3_col, Y4_col, Y1_lb, Y4_lb, predictors)
    feat = make_rsm_features(x);
    preds = struct();
    fields = fieldnames(models);
    for i = 1:numel(fields)
        preds.(fields{i}) = predict(models.(fields{i}), feat);
    end
    y3 = preds.(Y3_col);
    viol = max(0, Y1_lb - preds.(Y1_col)) + max(0, Y4_lb - preds.(Y4_col));
    obj = [y3, viol];
end

% --- GA helper functions ---
function flag = dominates(a, b)
    flag = all(a <= b) && any(a < b);
end

function fronts = fast_non_dominated_sort(objs)
    N = size(objs,1);
    S = cell(N,1);
    n = zeros(N,1);
    rank = zeros(N,1);
    fronts = {};
    F = [];
    for p = 1:N
        S{p} = [];
        n(p) = 0;
        for q = 1:N
            if dominates(objs(p,:), objs(q,:))
                S{p} = [S{p}, q];
            elseif dominates(objs(q,:), objs(p,:))
                n(p) = n(p) + 1;
            end
        end
        if n(p) == 0
            rank(p) = 1;
            F = [F, p];
        end
    end
    fronts{1} = F;
    i = 1;
    while ~isempty(fronts{i})
        Q = [];
        for p = fronts{i}
            for q = S{p}
                n(q) = n(q) - 1;
                if n(q) == 0
                    rank(q) = i + 1;
                    Q = [Q, q];
                end
            end
        end
        i = i + 1;
        fronts{i} = Q;
    end
    if isempty(fronts{end})
        fronts(end) = [];
    end
end

function dist = crowding_distance(front, objs)
    l = numel(front);
    dist = zeros(l,1);
    m = size(objs,2);
    for k = 1:m
        vals = objs(front,k);
        [sorted, idx] = sort(vals);
        dist(idx(1)) = inf;
        dist(idx(end)) = inf;
        fmin = sorted(1);
        fmax = sorted(end);
        if fmax == fmin
            continue;
        end
        for j = 2:l-1
            dist(idx(j)) = dist(idx(j)) + (sorted(j+1) - sorted(j-1))/(fmax - fmin);
        end
    end
end

function sel = tournament_selection(pop, objs)
    i1 = randi(numel(pop));
    i2 = randi(numel(pop));
    if dominates(objs(i1,:), objs(i2,:))
        sel = pop{i1};
    elseif dominates(objs(i2,:), objs(i1,:))
        sel = pop{i2};
    else
        sel = pop{randi([min(i1,i2), max(i1,i2)])};
    end
end

function [c1, c2] = sbx_crossover(p1, p2, low, up, eta_c)
    c1 = p1;
    c2 = p2;
    for i = 1:numel(p1)
        if rand <= 0.5
            if abs(p1(i) - p2(i)) > 1e-14
                x1 = min(p1(i), p2(i));
                x2 = max(p1(i), p2(i));
                rand_beta = rand;
                beta = 1 + 2*(x1 - low(i))/(x2 - x1);
                alpha = 2 - beta^-(eta_c+1);
                if rand_beta <= 1/alpha
                    betaq = (rand_beta*alpha)^(1/(eta_c+1));
                else
                    betaq = (1/(2 - rand_beta*alpha))^(1/(eta_c+1));
                end
                c1(i) = 0.5*((x1 + x2) - betaq*(x2 - x1));
                beta = 1 + 2*(up(i) - x2)/(x2 - x1);
                alpha = 2 - beta^-(eta_c+1);
                if rand_beta <= 1/alpha
                    betaq = (rand_beta*alpha)^(1/(eta_c+1));
                else
                    betaq = (1/(2 - rand_beta*alpha))^(1/(eta_c+1));
                end
                c2(i) = 0.5*((x1 + x2) + betaq*(x2 - x1));
                c1(i) = min(max(c1(i), low(i)), up(i));
                c2(i) = min(max(c2(i), low(i)), up(i));
            end
        end
    end
end

function child = polynomial_mutation(child, low, up, eta_m, mutprob)
    for i = 1:numel(child)
        if rand < mutprob
            x = child(i);
            delta1 = (x - low(i))/(up(i) - low(i));
            delta2 = (up(i) - x)/(up(i) - low(i));
            rand_mut = rand;
            mut_pow = 1/(eta_m + 1);
            if rand_mut < 0.5
                xy = 1 - delta1;
                val = 2*rand_mut + (1 - 2*rand_mut)*xy^(eta_m+1);
                deltaq = val^mut_pow - 1;
            else
                xy = 1 - delta2;
                val = 2*(1 - rand_mut) + 2*(rand_mut - 0.5)*xy^(eta_m+1);
                deltaq = 1 - val^mut_pow;
            end
            x = x + deltaq*(up(i) - low(i));
            child(i) = min(max(x, low(i)), up(i));
        end
    end
end

% --- GA main ---
low = bounds(:,1)';
up = bounds(:,2)';
pop = cell(POP_SIZE,1);
objs = zeros(POP_SIZE,2);
preds = cell(POP_SIZE,1);
for i = 1:POP_SIZE
    ind = low + (up - low).*rand(1, numel(low));
    pop{i} = ind;
    [objs(i,:), preds{i}] = evaluate_solution(ind, models, Y1_col, Y3_col, Y4_col, Y1_lb, Y4_lb, predictors);
end

for gen = 1:NGEN
    offspring = {};
    while numel(offspring) < POP_SIZE
        p1 = tournament_selection(pop, objs);
        p2 = tournament_selection(pop, objs);
        if rand < CXPB
            [c1, c2] = sbx_crossover(p1, p2, low, up, ETA_C);
        else
            c1 = p1;
            c2 = p2;
        end
        c1 = polynomial_mutation(c1, low, up, ETA_M, MUTPB);
        c2 = polynomial_mutation(c2, low, up, ETA_M, MUTPB);
        offspring{end+1} = c1;
        offspring{end+1} = c2;
    end
    off_objs = zeros(POP_SIZE,2);
    off_preds = cell(POP_SIZE,1);
    for i = 1:POP_SIZE
        [off_objs(i,:), off_preds{i}] = evaluate_solution(offspring{i}, models, Y1_col, Y3_col, Y4_col, Y1_lb, Y4_lb, predictors);
    end
    combined = [pop; offspring'];
    combined_objs = [objs; off_objs];
    combined_preds = [preds; off_preds];
    fronts = fast_non_dominated_sort(combined_objs);
    new_pop = {};
    new_objs = [];
    new_preds = {};
    for f = 1:numel(fronts)
        front = fronts{f};
        if numel(new_pop) + numel(front) > POP_SIZE
            dist = crowding_distance(front, combined_objs);
            [~, idx_sorted] = sort(dist, 'descend');
            for k = 1:numel(idx_sorted)
                idx = front(idx_sorted(k));
                if numel(new_pop) < POP_SIZE
                    new_pop{end+1} = combined{idx};
                    new_objs = [new_objs; combined_objs(idx,:)];
                    new_preds{end+1} = combined_preds{idx};
                end
            end
            break;
        else
            for k = 1:numel(front)
                idx = front(k);
                new_pop{end+1} = combined{idx};
                new_objs = [new_objs; combined_objs(idx,:)];
                new_preds{end+1} = combined_preds{idx};
            end
        end
    end
    pop = new_pop(:);
    objs = new_objs;
    preds = new_preds(:);
end

% Pareto front
fronts = fast_non_dominated_sort(objs);
pf_idx = fronts{1};
pf = cell(numel(pf_idx),1);
for i = 1:numel(pf_idx)
    pf{i}.vars = pop{pf_idx(i)};
    pf{i}.obj = objs(pf_idx(i),:);
    pf{i}.preds = preds{pf_idx(i)};
end

% Convert to table
pareto_tbl = table();
for i = 1:numel(pf)
    row = array2table(pf{i}.vars, 'VariableNames', predictors);
    preds_row = struct2table(pf{i}.preds);
    row.Y3_obj = pf{i}.obj(1);
    row.total_violation = pf{i}.obj(2);
    pareto_tbl = [pareto_tbl; [row, preds_row]];
end

feasible = pareto_tbl(pareto_tbl.total_violation <= 1e-9, :);
if isempty(feasible)
    [~, idx] = mink(pareto_tbl.total_violation, 10);
    feasible = pareto_tbl(idx, :);
end

[~, best_idx] = min(feasible.Y3_obj);
best = feasible(best_idx, :);

% Save results
[folder, ~, ~] = fileparts(file_path);
out_file = fullfile(folder, 'MOGA_Optimisation_Poly_results.xlsx');
writetable(pareto_tbl, out_file, 'Sheet', 'Pareto_front');
writetable(feasible, out_file, 'Sheet', 'Feasible');
metrics_tbl = struct2table(metrics);

% Convert nested struct columns to a flat table for R2 and MSE
fields = metrics_tbl.Properties.VariableNames;
n_fields = numel(fields);
Response = fields';
R2 = zeros(n_fields,1);
MSE = zeros(n_fields,1);
for i = 1:n_fields
    R2(i) = metrics_tbl.(fields{i}).R2;
    MSE(i) = metrics_tbl.(fields{i}).MSE;
end
metrics_flat = table(Response, R2, MSE);

writetable(metrics_flat, out_file, 'Sheet', 'Model_Metrics');

% Print surrogate model statistics (R2 and MSE) and the feasible solution
disp('Surrogate model metrics (R2 and MSE):');
disp(metrics_flat(:, {'Response', 'R2', 'MSE'}));

disp('Best feasible solution:');
disp(best(:, [predictors, responses, {'total_violation'}]));
fprintf('\nResults saved to %s\n', out_file);