% A Script that Creates Response Surface and Contour Plots from ANOVA Analysis using OLS regression on a dataset read from an Excel file
% Print the model formular with Linear and Quadratic terms, and the two most significant predictors (main effects) with their p-values
%` Assumptions:
%` - Excel file: 'Mixing_tank_single_response.xlsx' (loaded via GUI picker)
%` - Desired factor names: 'Baffle Thickness', 'Baffle Length', 'Blade Chord', 'Impeller RPM'
%` - Response column name is 'Average k' (kept for compatibility)

clc; clear; close all;

%% 1) Resolve path from this script folder
script_fullpath = mfilename('fullpath');
if isempty(script_fullpath)
    script_dir = pwd; % fallback
else
    script_dir = fileparts(script_fullpath);
end
excel_path = fullfile(script_dir, "Mixing_tank_single_response.xlsx");

%% 2) Read Excel file
fprintf("Reading Excel file...\n");
data = readtable(excel_path);
fprintf("Excel file read successfully.\n");

%% 3) Drop rows with missing values
data_clean = rmmissing(data);
if width(data_clean) < 3
    error('Dataset must contain at least 2 predictors and 1 response column.');
end

%% 4) Build and fit second-order model (linear + interaction + quadratic)
predictorVars = data_clean.Properties.VariableNames(1:end-1);
responseVar   = data_clean.Properties.VariableNames{end};

mainTerms = strjoin(predictorVars, ' + ');
quadTerms = strjoin(strcat(predictorVars, '^2'), ' + ');

interactionTermsCell = {};
for i = 1:numel(predictorVars)-1
    for j = i+1:numel(predictorVars)
        interactionTermsCell{end+1} = [predictorVars{i} ':' predictorVars{j}]; %#ok<AGROW>
    end
end

if isempty(interactionTermsCell)
    interactionTerms = '';
else
    interactionTerms = strjoin(interactionTermsCell, ' + ');
end

formula = [responseVar ' ~ ' mainTerms];
if ~isempty(interactionTerms)
    formula = [formula ' + ' interactionTerms];
end
formula = [formula ' + ' quadTerms];

fprintf('\nModel Formula (Readable):\n');
fprintf('  Response      : %s\n', responseVar);
fprintf('  Linear terms  : %s\n', mainTerms);
if ~isempty(interactionTerms)
    fprintf('  Interactions  : %s\n', interactionTerms);
else
    fprintf('  Interactions  : (none)\n');
end
fprintf('  Quadratic     : %s\n', quadTerms);
fprintf('  Full formula  : %s\n\n', formula);

fprintf("Fitting second-order polynomial model...\n");
mdl = fitlm(data_clean, formula);

%% 5) Show most significant predictors (linear and quadratic main effects) with p < 0.05
alpha = 0.05;
coefTable = mdl.Coefficients;
coefNames = coefTable.Properties.RowNames;
pvalsAll  = coefTable.pValue;

isIntercept   = strcmp(coefNames, '(Intercept)');
isInteraction = contains(coefNames, ':');
isQuadratic   = contains(coefNames, '^2');
isLinearMain  = ismember(coefNames, predictorVars);

% Keep only linear and quadratic main effects (exclude intercept/interactions)
targetMask = ~isIntercept & ~isInteraction & (isLinearMain | isQuadratic) & ~isnan(pvalsAll);

termNames = coefNames(targetMask);
termPvals = pvalsAll(targetMask);

termType = strings(numel(termNames),1);
for k = 1:numel(termNames)
    if contains(termNames{k}, '^2')
        termType(k) = "Quadratic";
    else
        termType(k) = "Linear";
    end
end

sigMask = termPvals < alpha;
sigNames = termNames(sigMask);
sigPvals = termPvals(sigMask);
sigType  = termType(sigMask);

[sigPvalsSorted, idxSig] = sort(sigPvals, 'ascend');
sigNamesSorted = sigNames(idxSig);
sigTypeSorted  = sigType(idxSig);

fprintf('Most significant predictors (linear + quadratic main effects) with p < %.2f:\n', alpha);
if isempty(sigNamesSorted)
    fprintf('  None\n\n');
else
    for k = 1:numel(sigNamesSorted)
        fprintf('  %d) %s [%s]: p-value = %.6g\n', k, sigNamesSorted{k}, sigTypeSorted(k), sigPvalsSorted(k));
    end
    fprintf('\n');
end

%% 6) Select two linear predictors for plotting
% Prefer significant linear terms; fallback to top 2 linear terms by p-value
linearMask = ~isIntercept & ~isInteraction & isLinearMain & ~isnan(pvalsAll);
linearNames = coefNames(linearMask);
linearPvals = pvalsAll(linearMask);

if numel(linearNames) < 2
    error('At least two linear main-effect predictors are required for plotting.');
end

[linearPvalsSorted, idxLinear] = sort(linearPvals, 'ascend');
linearNamesSorted = linearNames(idxLinear);

sigLinearMask = linearPvalsSorted < alpha;
sigLinearNames = linearNamesSorted(sigLinearMask);

if numel(sigLinearNames) >= 2
    plotPredictors = sigLinearNames(1:2);
else
    plotPredictors = linearNamesSorted(1:2);
    fprintf('Note: Fewer than 2 linear predictors met p < %.2f. Using top 2 linear predictors for plots.\n\n', alpha);
end

%% 7) Create 3D response surface and 2D contour plots
nGrid = 40;
x1 = linspace(min(data_clean.(plotPredictors{1})), max(data_clean.(plotPredictors{1})), nGrid);
x2 = linspace(min(data_clean.(plotPredictors{2})), max(data_clean.(plotPredictors{2})), nGrid);
[X1, X2] = meshgrid(x1, x2);

nRows = numel(X1);
predictData = zeros(nRows, numel(predictorVars));

for i = 1:numel(predictorVars)
    if strcmp(predictorVars{i}, plotPredictors{1})
        predictData(:, i) = X1(:);
    elseif strcmp(predictorVars{i}, plotPredictors{2})
        predictData(:, i) = X2(:);
    else
        predictData(:, i) = mean(data_clean.(predictorVars{i}), 'omitnan');
    end
end

predictTable = array2table(predictData, 'VariableNames', predictorVars);
Yhat = predict(mdl, predictTable);
YhatGrid = reshape(Yhat, nGrid, nGrid);

figure('Name', '3D Surface Plot and 2D Contour Plot', 'NumberTitle', 'off');

subplot(1,2,1);
surf(X1, X2, YhatGrid, 'EdgeColor', 'none');
xlabel(plotPredictors{1}, 'Interpreter', 'none');
ylabel(plotPredictors{2}, 'Interpreter', 'none');
zlabel(responseVar, 'Interpreter', 'none');
title('3D Surface Plot');
colorbar;
view(135, 30);
shading interp;

subplot(1,2,2);
contourf(X1, X2, YhatGrid, 20, 'LineColor', 'none');
xlabel(plotPredictors{1}, 'Interpreter', 'none');
ylabel(plotPredictors{2}, 'Interpreter', 'none');
title('2D Contour Plot');
colorbar;

fprintf('Predictors used for plots:\n');
fprintf('  1) %s\n', plotPredictors{1});
fprintf('  2) %s\n', plotPredictors{2});

% Repeat this sequence for the remaining responses in the Excel file 'Mixing_tank_multiple_responses.xlsx'