%  A Script that computes the mean effects of individual factors and their pairwise interactions, perform ANOVA to identify significant effects, 
%` and evaluates the goodness of fit for a linear model with main effects, interactions, and quadratic terms. 
%` It also includes visualizations of mean effects and interactions.
%` Assumptions:
%` - Excel file: 'Mixing_tank_single_response.xlsx' (in current folder) — adjust filename if needed.
%` - Desired factor names: 'Baffle Thickness', 'Baffle Length', 'Blade Chord', 'Impeller RPM'
%` - Response column name is 'Average k' (kept for compatibility).

clear; clc;

%% Load data
filePath = 'Mixing_tank_single_response.xlsx';
opts = detectImportOptions(filePath, 'VariableNamingRule', 'preserve');
T = readtable(filePath, opts);

% Clean and standardize variable names
origNames = strtrim(T.Properties.VariableNames);
safeNames = matlab.lang.makeValidName(origNames, 'ReplacementStyle', 'underscore');
T.Properties.VariableNames = safeNames;

% Define factors and response (human-friendly source names)
factorsOrig = {'Baffle Thickness', 'Baffle Length', 'Blade Chord', 'Impeller RPM'};
responseOrig = 'Average k';

% Convert to safe names used by MATLAB table/formula
factors = matlab.lang.makeValidName(strtrim(factorsOrig), 'ReplacementStyle', 'underscore');
response = matlab.lang.makeValidName(strtrim(responseOrig), 'ReplacementStyle', 'underscore');

% Validate required columns
requiredVars = [factors, {response}];
missingVars = setdiff(requiredVars, T.Properties.VariableNames);
if ~isempty(missingVars)
    error('Missing required columns in Excel file: %s', strjoin(missingVars, ', '));
end

% Drop rows with missing values in model variables
dataClean = rmmissing(T(:, requiredVars));

%% Mean effect of each factor
meanEffects = mean(dataClean{:, factors}, 1, 'omitnan');

figure('Color', 'w');
bar(categorical(factorsOrig), meanEffects);
title('Mean Effect of Each Factor');
ylabel('Mean Value');
xlabel('Factors');
grid on;

%% Mean pairwise interaction effects
pairs = nchoosek(1:numel(factors), 2);
interactionMeans = zeros(size(pairs, 1), 1);
interactionLabels = cell(size(pairs, 1), 1);

for k = 1:size(pairs, 1)
    i = pairs(k, 1);
    j = pairs(k, 2);
    interactionTerm = dataClean{:, factors{i}} .* dataClean{:, factors{j}};
    interactionMeans(k) = mean(interactionTerm, 'omitnan');
    interactionLabels{k} = sprintf('%s*%s', factorsOrig{i}, factorsOrig{j});
end

figure('Color', 'w');
bar(categorical(interactionLabels), interactionMeans);
title('Mean Interaction Effect of Factor Pairs');
ylabel('Mean Interaction Value');
xlabel('Factor Pairs');
xtickangle(45);
grid on;

%% Build model formula: main effects + 2-factor interactions + quadratic terms
terms = factors;

% Add pairwise interaction terms
interactionTerms = cell(size(pairs, 1), 1);
for k = 1:size(pairs, 1)
    i = pairs(k, 1);
    j = pairs(k, 2);
    interactionTerms{k} = sprintf('%s:%s', factors{i}, factors{j});
end
terms = [terms, interactionTerms'];

% Add quadratic terms as explicit columns
quadTerms = cell(1, numel(factors));
for i = 1:numel(factors)
    sqName = [factors{i}, '_sq'];
    dataClean.(sqName) = dataClean.(factors{i}).^2;
    quadTerms{i} = sqName;
end
terms = [terms, quadTerms];

formula = sprintf('%s ~ %s', response, strjoin(terms, ' + '));
disp('Model formula:');
disp(formula);

%% Fit model and ANOVA
mdl = fitlm(dataClean, formula);

anovaTable = anova(mdl, 'summary');
disp(' ');
disp('ANOVA completed and results printed on screen.');
disp(anovaTable);

%% Significant terms (p < 0.05)
alpha = 0.05;
if ismember('pValue', anovaTable.Properties.VariableNames)
    sigIdx = anovaTable.pValue < alpha;
    significant = anovaTable(sigIdx, :);
else
    significant = table(); % fallback if version differs
end

disp(' ');
disp('========== SIGNIFICANT TERMS (p < 0.05) ==========');
if ~isempty(significant)
    disp(significant);
else
    disp('No statistically significant effects found (p >= 0.05).');
end

%% Model summary and goodness-of-fit
disp(mdl);

[pModel, FModel] = coefTest(mdl); % overall model significance test

fprintf('\n========== GOODNESS OF FIT METRICS ==========\n');
fprintf('R-squared       : %.4f\n', mdl.Rsquared.Ordinary);
fprintf('Adj. R-squared  : %.4f\n', mdl.Rsquared.Adjusted);
fprintf('F-statistic     : %.4f\n', FModel);
fprintf('p-value (Model) : %.4e\n', pModel);
fprintf('RMSE            : %.4f\n', mdl.RMSE);

% Repeat this sequence for the remaining responses in the Excel file 'Mixing_tank_multiple_responses.xlsx'