% A script that loads Excel CFD Data File from a Directory and display the DoE Table
% Install the required MATLAB Toolboxes if you haven't already:
% - Statistics and Machine Learning Toolbox (for DoE analysis)

  file_path = 'Mixing_tank_single_response.xlsx';  % If the file is in the current directory
data = readtable(file_path);  % Load the data from Excel

fprintf('\n========== DATA PREVIEW (First 5 Rows) ==========\n');
disp(data(1:5, :));

fprintf('\n========== FULL CCD TABLE ==========\n');
fprintf('Number of runs: %d\n', height(data));
fprintf('Number of variables: %d\n\n', width(data));
disp(data);