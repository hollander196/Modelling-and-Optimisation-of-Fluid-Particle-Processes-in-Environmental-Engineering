% (iv) Prediction of total pressure drop increase in a constant flux run

% Define the parameters (replace with actual values)
clear;
n = 1.35; 
beta = 0.15; 
Pa = 1100; 
k0 = 1.89 * 10^(-14); 
eps0 = 0.3; s = 0.00398; 
mu = 0.00089; 
J = 9.722 * 10^(-6); 
Rm = 1.12 * 10^(11);

A = 1 - n - beta;
C = (2 * mu * s * A) / (k0 * Pa * (eps0 - s));


% Define the time interval and step
t_start = 300;         % start time
t_end = 90*60;      % end time in seconds (90 minutes)
dt = 300;            % time step in seconds
t_values = t_start:dt:t_end;  % time values
t_values=t_values';
% Initialize cumulative P value
P_total = 0;
n = 0; % Loop index
y = zeros(18, 1);

for t = 300:300:5400
    % Define the function for fsolve
    F = @(P) (P - ((((((J^2) * C * t) + 1).^(1 / A)) - 1) * Pa) - (mu * Rm * J));
    
    % Initial guess for fsolve
    P0 = 100;
    
    % Solve for P
    P = fsolve(F, P0);
    
    % Accumulate P into P_total
    P_total = P_total + P;
    
    % Store current P and cumulative P_total if needed
    y(n + 1, 1) = P;            % Current P at each time step
    y(n + 1, 2) = P_total;       % Cumulative P value up to this step
    
    TotalP_values = y(:,2);
    n = n + 1;  % Increment index
end

% Plot P_total vs. t
figure;
plot(t_values/60,  TotalP_values/ 1000);  % t_values/60 to convert to minutes
xlabel('Time (minutes)');
ylabel('Total pressure drop (kPa)');
title('Total pressure drop vs. Time');


