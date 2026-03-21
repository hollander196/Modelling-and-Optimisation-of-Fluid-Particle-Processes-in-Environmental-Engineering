% (ii) Solving all the model equations in the mathematical model

% Parameters (replace with actual values from experimental data fitting)
clear; 
% Define the parameters (replace with actual values) 
n = 1.8213;               % Example value for n 
beta = 0.1671;            % Example value for beta 
tau = 0.3660;             % Example value for tau 
theta = 0.0264;           % Example value for theta 
epsilon_so = 0.3010;      % Example value for epsilon_so 
k_o = 1.8739e-14;         % Example value for k_o 
DeltaP = 50000;           % Example value for Delta P 
mu = 0.00089;             % Example value for mu 
Rm = 1.37e11;             % Example value for Rm 
s=0.00398;                % Example value for s 
rhos=1095.2;              % Example value for density of the feed 
 
A=1-n-beta; 
Pa=(tau*exp((DeltaP/1000)*theta))*1000; 
C=(2*mu*s*A)/(k_o*Pa*(epsilon_so-s)); 
l=0; 
y=zeros(54,2); 
 
% Define the time interval and step 
t_start = 100;         % start time 
t_end = 90*60;      % end time in seconds (90 minutes) 
dt = 100;            % time step in seconds 
 
% Initialize variables 
t_values = t_start:dt:t_end;  % time values 
J_values = zeros(size(t_values));  % initialize J values 
DeltaPc_values = zeros(size(t_values));  % initialize DeltaPc values 
 
for t=100:100:5400 
    
    % Solve the equations simultaneously using fsolve 
    % Initial guesses for [J, DeltaPc] 
    x0 = [100 5*10^(-5)]; 
     
    % Define the system of equations as a function handle 
    F = @(x)[ x(1) - DeltaP + (mu*Rm.*x(2));
              x(2) + ( ((1 + (x(1)/Pa))^A - 1) ./ sqrt( C * (((1 + (x(1)/Pa))^A) - 1) * t ) ) ]; 
     
    % Solve using fsolve 
    m = fsolve(F, x0); 
     
    % Extract solutions 
    y(l+1,1)=m(1); 
    y(l+1,2)=m(2); 
    l=l+1; 
     
    % Store the values 
    J_values = y(:,2); 
    DeltaPc_values= y(:,1);
end 

%% Solve for Ps, alpha, and epsilon for all DeltaPc values 
f_values = 0:0.01:1;  % Values for f 
Ps_values = zeros(length(f_values), length(DeltaPc_values)); 
alpha_values = zeros(length(f_values), length(DeltaPc_values)); 
epsilon_values = zeros(length(f_values), length(DeltaPc_values)); 
 
alpha_o = 1 / (epsilon_so * k_o * rhos); 
 
% Iterate over each DeltaPc_value 
for j = 1:length(DeltaPc_values) 
    DeltaPc_value = DeltaPc_values(j);  % Use the j-th calculated DeltaPc value 
    V = (1 + DeltaPc_value / Pa)^A; 
     
    % Iterate over f values 
    for i = 1:length(f_values) 
        f = f_values(i); 
         
        % Solve for Ps 
        Ps0 = 40000;  % Initial guess for Ps 
        F1 = @(Ps) (Ps / Pa) - (1 + (1 - f) * (V - 1))^(1 / A) + 1; 
        Ps_values(i, j) = fsolve(F1, Ps0); 
         
        % Solve for epsilon 
        epsilon0 = 0.5;  % Initial guess for epsilon 
        F2 = @(epsilon) epsilon - 1 + epsilon_so * ((V - 1) * (1 - f) + 1)^(beta / A); 
        epsilon_values(i, j) = fsolve(F2, epsilon0); 
         
        % Solve for alpha 
        alpha0 = 4e13;  % Initial guess for alpha 
        F3 = @(alpha) alpha - alpha_o * ((V - 1) * (1 - f) + 1)^(n / A); 
        alpha_values(i, j) = fsolve(F3, alpha0); 
    end 
end 


figure (1); % Plot Permeate flux vs Time 
plot(t_values / 60, J_values * 1000 * 3600);  % Convert t_values to minutes 
xlabel('Time (minutes)'); 
ylabel('Permeate flux (LMH)'); 
title('J vs Time'); 
grid on; 
 
figure (2); % Plot solid compressive pressure profile across the fouling layer 
hold on 
for r = [1 54] 
  plot(f_values,Ps_values(:,r)) 
  xlabel("Relative cake thickness, x/L") 
  ylabel("Solid compressive pressure (kPa)") 
  title('Ps vs Time') 
  grid on; 
end   
 
figure (3); % Plot porosity profile across the cake layer 
hold on 
for r = [1 54] 
  plot(f_values,epsilon_values(:,r)) 
  xlabel("Relative cake thickness, x/L") 
  ylabel("Porosity") 
  title('Porosity vs Time') 
   grid on; 
end  
hold off 