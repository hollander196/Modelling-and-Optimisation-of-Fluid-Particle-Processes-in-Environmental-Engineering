% (iii) Prediction of flux decline for a constant pressure run

% Define the parameters (replace with actual values)
n = 1.37;       % Example value for n
beta = 0.15;    % Example value for beta
tau = 0.2877;       % Example value for tau
theta = 0.022;     % Example value for theta
epsilon_so = 0.27; % Example value for epsilon_so
k_o = 1.89*10^(-14);       % Example value for k_o
DeltaP = 50000;    % Example value for Delta P
mu = 0.00089;        % Example value for mu
Rm = 1.37*10^11;        % Example value for Rm
s=0.00398;

A=1-n-beta;
Pa=(tau*exp((DeltaP/1000)*theta))*1000;
C=(2*mu*s*A)/(k_o*Pa*(epsilon_so-s));
n=0;
p=0;
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
    F=@(x)[x(1)-DeltaP+(mu*Rm.*x(2)),x(2)+((((1+(x(1)/Pa))^A)-1)./(sqrt(C*(((1+(x(1)./Pa))^A)-1).*t)))];
    
    % Solve using fsolve
    m =fsolve(F,x0);
    
    % Extract solutions
    y(n+1,1)=m(1);
    y(n+1,2)=m(2);
    n=n+1; % Increment index
    
    % Store the values
    J_values = y(:,2);
    DeltaPc_values= y(:,1);
end

% Plot J vs. t
figure;
plot(t_values/60, J_values * 1000 * 3600);  % t_values/60 to convert to minutes
xlabel('Time (minutes)');
ylabel('Permeate flux (LMH)');
title('Permeate flux vs. Time');
grid on;

