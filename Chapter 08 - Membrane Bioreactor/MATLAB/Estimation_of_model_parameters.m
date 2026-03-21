% (i) Estimation of the model parameters from experimental data

function Estimation_of_model_parameters()
% Input experimental data (replace with actual values if needed)
Exp_t = [300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600, 3900, 4200, 4500, 4800, 5100, 5400];

Exp_J1 = [5.22222E-05, 3.67778E-05, 3.00741E-05, 2.58333E-05, 2.32444E-05, 2.12222E-05, 1.96825E-05, 1.83333E-05, 1.73827E-05, 1.64889E-05, 1.56566E-05, 1.4963E-05, 1.44103E-05, 1.38889E-05, 1.34074E-05, 0.000013, 1.25621E-05, 1.22469E-05];

Exp_DeltaPc = [33481.93333, 35416.98438, 36238.53236, 36781.70458, 37091.31275, 37352.03542, 37538.26589, 37713.58443, 37827.31111, 37938.66142, 38044.58, 38132.84549, 38201.26429, 38265.72869, 38327.02956, 38375.57557, 38432.78838, 38470.06491];

% Initial guess for model parameters
 initial_bestx = [1.37, 0.15,0.3877,0.055, 1.89E-14, 0.3];
    

% Estimate model parameters using fminsearch
estimated_parameters = fminsearch(@(bestx) para_function(bestx, Exp_t, Exp_J1, Exp_DeltaPc), initial_bestx);

% Display estimated parameters
 fprintf('Estimated Parameters:\n');
 fprintf('n = %.4f\n', estimated_parameters(1));
 fprintf('beta = %.4f\n', estimated_parameters(2));
 fprintf('tau = %.4f\n', estimated_parameters(3));
 fprintf('theta = %.4f\n', estimated_parameters(4));
 fprintf('k0 = %.4e\n', estimated_parameters(5));
 fprintf('eps0 = %.4f\n', estimated_parameters(6));

 % Plot the fitted curve vs experimental data
 figure;
 plot(Exp_t / 60, Exp_J1, 'ro', 'MarkerFaceColor', 'r');  % Experimental data
 hold on;

 y_fit = model_funct(estimated_parameters, Exp_DeltaPc, Exp_t);  % Fitted model
 plot(Exp_t/ 60, y_fit, 'b-', 'LineWidth', 2);
 xlabel('Time (minutes)');
 ylabel('Permeate flux (m3/m2.s)');
 legend('Experimental Data', 'Model');
 title('Data Fitting for J vs t');
 grid on;

 % Calculate goodness of fit statistics
J_fit = model_funct(estimated_parameters, Exp_DeltaPc, Exp_t);
residuals = Exp_J1 - J_fit;
SSE = sum(residuals.^2);
SST = sum((Exp_J1 - mean(Exp_J1)).^2);
R_squared = 1 - SSE/SST;
RMSE = sqrt(mean(residuals.^2));

 % Nested function to define the model
     function J_fitted = model_funct(bestx, Exp_DeltaPc, Exp_t)
       
        n = bestx(1);
        beta =(bestx(2));
        tau = bestx(3);
        theta = bestx(4);
        k0 = bestx(5);
        eps0 = bestx(6);

        s = 0.00398; 
        mu = 0.00089;
        DeltaP=40000;
        
        A = 1 - n - beta;
        Pa=(tau * exp((DeltaP / 1000) * theta))* 1000;
        C = (2 * mu * s * A) / (k0 * Pa * (eps0 - s));
        
      
        J_fitted = -((((1 + (Exp_DeltaPc ./ Pa)).^A) - 1) ./ (sqrt(C * (((1 + (Exp_DeltaPc ./ Pa)).^A) - 1) .* Exp_t)));
        
    end
    % Nested function to define the cost function
    function para = para_function(bestx, Exp_t, Exp_J1, Exp_DeltaPc)
        J_model = model_funct(bestx, Exp_DeltaPc, Exp_t);
        para = sum((Exp_J1 - J_model).^2);  % Sum of squared errors
    end
  end