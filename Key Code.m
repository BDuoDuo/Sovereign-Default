% Filename: Assignment_2.m
% Finite-Horizon Sovereign Default Model with Endowment Economy

clear;
clc;

% Part 2: Model

% Parameters
params.beta = 0.9;       % Discount factor
params.beta_star = 0.9;  % Lenders' discount factor
params.sigma = 2          % Relative risk aversion
params.pi = 0.6;          % Probability of high endowment
params.y_H = 1.5;         % High endowment
params.y_L = 0.5;         % Low endowment
params.phi_H = 0.3;       % Default cost in high state
params.phi_L = 0;         % Default cost in low state
params.T = 10;            % Number of periods
params.b_min = 0;         % Minimum debt level
params.b_max = 2;         % Maximum debt level
params.nb = 100;          % Number of grid points for debt
params.ny = 2;            % Number of possible income states

% Create grids
b_grid = linspace(params.b_min, params.b_max, params.nb)';
y_grid = [params.y_L; params.y_H];

% Initialize value functions and policy functions
V = zeros(params.nb, params.ny, params.T+1);    % Value function
V_R = zeros(params.nb, params.ny, params.T+1);  % Value of repayment
V_D = zeros(params.ny, params.T+1);             % Value of default
d_policy = zeros(params.nb, params.ny, params.T+1);  % Default policy
b_policy = zeros(params.nb, params.ny, params.T+1);  % Borrowing policy
q = zeros(params.nb, params.ny, params.T+1);    % Bond/debt price

% Solve model backwards
% Define key function 'solve_period' for optimal policies in each period
for t = params.T:-1:1
    [V(:,:,t), V_R(:,:,t), V_D(:,t), d_policy(:,:,t), b_policy(:,:,t), q(:,:,t)] = ...
        solve_period(t, V(:,:,t+1), b_grid, y_grid, params);
end

% Plot results: creates visualizations of the results
plot_results(V, V_R, V_D, d_policy, b_policy, q, b_grid, y_grid, params);

function [V, V_R, V_D, d_policy, b_policy, q] = solve_period(t, V_next, b_grid, y_grid, params)
    % Initialize outputs
    nb = length(b_grid);
    ny = length(y_grid);
    V = zeros(nb, ny);
    V_R = zeros(nb, ny);
    V_D = zeros(ny, 1);
    d_policy = zeros(nb, ny);
    b_policy = zeros(nb, ny);
    q = zeros(nb, ny);
    
    % Calculate value of default for each income state
    for iy = 1:ny
        y = y_grid(iy);
        phi = get_default_cost(y, params);
        c_default = y - phi;
        
        % Expected continuation value after default
        EV_next = params.pi * V_next(1,2) + (1-params.pi) * V_next(1,1);
        V_D(iy) = ((c_default)^(params.sigma)-1)/(1-params.sigma) + params.beta * EV_next;
    end
    
    % Calculate bond prices and value of repayment
    for ib = 1:nb
        for iy = 1:ny
            % Current state
            b = b_grid(ib);
            y = y_grid(iy);
            
            % Find optimal borrowing choice
            V_R_temp = -1e10 * ones(nb,1);
            for ib_next = 1:nb
                b_next = b_grid(ib_next);
                
                % Calculate expected probability of repayment next period
                prob_repay_H = 1 - d_policy(ib_next,2);
                prob_repay_L = 1 - d_policy(ib_next,1);
                exp_repay = params.pi * prob_repay_H + (1-params.pi) * prob_repay_L;
                
                % Bond price
                if b_next <= 0
                    q_temp = params.beta_star;
                else
                    q_temp = params.beta_star * exp_repay;
                end
                
                % Consumption and utility
                c = y + b - q_temp * b_next;
                if c > 0
                    % Expected continuation value
                    EV_next = params.pi * V_next(ib_next,2) + ...
                             (1-params.pi) * V_next(ib_next,1);
                    V_R_temp(ib_next) = (c^(params.sigma)-1)/(1-params.sigma) + params.beta * EV_next;
                end
            end
            
            % Find maximum value and optimal borrowing
            [V_R(ib,iy), idx] = max(V_R_temp);
            b_policy(ib,iy) = b_grid(idx);
            
            % Default decision
            if V_D(iy) > V_R(ib,iy)
                V(ib,iy) = V_D(iy);
                d_policy(ib,iy) = 1;
            else
                V(ib,iy) = V_R(ib,iy);
                d_policy(ib,iy) = 0;
            end
            
            % Update bond price
            if b_grid(ib) <= 0
                q(ib,iy) = params.beta_star;
            else
                prob_repay_H = 1 - d_policy(ib,2);
                prob_repay_L = 1 - d_policy(ib,1);
                q(ib,iy) = params.beta_star * (params.pi * prob_repay_H + ...
                           (1-params.pi) * prob_repay_L);
            end
        end
    end
end

function phi = get_default_cost(y, params)
    if abs(y - params.y_H) < 1e-10
        phi = params.phi_H;
    else
        phi = params.phi_L;
    end
end

% Visualisation of Period 1 (t=1) as an example
function plot_results(V, V_R, V_D, d_policy, b_policy, q, b_grid, y_grid, params)
    % Create figure with subplots
    figure('Position', [100 100 1200 800]);
    
    % Plot default policy
    subplot(2,2,1);
    plot(b_grid, d_policy(:,1,1), 'b-', 'LineWidth', 2);
    hold on;
    plot(b_grid, d_policy(:,2,1), 'r--', 'LineWidth', 2);
    xlabel('Current Debt (b_t)');
    ylabel('Default Decision');
    title('Default Policy (Period 1)');
    legend('Low Income', 'High Income', 'Location', 'southeast');
    grid on;
    
    % Plot borrowing policy
    subplot(2,2,2);
    plot(b_grid, b_policy(:,1,1), 'b-', 'LineWidth', 2);
    hold on;
    plot(b_grid, b_policy(:,2,1), 'r--', 'LineWidth', 2);
    plot(b_grid, b_grid, 'k:', 'LineWidth', 1);
    xlabel('Current Debt (b_t)');
    ylabel('Next Period Debt (b_{t+1})');
    title('Borrowing Policy (Period 1)');
    legend('Low Income', 'High Income', '45-degree line', 'Location', 'southeast');
    grid on;
    
    % Plot bond prices
    subplot(2,2,3);
    plot(b_grid, q(:,1,1), 'b-', 'LineWidth', 2);
    hold on;
    plot(b_grid, q(:,2,1), 'r--', 'LineWidth', 2);
    xlabel('Next Period Debt (b_{t+1})');
    ylabel('Bond Price (q_t)');
    title('Bond Prices (Period 1)');
    legend('Low Income', 'High Income', 'Location', 'southeast');
    grid on;
    
    % Plot value functions
    subplot(2,2,4);
    plot(b_grid, V(:,1,1), 'b-', 'LineWidth', 2);
    hold on;
    plot(b_grid, V(:,2,1), 'r--', 'LineWidth', 2);
    xlabel('Current Debt (b_t)');
    ylabel('Value');
    title('Value Functions (Period 1)');
    legend('Low Income', 'High Income', 'Location', 'southeast');
    grid on;
end


% Part 3: Compare value functions at t=1 and t=2 for different betas
% Define function 'analyze_value_functions' to compare value functions between t=1 and t=2
function analyze_value_functions(params_base)
    % Setup grids
    b_grid = linspace(params_base.b_min, params_base.b_max, params_base.nb)';
    y_grid = [params_base.y_L; params_base.y_H];
    
    % Solve model with base parameters
    [V1, ~, ~, ~, ~, ~] = solve_period(1, zeros(params_base.nb, params_base.ny), b_grid, y_grid, params_base);
    [V2, ~, ~, ~, ~, ~] = solve_period(2, V1, b_grid, y_grid, params_base);
    
    % Create high beta parameters
    params_high_beta = params_base;
    params_high_beta.beta = 0.99;  % Increased beta
    
    % Solve model with high beta
    [V1_high, ~, ~, ~, ~, ~] = solve_period(1, zeros(params_base.nb, params_base.ny), b_grid, y_grid, params_high_beta);
    [V2_high, ~, ~, ~, ~, ~] = solve_period(2, V1_high, b_grid, y_grid, params_high_beta);
    
    % Calculate differences
    V_diff_base = V2 - V1;
    V_diff_high = V2_high - V1_high;
    
    % Plot value function differences
    figure('Position', [100 100 1200 400]);
    
    % Base case
    subplot(1,2,1);
    plot(b_grid, V_diff_base(:,1), 'b-', 'LineWidth', 2);
    hold on;
    plot(b_grid, V_diff_base(:,2), 'r--', 'LineWidth', 2);
    xlabel('Debt Level');
    ylabel('Value Difference (V_2 - V_1)');
    title(sprintf('Value Function Differences (\\beta = %.2f)', params_base.beta));
    legend('Low Income', 'High Income', 'Location', 'best');
    grid on;
    
    % High beta case
    subplot(1,2,2);
    plot(b_grid, V_diff_high(:,1), 'b-', 'LineWidth', 2);
    hold on;
    plot(b_grid, V_diff_high(:,2), 'r--', 'LineWidth', 2);
    xlabel('Debt Level');
    ylabel('Value Difference (V_2 - V_1)');
    title(sprintf('Value Function Differences (\\beta = %.2f)', params_high_beta.beta));
    legend('Low Income', 'High Income', 'Location', 'best');
    grid on;
    
    % Print average absolute differences
    fprintf('Average absolute V2-V1 difference (beta=%.2f): %.4f\n', ...
            params_base.beta, mean(abs(V_diff_base(:))));
    fprintf('Average absolute V2-V1 difference (beta=%.2f): %.4f\n', ...
            params_high_beta.beta, mean(abs(V_diff_high(:))));
end

% Part 4: Analyze default behavior under different parameters
% Define function 'analyze_default_behaviour' to compare value functions between t=1 and t=2
function analyze_default_behavior(params_base)
    % Setup grids
    b_grid = linspace(params_base.b_min, params_base.b_max, params_base.nb)';
    y_grid = [params_base.y_L; params_base.y_H];
    
    % Base case
    [~, ~, ~, d_base, ~, q_base] = solve_period(1, zeros(params_base.nb, params_base.ny), ...
                                               b_grid, y_grid, params_base);
    
    % High default cost case
    params_high_cost = params_base;
    params_high_cost.phi_H = 0.5;  % Increase the high-state default cost
    [~, ~, ~, d_high_cost, ~, q_high_cost] = solve_period(1, zeros(params_base.nb, params_base.ny), ...
                                                         b_grid, y_grid, params_high_cost);
    
    % Patient lenders case
    params_patient_lender = params_base;
    params_patient_lender.beta_star = 0.99;  % More patient lenders
    [~, ~, ~, d_patient_lender, ~, q_patient_lender] = solve_period(1, zeros(params_base.nb, params_base.ny), ...
                                                     b_grid, y_grid, params_patient_lender);
    
    % Patient borrower case
    params_patient_borrower = params_base;
    params_patient_borrower.beta = 0.99;  % More patient borrowers
    [~, ~, ~, d_patient_borrower, ~, q_patient_borrower] = solve_period(1, zeros(params_base.nb, params_base.ny), ...
                                                                       b_grid, y_grid, params_patient_borrower);
    
    % Plot results
    figure('Position', [100 100 1200 800]);
    
    % Default policies
    subplot(2,3,1);
    plot(b_grid, d_base(:,1), 'b-', 'LineWidth', 2);
    hold on;
    plot(b_grid, d_high_cost(:,1), 'r--', 'LineWidth', 2);
    xlabel('Debt Level');
    ylabel('Default Decision');
    title('Default Policy: Base vs High Default Cost');
    legend('Base Cost', 'High Cost', 'Location', 'best');
    grid on;
    
    subplot(2,3,2);
    plot(b_grid, d_base(:,1), 'b-', 'LineWidth', 2);
    hold on;
    plot(b_grid, d_patient_lender(:,1), 'r--', 'LineWidth', 2);
    xlabel('Debt Level');
    ylabel('Default Decision');
    title('Default Policy: Base vs Patient Lenders');
    legend('Base Case', 'Patient Lenders', 'Location', 'best');
    grid on;
    
    subplot(2,3,3);
    plot(b_grid, d_base(:,1), 'b-', 'LineWidth', 2);
    hold on;
    plot(b_grid, d_patient_borrower(:,1), 'r--', 'LineWidth', 2);
    xlabel('Debt Level');
    ylabel('Default Decision');
    title('Default Policy: Base vs Patient Borrowers');
    legend('Base Case', 'Patient borrowers', 'Location', 'best');
    grid on;

    % Bond prices
    subplot(2,3,4);
    plot(b_grid, q_base(:,1), 'b-', 'LineWidth', 2);
    hold on;
    plot(b_grid, q_high_cost(:,1), 'r--', 'LineWidth', 2);
    xlabel('Debt Level');
    ylabel('Bond Price');
    title('Bond Prices: Base vs High Default Cost');
    legend('Base Cost', 'High Cost', 'Location', 'best');
    grid on;
    
    subplot(2,3,5);
    plot(b_grid, q_base(:,1), 'b-', 'LineWidth', 2);
    hold on;
    plot(b_grid, q_patient_lender(:,1), 'r--', 'LineWidth', 2);
    xlabel('Debt Level');
    ylabel('Bond Price');
    title('Bond Prices: Base vs Patient Lenders');
    legend('Base Case', 'Patient Lenders', 'Location', 'best');
    grid on;
    
    subplot(2,3,6);
    plot(b_grid, q_base(:,1), 'b-', 'LineWidth', 2);
    hold on;
    plot(b_grid, q_patient_borrower(:,1), 'r--', 'LineWidth', 2);
    xlabel('Debt Level');
    ylabel('Bond Price');
    title('Bond Prices: Base vs Patient Borrowers');
    legend('Base Case', 'Patient Borrowers', 'Location', 'best');
    grid on;

    % Print default thresholds
    fprintf('\nDefault thresholds (debt level where default occurs):\n');
    fprintf('Base case: %.4f\n', b_grid(find(diff(d_base(:,1))>0, 1)));
    fprintf('High default cost: %.4f\n', b_grid(find(diff(d_high_cost(:,1))>0, 1)));
    fprintf('Patient lenders: %.4f\n', b_grid(find(diff(d_patient_lender(:,1))>0, 1)));
    fprintf('Patient borrower: %.4f\n', b_grid(find(diff(d_patient_borrower(:,1))>0, 1)));
end

% Run the analysis to see the effect of changes in parameters
analyze_value_functions(params);
analyze_default_behavior(params);