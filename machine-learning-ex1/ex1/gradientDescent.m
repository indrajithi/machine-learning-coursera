function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    % Algo:
    % Repete untill converge
    % thetaJ := thetaJ - alpha d/(d thetaJ) of J(theta0,theta1)
    %
    % theta0 := theta0 - alpha*1/m sum_from_1_to_m(Htheta(xi) - yi)  
    % theta1 := theta1 - alpha*1/m sum_from_1_to_m(Htheta(xi) - yi) *xi

    %h=X*theta; 

    % delta=1/m*(X' * X*theta - X' * y); %derivative term
    % delta=1/m*(X' * X*theta - X' * y); %derivative term
    %theta=theta-alpha.*delta;

    htheta = X * theta;
    theta0 = theta(1) - alpha / m * sum((htheta - y) );
    theta1 = theta(2) - alpha / m * sum((htheta - y) .* X(:,2)); 
    theta = [theta0; theta1];



    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    

end

end
