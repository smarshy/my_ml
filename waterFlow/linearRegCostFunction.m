function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% The following variables are to be returned correctly 
J = 0;
grad = zeros(size(theta));

% =========================================================================

% Computed the cost and gradient of regularized linear 
% regression for a particular choice of theta

% ignored theta(1) for regularization, also preserve dimensions
J =(sum((X*theta-y).^2)+lambda*sum([0;theta(2:end)].^2))/(2*m);
grad=(1/m).*(X'*(X*theta-y))+(lambda/m)*[0;theta(2:end)];

% =========================================================================

grad = grad(:);

end
