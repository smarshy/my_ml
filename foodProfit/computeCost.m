function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

m = length(y); % number of training examples
J = 0;

% ============================================
% Computing the cost of a particular choice of theta
% 

K = zeros(m,1);
K = X * theta - y;
J = (sum(K.^2))/(2*m);


% =========================================================================

end
