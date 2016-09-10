function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.


m = length(y); % number of training examples
J = 0;
grad = zeros(size(theta));

% ============================================
% cost of a particular choice of theta and partial derivatives (grad)
% grad has the same dimensions as theta

k = zeros(m);
k = sigmoid(X*theta);
J = (-y'*log(k) - (1-y')*log(1 - k))/m;
grad = (X'*(k - y))/m;

% =============================================================

end
