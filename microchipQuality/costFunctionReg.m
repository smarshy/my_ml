function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

m = length(y); % number of training examples
J = 0;
grad = zeros(size(theta));

% ============================================

k = zeros(m);
k = sigmoid(X*theta);
l = theta([2:end],:);
J = (-y'*log(k) - (1-y')*log(1 - k))/m + (l'*l)*lambda/(2*m);
s = [0; lambda*l];
grad = (X'*(k - y))/m + s/m;

% =============================================================

end
