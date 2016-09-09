function [theta] = normalEqn(X, y)
%NORMALEQN Computes the closed-form solution to linear regression 
%   NORMALEQN(X,y) computes the closed-form solution to linear 
%   regression using the normal equations.

theta = zeros(size(X, 2), 1);

% ============================================
% code to compute the closed form solution for linear regression

theta=inv((X'*X))*X'*y;

% -------------------------------------------------------------


% ============================================================

end
