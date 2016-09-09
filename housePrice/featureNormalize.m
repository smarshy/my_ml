function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. 

X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ============================================

% Taking mean along rows for each column (or feature)
mu=mean(X,1);
sigma=std(X);
%X_norm=(X_norm-mu)/sigma;

for m=1:size(X,1)
	for n=1:size(X,2)
		X_norm(m,n)=(X_norm(m,n)-mu(n))/sigma(n);
	end
end
% ============================================================

end
