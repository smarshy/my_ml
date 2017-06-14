function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% Return the following variables correctly.
C = 1;
sigma = 0.3;

% =======================================================================
% returns the optimal C and sigma learning parameters found using the cross validation set.
% predict the labels on the cross validation set : predictions = svmPredict(model, Xval);
%  compute the prediction error using mean(double(predictions ~= yval))
%







% =========================================================================

end
