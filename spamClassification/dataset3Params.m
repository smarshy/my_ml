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

% taken a value based on observed outputs
min=10;

for temp_c = [0.01 0.03 0.1 0.3 1, 3, 10 30]
    for temp_sigma = [0.01 0.03 0.1 0.3 1, 3, 10 30]
        model = svmTrain(X, y, temp_c, @(x1, x2) gaussianKernel(x1, x2, temp_sigma));
        predictions = svmPredict(model, Xval);
        prediction_error = mean(double(predictions ~= yval));  
        if (prediction_error<min)
        	min=prediction_error;
        	C=temp_c;
        	sigma=temp_sigma;
        end      
    end
end

% =========================================================================

end
