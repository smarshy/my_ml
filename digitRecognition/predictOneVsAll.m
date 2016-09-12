function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);
p = zeros(size(X, 1), 1);

X = [ones(m, 1) X];

% ============================================

t = sigmoid(X * all_theta');
% column vector with max of each row
[max_val,p] = max(t, [], 2);

% =========================================================================


end
