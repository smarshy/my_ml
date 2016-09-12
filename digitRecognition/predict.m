function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

m = size(X, 1);
num_labels = size(Theta2, 1);
p = zeros(size(X, 1), 1);

% ============================================

a1 = [ones(m, 1) X];
z2 = sigmoid(a1*Theta1');
a2 = [ones(m, 1) z2];
z3 = sigmoid(a2*Theta2');
[max_val,p] = max(z3, [], 2);

% =========================================================================
end
