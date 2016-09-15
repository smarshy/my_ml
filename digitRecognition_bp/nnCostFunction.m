function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

m = size(X, 1);
         
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ============================================

% cost without regularization
a1 = [ones(m, 1) X];
a2= [ones(m, 1) sigmoid(a1*Theta1')];
a3 = sigmoid(a2*Theta2');
yVector = repmat([1:num_labels], m, 1) == repmat(y, 1, num_labels);
J = sum(sum((-yVector.*log(a3) - (1-yVector).*log(1 - a3))))/m;

% adding regularized cost
J= J + (lambda/(2*m))*(sum(sum(Theta1(:,2:end).^2 ))+ sum(sum(Theta2(:,2:end).^2)))

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
