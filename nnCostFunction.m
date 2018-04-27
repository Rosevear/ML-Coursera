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
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%VECTORIZED METHOD
%Modify the target matrix so that it is a matrix of 1 hot encoded vectors
eye_matrix = eye(num_labels);
y_matrix = eye_matrix(y,:);

%Forwardpropagate
%Get the current example, add the bias units, and compute the hidden layer activation
a1 = [ones(size(X, 1), 1) X];
z2 = a1 * transpose(Theta1);
a2 = sigmoid(z2);

%Add the bias unit to the hidden layer and compute the final output
a2 = [ones(size(a2, 1), 1) a2];
z3 = a2 * transpose(Theta2);
a3 = sigmoid(z3);

%Backpropagate
%Compute the error for the output layer
d3 = a3 - y_matrix;

%Compute the error for the hidden layer
d2 = (d3 * Theta2(:, 2:end)) .* sigmoidGradient(z2);

%Compute the vector of partial derivatives (gradient) for each layer
delta_2 = transpose(d3) * a2;
delta_1 = transpose(d2) * a1;

%Scale the gradients
Theta1_grad = delta_1 ./ m;
Theta2_grad = delta_2 ./ m;

%Compute the unregularized cost
cost_term_1 = transpose(-y_matrix) * log(a3);
cost_term_2 = transpose(1 - y_matrix) * log(1 - a3);
%We have to multiple by the identity matrix to eliminate the non diagonal
%terms, since they are not relevant to the cost function when y and h are
%matrices and not vectors
J = (sum(sum(eye_matrix .* cost_term_1)) - sum(sum((eye_matrix .* cost_term_2)))) ./ m

%Compute the regularized cost
regCost = (lambda ./ (2 .* m)) .* (sum(sum(Theta1(:, 2:end) .^ 2)) + sum(sum(Theta2(:, 2:end) .^ 2)));

%Total cost
J = J + regCost

%Modify the gradients to take into account the regularization
%We can modify the weight matrices to exclude the bias units, since we've already computed h(x)
Theta1(:,1) = 0;
Theta2(:,1) = 0;
Theta1 = Theta1 .* (lambda ./ m);
Theta2 = Theta2 .* (lambda ./ m);

Theta1_grad = Theta1_grad + Theta1;
Theta2_grad = Theta2_grad + Theta2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
