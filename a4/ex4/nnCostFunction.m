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

for i = 1:m
    %Get the current example and add the bias units
    a1 = [1 X(i, :)];
   
    %Convert the output label into a corresponding bit vector
    curExOutputLabel = y(i, :);
    curExOutputVec = zeros(num_labels, 1);
    curExOutputVec(curExOutputLabel) = 1;
    
    %Compute the hidden layer
    z1 = a1 * Theta1';
    a2 = sigmoid(z1);
    
    %Add the bias unit the the hidden layer (as a row)
    a2 = [1 a2];
    z2 = a2 * Theta2';
    a3 = sigmoid(z2);
  
    %Backpropogate
    delta3 = a3 - curExOutputVec';
    delta3
    %Remove the weights for the hidden layer bias unit: since it does not
    %connect to anything in the input layer, we don't backpropogate on it

      size(delta3)
      size(Theta2)
      size(sigmoidGradient(z2))
      delta2 = delta3 * Theta2 .* sigmoidGradient(z2)';
      delta2 = delta2(:, 2:end);     
      delta2
    
    %Compute the gradients
%     temp1 = delta2 * a1';
%     temp2 = delta3 * a2';
%     Theta1_grad = Theta1_grad + temp1;
%     Theta2_grad = Theta2_grad + temp2;
     
    %Compute the cost over all of the output units for the current example
    hypothesis = a3;
    for j = 1:num_labels
       curCost = (-curExOutputVec(j) .* log(hypothesis(j))) - ((1 - curExOutputVec(j)) .* log(1 - hypothesis(j)));
       J = J + curCost;
    end
end
J = J ./ m;

%Compute the final partial derivatives used by the optimization algorithm
Theta1_grad = Theta1_grad ./ m;
Theta2_grad = Theta2_grad ./ m;

%Regularize the cost function
regCost = 0;
params = {Theta1, Theta2};
for i = 1:length(params)
    curTheta = params{i};
    unBiasedTheta = curTheta(:, 2:end);
    for j = 1:size(unBiasedTheta, 1)
        for k = 1:size(unBiasedTheta, 2)
            regCost = regCost + (unBiasedTheta(j, k) ^ 2);
        end
    end
end

regCost = (lambda ./ (2 .* m)) .* regCost;
J = J + regCost;


















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
