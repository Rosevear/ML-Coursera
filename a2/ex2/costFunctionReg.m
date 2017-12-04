function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

hypothesis = sigmoid(X * theta);
%Set to zero since we don't want to penalize the first parameter (by convention)
theta(1) = 0;

%Compute the cost
J = ((sum((-y .* log(hypothesis)) - ((1 - y) .* log(1 - hypothesis)))) ./ m) + ((lambda ./ (2 .* m)) .* sum(theta .^ 2));

%Compute the gradients
%The first gradient for theta0 (which is unregularized by convention)
grad(1) = sum((hypothesis - y) .* X(:, 1)) ./ m ;

%The rest of the gradients (which are regularized)
for i = 2:numel(grad)
    grad(i) = (sum((hypothesis - y) .* X(:, i)) ./ m) + ((lambda ./ m) .* theta(i));
end




% =============================================================

end
