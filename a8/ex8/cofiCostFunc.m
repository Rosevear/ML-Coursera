function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user


%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

%Compute the unregularized cost function
J = (sum(sum((((X * Theta') - Y) .^ 2) .* R))) ./ 2;

%Regularize the cost function
J = J + ((lambda ./ 2) .* (sum(sum(Theta .^ 2)))) + ((lambda ./ 2) .* (sum(sum(X .^ 2))));

%Partial derivatives w.r.t parameter vectors
for j = 1:num_users
    %Indices for the movies that the current user has rated
    idx = find(R(:, j) == 1);
    %Movie ratings for the current user
    yTemp = Y(idx, j);
    %Feature vectors for the movies rated by the current user
    XTemp = X(idx, :);
    Theta_grad(j, :) = ((XTemp * Theta(j, :)' - yTemp))' * XTemp + (lambda * Theta(j, :));
    
end

%Partial derivatives w.r.t feature vectors
for i = 1:num_movies
    %Indices of users who rated the current movie
    idx = find(R(i, :) == 1);
    %Ratings by the subset of users who in fact rated the current movie
    yTemp = Y(i, idx);
    %Parameter vectors for the subset of users who rated the current movie
    thetaTemp = Theta(idx, :);
    X_grad(i, :) = (X(i, :) * thetaTemp' - yTemp) * thetaTemp + (lambda * X(i, :));
end














% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
