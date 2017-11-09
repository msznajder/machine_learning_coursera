function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
predictions = X * theta;                        % each row being one training data record
% predictions =  sum(theta' .* X, 2);           % alternative method
sumSqrError = sum((predictions - y).^2);        % sum cols to get squared error
J = 1 / (2 * m) * sumSqrError;                  % multiply to get cost
% =========================================================================

end
