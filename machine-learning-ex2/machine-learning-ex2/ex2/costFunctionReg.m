function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
Z = X * theta;
hx = sigmoid(Z);
Pa = log(hx);
Pb = log(1 - hx);
Pa = -y' * Pa;
Pb = (1 - y') * Pb;
J = Pa - Pb;
J = J / m;

regExp = theta' * theta;
regExp(1) = regExp(1) - theta(1) * theta(1);
regExp = regExp / (2 * m);
regExp = lambda * regExp;

J = J + regExp;

input = hx - y;
input = X' * input;
grad = input / m;

regExp2 = (theta * lambda) / m;

grad = grad + regExp2;
grad(1) = grad(1) - ((theta(1) * lambda) / m);


% =============================================================

end
