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
%
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
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% -------------------------------------------------------------

% Compute h(x) step by step, code from ex3 predict.m 
% Add bias
a1 = [ones(m, 1) X];
% Compute z2
z2 = Theta1 * a1';
% Compute a2
a2 = sigmoid(z2);
% Add bias
a2bias = [ones(size(a2'), 1) a2'];
% Compute z3
z3 = Theta2 * a2bias';
% Compute a3
a3 = sigmoid(z3)';
hx = a3;

% We need to map every value of y, to its value as vectors 0,1
% Identity matrix, which contains the possible values of y
% Example: 1 -> y(1) = {1,0,...,0}
I = eye(num_labels);
% Y matrix, with space for m values X num_label vector size
Y = zeros(m, num_labels);
% Store in Y(i), the value of y as a vector of 0,1
for i=1:m
  Y(i, :)= I(y(i), :);
end

% Now we can compute the cost function
J = sum(sum((1/m) * ((-Y .* log(hx)) - ((1 - Y) .* log(1 - hx)))));

% Regularized cost function
% Compute the regularization term
reg = (lambda/(2*m)) * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));
% Add reg
J = J + reg;

%% Backpropagation

% For the output layer, compute the difference, here, delta3
delta3 = a3 - Y;
% Fore hidden layer, compute delta, in this case only delta2
% We have to add the bias to the result of g'
delta2 = (delta3 * Theta2) .* [ones(size(z2', 1), 1) sigmoidGradient(z2')];
% Then we remove the first row, which correspond to the bias
delta2 = delta2(:,2:end);

% Compute grade
Theta1_grad = (1/m) * (delta2' * a1);
Theta2_grad = (1/m) * (delta3' * a2bias);

% Regularization
% We have to include a row of zeros first, because the first column of theta
% is not regularized, as it is used for the bias term
Theta1_grad += (lambda/m) * [zeros(size(Theta1, 1), 1) Theta1(:,2:end)];
Theta2_grad += (lambda/m) * [zeros(size(Theta2, 1), 1) Theta2(:,2:end)];



% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
