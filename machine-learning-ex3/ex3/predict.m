function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Following fig2 formulas
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

% Now we have a matrix of m*num_labels, in which each row correspond
% to a vector with the fitness value of each example for each class
% We have to get the index (class) with maximum value
[maximum, index] = max((a3), [], 2);

p = index;

% =========================================================================


end
