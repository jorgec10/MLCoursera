function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       

% For each example (X row), we have to predict which class correspond to that example (row)

% This for loop computes the max value for ith example
% It helps to understand what has to be done
% for i=1:num_labels
%	% Get ith row, an example
% 	row = X(i,:);
%
%	% If we multiply the example by each possible value of theta, which is
%	% a trained logistic regression theta vector for the j-th class, we will
%	% see how "well" the example fits to each class
% 	values = row * all_theta';
%
%	% Now we find the max of these values, which correspond to the more
%	% suitable class. The index of the position of that max value will give us
%	% the class of that example.
% 	max(values)
% end;

% Now, the final, vectorized, version
% To vectorize the previous for loop, we perform directly the matrix multiplication
% And we will have a matrix with the fitness values of each example to each class
fitness = X * all_theta';

% The values of each row of fitness matrix correspond to the "level" of fitness
% of the index-th class applied to each example. 
% We want to know the index with the maximum level, which correspond to the 
% class identifier
[maximum, index] = max((fitness), [], 2);

% Return index
p = index;


% =========================================================================


end
