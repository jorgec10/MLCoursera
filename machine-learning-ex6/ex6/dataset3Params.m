function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% Possible values of C and sigma
values = [0.01 0.03 0.1 0.3 1 3 10 30]';
valueslen = length(values);

% Initialize vectors
predictionError = zeros(valueslen, valueslen);
result = zeros(valueslen*2, 3);

row = 1;

for i = 1:valueslen
	for j = 1:valueslen
		% Get the value of C and sigma
		Cval = values(i);
		sigmaval = values(j);

		% Train the model with these values
		model = svmTrain(X, y, Cval, @(x1, x2) gaussianKernel(x1, x2, sigmaval));
		% Predict the values with the cross validation set
		predictions = svmPredict(model, Xval);
		% Compute error of the predictions
		predictionError(i,j) = mean(double(predictions ~= yval));
		% Store the result
		result(row,:) = [predictionError(i,j), Cval, sigmaval];

		row = row + 1;
	end;
end;

% Sort rows in ascending order, to get the min in row 1
sortedResults = sortrows(result, 1);

% Return the optimum values
C = sortedResults(1,2);
sigma = sortedResults(1,3);

% =========================================================================

end
