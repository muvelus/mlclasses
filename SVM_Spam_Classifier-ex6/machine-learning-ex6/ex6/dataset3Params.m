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
sampleC = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sampleSigma = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

min = 99;

for i = 1:size(sampleC, 2)
    for j = 1:size(sampleSigma, 2)
        predictions = svmPredict(svmTrain(X, y, sampleC(i), @(x1, x2) gaussianKernel(x1, x2, sampleSigma(j))), Xval);
        predError = mean(double(predictions ~= yval));
        
        if predError < min
            min = predError;
            C = sampleC(i);
            sigma = sampleSigma(j);
        end
    end
end

% =========================================================================

end
