function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% Some useful variables
m = size(X, 1);
n = size(X, 2);

% You need to return the following variables correctly 
% k*(N+1) classifiers parameter (+1 bias unit)
all_theta = zeros(num_labels, n + 1); % 10 * 401

% Add ones to the X data matrix
X = [ones(m, 1) X]; % 100 * 401

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the following code to train num_labels
%               logistic regression classifiers with regularization
%               parameter lambda. 
%
% Hint: theta(:) will return lca column vector.
%
% Hint: You can use y == c to obtain a vector of 1's and 0's that tell you
%       whether the ground truth is true/false for this class.
%
% Note: For this assignment, we recommend using fmincg to optimize the cost
%       function. It is okay to use a for-loop (for c = 1:num_labels) to
%       loop over the different classes.
%
%       fmincg works similarly to fminunc, but is more efficient when we
%       are dealing with large number of parameters.
%
% Example Code for fmincg
%
%     % Set Initial theta
%     initial_theta = zeros(n + 1, 1);
%     
%     % Set options for fminunc
%     options = optimset('GradObj', 'on', 'MaxIter', 50);
% 
%     % Run fmincg to obtain the optimal theta
%     % This function will return theta and the cost 
%     [theta] = ...
%         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
%                 initial_theta, options);
%

options = optimset('GradObj', 'on', 'MaxIter', 50);
% 可以自动选择合适的学习速率的梯度下降算法
%[optTheta, functionVal, exitFlag] = fminunc(@costFunction, initialTheta, options);
%最终我们会得到三个返回值，分别是满足最小化代价函数J(θ)的θ值optTheta，
%costFunction中定义的jVal的值functionVal，
%以及标记是否已经收敛的状态值exitFlag，如果已收敛，标记为1，否则为0。
for c = 1 : num_labels
	[all_theta(c,:)]= fmincg(@(t)(lrCostFunction(t, X, (y==c), lambda)), all_theta(c,:)', options);
	
end

%for c = 1 : num_labels
%	[~, all_theta(c,:)] = lrCostFunction(all_theta(c,:)', X, y, lambda);
%end

all_theta(:);

% =========================================================================


end
