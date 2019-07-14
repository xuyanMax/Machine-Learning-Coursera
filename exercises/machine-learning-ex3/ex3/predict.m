function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)
% The matrices Theta1 and Theta2 will now be in your Octave environment
% Theta1 has size 25 x 401
% Theta2 has size 10 x 26

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

X = [ones(m, 1) X]; % 100 * 401
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

z_2 = Theta1 * X'; % 25 * 5000
a_2 = sigmoid(z_2); 
a_2 = [ones(1,size(a_2,2));a_2]; %26*5000
z_3 = Theta2 * a_2; %10*5000
a_3 = sigmoid(z_3);
a_3 = a_3'; % 5000*10
[~,p] = max(a_3, [], 2);

% =========================================================================


end
