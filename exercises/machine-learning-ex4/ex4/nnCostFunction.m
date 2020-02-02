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
m = size(X, 1);%5000
         
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


% part 1 implementation

a1 = [ones(m,1) X]; % 5000 * 401
z2 = Theta1 * a1'; % 25 * 401 * 401 * 5000 = 25 * 5000
a2 = sigmoid(z2);
a2 = [ones(1, size(a2,2)); a2]; % 26 * 5000
z3 = Theta2 * a2; % 10 * 26 * 26 * 100 = 10 * 100
a3 = sigmoid(z3); % 10 * 5000
h_theta_X = a3';  % 5000 * 10

y_vec = zeros(m, num_labels);

% y m*k = 5000 * 10
for i = 1:m
	y_vec(i, y(i)) = 1;
end

J = -(1/m) * sum( sum(y_vec .* log(h_theta_X) + (1-y_vec) .* log(1-h_theta_X)) ); % sum m*k matrix

% follow exactly to the formula
regularization = (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2))) * (lambda/(2*m));

J = J + regularization;


% part 2 implementation

for t = 1:m

	% for input layer
	a1  = [1; X(t,:)'];%401*1

	% for hidden layer
	z2 = Theta1 * a1;%25*1
	a2 = [1;sigmoid(z2)];

	% for output layer
	z3 = Theta2 * a2;
	a3 = sigmoid(z3);%10*1

	yy = ([1:num_labels] == y(t))'; % y: 10*1 y(t) is a scalar. Ex. [1:5] ==[2] =>[0 1 0 0 0]

	delta_3 = a3 - yy; % 

	delta_2 = Theta2' * delta_3 .* [1;sigmoidGradient(z2)];

	% remove bias unit from delta_2
	delta_2 = delta_2(2:end);%25*1

	% update triangle delta 
	Theta1_grad = Theta1_grad + delta_2 * a1';
	Theta2_grad = Theta2_grad + delta_3 * a2';

end

% update big Delta

% it does not update simultaneously, so wrong 
% Theta1_grad(:,2:end)  = (1/m)*Theta1 + (lambda/m) * Theta1(:,2:end);
% Theta1_grad(:,1) = (1/m)*Theta1(:,1);

%Theta2_grad(:,2:end) = (1/m)*Theta2_grad + (lambda/m) * Theta2(:,2:end);
%Theta2_grad(:,1) = (1/m)*Theta2(:,1);

% update big Delta simultaneously
% regularization
Theta1_grad = (1/m)*Theta1_grad + (lambda/m)* [zeros(size(Theta1_grad,1),1) Theta1(:,2:end)];

Theta2_grad = (1/m)*Theta2_grad + (lambda/m)* [zeros(size(Theta2_grad,1),1) Theta2(:,2:end)];


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
