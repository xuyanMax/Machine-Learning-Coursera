function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

for k =1:K
	
	%c_i = find(idx==k); % find all indices of id = k from m*1 matrix
	c_i = idx == k;
	%count = size(c_i,2);
	k_num = sum(c_i);
%	idx_sum = zeros(1,n);

%	for i=1:count

%		idx_sum += X(c_i(1,i),:); %
%
	%end
	c_i_matrix = repmat(c_i, 1, n);
	X_c_i = X .* c_i_matrix;

	centroids(k,:) = sum(X_c_i) ./ k_num;
	%centroids(k,:) = idx_sum / count;

end







% =============================================================


end

