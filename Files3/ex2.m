% Load the data
load('threes.mat');

% Compute the mean and display it
mean_3 = mean(threes);
mean_3_image = reshape(mean_3, 16, 16);
colormap('gray');
imagesc(mean_3_image);

% Compute the covariance matrix and its eigenvalues/eigenvectors
covariance_matrix = cov(threes);
[V, D] = eig(covariance_matrix);
eigenvalues = diag(D);
plot(eigenvalues);

% Compress the dataset by projecting it onto one, two, three, and four principal components
figure;
for i = 1:4
    compressed_data = threes * V(:,1:i);
    reconstructed_data = compressed_data * V(:,1:i)';
    subplot(2,2,i);
    imagesc(reshape(reconstructed_data(1,:), 16, 16),[0,1]);
end

% Define a function to compress and reconstruct the dataset with q principal components
function [reconstructed_data, error] = compress_and_reconstruct(data, q)
    % Compute the principal components
    covariance_matrix = cov(data);
    [V, D] = eig(covariance_matrix);
    eigenvalues = diag(D);
    
    % Project the data onto the q principal components
    compressed_data = data * V(:,1:q);
    
    % Reconstruct the data from the compressed representation
    reconstructed_data = compressed_data * V(:,1:q)';
    
    % Compute the reconstruction error
    error = norm(data - reconstructed_data, 'fro')^2 / size(data, 1);
end

% Compute the reconstruction error as a function of q
errors = zeros(50, 1);
for q = 1:50
    [~, errors(q)] = compress_and_reconstruct(threes, q);
end
plot(errors);

% Compute the sum of the eigenvalues and compare to the reconstruction error
eigenvalue_sum = cumsum(eigenvalues, 'reverse');
figure;
plot(eigenvalue_sum(1:50));
hold on;
plot(errors);
legend('Cumulative sum of eigenvalues', 'Reconstruction error');
