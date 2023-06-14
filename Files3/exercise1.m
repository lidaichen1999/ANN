% Generate random data matrix
X = randn(50, 500);

% Calculate covariance matrix
C = cov(X');

% Calculate eigenvectors and eigenvalues
[V, D] = eigs(C, 10);

% Project data onto eigenvectors
Y = V' * X;

% Reconstruct data from reduced representation
Xhat = V * Y;

% Calculate reconstruction error
error = sqrt(mean(mean((X - Xhat).^2)));
disp(['Reconstruction error: ' num2str(error)]);
%%
% Test different reduced dimensions
dimensions = [5 10 20 30];
num_dims = length(dimensions);

% Plot original and reconstructed data for each dimension
figure;
for i = 1:num_dims
    % Calculate eigenvectors and eigenvalues
    [V, D] = eigs(C, dimensions(i));
    
    % Project data onto eigenvectors
    Y = V' * X;
    
    % Reconstruct data from reduced representation
    Xhat = V * Y;
    
    % Calculate reconstruction error
    error = sqrt(mean(mean((X - Xhat).^2)));
    
    % Plot original and reconstructed data
    subplot(num_dims, 2, (i-1)*2+1);
    imagesc(X);
    title(['Original data (dim=' num2str(size(X, 1)) ')']);
    subplot(num_dims, 2, (i-1)*2+2);
    imagesc(Xhat);
    title(['Reduced data (dim=' num2str(dimensions(i)) '), error=' num2str(error)]);
end
%%
% Load data
load choles_all;

% Calculate covariance matrix
C = cov(p');

% Calculate eigenvectors and eigenvalues
[V, D] = eigs(C, 10);

% Project data onto eigenvectors
Y = V' * p;

% Reconstruct data from reduced representation
Xhat = V * Y;

% Calculate reconstruction error
error = sqrt(mean(mean((p - Xhat).^2)));
disp(['Reconstruction error: ' num2str(error)]);
%%
% Generate highly correlated data
X_corr = [1 0.5 0.3; 0.5 1 0.4; 0.3 0.4 1] * randn(3, 500);

% Calculate covariance matrices for random and correlated data
C_rand = cov(randn(50, 500)');
C_corr = cov(X_corr');

% Calculate eigenvectors and eigenvalues for random and correlated data
[V_rand, D_rand] = eigs(C_rand, 10);
[V_corr, D_corr] = eigs(C_corr, 10);

% Project data onto eigenvectors for random and correlated data
Y_rand = V_rand' * randn(50, 500);
Y_corr = V_corr' * X_corr;

% Reconstruct data from reduced representation for random and correlated data
Xhat_rand = V_rand * Y_rand;
Xhat_corr = V_corr * Y_corr;

% Calculate reconstruction errors for random and correlated data
%%
