% Generate random data
data = randn(50, 500);

% Compute the covariance matrix
covarianceMatrix = cov(data');

% Compute the eigenvalues and eigenvectors
[eigenvecs, eigenvals] = eigs(covarianceMatrix, 21); % Reduce to 21 dimensions

% Project the data onto the eigenvectors
reducedData = eigenvecs' * data;

% Reconstruct the original data
reconstructedData = eigenvecs * reducedData;

% Compute the root mean square difference between the reconstructed and original data
error = sqrt(mean(mean((data - reconstructedData).^2)));

% Load cholesall data
load choles_all.mat;

% Extract the p component (21 x 264 matrix)
dataCholesall = p;

% Compute the covariance matrix
covarianceMatrixCholesall = cov(dataCholesall');

% Compute the eigenvalues and eigenvectors
[eigenvecsCholesall, eigenvalsCholesall] = eigs(covarianceMatrixCholesall, 21); % Reduce to 21 dimensions

% Project the data onto the eigenvectors
reducedDataCholesall = eigenvecsCholesall' * dataCholesall;

% Reconstruct the original data
reconstructedDataCholesall = eigenvecsCholesall * reducedDataCholesall;

% Compute the root mean square difference between the reconstructed and original data
errorCholesall = sqrt(mean(mean((dataCholesall - reconstructedDataCholesall).^2)));

% Display the errors
disp("Random Data Error: " + error);
disp("Cholesall Data Error: " + errorCholesall);
