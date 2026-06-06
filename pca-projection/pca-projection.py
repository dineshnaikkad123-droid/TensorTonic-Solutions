import numpy as np

def pca_projection(X, k):
    X = np.array(X, dtype=float)

    # Center the data
    X_centered = X - np.mean(X, axis=0)

    # Covariance matrix
    cov = np.cov(X_centered, rowvar=False)

    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort by descending eigenvalue
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]

    # Top-k principal components
    components = eigenvectors[:, :k]

    # Project data
    projected = X_centered @ components

    return projected.tolist()