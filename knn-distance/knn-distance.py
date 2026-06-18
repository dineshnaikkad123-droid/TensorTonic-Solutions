import numpy as np

def knn_distance(X_train, X_test, k):
    """
    Compute pairwise distances and return k nearest neighbor indices.
    """
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)

    # Handle 1D data
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)

    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)

    # Pairwise Euclidean distances
    distances = np.linalg.norm(
        X_test[:, None, :] - X_train[None, :, :],
        axis=2
    )

    n_train = len(X_train)
    k_actual = min(k, n_train)

    # Indices of nearest neighbors
    neighbors = np.argsort(distances, axis=1)[:, :k_actual]

    # Pad with -1 if k > n_train
    if k_actual < k:
        padding = np.full((len(X_test), k - k_actual), -1, dtype=int)
        neighbors = np.hstack([neighbors, padding])

    return neighbors.astype(int)