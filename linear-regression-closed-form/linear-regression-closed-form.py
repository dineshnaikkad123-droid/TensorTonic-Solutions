import numpy as np

def linear_regression_closed_form(X, y):
    """
    Compute the optimal weight vector using the normal equation.
    """
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    XtX = X.T @ X
    Xty = X.T @ y

    w = np.linalg.inv(XtX) @ Xty

    return w.tolist()