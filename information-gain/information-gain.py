import numpy as np

def _entropy(y):
    """
    Helper: Compute Shannon entropy (base 2) for labels y.
    """
    y = np.asarray(y)
    if y.size == 0:
        return 0.0
    vals, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) if p.size else 0.0

def information_gain(y, split_mask):
    """
    Compute Information Gain of a binary split on labels y.
    Use the _entropy() helper above.
    """
    y = np.asarray(y)
    split_mask = np.asarray(split_mask, dtype=bool)

    y_left = y[split_mask]
    y_right = y[~split_mask]

    n_left = y_left.size
    n_right = y_right.size
    n_total = y.size

    # Empty side => invalid split
    if n_left == 0 or n_right == 0:
        return 0.0

    h_parent = _entropy(y)
    h_left = _entropy(y_left)
    h_right = _entropy(y_right)

    weighted_children = (
        (n_left / n_total) * h_left +
        (n_right / n_total) * h_right
    )

    return float(h_parent - weighted_children)