import numpy as np

def decision_tree_split(X, y):
    X = np.array(X)
    y = np.array(y)

    def gini(labels):
        if len(labels) == 0:
            return 0.0

        _, counts = np.unique(labels, return_counts=True)
        probs = counts / len(labels)
        return 1.0 - np.sum(probs ** 2)

    n_samples = len(y)
    n_features = len(X[0])

    parent_gini = gini(y)

    best_gain = -1
    best_feature = 0
    best_threshold = 0.0

    for feature in range(n_features):
        values = sorted(set(X[:, feature]))

        for i in range(len(values) - 1):
            threshold = (values[i] + values[i + 1]) / 2.0

            left_mask = X[:, feature] <= threshold
            right_mask = X[:, feature] > threshold

            if not np.any(left_mask) or not np.any(right_mask):
                continue

            y_left = y[left_mask]
            y_right = y[right_mask]

            weighted_gini = (
                len(y_left) / n_samples * gini(y_left)
                + len(y_right) / n_samples * gini(y_right)
            )

            gain = parent_gini - weighted_gini

            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = float(threshold)

    return [best_feature, best_threshold]