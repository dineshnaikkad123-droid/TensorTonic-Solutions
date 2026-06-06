import numpy as np

def majority_classifier(y_train, X_test):
    """
    Predict the most frequent label in training data for all test samples.
    """
    y_train = np.asarray(y_train)

    # Find majority class with stable tie-breaking
    counts = {}
    for label in y_train:
        counts[label] = counts.get(label, 0) + 1

    majority_class = max(counts.items(), key=lambda x: x[1])[0]

    # Number of test samples
    n_test = len(X_test)

    return np.full(n_test, majority_class, dtype=int)