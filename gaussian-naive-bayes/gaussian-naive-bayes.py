import math
from collections import defaultdict

def gaussian_naive_bayes(X_train, y_train, X_test):
    classes = sorted(set(y_train))

    priors = {}
    means = {}
    variances = {}

    n_samples = len(y_train)
    n_features = len(X_train[0])

    # Train
    for c in classes:
        X_c = [X_train[i] for i in range(n_samples) if y_train[i] == c]

        priors[c] = len(X_c) / n_samples

        means[c] = []
        variances[c] = []

        for j in range(n_features):
            feature = [row[j] for row in X_c]

            mean = sum(feature) / len(feature)
            var = sum((x - mean) ** 2 for x in feature) / len(feature)

            means[c].append(mean)
            variances[c].append(var + 1e-9)  # avoid division by zero

    # Predict
    predictions = []

    for sample in X_test:
        best_class = None
        best_score = float('-inf')

        for c in classes:
            score = math.log(priors[c])

            for j in range(n_features):
                mean = means[c][j]
                var = variances[c][j]

                score += (
                    -0.5 * math.log(2 * math.pi * var)
                    - ((sample[j] - mean) ** 2) / (2 * var)
                )

            if score > best_score:
                best_score = score
                best_class = c

        predictions.append(best_class)

    return predictions