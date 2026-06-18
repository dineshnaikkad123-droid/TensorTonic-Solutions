import numpy as np

def naive_bayes_bernoulli(X_train, y_train, X_test):
    """
    Compute log-likelihood P(y|x) for Bernoulli Naive Bayes.
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)

    classes = np.sort(np.unique(y_train))
    n_classes = len(classes)
    n_features = X_train.shape[1]
    n_train = X_train.shape[0]

    # Log priors
    log_priors = np.zeros(n_classes)

    # Theta[c, j] = P(x_j = 1 | class c)
    theta = np.zeros((n_classes, n_features))

    for i, c in enumerate(classes):
        X_c = X_train[y_train == c]
        n_c = X_c.shape[0]

        log_priors[i] = np.log(n_c / n_train)

        # Laplace smoothing (alpha = 1)
        theta[i] = (X_c.sum(axis=0) + 1) / (n_c + 2)

    log_theta = np.log(theta)
    log_one_minus_theta = np.log(1 - theta)

    # Compute log posteriors
    scores = (
        X_test @ log_theta.T
        + (1 - X_test) @ log_one_minus_theta.T
        + log_priors
    )

    return scores