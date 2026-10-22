import numpy as np

# Helper functions


def _mse_loss(y, tx, w):
    """Mean squared error with the factor 1/2: (1/2N) * sum (y - tx.w)^2."""
    e = y - tx @ w
    return 0.5 * np.mean(e**2)


def _mse_gradient(y, tx, w):
    """Gradient of the MSE loss with respect to w."""
    e = y - tx @ w
    return -(tx.T @ e) / len(y)


def _sigmoid(z):
    """Numerically stable sigmoid: 1 / (1 + exp(-z))."""
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def _logistic_loss(y, tx, w):
    """Negative log-likelihood for labels in {0, 1}, averaged over samples."""
    z = tx @ w
    return np.mean(np.logaddexp(0.0, z) - y * z)


def _logistic_gradient(y, tx, w):
    """Gradient of the logistic loss with respect to w."""
    return tx.T @ (_sigmoid(tx @ w) - y) / len(y)


# The six required functions


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent.

    Args:
        y: shape (N,), targets.
        tx: shape (N, D), data matrix.
        initial_w: shape (D,), starting weights.
        max_iters: number of gradient steps.
        gamma: step size.

    Returns:
        w: shape (D,), weights after the last step.
        loss: MSE loss (with factor 1/2) at the final w.
    """
    w = np.array(initial_w, dtype=float)
    for _ in range(max_iters):
        w = w - gamma * _mse_gradient(y, tx, w)
    return w, _mse_loss(y, tx, w)


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent (batch size 1).

    Args:
        y: shape (N,), targets.
        tx: shape (N, D), data matrix.
        initial_w: shape (D,), starting weights.
        max_iters: number of SGD steps.
        gamma: step size.

    Returns:
        w: shape (D,), weights after the last step.
        loss: MSE loss (with factor 1/2) on the full data at the final w.
    """
    w = np.array(initial_w, dtype=float)
    n = len(y)
    for _ in range(max_iters):
        i = np.random.randint(0, n)
        grad = -tx[i] * (y[i] - tx[i] @ w)
        w = w - gamma * grad
    return w, _mse_loss(y, tx, w)


def least_squares(y, tx):
    """Least squares regression using the normal equations.

    Solves (tx^T tx) w = tx^T y.

    Args:
        y: shape (N,), targets.
        tx: shape (N, D), data matrix.

    Returns:
        w: shape (D,), optimal weights.
        loss: MSE loss (with factor 1/2) at w.
    """
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    return w, _mse_loss(y, tx, w)


def ridge_regression(y, tx, lambda_):
    """Ridge regression using the normal equations.

    Solves (tx^T tx + 2 N lambda_ I) w = tx^T y. The returned loss is the
    plain MSE and does not include the penalty term.

    Args:
        y: shape (N,), targets.
        tx: shape (N, D), data matrix.
        lambda_: regularization strength (>= 0).

    Returns:
        w: shape (D,), optimal weights.
        loss: MSE loss (with factor 1/2) at w, without the penalty.
    """
    n, d = tx.shape
    a = tx.T @ tx + 2.0 * n * lambda_ * np.eye(d)
    w = np.linalg.solve(a, tx.T @ y)
    return w, _mse_loss(y, tx, w)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent.

    Args:
        y: shape (N,), labels in {0, 1}.
        tx: shape (N, D), data matrix.
        initial_w: shape (D,), starting weights.
        max_iters: number of gradient steps.
        gamma: step size.

    Returns:
        w: shape (D,), weights after the last step.
        loss: negative log-likelihood (mean over samples) at the final w.
    """
    w = np.array(initial_w, dtype=float)
    for _ in range(max_iters):
        w = w - gamma * _logistic_gradient(y, tx, w)
    return w, _logistic_loss(y, tx, w)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent.

    The penalty is lambda_ * ||w||^2. The returned loss does not include
    the penalty term.

    Args:
        y: shape (N,), labels in {0, 1}.
        tx: shape (N, D), data matrix.
        lambda_: regularization strength (>= 0).
        initial_w: shape (D,), starting weights.
        max_iters: number of gradient steps.
        gamma: step size.

    Returns:
        w: shape (D,), weights after the last step.
        loss: negative log-likelihood (mean over samples) at the final w,
            without the penalty.
    """
    w = np.array(initial_w, dtype=float)
    for _ in range(max_iters):
        grad = _logistic_gradient(y, tx, w) + 2.0 * lambda_ * w
        w = w - gamma * grad
    return w, _logistic_loss(y, tx, w)
