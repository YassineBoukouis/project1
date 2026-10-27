"""Class-weighted logistic regression by gradient descent.

Every row with label 1 counts pos_weight times as much as a row with label 0
in the loss and the gradient. Both are divided by the sum of the row weights,
so the size of the gradient stays comparable to the unweighted version, and
pos_weight = 1 gives exactly regularized logistic regression.

Command line: ``python weighted_logistic.py`` runs a check on made-up data.
"""

import numpy as np

from implementations import reg_logistic_regression


def sigmoid(z):
    """Numerically stable sigmoid, 1 / (1 + exp(-z))."""
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def sample_weights(y, pos_weight):
    """Weight of every row: pos_weight where the label is 1, else 1."""
    return np.where(y == 1, pos_weight, 1.0)


def weighted_loss(y, tx, w, pos_weight):
    """Weighted negative log-likelihood (labels in {0, 1}), no penalty term."""
    z = tx @ w
    s = sample_weights(y, pos_weight)
    return np.sum(s * (np.logaddexp(0.0, z) - y * z)) / np.sum(s)


def weighted_gradient(y, tx, w, pos_weight):
    """Gradient of the weighted loss with respect to w."""
    s = sample_weights(y, pos_weight)
    return tx.T @ (s * (sigmoid(tx @ w) - y)) / np.sum(s)


def weighted_logistic_regression(
    y, tx, lambda_, initial_w, max_iters, gamma, pos_weight
):
    """Class-weighted logistic regression with penalty lambda_ * ||w||^2.

    Args:
        y: shape (N,), labels in {0, 1}.
        tx: shape (N, D), data matrix.
        lambda_: regularization strength (>= 0).
        initial_w: shape (D,), starting weights.
        max_iters: number of gradient steps.
        gamma: step size.
        pos_weight: weight of the rows with label 1 (rows with label 0 get 1).

    Returns:
        w: shape (D,), weights after the last step.
        loss: weighted negative log-likelihood at the final w, without the
            penalty term.
    """
    w = np.array(initial_w, dtype=float)
    for _ in range(max_iters):
        grad = weighted_gradient(y, tx, w, pos_weight) + 2.0 * lambda_ * w
        w = w - gamma * grad
    return w, weighted_loss(y, tx, w, pos_weight)


def main():
    """Check the code on made-up data (no real data is used)."""
    rng = np.random.RandomState(0)
    n, d = 2000, 6
    tx = np.hstack([np.ones((n, 1)), rng.randn(n, d - 1)])
    true_w = np.array([-2.5, 1.0, -0.5, 0.8, 0.0, 0.3])
    y = (rng.rand(n) < sigmoid(tx @ true_w)).astype(float)
    print(f"Share of 1 in the made-up labels: {y.mean():.3f}")

    # 1. Weight 1 must give the same result as reg_logistic_regression.
    w0 = np.zeros(d)
    w_ref, loss_ref = reg_logistic_regression(y, tx, 1e-4, w0, 100, 0.1)
    w_one, loss_one = weighted_logistic_regression(y, tx, 1e-4, w0, 100, 0.1, 1.0)
    print(
        f"Weight 1 against reg_logistic_regression: max |w difference| "
        f"{np.max(np.abs(w_one - w_ref)):.2e}, loss difference "
        f"{abs(loss_one - loss_ref):.2e}"
    )

    # 2. The gradient must match a finite-difference estimate (weight 10).
    w = rng.randn(d) * 0.3
    numeric = np.zeros(d)
    for j in range(d):
        step = np.zeros(d)
        step[j] = 1e-6
        up = weighted_loss(y, tx, w + step, 10.0)
        down = weighted_loss(y, tx, w - step, 10.0)
        numeric[j] = (up - down) / 2e-6
    analytic = weighted_gradient(y, tx, w, 10.0)
    print(f"Gradient check: max difference {np.max(np.abs(analytic - numeric)):.2e}")

    # 3. A larger weight should push the predicted probabilities up.
    w_ten, _ = weighted_logistic_regression(y, tx, 1e-4, w0, 300, 0.1, 10.0)
    print(
        f"Mean predicted probability: weight 1 {sigmoid(tx @ w_ref).mean():.3f}, "
        f"weight 10 {sigmoid(tx @ w_ten).mean():.3f}"
    )


if __name__ == "__main__":
    main()
