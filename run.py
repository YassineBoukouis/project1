"""Train the final model and write the submission file.

Trains class-weighted regularized logistic regression on all training rows
and predicts the test rows. Gradient descent starts from zero weights and
uses no random numbers, so every run writes the same submission.csv.

Usage:
    python run.py

Expects the data in data/dataset/ (see README.md).
"""

import time

import numpy as np

from data_loading import get_test_data, get_train_data, labels_to_01
from helpers import create_csv_submission
from preprocessing import DEFAULT_CONFIG, fit, transform
from weighted_logistic import sigmoid, weighted_logistic_regression

LAMBDA = 1e-4
GAMMA = 0.1
MAX_ITERS = 600
POS_WEIGHT = 3.0
# Cut-off with the best F1 on the out-of-fold scores of this model.
THRESHOLD = 0.42
OUTPUT_FILE = "submission.csv"


def predict(tx, w, threshold):
    """Return labels in {-1, +1}: +1 where the probability exceeds threshold."""
    return np.where(sigmoid(tx @ w) > threshold, 1, -1)


def print_metrics(y, pred, title):
    """Print accuracy, precision, recall and F1 (labels in {-1, +1})."""
    tp = np.sum((pred == 1) & (y == 1))
    fp = np.sum((pred == 1) & (y == -1))
    fn = np.sum((pred == -1) & (y == 1))
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if tp > 0 else 0.0
    print(
        f"{title}: accuracy {np.mean(pred == y):.4f}, precision {precision:.4f}, "
        f"recall {recall:.4f}, F1 {f1:.4f}, predicted +1: {np.sum(pred == 1)}"
    )


def main():
    x, y, ids, names = get_train_data()
    print(f"Training rows: {len(y)}, share of +1: {np.mean(y == 1):.4f}")

    config = dict(DEFAULT_CONFIG)
    config["clip_standardized"] = 10.0
    config["add_bias"] = True
    start = time.time()
    state = fit(x, names, config)
    tx = transform(x, state).astype(np.float64)
    del x
    print(f"Cleaned matrix {tx.shape} in {time.time() - start:.1f} s")

    start = time.time()
    w, loss = weighted_logistic_regression(
        labels_to_01(y), tx, LAMBDA, np.zeros(tx.shape[1]), MAX_ITERS, GAMMA, POS_WEIGHT
    )
    print(f"Trained {MAX_ITERS} steps, loss {loss:.5f} ({time.time() - start:.1f} s)")
    print_metrics(y, predict(tx, w, THRESHOLD), "Training rows")
    del tx

    x_test, ids_test = get_test_data()
    pred_test = predict(transform(x_test, state).astype(np.float64), w, THRESHOLD)
    print(f"Test rows: {len(pred_test)}, predicted +1: {np.sum(pred_test == 1)}")
    create_csv_submission(ids_test, pred_test, OUTPUT_FILE)
    print(f"Wrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
