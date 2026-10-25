"""Train a regularized logistic regression and write a submission file.

Usage:
    python make_submission.py 20000   # stratified sample, no file written
    python make_submission.py         # all rows, writes submission.csv
"""

import sys
import time

import numpy as np

from data_loading import get_test_data, get_train_data, labels_to_01
from helpers import create_csv_submission
from implementations import reg_logistic_regression
from leakage_check import stratified_indices
from preprocessing import DEFAULT_CONFIG, fit, transform

LAMBDA = 1e-4
GAMMA = 0.1
CHUNKS = 6
ITERS_PER_CHUNK = 50
# Cut-off with the best F1 on the out-of-fold scores of the threshold sweep.
THRESHOLD = 0.19
OUTPUT_FILE = "submission.csv"


def sigmoid(z):
    """Numerically stable sigmoid, 1 / (1 + exp(-z))."""
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def predict(tx, w, threshold):
    """Return labels in {-1, +1}: +1 where the probability exceeds threshold."""
    return np.where(sigmoid(tx @ w) > threshold, 1, -1)


def print_metrics(y, pred, title):
    """Print accuracy, precision, recall and F1 (labels in {-1, +1})."""
    tp = np.sum((pred == 1) & (y == 1))
    fp = np.sum((pred == 1) & (y == -1))
    fn = np.sum((pred == -1) & (y == 1))
    accuracy = np.mean(pred == y)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if tp > 0 else 0.0
    print(
        f"{title}: accuracy {accuracy:.4f}, precision {precision:.4f}, "
        f"recall {recall:.4f}, F1 {f1:.4f}, predicted +1: {np.sum(pred == 1)}"
    )


def main():
    n_sample = int(sys.argv[1]) if len(sys.argv) > 1 else None
    x, y, ids, names = get_train_data()
    if n_sample is not None:
        idx = stratified_indices(y, n_sample, seed=0)
        x, y = x[idx], y[idx]
    print(f"Training rows: {len(y)}, share of +1: {np.mean(y == 1):.4f}")
    print(f"All -1 accuracy: {np.mean(y == -1):.4f}")

    config = dict(DEFAULT_CONFIG)
    config["clip_standardized"] = 10.0
    config["add_bias"] = True
    start = time.time()
    state = fit(x, names, config)
    tx = transform(x, state).astype(np.float64)
    del x
    print(f"Cleaned matrix {tx.shape} in {time.time() - start:.1f} s")

    y01 = labels_to_01(y)
    w = np.zeros(tx.shape[1])
    for chunk in range(CHUNKS):
        start = time.time()
        w, loss = reg_logistic_regression(y01, tx, LAMBDA, w, ITERS_PER_CHUNK, GAMMA)
        done = (chunk + 1) * ITERS_PER_CHUNK
        print(f"after {done} steps: loss {loss:.5f} ({time.time() - start:.1f} s)")

    print_metrics(y, predict(tx, w, THRESHOLD), "Training rows")
    del tx

    x_test, ids_test = get_test_data()
    tx_test = transform(x_test, state).astype(np.float64)
    pred_test = predict(tx_test, w, THRESHOLD)
    print(f"Test rows: {len(pred_test)}, predicted +1: {np.sum(pred_test == 1)}")
    if n_sample is None:
        create_csv_submission(ids_test, pred_test, OUTPUT_FILE)
        print(f"Wrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
