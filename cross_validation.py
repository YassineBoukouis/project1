"""Stratified k-fold cross-validation with out-of-fold predictions.

Every training row is predicted by a model that never saw it. Inside each
fold the preprocessing is fitted on the training part only and then applied
to the validation part, so the validation scores are not too optimistic.

Usage:
    python cross_validation.py 20000   # stratified sample of 20000 rows
    python cross_validation.py         # all training rows

The out-of-fold scores are saved in the cache folder (ignored by Git).
"""

import os
import sys
import time

import numpy as np

from data_loading import CACHE_DIR, get_train_data, labels_to_01
from implementations import (
    least_squares,
    logistic_regression,
    reg_logistic_regression,
    ridge_regression,
)
from leakage_check import stratified_indices
from preprocessing import DEFAULT_CONFIG, fit, transform

N_FOLDS = 5
FOLD_SEED = 1
SAMPLE_SEED = 0
GAMMA = 0.1
MAX_ITERS = 300
LAMBDA = 1e-4
THRESHOLD = 0.5
METRICS = ("accuracy", "precision", "recall", "f1")


def sigmoid(z):
    """Numerically stable sigmoid, 1 / (1 + exp(-z))."""
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def stratified_folds(y, n_folds=N_FOLDS, seed=FOLD_SEED):
    """Return the fold number (0 to n_folds - 1) of every row.

    Labels are in {-1, +1}. The rows of each class are shuffled with a fixed
    seed and cut into n_folds parts of almost equal size, so every fold has
    almost the same share of positive rows as the whole data.
    """
    rng = np.random.RandomState(seed)
    fold_of = np.zeros(len(y), dtype=int)
    for label in (-1, 1):
        rows = rng.permutation(np.where(y == label)[0])
        for k, part in enumerate(np.array_split(rows, n_folds)):
            fold_of[part] = k
    return fold_of


def compute_metrics(y, pred):
    """Return accuracy, precision, recall and F1 (labels in {-1, +1}).

    Precision is the share of predicted +1 that are really +1, recall is the
    share of real +1 that were found, F1 is their harmonic mean, written here
    as 2 tp / (2 tp + fp + fn). A metric with an empty denominator is 0.
    """
    tp = np.sum((pred == 1) & (y == 1))
    fp = np.sum((pred == 1) & (y == -1))
    fn = np.sum((pred == -1) & (y == 1))
    return {
        "accuracy": float(np.mean(pred == y)),
        "precision": float(tp / (tp + fp)) if tp + fp > 0 else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn > 0 else 0.0,
        "f1": float(2 * tp / (2 * tp + fp + fn)) if tp > 0 else 0.0,
        "predicted_positive": int(tp + fp),
    }


def scores_all_negative(tx_train, y01, tx_val):
    """Baseline without a model: every row gets the score 0 (always -1)."""
    return np.zeros(tx_val.shape[0])


def scores_least_squares(tx_train, y01, tx_val):
    """Least squares on the 0/1 labels; the score estimates the label."""
    try:
        w, _ = least_squares(y01, tx_train)
    except np.linalg.LinAlgError:
        print("  least squares: singular matrix, using ridge with lambda 1e-8")
        w, _ = ridge_regression(y01, tx_train, 1e-8)
    return tx_val @ w


def scores_logistic(tx_train, y01, tx_val):
    """Logistic regression by gradient descent; the score is a probability."""
    w0 = np.zeros(tx_train.shape[1])
    w, _ = logistic_regression(y01, tx_train, w0, MAX_ITERS, GAMMA)
    return sigmoid(tx_val @ w)


def scores_reg_logistic(tx_train, y01, tx_val):
    """Regularized logistic regression; the score is a probability."""
    w0 = np.zeros(tx_train.shape[1])
    w, _ = reg_logistic_regression(y01, tx_train, LAMBDA, w0, MAX_ITERS, GAMMA)
    return sigmoid(tx_val @ w)


# (key used in file names, title in tables, function returning validation scores)
MODELS = [
    ("all_negative", "all -1", scores_all_negative),
    ("least_squares", "least squares", scores_least_squares),
    ("logistic", "logistic", scores_logistic),
    ("reg_logistic", "reg. logistic", scores_reg_logistic),
]


def cross_validate(x, y, names, config, models, n_folds=N_FOLDS, seed=FOLD_SEED):
    """Run stratified k-fold cross-validation for several models.

    Returns the out-of-fold scores (one array per model key, one score per
    row), the metrics of every fold at THRESHOLD, and the fold number of
    every row. All models see exactly the same folds and the same cleaned
    matrices.
    """
    fold_of = stratified_folds(y, n_folds, seed)
    y01 = labels_to_01(y)
    oof = {key: np.zeros(len(y)) for key, _, _ in models}
    per_fold = {key: [] for key, _, _ in models}
    for k in range(n_folds):
        start = time.time()
        val = np.where(fold_of == k)[0]
        train = np.where(fold_of != k)[0]
        state = fit(x[train], names, config)
        tx_train = transform(x[train], state).astype(np.float64)
        tx_val = transform(x[val], state).astype(np.float64)
        print(
            f"Fold {k + 1}/{n_folds}: train {len(train)}, validation {len(val)}, "
            f"columns {tx_train.shape[1]}, cleaning {time.time() - start:.1f} s"
        )
        for key, title, model in models:
            start = time.time()
            scores = model(tx_train, y01[train], tx_val)
            oof[key][val] = scores
            pred = np.where(scores > THRESHOLD, 1, -1)
            per_fold[key].append(compute_metrics(y[val], pred))
            print(f"  {title}: {time.time() - start:.1f} s")
        del tx_train, tx_val
    return oof, per_fold, fold_of


def print_fold_table(per_fold, models):
    """Print mean +/- standard deviation over the folds for every model."""
    print()
    print(f"Metrics at threshold {THRESHOLD} (mean +/- std over folds)")
    print(f"{'model':<15}" + "".join(f"{m:<19}" for m in METRICS))
    for key, title, _ in models:
        row = f"{title:<15}"
        for metric in METRICS:
            values = np.array([r[metric] for r in per_fold[key]])
            row += f"{values.mean():.4f} +/- {values.std(ddof=1):.4f}  "
        print(row)


def print_pooled(y, oof, models):
    """Print the metrics of all out-of-fold predictions taken together."""
    print()
    print(f"All out-of-fold rows together, threshold {THRESHOLD}")
    for key, title, _ in models:
        pred = np.where(oof[key] > THRESHOLD, 1, -1)
        r = compute_metrics(y, pred)
        print(
            f"{title:<15} accuracy {r['accuracy']:.4f}, "
            f"precision {r['precision']:.4f}, recall {r['recall']:.4f}, "
            f"F1 {r['f1']:.4f}, predicted +1: {r['predicted_positive']}"
        )


def main():
    n_sample = int(sys.argv[1]) if len(sys.argv) > 1 else None
    x, y, ids, names = get_train_data()
    if n_sample is not None:
        idx = stratified_indices(y, n_sample, seed=SAMPLE_SEED)
        x, y = x[idx], y[idx]
    print(f"Rows: {len(y)}, share of +1: {np.mean(y == 1):.4f}")
    print(f"All -1 accuracy: {np.mean(y == -1):.4f}")

    config = dict(DEFAULT_CONFIG)
    config["clip_standardized"] = 10.0
    config["add_bias"] = True

    start = time.time()
    oof, per_fold, fold_of = cross_validate(x, y, names, config, MODELS)
    print(f"\nCross-validation took {time.time() - start:.1f} s")
    print_fold_table(per_fold, MODELS)
    print_pooled(y, oof, MODELS)

    tag = "full" if n_sample is None else f"sample{n_sample}"
    path = os.path.join(str(CACHE_DIR), f"oof_scores_{tag}.npz")
    np.savez(path, y=y, fold=fold_of, **oof)
    print(f"\nSaved out-of-fold scores to {path}")


if __name__ == "__main__":
    main()
