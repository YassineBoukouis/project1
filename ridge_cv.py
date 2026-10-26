"""Ridge regression over a grid of penalty strengths, by cross-validation.

Uses exactly the same folds, cleaning and threshold sweep as
cross_validation.py and threshold_sweep.py, but trains only ridge regression,
so the slower logistic models are not run again. Ridge scores estimate the
0/1 label (like least squares), so they are compared at their own best
threshold, not at 0.5.

Usage:
    python ridge_cv.py 20000   # stratified sample of 20000 rows
    python ridge_cv.py         # all training rows

The out-of-fold scores are saved in the cache folder (ignored by Git).
"""

import os
import sys
import time

import numpy as np

from cross_validation import FOLD_SEED, N_FOLDS, SAMPLE_SEED, stratified_folds
from data_loading import CACHE_DIR, get_train_data, labels_to_01
from implementations import ridge_regression
from leakage_check import stratified_indices
from preprocessing import DEFAULT_CONFIG, fit, transform
from threshold_sweep import THRESHOLDS, fold_stats, pick_best, sweep

LAMBDAS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]


def lambda_key(lambda_):
    """Name used for the scores of one penalty strength in the saved file."""
    return f"ridge_lambda_{lambda_:g}"


def cross_validate_ridge(x, y, names, config, lambdas):
    """Out-of-fold ridge scores for every lambda, on the same folds as before.

    The cleaning is fitted once per fold on the training part only. Returns
    one array of scores per lambda (one score per row) and the fold number of
    every row.
    """
    fold_of = stratified_folds(y, N_FOLDS, FOLD_SEED)
    y01 = labels_to_01(y)
    oof = {lambda_key(lam): np.zeros(len(y)) for lam in lambdas}
    for k in range(N_FOLDS):
        start = time.time()
        val = np.where(fold_of == k)[0]
        train = np.where(fold_of != k)[0]
        state = fit(x[train], names, config)
        tx_train = transform(x[train], state).astype(np.float64)
        tx_val = transform(x[val], state).astype(np.float64)
        print(
            f"Fold {k + 1}/{N_FOLDS}: train {len(train)}, validation {len(val)}, "
            f"columns {tx_train.shape[1]}, cleaning {time.time() - start:.1f} s"
        )
        for lam in lambdas:
            start = time.time()
            w, _ = ridge_regression(y01[train], tx_train, lam)
            oof[lambda_key(lam)][val] = tx_val @ w
            print(f"  lambda {lam:g}: {time.time() - start:.1f} s")
        del tx_train, tx_val
    return oof, fold_of


def print_ridge_table(y, fold_of, oof, lambdas):
    """Print, for every lambda, the best threshold and its F1 over the folds."""
    print()
    print("Ridge at its own best-F1 threshold (all out-of-fold rows together)")
    print("  lambda   threshold  accuracy   F1 (pooled)   F1 over folds")
    for lam in lambdas:
        score = oof[lambda_key(lam)]
        rows = sweep(y, score, THRESHOLDS)
        best = pick_best(rows, "f1")
        stats = fold_stats(y, score, fold_of, best["threshold"])
        mean_f1, std_f1 = stats["f1"]
        print(
            f"  {lam:<8g} {best['threshold']:9.2f}  {best['accuracy']:8.4f}"
            f"  {best['f1']:11.4f}   {mean_f1:.4f} +/- {std_f1:.4f}"
        )
        if best["threshold"] in (THRESHOLDS[0], THRESHOLDS[-1]):
            print("    Note: best threshold is at the edge of the grid.")


def print_reference(path, y, fold_of):
    """Print least squares from the earlier run, if it used the same rows."""
    if not os.path.exists(path):
        print(f"\nNo earlier scores found at {path}; no reference row.")
        return
    data = np.load(path)
    same_rows = len(data["y"]) == len(y) and np.array_equal(data["y"], y)
    if not same_rows or not np.array_equal(data["fold"], fold_of):
        print("\nEarlier scores use different rows or folds; no reference row.")
        return
    if "least_squares" not in data.files:
        return
    score = data["least_squares"]
    best = pick_best(sweep(y, score, THRESHOLDS), "f1")
    mean_f1, std_f1 = fold_stats(y, score, fold_of, best["threshold"])["f1"]
    print(
        f"\nReference, least squares (earlier run): threshold "
        f"{best['threshold']:.2f}, F1 pooled {best['f1']:.4f}, "
        f"over folds {mean_f1:.4f} +/- {std_f1:.4f}"
    )


def main():
    n_sample = int(sys.argv[1]) if len(sys.argv) > 1 else None
    x, y, ids, names = get_train_data()
    if n_sample is not None:
        idx = stratified_indices(y, n_sample, seed=SAMPLE_SEED)
        x, y = x[idx], y[idx]
    print(f"Rows: {len(y)}, share of +1: {np.mean(y == 1):.4f}")

    # Same cleaning settings as in cross_validation.py.
    config = dict(DEFAULT_CONFIG)
    config["clip_standardized"] = 10.0
    config["add_bias"] = True

    start = time.time()
    oof, fold_of = cross_validate_ridge(x, y, names, config, LAMBDAS)
    print(f"\nRidge cross-validation took {time.time() - start:.1f} s")
    print_ridge_table(y, fold_of, oof, LAMBDAS)

    tag = "full" if n_sample is None else f"sample{n_sample}"
    print_reference(os.path.join(str(CACHE_DIR), f"oof_scores_{tag}.npz"), y, fold_of)
    path = os.path.join(str(CACHE_DIR), f"oof_scores_ridge_{tag}.npz")
    np.savez(path, y=y, fold=fold_of, **oof)
    print(f"\nSaved out-of-fold scores to {path}")


if __name__ == "__main__":
    main()
