"""Cross-validation of class-weighted logistic regression.

Same folds and cleaning as cross_validation.py. For every class weight the
model is trained in one run and the scores are saved after each checkpoint
number of gradient steps, so different numbers of steps are compared without
training twice. Every model is compared at its own best-F1 threshold.

Usage:
    python weighted_cv.py 20000   # stratified sample of 20000 rows
    python weighted_cv.py         # all training rows
"""

import os
import sys
import time

import numpy as np

from cross_validation import FOLD_SEED, N_FOLDS, SAMPLE_SEED, stratified_folds
from cross_validation import GAMMA, LAMBDA
from data_loading import CACHE_DIR, get_train_data, labels_to_01
from leakage_check import stratified_indices
from preprocessing import DEFAULT_CONFIG, fit, transform
from threshold_sweep import fold_stats, pick_best, sweep
from weighted_logistic import sigmoid, weighted_logistic_regression

WEIGHTS = [3.0, 10.0]
CHECKPOINTS = [300, 600]
THRESHOLDS = np.round(np.arange(0.05, 0.9501, 0.01), 2)


def score_key(weight, steps):
    """Name of the scores of one weight and number of steps in the file."""
    return f"weight{weight:g}_steps{steps}"


def main():
    n_sample = int(sys.argv[1]) if len(sys.argv) > 1 else None
    x, y, ids, names = get_train_data()
    if n_sample is not None:
        idx = stratified_indices(y, n_sample, seed=SAMPLE_SEED)
        x, y = x[idx], y[idx]
    print(f"Rows: {len(y)}, share of +1: {np.mean(y == 1):.4f}")

    config = dict(DEFAULT_CONFIG)
    config["clip_standardized"] = 10.0
    config["add_bias"] = True

    fold_of = stratified_folds(y, N_FOLDS, FOLD_SEED)
    y01 = labels_to_01(y)
    keys = [score_key(wt, s) for wt in WEIGHTS for s in CHECKPOINTS]
    oof = {key: np.zeros(len(y)) for key in keys}
    total = time.time()
    for k in range(N_FOLDS):
        start = time.time()
        val = np.where(fold_of == k)[0]
        train = np.where(fold_of != k)[0]
        state = fit(x[train], names, config)
        tx_train = transform(x[train], state).astype(np.float64)
        tx_val = transform(x[val], state).astype(np.float64)
        print(f"Fold {k + 1}/{N_FOLDS}: cleaning {time.time() - start:.1f} s")
        for weight in WEIGHTS:
            w = np.zeros(tx_train.shape[1])
            done = 0
            for steps in CHECKPOINTS:
                start = time.time()
                w, _ = weighted_logistic_regression(
                    y01[train], tx_train, LAMBDA, w, steps - done, GAMMA, weight
                )
                done = steps
                oof[score_key(weight, steps)][val] = sigmoid(tx_val @ w)
                print(
                    f"  weight {weight:g}, {steps} steps: {time.time() - start:.1f} s"
                )
        del tx_train, tx_val
    print(f"\nCross-validation took {time.time() - total:.1f} s")

    print()
    print("Each model at its own best-F1 threshold (all out-of-fold rows)")
    print("  model                  thr  accuracy  F1 pooled  F1 over folds")
    for key in keys:
        best = pick_best(sweep(y, oof[key], THRESHOLDS), "f1")
        mean_f1, std_f1 = fold_stats(y, oof[key], fold_of, best["threshold"])["f1"]
        print(
            f"  {key:<22} {best['threshold']:.2f}  {best['accuracy']:8.4f}"
            f"  {best['f1']:9.4f}  {mean_f1:.4f} +/- {std_f1:.4f}"
        )
        if best["threshold"] in (THRESHOLDS[0], THRESHOLDS[-1]):
            print("    Note: best threshold is at the edge of the grid.")

    tag = "full" if n_sample is None else f"sample{n_sample}"
    path = os.path.join(str(CACHE_DIR), f"oof_scores_weighted_{tag}.npz")
    np.savez(path, y=y, fold=fold_of, **oof)
    print(f"\nSaved out-of-fold scores to {path}")


if __name__ == "__main__":
    main()
