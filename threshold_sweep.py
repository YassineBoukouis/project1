"""Threshold sweep on out-of-fold scores.

Reads the scores saved by cross_validation.py and shows how accuracy,
precision, recall and F1 change when the cut-off between "predict -1" and
"predict +1" moves. A row is predicted +1 when its score is above the
threshold.

Usage:
    python threshold_sweep.py 20000   (scores of the 20,000-row sample)
    python threshold_sweep.py         (scores of all rows)
"""

import os
import sys

import numpy as np

from data_loading import CACHE_DIR

MODEL_TITLES = {
    "logistic": "logistic regression",
    "reg_logistic": "regularized logistic regression",
    "least_squares": "least squares",
}
THRESHOLDS = np.round(np.arange(0.05, 0.6001, 0.01), 2)
METRIC_NAMES = ("accuracy", "precision", "recall", "f1")


def confusion_counts(y, predicted_positive):
    """Return (tp, fp, fn, tn) for labels y in {-1, +1} and a boolean array."""
    actual = y == 1
    tp = int(np.sum(predicted_positive & actual))
    fp = int(np.sum(predicted_positive & ~actual))
    fn = int(np.sum(~predicted_positive & actual))
    tn = len(y) - tp - fp - fn
    return tp, fp, fn, tn


def metrics_from_counts(tp, fp, fn, tn):
    """Accuracy, precision, recall, F1 and number of predicted +1."""
    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    denominator = 2 * tp + fp + fn
    return {
        "accuracy": (tp + tn) / total,
        "precision": precision,
        "recall": recall,
        "f1": 2 * tp / denominator if denominator > 0 else 0.0,
        "predicted_positive": tp + fp,
    }


def sweep(y, score, thresholds):
    """Metrics on all given rows for every threshold (one dict per threshold)."""
    rows = []
    for threshold in thresholds:
        counts = confusion_counts(y, score > threshold)
        row = metrics_from_counts(*counts)
        row["threshold"] = float(threshold)
        rows.append(row)
    return rows


def pick_best(rows, key, min_accuracy=None):
    """Row with the largest value of key, optionally with accuracy >= min_accuracy."""
    candidates = [
        r for r in rows if min_accuracy is None or r["accuracy"] >= min_accuracy
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda r: r[key])


def fold_stats(y, score, fold, threshold):
    """Mean and standard deviation (ddof=1) of each metric over the folds."""
    per_fold = []
    for k in np.unique(fold):
        mask = fold == k
        counts = confusion_counts(y[mask], score[mask] > threshold)
        per_fold.append(metrics_from_counts(*counts))
    stats = {}
    for name in METRIC_NAMES:
        values = np.array([m[name] for m in per_fold])
        stats[name] = (values.mean(), values.std(ddof=1))
    return stats


def print_table(rows):
    """Print every fifth row of the sweep (steps of 0.05)."""
    print("  threshold  accuracy  precision  recall      F1  predicted +1")
    for r in rows:
        if int(round(r["threshold"] * 100)) % 5 != 0:
            continue
        print(
            f"  {r['threshold']:9.2f}  {r['accuracy']:8.4f}  {r['precision']:9.4f}"
            f"  {r['recall']:6.4f}  {r['f1']:6.4f}  {r['predicted_positive']:12d}"
        )


def print_option(label, row, y, score, fold):
    """Print one option: pooled numbers and mean +/- std over the folds."""
    if row is None:
        print(f"  {label}: no threshold qualifies")
        return
    stats = fold_stats(y, score, fold, row["threshold"])
    parts = [f"{n} {stats[n][0]:.4f} +/- {stats[n][1]:.4f}" for n in METRIC_NAMES]
    print(
        f"  {label:<26} threshold {row['threshold']:.2f}"
        f" ({row['predicted_positive']} predicted +1)"
    )
    print("      folds: " + ", ".join(parts))


def main():
    if len(sys.argv) > 1:
        file_name = f"oof_scores_sample{sys.argv[1]}.npz"
    else:
        file_name = "oof_scores_full.npz"
    path = os.path.join(str(CACHE_DIR), file_name)
    data = np.load(path)
    y = data["y"]
    fold = data["fold"]
    baseline = float(np.mean(y == -1))
    print(f"Loaded {path}")
    print(
        f"{len(y)} rows, {int(np.sum(y == 1))} positives, {len(np.unique(fold))} folds"
    )
    print(f"Accuracy of predicting -1 for everyone: {baseline:.4f}")

    for key in ("logistic", "reg_logistic", "least_squares"):
        if key not in data.files:
            continue
        score = data[key]
        rows = sweep(y, score, THRESHOLDS)
        print()
        print(f"=== {MODEL_TITLES[key]} (all out-of-fold rows together) ===")
        print_table(rows)

        best_f1 = pick_best(rows, "f1")
        best_acc = pick_best(rows, "accuracy")
        compromise = pick_best(rows, "f1", min_accuracy=baseline)
        print()
        print("  Options (threshold from the 0.01 grid, then mean +/- std over folds):")
        print_option("best F1", best_f1, y, score, fold)
        print_option("best F1, accuracy >= all -1", compromise, y, score, fold)
        print_option("best accuracy", best_acc, y, score, fold)
        at_half = [r for r in rows if abs(r["threshold"] - 0.5) < 1e-9]
        if at_half:
            print_option("threshold 0.50 (for reference)", at_half[0], y, score, fold)
        edges = (THRESHOLDS[0], THRESHOLDS[-1])
        for label, row in (("best F1", best_f1), ("best accuracy", best_acc)):
            if row is not None and row["threshold"] in edges:
                print(f"  Note: {label} is at the edge of the grid; widen the grid.")


if __name__ == "__main__":
    main()
