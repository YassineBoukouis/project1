"""Summarise columns with many distinct values (skipped by the special codes scan).

Read-only. Prints a short summary and writes one line per column to
data/numeric_report.txt (the data/ folder is ignored by Git).
"""

import sys

import numpy as np

from data_loading import get_train_data
from leakage_check import stratified_indices

MIN_DISTINCT = 31
TOP_K = 5
LARGEST_K = 3
REPORT_PATH = "data/numeric_report.txt"


def fmt(value):
    """Format a value without a trailing .0 for whole numbers."""
    if float(value).is_integer():
        return str(int(value))
    return "%.4g" % value


def describe_column(name, col):
    """Return one report line for a column, or None if it has few distinct values."""
    valid = col[~np.isnan(col)]
    values, counts = np.unique(valid, return_counts=True)
    if len(values) < MIN_DISTINCT:
        return None
    nan_share = 100.0 * (1.0 - len(valid) / len(col))
    order = np.argsort(-counts, kind="stable")[:TOP_K]
    top = " ".join("%s:%d" % (fmt(values[i]), counts[i]) for i in order)
    last = len(values) - 1
    largest = " ".join(
        "%s:%d" % (fmt(values[i]), counts[i]) for i in range(last, last - LARGEST_K, -1)
    )
    return "%-10s distinct=%6d nan=%5.1f%% min=%s | top %s | largest %s" % (
        name,
        len(values),
        nan_share,
        fmt(values[0]),
        top,
        largest,
    )


def main():
    x, y, ids, names = get_train_data()
    if len(sys.argv) > 1:
        x = x[stratified_indices(y, int(sys.argv[1]))]
    lines = []
    for j in range(x.shape[1]):
        line = describe_column(str(names[j]), x[:, j])
        if line is not None:
            lines.append(line)
    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("Rows used: %d" % x.shape[0])
    print("Columns with >= %d distinct values: %d" % (MIN_DISTINCT, len(lines)))
    print("Full report saved to %s" % REPORT_PATH)


if __name__ == "__main__":
    main()
