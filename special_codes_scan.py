"""Scan low-cardinality columns for BRFSS special codes.

Read-only analysis. Prints a summary and writes a full per-column report
to data/special_codes_report.txt (the data/ folder is ignored by Git).
"""

import sys

import numpy as np

from data_loading import get_train_data, labels_to_01
from leakage_check import stratified_indices

SPECIAL_CODES = [7, 9, 77, 88, 99, 777, 999]
MAX_DISTINCT = 30
TOP_K = 6
PRINT_LINES = 20
REPORT_PATH = "data/special_codes_report.txt"


def scan_column(col):
    """Return (nan_share, values, counts) of the non-NaN entries of a column."""
    valid = col[~np.isnan(col)]
    values, counts = np.unique(valid, return_counts=True)
    nan_share = 1.0 - len(valid) / len(col)
    return nan_share, values, counts


def format_value(v):
    """Print whole numbers without a decimal point."""
    return str(int(v)) if float(v) == int(v) else str(float(v))


def describe_column(name, nan_share, values, counts):
    """Build one report line for a column, or None if no special code occurs."""
    found = []
    for code in SPECIAL_CODES:
        n = int(counts[values == code].sum())
        if n > 0:
            found.append((code, n))
    if not found:
        return None, found
    specials = " ".join("%d:%d" % (c, n) for c, n in found)
    order = np.argsort(-counts)[:TOP_K]
    top = " ".join("%s:%d" % (format_value(values[i]), counts[i]) for i in order)
    line = "%-14s distinct=%2d nan=%5.1f%% | specials %s | top %s" % (
        name,
        len(values),
        100 * nan_share,
        specials,
        top,
    )
    return line, found


def main():
    n_rows = int(sys.argv[1]) if len(sys.argv) > 1 else None
    x, y, ids, names = get_train_data()
    y01 = labels_to_01(y)
    if n_rows is not None:
        idx = stratified_indices(y, n_rows)
        x = x[idx]
        y01 = y01[idx]
    print("Rows used: %d, positives: %.2f%%" % (len(x), 100 * y01.mean()))

    lines = []
    n_scanned = 0
    columns_with_code = {code: 0 for code in SPECIAL_CODES}
    for j in range(x.shape[1]):
        nan_share, values, counts = scan_column(x[:, j])
        if len(values) == 0 or len(values) > MAX_DISTINCT:
            continue
        n_scanned += 1
        line, found = describe_column(str(names[j]), nan_share, values, counts)
        if line is not None:
            lines.append(line)
            for code, _ in found:
                columns_with_code[code] += 1

    print("Columns scanned (<= %d distinct values): %d" % (MAX_DISTINCT, n_scanned))
    print("Columns where at least one special code occurs: %d" % len(lines))
    print("Number of columns containing each code:")
    for code in SPECIAL_CODES:
        print("  code %3d: %d columns" % (code, columns_with_code[code]))

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\nFull report saved to %s (%d lines)." % (REPORT_PATH, len(lines)))
    print("First %d lines:" % PRINT_LINES)
    for line in lines[:PRINT_LINES]:
        print(line)


if __name__ == "__main__":
    main()
