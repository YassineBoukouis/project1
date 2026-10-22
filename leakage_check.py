"""Step 1b: look for feature columns that leak the label (_MICHD)."""
import sys

import numpy as np

from data_loading import get_train_data

KEYWORDS = ("MICHD", "CVDINFR", "CVDCRHD")


def stratified_indices(y, n, seed=0):
    """Pick n rows keeping the share of +1 labels the same as in y."""
    rng = np.random.RandomState(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == -1)[0]
    n_pos = int(round(n * len(pos) / len(y)))
    n_neg = n - n_pos
    idx = np.concatenate(
        [
            rng.choice(pos, n_pos, replace=False),
            rng.choice(neg, n_neg, replace=False),
        ]
    )
    idx.sort()
    return idx


def corr_with_label(col, y01):
    """Pearson correlation between a column and the label, ignoring NaN."""
    ok = ~np.isnan(col)
    if ok.sum() < 100:
        return np.nan
    a = col[ok].astype(np.float64)
    b = y01[ok]
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return np.corrcoef(a, b)[0, 1]


def lookup_rule(col, y01, max_levels=30):
    """Best 'one prediction per distinct value' rule using only this column.

    NaN counts as its own value. Returns None if there are too many values.
    Otherwise returns (accuracy, levels, counts, n_positive_per_level).
    """
    filled = np.where(np.isnan(col), -999999.0, col)
    levels, inverse = np.unique(filled, return_inverse=True)
    if len(levels) > max_levels:
        return None
    inverse = inverse.ravel()
    counts = np.bincount(inverse, minlength=len(levels))
    pos = np.bincount(inverse, weights=y01, minlength=len(levels))
    acc = np.maximum(pos, counts - pos).sum() / len(y01)
    return acc, levels, counts, pos


def main():
    n_rows = int(sys.argv[1]) if len(sys.argv) > 1 else None
    x, y, ids, names = get_train_data()
    if n_rows is not None:
        idx = stratified_indices(y, n_rows)
        x, y = x[idx], y[idx]
    y01 = (y == 1).astype(np.float64)
    n_cols = x.shape[1]
    nan_share = np.isnan(x).mean(axis=0)

    print("Rows used: %d, columns: %d" % (x.shape[0], n_cols))
    print("Share of +1: %.4f" % y01.mean())
    print("Baseline accuracy (all -1): %.4f" % (1 - y01.mean()))

    # 1. Correlations
    corrs = np.array([corr_with_label(x[:, j], y01) for j in range(n_cols)])
    order = np.argsort(-np.nan_to_num(np.abs(corrs)))
    print("\n=== Top 15 columns by |correlation| with the label ===")
    for j in order[:15]:
        print("%-12s corr=%+.4f  NaN share=%.3f" % (names[j], corrs[j], nan_share[j]))

    # 2. Single-column lookup rules
    results = []
    for j in range(n_cols):
        r = lookup_rule(x[:, j], y01)
        if r is not None:
            results.append((r[0], j, len(r[1])))
    results.sort(reverse=True)
    print("\n=== Top 15 columns by single-column rule accuracy ===")
    print("(only columns with at most 30 distinct values, NaN counted as a value)")
    for acc, j, n_levels in results[:15]:
        print(
            "%-12s accuracy=%.4f  levels=%d  NaN share=%.3f"
            % (names[j], acc, n_levels, nan_share[j])
        )
    n_flag = sum(1 for acc, _, _ in results if acc >= 0.99)
    print("Columns with rule accuracy >= 0.99: %d" % n_flag)

    # 3. Columns whose name hints at the label
    print("\n=== Columns whose name contains %s ===" % ", ".join(KEYWORDS))
    found = False
    for j in range(n_cols):
        if any(k in str(names[j]).upper() for k in KEYWORDS):
            found = True
            print("\n%s (NaN share=%.3f)" % (names[j], nan_share[j]))
            r = lookup_rule(x[:, j], y01)
            if r is None:
                print("  too many distinct values to tabulate")
                continue
            acc, levels, counts, pos = r
            print("  rule accuracy = %.4f" % acc)
            for lv, c, p in zip(levels, counts, pos):
                label = "NaN" if lv == -999999.0 else "%g" % lv
                print("  value %-6s n=%-7d share +1 = %.4f" % (label, c, p / c))
    if not found:
        print("No column name matches.")


if __name__ == "__main__":
    main()