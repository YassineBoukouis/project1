"""Exploratory data analysis (EDA) of the BRFSS training data.

Usage:
    python eda.py 20000   # stratified sample of 20000 rows, figures in figures_sample/
    python eda.py         # full training data, figures in figures/
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")  # draw to files only, no window
import matplotlib.pyplot as plt
import numpy as np

from data_loading import get_train_data, labels_to_01
from leakage_check import corr_with_label, stratified_indices

SPECIAL_CODES = [7, 9, 77, 88, 99, 777, 999]
EXAMPLE_COLUMNS = ["GENHLTH", "PHYSHLTH", "MENTHLTH", "SLEPTIM1", "CHILDREN"]


def save_fig(fig_dir, name):
    """Save the current matplotlib figure into fig_dir and close it."""
    os.makedirs(fig_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, name), dpi=150)
    plt.close()


def plot_class_balance(y, fig_dir):
    """Print and plot the number of -1 and +1 labels."""
    n_neg = int(np.sum(y == -1))
    n_pos = int(np.sum(y == 1))
    share = 100.0 * n_pos / (n_pos + n_neg)
    print(
        "Class balance: -1 = {}, +1 = {} ({:.2f}% positives)".format(
            n_neg, n_pos, share
        )
    )
    plt.figure(figsize=(4, 3))
    plt.bar(["-1 (no)", "+1 (yes)"], [n_neg, n_pos], color=["tab:blue", "tab:red"])
    plt.ylabel("Number of people")
    plt.title("Class balance")
    save_fig(fig_dir, "class_balance.png")


def analyse_nan(x, fig_dir):
    """Print and plot the share of NaN in every column."""
    nan_frac = np.mean(np.isnan(x), axis=0)
    print("\nOverall NaN share: {:.3f}".format(np.mean(nan_frac)))
    print("Columns with no NaN:     {}".format(int(np.sum(nan_frac == 0))))
    print("Columns with > 50% NaN:  {}".format(int(np.sum(nan_frac > 0.5))))
    print("Columns with > 80% NaN:  {}".format(int(np.sum(nan_frac > 0.8))))
    print("Columns with > 95% NaN:  {}".format(int(np.sum(nan_frac > 0.95))))
    plt.figure(figsize=(5, 3))
    plt.hist(nan_frac, bins=20, color="tab:gray")
    plt.xlabel("Share of NaN in the column")
    plt.ylabel("Number of columns")
    plt.title("Missing values per column")
    save_fig(fig_dir, "nan_share.png")
    return nan_frac


def count_distinct(x, fig_dir):
    """Count distinct non-NaN values per column and group the columns."""
    counts = np.zeros(x.shape[1], dtype=int)
    for j in range(x.shape[1]):
        col = x[:, j]
        counts[j] = len(np.unique(col[~np.isnan(col)]))
    groups = [
        ("0 (all NaN)", counts == 0),
        ("1 (constant)", counts == 1),
        ("2 (binary)", counts == 2),
        ("3-10", (counts >= 3) & (counts <= 10)),
        ("11-50", (counts >= 11) & (counts <= 50)),
        (">50 (continuous)", counts > 50),
    ]
    print("\nColumns grouped by number of distinct values:")
    for label, mask in groups:
        print("  {:<18} {}".format(label, int(np.sum(mask))))
    plt.figure(figsize=(6, 3))
    plt.bar(
        [g[0] for g in groups], [int(np.sum(g[1])) for g in groups], color="tab:green"
    )
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Number of columns")
    plt.title("Distinct values per column")
    save_fig(fig_dir, "distinct_values.png")
    return counts


def show_special_codes(x, names):
    """Print how often the BRFSS special codes occur in a few example columns."""
    name_list = [str(n) for n in names]
    print("\nSpecial codes in example columns:")
    for name in EXAMPLE_COLUMNS:
        if name not in name_list:
            print("\n{}: column not found".format(name))
            continue
        col = x[:, name_list.index(name)]
        valid = col[~np.isnan(col)]
        values, counts = np.unique(valid, return_counts=True)
        n_nan = int(np.sum(np.isnan(col)))
        print("\n{}: {} NaN, {} distinct values".format(name, n_nan, len(values)))
        for code in SPECIAL_CODES:
            n_code = int(np.sum(valid == code))
            if n_code > 0:
                print("   code {:>3}: {} rows".format(code, n_code))
        top = np.argsort(-counts)[:6]
        print(
            "   most frequent values:",
            [(float(values[i]), int(counts[i])) for i in top],
        )


def plot_top_correlations(x, y01, names, fig_dir):
    """Plot the 15 columns with the largest |correlation| with the label."""
    corr = np.array([corr_with_label(x[:, j], y01) for j in range(x.shape[1])])
    order = np.argsort(-np.abs(np.nan_to_num(corr)))[:15]
    print("\nTop 15 columns by |correlation| with the label:")
    for j in order:
        print("  {:<10} {:+.4f}".format(str(names[j]), corr[j]))
    plt.figure(figsize=(5, 4))
    plt.barh(
        [str(names[j]) for j in order][::-1], corr[order][::-1], color="tab:purple"
    )
    plt.xlabel("Correlation with label (+1 = heart disease)")
    plt.title("Top 15 correlations")
    save_fig(fig_dir, "top_correlations.png")


def plot_rate_by_age(x, y01, names, fig_dir):
    """Plot the share of +1 for each 5-year age group (column _AGEG5YR)."""
    name_list = [str(n) for n in names]
    col = x[:, name_list.index("_AGEG5YR")]
    levels = np.unique(col[~np.isnan(col)])
    rates = []
    print(
        "\nShare of +1 by age group (_AGEG5YR; 1 = 18-24 ... 13 = 80+, 14 = unknown):"
    )
    for level in levels:
        mask = col == level
        rates.append(100.0 * np.mean(y01[mask]))
        print(
            "  group {:>4.0f}: {:>7} people, {:.2f}% positive".format(
                level, int(np.sum(mask)), rates[-1]
            )
        )
    plt.figure(figsize=(5, 3))
    plt.bar([str(int(v)) for v in levels], rates, color="tab:red")
    plt.xlabel("Age group (_AGEG5YR)")
    plt.ylabel("% with heart disease")
    plt.title("Heart disease by age group")
    save_fig(fig_dir, "positive_rate_by_age.png")


def main():
    n_sample = int(sys.argv[1]) if len(sys.argv) > 1 else None
    x, y, ids, names = get_train_data()
    if n_sample is not None:
        idx = stratified_indices(y, n_sample, seed=0)
        x, y = x[idx], y[idx]
        fig_dir = "figures_sample"
    else:
        fig_dir = "figures"
    y01 = labels_to_01(y)
    print("Data shape: {} rows x {} columns".format(x.shape[0], x.shape[1]))
    plot_class_balance(y, fig_dir)
    analyse_nan(x, fig_dir)
    count_distinct(x, fig_dir)
    show_special_codes(x, names)
    plot_top_correlations(x, y01, names, fig_dir)
    plot_rate_by_age(x, y01, names, fig_dir)
    print("\nFigures saved in {}/".format(fig_dir))


if __name__ == "__main__":
    main()
