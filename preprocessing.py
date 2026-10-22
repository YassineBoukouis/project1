"""Preprocessing: learn statistics on training rows, apply them to any rows.

``fit`` learns everything that depends on the data (which columns to keep,
medians, clipping bounds, means, standard deviations) from the rows it is
given. ``transform`` applies those stored numbers to any rows, so rows that
were not used in ``fit`` (validation folds, test data) never influence them.
Every part can be switched off through the config dictionary.

Command line: ``python preprocessing.py 20000`` runs a quick check on a
stratified sample, ``python preprocessing.py`` runs it on all training rows.
"""

import sys
import time

import numpy as np

from preprocessing_rules import (
    COLUMN_RULES,
    DEFAULT_MAX_DISTINCT,
    DEFAULT_RULE,
    DROP_CATEGORY_OR_UNIT,
    DROP_ID_DATE,
    DROP_RAW_WITH_TWIN,
    KEEP_CODES_AS_VALUES,
    n_distinct,
)

DEFAULT_CONFIG = {
    # Apply the per-column rules (special codes become NaN or 0).
    "use_rules": True,
    # Apply the default rule (7 and 9 become NaN) to the other small columns.
    "use_default_rule": True,
    # Drop the identifier, raw-twin and category-label columns.
    "drop_listed": True,
    # Drop columns that have a single distinct value.
    "drop_constant": True,
    # Drop columns whose share of NaN (after the rules) is above this value.
    "max_nan_share": 0.95,
    # Add a 0/1 column marking where a value was missing.
    "add_missing_indicators": True,
    # Only columns with at least this share of NaN get an indicator.
    "indicator_min_share": 0.01,
    # Clip columns with many distinct values to these percentiles (None = off).
    "clip_percentiles": (1, 99),
    # Subtract the mean and divide by the standard deviation.
    "standardize": True,
    # Clip standardized values to [-limit, limit] (None = off).
    "clip_standardized": None,
    # Put a column of ones first (the intercept of a linear model).
    "add_bias": True,
}


def _make_config(config):
    """Return the default config updated with the given overrides."""
    cfg = dict(DEFAULT_CONFIG)
    if config:
        unknown = [key for key in config if key not in cfg]
        if unknown:
            raise ValueError("Unknown config keys: %s" % unknown)
        cfg.update(config)
    return cfg


def apply_rules(x, names, default_cols, cfg):
    """Return a copy of x where special codes became NaN, 0 or another value.

    The input array is never modified. All masks are computed on the original
    column, so one replacement cannot trigger another.
    """
    out = x.copy()
    index = {name: j for j, name in enumerate(names)}
    rules = {}
    if cfg["use_default_rule"]:
        for name in default_cols:
            rules[name] = DEFAULT_RULE
    if cfg["use_rules"]:
        rules.update(COLUMN_RULES)
    for name, rule in rules.items():
        j = index.get(name)
        if j is None:
            continue
        col = x[:, j]
        nan_mask = np.isin(col, rule.get("nan", []))
        zero_mask = np.isin(col, rule.get("zero", []))
        set_masks = [(col == code, v) for code, v in rule.get("set", {}).items()]
        target = out[:, j]
        target[nan_mask] = np.nan
        target[zero_mask] = 0
        for mask, value in set_masks:
            target[mask] = value
    return out


def _numeric_block(cleaned, state):
    """Keep the chosen columns, fill NaN with the medians and clip."""
    block = cleaned[:, state["keep"]]
    block = np.where(np.isnan(block), state["medians"], block)
    np.clip(block, state["lows"], state["highs"], out=block)
    return block


def fit(x, names, config=None):
    """Learn the preprocessing from the rows of x and return it as a dict."""
    cfg = _make_config(config)
    names = [str(s) for s in names]
    n_rows = x.shape[0]
    distinct = np.array([n_distinct(x[:, j]) for j in range(len(names))])

    # Small columns without their own rule get the default rule.
    skip = set(DROP_ID_DATE + DROP_RAW_WITH_TWIN + DROP_CATEGORY_OR_UNIT)
    skip |= set(COLUMN_RULES) | set(KEEP_CODES_AS_VALUES)
    default_cols = [
        n
        for n, d in zip(names, distinct)
        if n not in skip and 2 <= d <= DEFAULT_MAX_DISTINCT
    ]

    cleaned = apply_rules(x, names, default_cols, cfg)
    nan_share = np.isnan(cleaned).sum(axis=0) / float(n_rows)

    drop = set()
    if cfg["drop_listed"]:
        drop |= set(DROP_ID_DATE + DROP_RAW_WITH_TWIN + DROP_CATEGORY_OR_UNIT)
    if cfg["drop_constant"]:
        drop |= set(n for n, d in zip(names, distinct) if d <= 1)
    keep = [
        j
        for j, n in enumerate(names)
        if n not in drop and nan_share[j] <= cfg["max_nan_share"]
    ]

    # One indicator per distinct missing pattern (many columns are missing
    # for exactly the same people because of the survey skip logic).
    indicator_cols = []
    if cfg["add_missing_indicators"]:
        seen = set()
        for j in keep:
            if nan_share[j] < cfg["indicator_min_share"]:
                continue
            key = np.packbits(np.isnan(cleaned[:, j])).tobytes()
            if key not in seen:
                seen.add(key)
                indicator_cols.append(j)

    medians = np.zeros(len(keep), dtype=np.float32)
    lows = np.full(len(keep), -np.inf, dtype=np.float32)
    highs = np.full(len(keep), np.inf, dtype=np.float32)
    for k, j in enumerate(keep):
        values = cleaned[:, j]
        values = values[~np.isnan(values)]
        if values.size == 0:
            continue
        medians[k] = np.median(values)
        clip = cfg["clip_percentiles"]
        if clip is not None and distinct[j] > DEFAULT_MAX_DISTINCT:
            lows[k], highs[k] = np.percentile(values, clip)

    state = {
        "config": cfg,
        "names": names,
        "default_cols": default_cols,
        "keep": keep,
        "indicator_cols": indicator_cols,
        "medians": medians,
        "lows": lows,
        "highs": highs,
        "mean": np.zeros(len(keep), dtype=np.float32),
        "std": np.ones(len(keep), dtype=np.float32),
    }
    if cfg["standardize"]:
        block = _numeric_block(cleaned, state)
        state["mean"] = block.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = block.std(axis=0, dtype=np.float64)
        std[std < 1e-8] = 1.0
        state["std"] = std.astype(np.float32)

    state["out_names"] = (
        (["bias"] if cfg["add_bias"] else [])
        + [names[j] for j in keep]
        + ["missing_" + names[j] for j in indicator_cols]
    )
    return state


def transform(x, state):
    """Apply a fitted preprocessing to the rows of x (x is not modified)."""
    cfg = state["config"]
    if x.shape[1] != len(state["names"]):
        raise ValueError(
            "x has %d columns, expected %d" % (x.shape[1], len(state["names"]))
        )
    cleaned = apply_rules(x, state["names"], state["default_cols"], cfg)
    block = _numeric_block(cleaned, state)
    if cfg["standardize"]:
        block -= state["mean"]
        block /= state["std"]
        if cfg["clip_standardized"] is not None:
            limit = cfg["clip_standardized"]
            np.clip(block, -limit, limit, out=block)
    parts = []
    if cfg["add_bias"]:
        parts.append(np.ones((x.shape[0], 1), dtype=block.dtype))
    parts.append(block)
    if state["indicator_cols"]:
        missing = np.isnan(cleaned[:, state["indicator_cols"]])
        parts.append(missing.astype(block.dtype))
    return np.hstack(parts)


def main():
    """Fit on 75% of the rows, transform both parts and print sanity checks."""
    from data_loading import get_train_data
    from leakage_check import stratified_indices

    x, y, ids, names = get_train_data()
    if len(sys.argv) > 1:
        idx = stratified_indices(y, int(sys.argv[1]), seed=0)
    else:
        idx = np.arange(len(y))
    idx = np.random.RandomState(1).permutation(idx)
    cut = int(0.75 * len(idx))
    fit_rows, held_rows = idx[:cut], idx[cut:]

    start = time.time()
    state = fit(x[fit_rows], names)
    print("Fit took %.1f s" % (time.time() - start))
    train = transform(x[fit_rows], state)
    held = transform(x[held_rows], state)

    n_keep = len(state["keep"])
    offset = 1 if state["config"]["add_bias"] else 0
    print("Rows: fit %d, held out %d" % (len(fit_rows), len(held_rows)))
    print(
        "Columns kept: %d of %d, missing indicators: %d"
        % (n_keep, len(names), len(state["indicator_cols"]))
    )
    print("Output shapes: %s and %s, dtype %s" % (train.shape, held.shape, train.dtype))
    print("NaN left: %d and %d" % (np.isnan(train).sum(), np.isnan(held).sum()))
    block = train[:, offset : offset + n_keep]
    held_block = held[:, offset : offset + n_keep]
    print("Largest |column mean| on fit rows: %.5f" % np.abs(block.mean(axis=0)).max())
    stds = block.std(axis=0)
    print("Column std on fit rows: min %.3f, max %.3f" % (stds.min(), stds.max()))
    print("Largest |value| on held-out rows: %.2f" % np.abs(held_block).max())
    worst = np.argsort(np.abs(held_block).max(axis=0))[-5:]
    for k in worst:
        name = names[state["keep"][k]]
        print("  %-10s max |value| %.1f" % (name, np.abs(held_block[:, k]).max()))


if __name__ == "__main__":
    main()
