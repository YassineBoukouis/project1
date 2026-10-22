"""Fast loading of the BRFSS csv files, with a .npy cache.

The csv files are parsed once and stored as float32 NumPy arrays in
data/cache/, so later runs load in about a second.
"""

import csv
import os
import time

import numpy as np

DATA_DIR = os.path.join("data", "dataset")
CACHE_DIR = os.path.join("data", "cache")
CHUNK_ROWS = 10000


def _parse_field(text):
    """Convert one csv field to float; empty fields become NaN."""
    return float(text) if text != "" else np.nan


def _read_csv_chunked(path, max_rows=None):
    """Read a numeric csv with a header row.

    Returns (header, array) where array is float32 of shape (rows, columns).
    The csv file is read in chunks to keep memory usage low.
    """
    chunks = []
    buffer = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break
            buffer.append([_parse_field(v) for v in row])
            if len(buffer) == CHUNK_ROWS:
                chunks.append(np.array(buffer, dtype=np.float32))
                buffer = []
    if buffer:
        chunks.append(np.array(buffer, dtype=np.float32))
    return header, np.vstack(chunks)


def load_train_raw(max_rows=None):
    """Parse x_train.csv and y_train.csv directly (no cache).

    Returns (x, y, ids, names): features (float32, Id column removed),
    labels in {-1, +1}, row ids, and the list of feature names.
    """
    x_header, x_all = _read_csv_chunked(os.path.join(DATA_DIR, "x_train.csv"), max_rows)
    _, y_all = _read_csv_chunked(os.path.join(DATA_DIR, "y_train.csv"), max_rows)
    ids = x_all[:, 0].astype(np.int64)
    y_ids = y_all[:, 0].astype(np.int64)
    if not np.array_equal(ids, y_ids):
        raise ValueError("Ids of x_train.csv and y_train.csv do not match.")
    return x_all[:, 1:], y_all[:, 1], ids, x_header[1:]


def get_train_data():
    """Load the full training data, using the .npy cache if it exists."""
    paths = {
        name: os.path.join(CACHE_DIR, name + ".npy")
        for name in ["x_train", "y_train", "ids_train", "names"]
    }
    if all(os.path.exists(p) for p in paths.values()):
        return (
            np.load(paths["x_train"]),
            np.load(paths["y_train"]),
            np.load(paths["ids_train"]),
            list(np.load(paths["names"])),
        )
    x, y, ids, names = load_train_raw()
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(paths["x_train"], x)
    np.save(paths["y_train"], y)
    np.save(paths["ids_train"], ids)
    np.save(paths["names"], np.array(names))
    return x, y, ids, names


def _read_header(path):
    """Return the first row (column names) of a csv file."""
    with open(path, "r") as f:
        return next(csv.reader(f))


def load_test_raw(max_rows=None):
    """Parse x_test.csv directly (no cache).

    Returns (x, ids, names) with the same column layout as the training
    features (Id column removed). Raises an error if the header of
    x_test.csv differs from the header of x_train.csv.
    """
    x_header, x_all = _read_csv_chunked(os.path.join(DATA_DIR, "x_test.csv"), max_rows)
    train_header = _read_header(os.path.join(DATA_DIR, "x_train.csv"))
    if x_header != train_header:
        raise ValueError("Header of x_test.csv differs from x_train.csv.")
    ids = x_all[:, 0].astype(np.int64)
    return x_all[:, 1:], ids, x_header[1:]


def get_test_data():
    """Load the test features and ids, using the .npy cache if it exists.

    Returns (x, ids). Feature names are the same as for the training data.
    """
    paths = {
        name: os.path.join(CACHE_DIR, name + ".npy") for name in ["x_test", "ids_test"]
    }
    if all(os.path.exists(p) for p in paths.values()):
        return np.load(paths["x_test"]), np.load(paths["ids_test"])
    x, ids, _ = load_test_raw()
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(paths["x_test"], x)
    np.save(paths["ids_test"], ids)
    return x, ids


def labels_to_01(y):
    """Map labels from {-1, +1} to {0, 1} (needed by the logistic methods)."""
    return (y == 1).astype(np.float64)


def labels_to_pm1(y01):
    """Map labels or predictions from {0, 1} back to {-1, +1}."""
    return np.where(y01 > 0.5, 1, -1)


if __name__ == "__main__":
    import sys

    start = time.time()
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        x, ids = get_test_data()
        print("x_test shape:", x.shape, "dtype:", x.dtype)
        print("first 3 ids:", ids[:3], "last 3 ids:", ids[-3:])
        print("unique ids:", len(np.unique(ids)))
        print("fraction of NaN:", round(float(np.isnan(x).mean()), 3))
    else:
        n_rows = int(sys.argv[1]) if len(sys.argv) > 1 else None
        if n_rows is None:
            x, y, ids, names = get_train_data()
        else:
            x, y, ids, names = load_train_raw(max_rows=n_rows)
        print("x shape:", x.shape, "dtype:", x.dtype)
        print("number of feature names:", len(names))
        print("first 5 names:", names[:5])
        print("fraction of NaN:", round(float(np.isnan(x).mean()), 3))
        print("labels -1 / +1:", int((y == -1).sum()), "/", int((y == 1).sum()))
        y01 = labels_to_01(y)
        print("labels 0 / 1:", int((y01 == 0).sum()), "/", int((y01 == 1).sum()))
        print("round trip ok:", bool(np.array_equal(labels_to_pm1(y01), y)))
    print("seconds:", round(time.time() - start, 1))
