"""Field-error metrics shared by every surrogate (POD baseline, MLP, ensembles).

All functions take arrays of shape (N, n, n): N predicted / true fields.
"""
from __future__ import annotations

import numpy as np


def rel_l2(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Per-sample relative L2 error  ||pred - true||_2 / ||true||_2   -> (N,)."""
    diff = (pred - true).reshape(len(true), -1)
    ref = true.reshape(len(true), -1)
    return np.linalg.norm(diff, axis=1) / np.linalg.norm(ref, axis=1)


def rmse(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Per-sample root-mean-square error in degrees -> (N,)."""
    return np.sqrt(((pred - true) ** 2).reshape(len(true), -1).mean(axis=1))


def max_t_error(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Per-sample error in the peak temperature, |max(pred) - max(true)| -> (N,)."""
    flat = lambda a: a.reshape(len(a), -1)
    return np.abs(flat(pred).max(axis=1) - flat(true).max(axis=1))


def hotspot_error(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Per-sample distance between the predicted and true hottest points,
    in plate lengths (the plate is the unit square) -> (N,)."""
    n = true.shape[-1]
    h = 1.0 / (n - 1)

    def argmax_xy(a):
        j, i = np.unravel_index(a.reshape(len(a), -1).argmax(axis=1), (n, n))
        return np.stack([i * h, j * h], axis=1)          # rows are y, columns are x

    return np.linalg.norm(argmax_xy(pred) - argmax_xy(true), axis=1)


def summarize(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    """Mean and 95th percentile of every metric, as plain floats (JSON-ready)."""
    out = {}
    for name, fn in [("rel_l2", rel_l2), ("rmse", rmse),
                     ("max_t_err", max_t_error), ("hotspot_err", hotspot_error)]:
        v = fn(pred, true)
        out[f"{name}_mean"] = float(v.mean())
        out[f"{name}_p95"] = float(np.percentile(v, 95))
    return out
