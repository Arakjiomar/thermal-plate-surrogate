"""Helpers for multi-seed experiments: one run = one (config, seed) pair."""
from __future__ import annotations

import numpy as np

from .metrics import rel_l2
from .mlp import FieldSurrogate, TrainConfig


def run_once(X, T, X_val, T_val, X_test, T_test, X_ood, T_ood,
             cfg: TrainConfig, kind: str = "pod") -> dict:
    """Train one surrogate and return its errors plus the train/val gap.

    The gap is measured AT THE KEPT EPOCH (best validation loss), i.e. for the
    model that is actually used, not for the last epoch.
    """
    m = FieldSurrogate(kind, physics=True).fit(X, T, X_val, T_val, cfg)
    i = m.history["epoch"].index(m.best_epoch)
    train_mse, val_mse = m.history["train"][i], m.history["val"][i]
    return {
        "seed": cfg.seed,
        "best_epoch": m.best_epoch,
        "train_mse": train_mse,
        "val_mse": val_mse,
        "gap": val_mse / train_mse,                     # >> 1 means overfitting
        "val_rel_l2": float(rel_l2(m.predict(X_val), T_val).mean()),
        "test_rel_l2": float(rel_l2(m.predict(X_test), T_test).mean()),
        "ood_rel_l2": float(rel_l2(m.predict(X_ood), T_ood).mean()),
    }


def seed_stats(runs: list[dict], key: str) -> dict:
    """Mean, sample standard deviation and standard error of `key` across seeds."""
    v = np.array([r[key] for r in runs], dtype=float)
    sd = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    return {"mean": float(v.mean()), "std": sd, "sem": sd / np.sqrt(len(v)), "n": len(v)}


def subsample(X, T, n: int, seed: int):
    """A random size-n subset of the training set (different subset per seed)."""
    idx = np.random.default_rng(seed).choice(len(X), size=n, replace=False)
    return X[idx], T[idx]
