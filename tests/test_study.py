"""Tests for the multi-seed study helpers."""
import numpy as np
import pytest

from thermal_surrogate.mlp import TrainConfig
from thermal_surrogate.study import run_once, seed_stats, subsample

rng = np.random.default_rng(0)


def toy(N=60, n=6):
    X = rng.uniform(0.5, 1.5, size=(N, 9))
    xs = np.linspace(0, 1, n); gx, gy = np.meshgrid(xs, xs)
    T = X[:, 0, None, None] * np.sin(np.pi * gx) + X[:, 1, None, None] * gy
    return X, T


def test_seed_stats():
    runs = [{"e": v} for v in (1.0, 2.0, 3.0, 4.0)]
    s = seed_stats(runs, "e")
    assert s["mean"] == 2.5
    assert s["std"] == pytest.approx(np.std([1, 2, 3, 4], ddof=1))   # sample std, ddof=1
    assert s["sem"] == pytest.approx(s["std"] / 2)                   # / sqrt(4)
    assert s["n"] == 4


def test_subsample_is_a_subset_and_depends_on_seed():
    X, T = toy()
    a, Ta = subsample(X, T, 20, seed=1)
    b, _ = subsample(X, T, 20, seed=2)
    assert a.shape == (20, 9) and Ta.shape == (20, 6, 6)
    assert len(np.unique(a, axis=0)) == 20                              # no repeats
    assert all(any(np.array_equal(r, x) for x in X) for r in a)         # rows come from X
    assert not np.array_equal(a, b)
    # inputs and fields stay paired
    i = np.where((X == a[0]).all(axis=1))[0][0]
    np.testing.assert_array_equal(Ta[0], T[i])


def test_run_once_reports_consistent_numbers():
    X, T = toy()
    cfg = TrainConfig(width=16, depth=2, epochs=60, batch_size=16, eval_every=5, seed=3)
    r = run_once(X[:40], T[:40], X[40:50], T[40:50], X[50:], T[50:], X[50:], T[50:], cfg)
    assert r["seed"] == 3
    assert r["gap"] == pytest.approx(r["val_mse"] / r["train_mse"])
    assert r["test_rel_l2"] == r["ood_rel_l2"]      # same arrays passed for both
    assert 0 < r["best_epoch"] <= 60


def test_different_seeds_give_different_models():
    X, T = toy()
    args = (X[:40], T[:40], X[40:50], T[40:50], X[50:], T[50:], X[50:], T[50:])
    cfg = dict(width=16, depth=2, epochs=40, batch_size=16, eval_every=5)
    r0 = run_once(*args, TrainConfig(**cfg, seed=0))
    r1 = run_once(*args, TrainConfig(**cfg, seed=1))
    assert r0["test_rel_l2"] != r1["test_rel_l2"]
