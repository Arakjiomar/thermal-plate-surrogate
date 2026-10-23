"""Tests for the deep-ensemble UQ helpers."""
import numpy as np
import pytest

from thermal_surrogate.mlp import TrainConfig
from thermal_surrogate.uq import (Ensemble, auroc, coverage, interior, per_sample_rmse,
                                  per_sample_spread)

rng = np.random.default_rng(0)


def test_coverage_of_a_perfectly_calibrated_gaussian():
    # If errors really are N(0, sigma^2), the p-interval contains a fraction p of the truth.
    mu = np.zeros((200, 12, 12)); sigma = np.full_like(mu, 2.0)
    T = mu + sigma * rng.standard_normal(mu.shape)
    cov = coverage(T, mu, sigma, levels=(0.5, 0.9, 0.99))
    for p, c in cov.items():
        assert c == pytest.approx(p, abs=0.01)


def test_coverage_detects_overconfidence():
    mu = np.zeros((200, 12, 12)); true_sigma = 2.0
    T = true_sigma * rng.standard_normal(mu.shape)
    cov = coverage(T, mu, np.full_like(mu, 0.5 * true_sigma), levels=(0.9,))
    assert cov[0.9] < 0.6          # intervals half as wide as they should be -> far below 90%


def test_interior_and_per_sample_reductions():
    F = np.ones((3, 6, 6)); F[:, 0, :] = 100.0          # a hot boundary row must be ignored
    assert interior(F).shape == (3, 4, 4)
    np.testing.assert_allclose(per_sample_spread(F), 1.0)
    np.testing.assert_allclose(per_sample_rmse(F, np.zeros_like(F)), 1.0)


def test_auroc():
    assert auroc([0, 1, 2], [3, 4, 5]) == 1.0           # perfect separation
    assert auroc([3, 4, 5], [0, 1, 2]) == 0.0
    assert auroc([1, 1], [1, 1]) == 0.5                 # all ties
    a, b = rng.normal(size=2000), rng.normal(size=2000)
    assert auroc(a, b) == pytest.approx(0.5, abs=0.03)  # identical distributions


def toy(N=60, n=6):
    X = rng.uniform(0.5, 1.5, size=(N, 9))
    xs = np.linspace(0, 1, n); gx, gy = np.meshgrid(xs, xs)
    T = X[:, 0, None, None] * np.sin(np.pi * gx) + X[:, 1, None, None] * gy
    return X, T


def test_ensemble_spread_calibration_and_roundtrip(tmp_path):
    X, T = toy()
    cfg = TrainConfig(width=16, depth=2, epochs=60, batch_size=16, eval_every=5)
    ens = Ensemble.train(X[:40], T[:40], X[40:50], T[40:50], cfg, seeds=[0, 1, 2])
    P = ens.predict_all(X[50:])
    assert P.shape == (3, 10, 6, 6)
    assert not np.allclose(P[0], P[1])                  # different seeds -> different members
    mu, sigma = ens.predict(X[50:])
    np.testing.assert_allclose(mu, P.mean(axis=0))
    s = ens.calibrate(X[40:50], T[40:50])
    z = interior(T[40:50] - ens.predict(X[40:50])[0]) / interior(ens.predict(X[40:50])[1])
    assert np.mean(z**2) == pytest.approx(1.0, rel=1e-6)   # calibrated on val by construction
    ens.save(tmp_path / "e")
    back = Ensemble.load(tmp_path / "e")
    assert back.scale == pytest.approx(s)
    np.testing.assert_allclose(back.predict(X[50:])[1], ens.predict(X[50:])[1], rtol=1e-6)


def test_ensemble_needs_two_members():
    with pytest.raises(ValueError):
        Ensemble([object()])
