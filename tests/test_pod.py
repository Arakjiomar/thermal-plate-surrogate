"""Tests for POD, ridge regression and the field metrics."""
import numpy as np
import pytest

from thermal_surrogate.metrics import hotspot_error, max_t_error, rel_l2, rmse, summarize
from thermal_surrogate.pod import (
    POD, PODRidge, physics_features, polynomial_features, ridge_fit)
from thermal_surrogate.solver import PlateParams, solve_plate

rng = np.random.default_rng(0)


# ---------------------------------------------------------------- POD
def random_fields(N=30, n=8):
    return rng.normal(size=(N, n, n)) + 5.0


def test_modes_are_orthonormal():
    pod = POD().fit(random_fields())
    G = pod.modes.T @ pod.modes
    np.testing.assert_allclose(G, np.eye(G.shape[0]), atol=1e-10)


def test_svd_energies_equal_covariance_eigenvalues():
    T = random_fields()
    pod = POD().fit(T)
    Y = (T.reshape(len(T), -1) - pod.mean).T
    C = Y @ Y.T / len(T)                                # the M x M covariance, formed explicitly
    eig = np.sort(np.linalg.eigvalsh(C))[::-1][: len(pod.energy)]
    np.testing.assert_allclose(pod.energy, eig, atol=1e-10)


def test_all_modes_reconstruct_exactly():
    T = random_fields()
    pod = POD().fit(T)
    R = len(pod.energy)
    np.testing.assert_allclose(pod.reconstruct(pod.project(T, R)), T, atol=1e-10)


def test_truncation_error_equals_dropped_energy():
    # Mean squared reconstruction error with R modes = sum of the dropped lambdas.
    T = random_fields()
    pod = POD().fit(T)
    for R in (1, 5, 12):
        resid = pod.reconstruct(pod.project(T, R)) - T
        mse = (resid.reshape(len(T), -1) ** 2).sum(axis=1).mean()
        assert mse == pytest.approx(pod.energy[R:].sum(), rel=1e-9)


def test_edge_temperatures_span_exactly_four_modes():
    # Superposition: with no source, every field is a linear combination of the
    # responses to the four edges, so the centred fields have rank <= 4.
    fields = [solve_plate(PlateParams(amp=0.0, t_left=a, t_right=b, t_bottom=c, t_top=e), 17)
              for a, b, c, e in rng.uniform(0, 100, size=(25, 4))]
    pod = POD().fit(np.array(fields))
    frac = pod.energy / pod.energy.sum()
    assert frac[:4].sum() > 1 - 1e-12
    assert np.all(frac[4:] < 1e-12)


# ---------------------------------------------------------------- features and ridge
def test_polynomial_feature_counts():
    Z = rng.normal(size=(5, 9))
    assert [polynomial_features(Z, d).shape[1] for d in (1, 2, 3)] == [9, 54, 219]
    np.testing.assert_allclose(polynomial_features(Z, 2)[:, 9], Z[:, 0] ** 2)


def test_physics_features():
    X = np.array([[2.0, 1000.0] + [0.5] * 7])
    np.testing.assert_allclose(physics_features(X)[0, -2:], [500.0, 0.5])


def test_ridge_alpha_zero_is_least_squares_and_alpha_shrinks():
    F = rng.normal(size=(50, 6)); A = rng.normal(size=(50, 3))
    W0 = ridge_fit(F, A, 0.0)
    np.testing.assert_allclose(W0, np.linalg.lstsq(F, A, rcond=None)[0], atol=1e-10)
    norms = [np.linalg.norm(ridge_fit(F, A, a)) for a in (0.0, 1.0, 100.0)]
    assert norms[0] > norms[1] > norms[2]


def test_pod_ridge_recovers_a_linear_map_and_roundtrips(tmp_path):
    # Fields that depend linearly on the inputs are fitted (almost) exactly by degree 1.
    X = rng.uniform(1, 2, size=(60, 9))
    basis = rng.normal(size=(9, 6, 6))
    T = np.einsum("ij,jkl->ikl", X, basis)
    m = PODRidge(n_modes=9, degree=1, alpha=1e-10).fit(X, T)
    assert rel_l2(m.predict(X), T).max() < 1e-6
    m.save(tmp_path / "m.npz")
    np.testing.assert_allclose(PODRidge.load(tmp_path / "m.npz").predict(X), m.predict(X))


# ---------------------------------------------------------------- metrics
def test_metrics_on_known_fields():
    true = np.zeros((1, 5, 5)); true[0, 1, 3] = 4.0     # hot spot at x = 3h, y = h
    pred = np.zeros((1, 5, 5)); pred[0, 1, 1] = 3.0     # hot spot at x = h,  y = h
    assert rmse(pred, true)[0] == pytest.approx(np.sqrt((16 + 9) / 25))
    assert rel_l2(pred, true)[0] == pytest.approx(5.0 / 4.0)
    assert max_t_error(pred, true)[0] == pytest.approx(1.0)
    assert hotspot_error(pred, true)[0] == pytest.approx(2 * 0.25)   # h = 1/4
    s = summarize(pred, true)
    assert set(s) == {f"{k}_{q}" for k in ("rel_l2", "rmse", "max_t_err", "hotspot_err")
                      for q in ("mean", "p95")}
