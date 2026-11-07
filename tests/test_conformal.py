"""Tests for split conformal prediction, including the coverage guarantee itself."""
import math

import numpy as np
import pytest

from thermal_surrogate.conformal import (ConformalCalibrator, conformal_quantile, coverage_of,
                                         scores)


# ---------------------------------------------------------------- the quantile
def test_quantile_rank_is_ceil_n_plus_1_times_1_minus_alpha():
    s = np.arange(1, 100, dtype=float)                  # n = 99 scores: 1, 2, ..., 99
    assert conformal_quantile(s, 0.1) == 90.0           # ceil(100 * 0.9) = 90th smallest
    assert conformal_quantile(np.arange(1, 10.0), 0.1) == 9.0   # n = 9: ceil(9.0) = 9 -> the max
    assert conformal_quantile(s[::-1], 0.1) == 90.0     # order of the input doesn't matter


def test_too_few_calibration_points_cannot_certify():
    assert math.isinf(conformal_quantile(np.arange(8.0), 0.1))   # ceil(9 * 0.9) = 9 > 8
    with pytest.raises(ValueError):
        conformal_quantile(np.arange(10.0), 0.0)


# ---------------------------------------------------------------- the guarantee
@pytest.mark.parametrize("dist", ["lognormal", "cauchy"])
def test_finite_sample_coverage_guarantee(dist):
    # Exchangeable scores from ANY continuous distribution (heavy-tailed included):
    # coverage averaged over repeated draws lies in [1 - alpha, 1 - alpha + 1/(n + 1)].
    rng = np.random.default_rng(0)
    n, alpha, trials = 50, 0.1, 20_000
    draw = {"lognormal": lambda size: rng.lognormal(0, 1.5, size),
            "cauchy": lambda size: np.abs(rng.standard_cauchy(size))}[dist]
    S = draw((trials, n + 1))
    k = math.ceil((n + 1) * (1 - alpha))
    q = np.sort(S[:, :n], axis=1)[:, k - 1]
    cov = np.mean(S[:, n] <= q)
    se = np.sqrt(cov * (1 - cov) / trials)
    assert 1 - alpha - 4 * se <= cov <= 1 - alpha + 1 / (n + 1) + 4 * se


def test_no_guarantee_without_exchangeability():
    # Calibrate on one distribution, test on a shifted one: coverage collapses.
    # This is the U2-4 situation (calibrated in range, queried out of range).
    rng = np.random.default_rng(1)
    q = conformal_quantile(rng.lognormal(0, 1, 500), 0.1)
    assert np.mean(rng.lognormal(1.0, 1, 20_000) <= q) < 0.75


# ---------------------------------------------------------------- fields and the calibrator
def fake_fields(m, rng, err_scale=1.0, n=12):
    """Ensemble-like predictions: sigma varies by plate, error = sigma * heavy-tailed noise."""
    T = rng.normal(50, 10, (m, n, n))
    sigma = rng.lognormal(0, 0.5, (m, 1, 1)) * np.ones((m, n, n))
    noise = err_scale * rng.standard_t(3, (m, n, n))
    return T, T - sigma * noise, sigma


def test_scores_shapes_and_definitions():
    rng = np.random.default_rng(2)
    T, mu, sigma = fake_fields(5, rng)
    g, b, p = (scores(k, T, mu, sigma) for k in ("gate", "band", "pixel"))
    assert g.shape == (5,) and b.shape == (5,) and p.shape == (5 * 10 * 10,)
    # with sigma constant within a plate, max|z| >= RMS|z| = gate score
    assert np.all(b >= g - 1e-12)
    with pytest.raises(ValueError):
        scores("wrong", T, mu, sigma)


@pytest.mark.parametrize("kind", ["gate", "band"])
def test_calibrator_covers_new_exchangeable_plates(kind):
    rng = np.random.default_rng(3)
    covs = []
    for _ in range(200):
        cal = ConformalCalibrator(0.1, kind).fit(*fake_fields(60, rng))
        covs.append(coverage_of(cal, *fake_fields(60, rng)))
    assert np.mean(covs) >= 0.89                        # ~0.9 + 1/61 on average


def test_gate_bound_is_q_times_spread():
    rng = np.random.default_rng(4)
    T, mu, sigma = fake_fields(30, rng)
    cal = ConformalCalibrator(0.1, "gate").fit(T, mu, sigma)
    np.testing.assert_allclose(cal.rmse_bound(sigma), cal.q[0] * sigma[:, 0, 0])
    with pytest.raises(ValueError):
        ConformalCalibrator(0.1, "band").fit(T, mu, sigma).rmse_bound(sigma)


def test_mondrian_covers_each_group_where_one_quantile_does_not():
    # Group 1 (far from the training data) has 3x larger errors relative to sigma:
    # one pooled q_hat under-covers it; per-group q_hat covers both. The guarantee
    # is an average over calibration draws, so average over repeated draws.
    rng = np.random.default_rng(5)
    def batch(m):
        a, b = fake_fields(m, rng, 1.0), fake_fields(m, rng, 3.0)
        dist = np.r_[np.zeros(m), np.full(m, 0.3)]
        return [np.concatenate([x, y]) for x, y in zip(a, b)], dist
    pooled_far, mond_far, mond_near, q = [], [], [], []
    for _ in range(100):
        (Tc, Mc, Sc), dc = batch(60)
        (Tt, Mt, St), dt = batch(60)
        pooled = ConformalCalibrator(0.1, "gate").fit(Tc, Mc, Sc)
        mond = ConformalCalibrator(0.1, "gate", bins=(0.0,)).fit(Tc, Mc, Sc, dc)
        far = dt > 0
        pooled_far.append(coverage_of(pooled, Tt[far], Mt[far], St[far]))
        mond_far.append(coverage_of(mond, Tt[far], Mt[far], St[far], dt[far]))
        mond_near.append(coverage_of(mond, Tt[~far], Mt[~far], St[~far], dt[~far]))
        q.append(mond.q)
    assert np.mean(pooled_far) < 0.85
    assert np.mean(mond_far) >= 0.89 and np.mean(mond_near) >= 0.89
    assert np.all(np.array(q)[:, 1] > np.array(q)[:, 0])     # wider where the model is worse


def test_mondrian_groups_and_empty_group_is_uncertifiable():
    cal = ConformalCalibrator(0.1, "gate", bins=(0.0, 0.1))
    np.testing.assert_array_equal(cal.group(np.array([0.0, 0.05, 0.1, 0.2])), [0, 1, 1, 2])
    rng = np.random.default_rng(6)
    T, mu, sigma = fake_fields(40, rng)
    cal.fit(T, mu, sigma, np.zeros(40))                  # nothing in groups 1 and 2
    assert math.isinf(cal.q[1]) and math.isinf(cal.q[2]) and cal.n == [40, 0, 0]
    with pytest.raises(ValueError):
        ConformalCalibrator(0.1, "gate", bins=(0.0,)).fit(T, mu, sigma)   # needs distances


def test_interval_and_round_trip():
    rng = np.random.default_rng(7)
    T, mu, sigma = fake_fields(20, rng)
    cal = ConformalCalibrator(0.2, "band", bins=(0.0,)).fit(T, mu, sigma, np.r_[np.zeros(10), np.ones(10)])
    lo, hi = cal.interval(mu, sigma, np.r_[np.zeros(10), np.ones(10)])
    assert lo.shape == mu.shape and np.all(hi >= lo)
    back = ConformalCalibrator.from_dict(cal.to_dict())
    assert back.q == cal.q and back.bins == cal.bins and back.kind == "band"
