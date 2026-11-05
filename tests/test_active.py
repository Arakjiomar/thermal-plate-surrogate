"""Tests for the active-learning selection rules and loop."""
import numpy as np
import pytest

from thermal_surrogate.active import (ALConfig, equivalent_n, fit_power_law, relative_spread, run_loop,
                                      select_cluster,
                                      select_diverse, select_random, select_top, unit_cube)
from thermal_surrogate.dataset import latin_hypercube, scale, solve_batch
from thermal_surrogate.mlp import TrainConfig

LOWS = np.array([0.5, 500, 0.2, 0.2, 0.05, 0, 0, 0, 0], dtype=float)
HIGHS = np.array([5.0, 5000, 0.8, 0.8, 0.20, 100, 100, 100, 100], dtype=float)


# ---------------------------------------------------------------- selection rules
def test_select_top_returns_the_largest_scores_in_order():
    s = np.array([0.1, 5.0, 3.0, 4.0, 0.2])
    assert select_top(s, 3).tolist() == [1, 3, 2]


def test_select_random_is_distinct_in_range_and_seeded():
    a = select_random(100, 30, np.random.default_rng(1))
    b = select_random(100, 30, np.random.default_rng(1))
    assert len(set(a.tolist())) == 30 and a.min() >= 0 and a.max() < 100
    np.testing.assert_array_equal(a, b)


def test_select_diverse_avoids_near_duplicates():
    # 10 near-identical candidates with the highest scores, plus 5 spread-out ones
    # with slightly lower scores. Greedy top-5 takes 5 of the clump; diverse must not.
    rng = np.random.default_rng(0)
    clump = 0.5 + 1e-3 * rng.standard_normal((10, 2))
    far = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.0]])
    Z = np.vstack([clump, far])
    s = np.concatenate([np.full(10, 2.0) + 1e-3 * np.arange(10), np.full(5, 1.0)])
    top = select_top(s, 5)
    div = select_diverse(s, Z, 5, factor=3)
    assert set(top.tolist()) <= set(range(10))                 # greedy: all in the clump
    assert div[0] == np.argmax(s)                               # starts at the best score
    assert len(set(div.tolist())) == 5
    assert sum(i < 10 for i in div) == 1                        # only one clump point kept


def test_select_diverse_stays_inside_the_shortlist():
    rng = np.random.default_rng(3)
    s, Z = rng.random(500), rng.random((500, 9))
    div = select_diverse(s, Z, 10, factor=4)
    assert set(div.tolist()) <= set(select_top(s, 40).tolist())


def test_unit_cube_maps_the_box_corners():
    np.testing.assert_allclose(unit_cube(np.stack([LOWS, HIGHS]), LOWS, HIGHS),
                               [[0] * 9, [1] * 9])


# ---------------------------------------------------------------- learning-curve summary
def test_power_law_fit_and_inverse():
    n = np.array([50, 100, 200, 400])
    c, alpha = fit_power_law(n, 3.0 * n ** -0.8)
    assert c == pytest.approx(3.0) and alpha == pytest.approx(0.8)
    np.testing.assert_allclose(equivalent_n(3.0 * n ** -0.8, c, alpha), n)


# ---------------------------------------------------------------- the loop (tiny, fast)
@pytest.fixture(scope="module")
def tiny():
    rng = np.random.default_rng(0)
    draw = lambda m: scale(latin_hypercube(m, 9, rng), LOWS, HIGHS)
    n = 17
    X0, Xv, Xt, pool = draw(20), draw(10), draw(10), draw(200)
    return dict(X0=X0, T0=solve_batch(X0, n), Xv=Xv, Tv=solve_batch(Xv, n),
                Xt=Xt, Tt=solve_batch(Xt, n), pool=pool, n=n)


def _run(d, strategy, seed=0):
    cfg = ALConfig(strategy=strategy, batch=10, budget=40, members=2, seed=seed,
                   train=TrainConfig(width=16, depth=1, epochs=20, eval_every=10))
    return run_loop(d["X0"], d["T0"], d["pool"], d["Xv"], d["Tv"],
                    {"test": (d["Xt"], d["Tt"])}, cfg, LOWS, HIGHS, n_grid=d["n"])


@pytest.mark.parametrize("strategy", ["random", "spread", "diverse", "cluster"])
def test_loop_spends_exactly_the_budget_without_reusing_or_leaking(tiny, strategy):
    h = _run(tiny, strategy)
    assert [r["n"] for r in h["rounds"]] == [20, 30, 40]        # 3 trainings, 2 batches
    picks = np.concatenate([np.array(p) for p in h["picks"]])
    assert len(picks) == 20
    assert len(np.unique(picks, axis=0)) == 20                  # no candidate picked twice
    for X in (tiny["X0"], tiny["Xv"], tiny["Xt"]):              # picks come only from the pool
        assert not (picks[:, None, :] == X[None, :, :]).all(axis=2).any()
    assert all(np.isfinite(r["test"]["rel_l2"]) for r in h["rounds"])
    assert all(0.0 <= r["test"]["defer"] <= 1.0 for r in h["rounds"])


def test_loop_is_deterministic(tiny):
    a, b = _run(tiny, "diverse", seed=1), _run(tiny, "diverse", seed=1)
    assert a["picks"] == b["picks"]


def test_loop_rejects_bad_config(tiny):
    with pytest.raises(ValueError):
        _run(tiny, "uncertainty")
    cfg = ALConfig(budget=5)
    with pytest.raises(ValueError):
        run_loop(tiny["X0"], tiny["T0"], tiny["pool"], tiny["Xv"], tiny["Tv"], {}, cfg,
                 LOWS, HIGHS, n_grid=tiny["n"])


def test_first_ensemble_is_reused_and_gives_identical_results(tiny):
    from thermal_surrogate.active import member_seeds_for
    from thermal_surrogate.uq import Ensemble
    cfg = ALConfig(strategy="diverse", batch=10, budget=40, members=2, seed=1,
                   train=TrainConfig(width=16, depth=1, epochs=20, eval_every=10))
    args = (tiny["X0"], tiny["T0"], tiny["pool"], tiny["Xv"], tiny["Tv"],
            {"test": (tiny["Xt"], tiny["Tt"])}, cfg, LOWS, HIGHS)
    ens0 = Ensemble.train(tiny["X0"], tiny["T0"], tiny["Xv"], tiny["Tv"], cfg.train,
                          member_seeds_for(cfg))
    a = run_loop(*args, n_grid=tiny["n"])
    b = run_loop(*args, n_grid=tiny["n"], first_ensemble=ens0)
    assert a["picks"] == b["picks"]
    assert a["rounds"][0]["test"] == b["rounds"][0]["test"]


def test_relative_spread_ignores_how_hot_the_plate_is():
    # Plate 2 is plate 1 scaled 5x: its spread in deg C is 5x larger, its RELATIVE spread is equal.
    from thermal_surrogate.uq import per_sample_spread
    rng = np.random.default_rng(0)
    base = 50 + rng.standard_normal((5, 1, 10, 10))           # 5 members, 1 plate
    P = np.concatenate([base, 5 * base], axis=1)               # (5, 2, 10, 10)
    absolute = per_sample_spread(P.std(axis=0, ddof=1))
    rel = relative_spread(P)
    assert absolute[1] == pytest.approx(5 * absolute[0])
    assert rel[1] == pytest.approx(rel[0])


def test_relative_score_runs_in_the_loop(tiny):
    cfg = ALConfig(strategy="diverse", score="rel", batch=10, budget=30, members=2,
                   train=TrainConfig(width=16, depth=1, epochs=20, eval_every=10))
    h = run_loop(tiny["X0"], tiny["T0"], tiny["pool"], tiny["Xv"], tiny["Tv"],
                 {"test": (tiny["Xt"], tiny["Tt"])}, cfg, LOWS, HIGHS, n_grid=tiny["n"])
    assert [r["n"] for r in h["rounds"]] == [20, 30]
    with pytest.raises(ValueError):
        run_loop(tiny["X0"], tiny["T0"], tiny["pool"], tiny["Xv"], tiny["Tv"], {},
                 ALConfig(score="relative"), LOWS, HIGHS, n_grid=tiny["n"])


def test_select_cluster_takes_one_uncertain_point_per_region():
    # 4 well-separated groups of 50 points; group 0 has ALL the highest scores.
    # Top-4 would take 4 points from group 0; cluster selection takes one per group,
    # and within each group the highest-scoring point.
    rng = np.random.default_rng(0)
    centres = np.array([[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]])
    Z = np.concatenate([c + 0.02 * rng.standard_normal((50, 2)) for c in centres])
    s = rng.random(200); s[:50] += 10
    picks = select_cluster(s, Z, 4, np.random.default_rng(1))
    assert sorted(p // 50 for p in picks) == [0, 1, 2, 3]
    for p in picks:
        g = slice(50 * (p // 50), 50 * (p // 50 + 1))
        assert s[p] == s[g].max()
