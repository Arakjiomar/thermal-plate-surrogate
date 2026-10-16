"""Tests for sampling and dataset generation."""
from pathlib import Path

import numpy as np
import pytest

from thermal_surrogate.dataset import (
    DatasetConfig,
    PARAM_NAMES,
    fingerprint,
    generate,
    latin_hypercube,
    load,
    ood_distance,
    sample_ood,
    save,
)
from thermal_surrogate.solver import PlateParams, solve_plate

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def small_cfg(**overrides):
    cfg = DatasetConfig.from_yaml(CONFIG)
    cfg.n, cfg.n_train, cfg.n_val, cfg.n_test, cfg.n_ood = 17, 12, 4, 4, 6
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def test_latin_hypercube_one_point_per_stratum():
    n, d = 50, 9
    u = latin_hypercube(n, d, np.random.default_rng(1))
    assert u.shape == (n, d)
    assert np.all((u >= 0) & (u < 1))
    for j in range(d):
        strata = np.floor(u[:, j] * n).astype(int)
        assert sorted(strata) == list(range(n))   # every slice hit exactly once


def test_latin_hypercube_axes_are_shuffled_independently():
    u = latin_hypercube(200, 2, np.random.default_rng(2))
    # Without the per-axis shuffle both columns would be sorted together (r ~ 1).
    assert abs(np.corrcoef(u[:, 0], u[:, 1])[0, 1]) < 0.3


def test_config_matches_param_order():
    cfg = DatasetConfig.from_yaml(CONFIG)
    assert len(cfg.lows) == len(PARAM_NAMES) == 9
    assert np.all(cfg.lows < cfg.highs)


def test_ood_distance():
    lows, highs = np.zeros(2), np.array([1.0, 10.0])
    X = np.array([[0.5, 5.0], [1.5, 5.0], [0.5, -5.0]])
    np.testing.assert_allclose(ood_distance(X, lows, highs), [0.0, 0.5, 0.5])


def test_ood_samples_leave_the_box_but_respect_physics():
    cfg = small_cfg(n_ood=300)
    X = sample_ood(cfg, np.random.default_rng(3))
    assert len(X) == 300
    assert np.all(ood_distance(X, cfg.lows, cfg.highs) > 0)
    assert np.all(X >= cfg.phys_lows) and np.all(X <= cfg.phys_highs)
    width = cfg.highs - cfg.lows
    assert np.all(X >= cfg.lows - cfg.ood_expand * width - 1e-12)
    assert np.all(X <= cfg.highs + cfg.ood_expand * width + 1e-12)


def test_generate_shapes_splits_and_consistency():
    cfg = small_cfg()
    data = generate(cfg)
    for s, size in [("train", 12), ("val", 4), ("test", 4), ("ood", 6)]:
        assert data[f"X_{s}"].shape == (size, 9)
        assert data[f"T_{s}"].shape == (size, 17, 17)
        assert data[f"T_{s}"].dtype == np.float32
    for s in ("train", "val", "test"):
        X = data[f"X_{s}"]
        assert np.all((X >= cfg.lows) & (X <= cfg.highs))
        assert np.all(data[f"ood_dist_{s}"] == 0)
    assert np.all(data["ood_dist_ood"] > 0)
    # no sample appears in two in-distribution splits
    allx = np.concatenate([data[f"X_{s}"] for s in ("train", "val", "test")])
    assert len(np.unique(allx, axis=0)) == len(allx)
    # stored field really is the solver's answer for the stored parameters
    T_ref = solve_plate(PlateParams(*data["X_test"][0]), 17)
    np.testing.assert_allclose(data["T_test"][0], T_ref, rtol=1e-5, atol=1e-4)


def test_same_seed_same_data_different_seed_different_data():
    a, b = generate(small_cfg()), generate(small_cfg())
    c = generate(small_cfg(seed=1))
    assert fingerprint(a) == fingerprint(b)
    assert fingerprint(a) != fingerprint(c)


def test_save_load_roundtrip(tmp_path):
    data = generate(small_cfg())
    save(data, tmp_path / "d.npz", meta={"note": "test"})
    back = load(tmp_path / "d.npz")
    for k, v in data.items():
        np.testing.assert_array_equal(back[k], v)
    assert "meta" in back
