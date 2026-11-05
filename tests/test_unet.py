"""Tests for the field encoding and the field-to-field U-Net surrogate."""
import numpy as np
import pytest
import torch

from thermal_surrogate.fields import (boundary_lift, encode, interior_mask, plate_scale,
                                      source_over_k)
from thermal_surrogate.solver import PlateParams, gaussian_source, make_grid, solve_plate
from thermal_surrogate.unet import FieldUNet, UNet, UNetConfig, receptive_field

rng = np.random.default_rng(0)
LOWS = np.array([0.5, 500, 0.2, 0.2, 0.05, 0, 0, 0, 0], dtype=float)
HIGHS = np.array([5.0, 5000, 0.8, 0.8, 0.20, 100, 100, 100, 100], dtype=float)


def draw(m):
    return LOWS + rng.random((m, 9)) * (HIGHS - LOWS)


def scaled(X, c):
    """Multiply the source amplitude and all four edge temperatures by c."""
    Xc = X.copy()
    Xc[:, 1] *= c
    Xc[:, 5:9] *= c
    return Xc


# ---------------------------------------------------------------- encoding
def test_source_channel_is_the_solver_source_divided_by_k():
    X = draw(3)
    Xg, Yg, _ = make_grid(32)
    for row, S in zip(X, source_over_k(X, 32)):
        p = PlateParams(*row)
        np.testing.assert_allclose(S, gaussian_source(p)(Xg, Yg) / p.k, rtol=1e-12)


def test_A_and_k_enter_only_as_A_over_k():
    # Same A/k, different A and k: identical source channel AND identical solved field
    # (edges identical), so encoding A/k loses nothing.
    a = np.array([[1.0, 1000, 0.4, 0.6, 0.1, 10, 20, 30, 40]])
    b = a.copy(); b[0, 0], b[0, 1] = 4.0, 4000
    np.testing.assert_allclose(source_over_k(a, 32), source_over_k(b, 32), rtol=1e-12)
    np.testing.assert_allclose(solve_plate(PlateParams(*a[0]), 32),
                               solve_plate(PlateParams(*b[0]), 32), atol=1e-9)


def test_lift_matches_the_solver_exactly_on_the_boundary_ring():
    X = draw(4)
    L = boundary_lift(X, 32)
    for row, Li in zip(X, L):
        T = solve_plate(PlateParams(*row), 32)
        ring = interior_mask(32) == 0
        np.testing.assert_array_equal(Li[ring], T[ring])


def test_lift_interior_stays_between_the_edge_temperatures():
    X = draw(20)
    L = boundary_lift(X, 32)[:, 1:-1, 1:-1]
    lo, hi = X[:, 5:9].min(axis=1), X[:, 5:9].max(axis=1)
    assert np.all(L >= lo[:, None, None] - 1e-9) and np.all(L <= hi[:, None, None] + 1e-9)


def test_plate_scale_is_positive_and_homogeneous_of_degree_one():
    X = draw(10)
    s = plate_scale(X, 32)
    assert np.all(s > 0)
    for c in (0.1, 2.0, 7.5):
        np.testing.assert_allclose(plate_scale(scaled(X, c), 32), c * s, rtol=1e-10)


def test_encode_shape_and_coordinate_channels():
    E = encode(draw(2), 16)
    assert E.shape == (2, 4, 16, 16)
    Xg, Yg, _ = make_grid(16)
    np.testing.assert_array_equal(E[0, 2], Xg)
    np.testing.assert_array_equal(E[1, 3], Yg)


# ---------------------------------------------------------------- network
def test_unet_keeps_the_grid_size_and_rejects_bad_norm():
    net = UNet(4, 1, (8, 16, 32, 32, 32))
    assert net(torch.zeros(2, 4, 64, 64)).shape == (2, 1, 64, 64)
    with pytest.raises(ValueError):
        UNet(norm="batch")


def test_receptive_field_formula_and_the_plate_needs_five_levels():
    # Lecture 5, problem 3: steady heat is elliptic, so an output pixel must see the
    # whole plate: 127 pixels on a 64 x 64 grid (corner to corner).
    assert receptive_field(4) < 127 <= receptive_field(5)


@pytest.mark.parametrize("levels", [2, 3])
def test_receptive_field_matches_an_empirical_gradient_footprint(levels):
    # Without GroupNorm (whose statistics couple every pixel), the set of input
    # pixels that the centre output pixel depends on is exactly the receptive field.
    torch.manual_seed(0)
    widths = tuple(8 * 2**i for i in range(levels))
    net = UNet(1, 1, widths, norm="none").double()
    n = 64
    x = torch.randn(1, 1, n, n, dtype=torch.float64, requires_grad=True)
    net(x)[0, 0, n // 2, n // 2].backward()
    cols = np.flatnonzero(x.grad[0, 0].abs().sum(dim=0).numpy() > 0)
    assert cols.max() - cols.min() + 1 == receptive_field(levels)


# ---------------------------------------------------------------- surrogate (tiny, fast)
@pytest.fixture(scope="module")
def tiny():
    n = 16
    X, Xv = draw(24), draw(8)
    T = np.stack([solve_plate(PlateParams(*r), n) for r in X])
    Tv = np.stack([solve_plate(PlateParams(*r), n) for r in Xv])
    cfg = UNetConfig(widths=(8, 16, 16), epochs=20, batch_size=8, eval_every=5)
    on = FieldUNet(cfg, n).fit(X, T, Xv, Tv)
    off = FieldUNet(UNetConfig(**{**cfg.__dict__, "scale_norm": False}), n).fit(X, T, Xv, Tv)
    return dict(n=n, X=X, T=T, Xv=Xv, on=on, off=off)


def test_predictions_have_exact_edges(tiny):
    P = tiny["on"].predict(tiny["Xv"])
    ring = interior_mask(tiny["n"]) == 0
    for row, Pi in zip(tiny["Xv"], P):
        np.testing.assert_allclose(Pi[ring], solve_plate(PlateParams(*row), tiny["n"])[ring],
                                   atol=1e-12)


def test_scale_norm_makes_the_model_exactly_scale_equivariant(tiny):
    X = tiny["Xv"]
    for c in (0.3, 4.0):
        np.testing.assert_allclose(tiny["on"].predict(scaled(X, c)),
                                   c * tiny["on"].predict(X), rtol=1e-6, atol=1e-6)


def test_without_scale_norm_equivariance_is_not_automatic(tiny):
    X = tiny["Xv"]
    a, b = tiny["off"].predict(scaled(X, 4.0)), 4.0 * tiny["off"].predict(X)
    assert np.abs(a - b).max() > 1e-3


def test_training_reduces_the_loss_and_keeps_the_best_epoch(tiny):
    h = tiny["on"].history
    assert h["train"][-1] < h["train"][0]
    assert tiny["on"].best_epoch in h["epoch"]


def test_save_and_load_give_identical_predictions(tiny, tmp_path):
    tiny["on"].save(tmp_path / "unet.pt")
    m = FieldUNet.load(tmp_path / "unet.pt")
    np.testing.assert_array_equal(m.predict(tiny["Xv"]), tiny["on"].predict(tiny["Xv"]))
