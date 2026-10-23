"""Verification tests for the finite-difference heat solver."""
import numpy as np
import pytest

from thermal_surrogate.solver import (
    PlateParams,
    make_grid,
    solve_plate,
    solve_poisson,
)

K = 2.5  # non-unit conductivity so a missing factor of k cannot hide


# Manufactured solution with non-trivial values on every edge.
def t_exact(X, Y):
    return np.sin(2 * np.pi * X) * np.cos(np.pi * Y) + X**2 + Y


def q_exact(X, Y):
    # -k lap(T) with lap(T) = -5 pi^2 sin(2 pi x) cos(pi y) + 2
    return K * (5 * np.pi**2 * np.sin(2 * np.pi * X) * np.cos(np.pi * Y) - 2.0)


def max_error(n):
    X, Y, _ = make_grid(n)
    T = solve_poisson(n, K, q_exact, t_exact)
    return np.max(np.abs(T - t_exact(X, Y)))


def test_second_order_convergence():
    ns = [17, 33, 65, 129]                    # h halves each step
    hs = np.array([1.0 / (n - 1) for n in ns])
    errs = np.array([max_error(n) for n in ns])
    # Observed order between successive refinements, and a global fit.
    local = np.log(errs[:-1] / errs[1:]) / np.log(hs[:-1] / hs[1:])
    slope = np.polyfit(np.log(hs), np.log(errs), 1)[0]
    assert np.all(errs[1:] < errs[:-1]), errs
    assert np.all(np.abs(local - 2.0) < 0.1), local
    assert abs(slope - 2.0) < 0.05, slope


def test_quadratic_is_exact():
    # The 5-point stencil has zero truncation error for quadratics.
    f = lambda X, Y: 1 + 2 * X - 3 * Y + X**2 + 0.5 * Y**2
    q = lambda X, Y: -K * (2.0 + 1.0) * np.ones_like(X)
    X, Y, _ = make_grid(20)
    T = solve_poisson(20, K, q, f)
    np.testing.assert_allclose(T, f(X, Y), atol=1e-10)


def test_no_source_uniform_boundary_gives_uniform_field():
    p = PlateParams(amp=0.0, t_left=7, t_right=7, t_bottom=7, t_top=7)
    np.testing.assert_allclose(solve_plate(p, 32), 7.0, atol=1e-12)


def test_maximum_principle_with_positive_source():
    # q >= 0 means heat only enters, so T can't drop below the coldest edge.
    p = PlateParams(amp=80.0, t_left=10, t_right=20, t_bottom=15, t_top=5)
    T = solve_plate(p, 48)
    assert T.min() >= 5.0 - 1e-12
    # Superposition: T = T_boundary-only + T_source-only, and the source-only
    # part (zero BCs, q > 0) is strictly positive inside the plate.
    T0 = solve_plate(PlateParams(amp=0.0, t_left=10, t_right=20, t_bottom=15, t_top=5), 48)
    assert np.all(T[1:-1, 1:-1] > T0[1:-1, 1:-1])


def test_mirror_symmetry():
    # Source centred in x with equal left/right temperatures -> T(x) = T(1-x).
    p = PlateParams(x0=0.5, y0=0.3, t_left=3, t_right=3, t_bottom=0, t_top=9)
    T = solve_plate(p, 41)
    np.testing.assert_allclose(T, T[:, ::-1], atol=1e-10)


def test_higher_conductivity_lowers_peak():
    base = PlateParams(amp=50.0)
    t1 = solve_plate(base, 32).max()
    t2 = solve_plate(PlateParams(k=4.0, amp=50.0), 32).max()
    # With zero boundaries T is exactly proportional to 1/k.
    assert t2 == pytest.approx(t1 / 4.0, rel=1e-10)


def test_rejects_bad_inputs():
    with pytest.raises(ValueError):
        solve_poisson(2, 1.0, lambda X, Y: X, lambda X, Y: X)
    with pytest.raises(ValueError):
        solve_poisson(10, -1.0, lambda X, Y: X, lambda X, Y: X)


def test_batch_solver_matches_reference():
    from thermal_surrogate.solver import PlateBatchSolver
    rng = np.random.default_rng(0)
    rows = np.column_stack([rng.uniform(0.5, 5, 6), rng.uniform(500, 5000, 6),
                            rng.uniform(0.2, 0.8, (6, 2)), rng.uniform(0.05, 0.2, 6),
                            rng.uniform(0, 100, (6, 4))])
    fast = PlateBatchSolver(33).solve(rows)
    for row, T in zip(rows, fast):
        np.testing.assert_allclose(T, solve_plate(PlateParams(*row), 33), rtol=1e-10, atol=1e-10)
