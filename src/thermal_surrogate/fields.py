"""Encode plate parameters as images on the grid, for field-to-field models.

Every channel is computed from the 9 parameters with closed-form expressions;
no solver call is involved.

    source  q(x, y) / k      the heat source divided by conductivity
    lift    L(x, y)          a cheap field that equals the edge temperatures on
                             the boundary ring (and blends them inside)
    x, y                     coordinate channels (the walls break translation
                             symmetry, so the network needs to know where it is)

Why q / k: the solution is T = T_b + u / k with -lap(u) = q and T_b harmonic with
the edge temperatures (P01 solver drill, Q4). So A and k only ever appear as A/k,
and q/k carries both exactly.

Why a lift: a field model predicts T = L + mask * correction, with mask = 0 on the
ring. The edges are then exact by construction, whatever the network outputs.
"""
from __future__ import annotations

import numpy as np

from .solver import PlateParams, edge_values, make_grid

CHANNELS = ("source", "lift", "x", "y")


def source_over_k(X: np.ndarray, n: int = 64) -> np.ndarray:
    """(N, 9) parameters -> (N, n, n) Gaussian source divided by conductivity."""
    Xg, Yg, _ = make_grid(n)
    k, amp, x0, y0, sig = (X[:, i, None, None] for i in range(5))
    r2 = (Xg - x0) ** 2 + (Yg - y0) ** 2
    return (amp / k) * np.exp(-r2 / (2.0 * sig**2))


def boundary_lift(X: np.ndarray, n: int = 64, power: float = 2.0) -> np.ndarray:
    """(N, 9) -> (N, n, n) field that equals the solver's boundary ring exactly.

    Interior: inverse-distance blend of the four edge temperatures,
        L = sum_s w_s t_s / sum_s w_s,   w_s = 1 / d_s^power,
    where d_s is the distance to side s. It is NOT the true harmonic field T_b
    (that would need a solve); it only has to match the edges, so the network
    learns a correction that is zero on the boundary.
    Ring: overwritten with `edge_values`, i.e. exactly what the solver outputs
    there (edge temperatures, corners = mean of the two edges that meet).
    """
    Xg, Yg, _ = make_grid(n)
    d = np.stack([Xg, 1 - Xg, Yg, 1 - Yg])                    # left, right, bottom, top
    inner = (slice(1, -1), slice(1, -1))
    w = np.zeros_like(d)
    w[(slice(None),) + inner] = 1.0 / d[(slice(None),) + inner] ** power
    w_sum = w.sum(axis=0)
    t = X[:, 5:9]                                             # (N, 4): left, right, bottom, top
    L = np.einsum("ns,sij->nij", t, w)
    L[(slice(None),) + inner] /= w_sum[inner]
    for i, row in enumerate(X):                               # exact ring, same rule as the solver
        G = edge_values(PlateParams(*row), n)
        L[i, 0, :], L[i, -1, :], L[i, :, 0], L[i, :, -1] = G[0, :], G[-1, :], G[:, 0], G[:, -1]
    return L


def interior_mask(n: int = 64) -> np.ndarray:
    """(n, n): 1 on interior nodes, 0 on the boundary ring."""
    m = np.zeros((n, n))
    m[1:-1, 1:-1] = 1.0
    return m


def plate_scale(X: np.ndarray, n: int = 64) -> np.ndarray:
    """A per-plate temperature scale s(X) > 0 that is homogeneous of degree 1.

    The solution is linear in (source, edge temperatures) jointly: multiplying
    A and all four edge temperatures by c > 0 multiplies T by c. If s has the
    same property, s(cX) = c s(X), then a model of the form
        T = s * g(inputs / s)
    is EXACTLY scale-equivariant, for any network g.

    s = RMS of the lift (edge-temperature scale) + 2 (A/k) sigma^2 (the order of
    the source-only temperature rise; -lap(u) = q gives u ~ q * width^2).
    Both terms scale by c, so their sum does too.
    """
    L = boundary_lift(X, n)
    edge_part = np.sqrt((L**2).reshape(len(X), -1).mean(axis=1))
    source_part = 2.0 * X[:, 1] / X[:, 0] * X[:, 4] ** 2
    return edge_part + source_part + 1e-12


def encode(X: np.ndarray, n: int = 64) -> np.ndarray:
    """(N, 9) parameters -> (N, 4, n, n) input images: source, lift, x, y."""
    Xg, Yg, _ = make_grid(n)
    N = len(X)
    out = np.empty((N, 4, n, n))
    out[:, 0] = source_over_k(X, n)
    out[:, 1] = boundary_lift(X, n)
    out[:, 2] = Xg
    out[:, 3] = Yg
    return out
