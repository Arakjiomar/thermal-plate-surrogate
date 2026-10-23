"""Finite-difference solver for steady 2D heat conduction on the unit square.

Problem
-------
    -k * laplacian(T) = q(x, y)     on (0, 1) x (0, 1)
    T = g(x, y)                      on the boundary (Dirichlet)

with uniform conductivity ``k > 0``.

Discretisation
--------------
Uniform grid of ``n`` nodes per side *including* the boundary, spacing
``h = 1 / (n - 1)``. Unknowns are the ``m = n - 2`` interior nodes per side.
The 5-point stencil

    k/h^2 * (4 T[i,j] - T[i+1,j] - T[i-1,j] - T[i,j+1] - T[i,j-1]) = q[i,j]

has truncation error O(h^2). Interior unknowns are ordered row-major with
``T[j, i]`` meaning y-index ``j`` and x-index ``i`` (so arrays plot directly
with ``imshow(origin="lower")``). Neighbours that sit on the boundary are known,
so they move to the right-hand side.

The matrix is assembled with Kronecker products:

    A = k * (I_m kron D + D kron I_m),   D = tridiag(-1, 2, -1) / h^2

and solved with a sparse direct LU factorisation.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

ArrayFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


# ----------------------------------------------------------------------------
# Grid and operator
# ----------------------------------------------------------------------------
def make_grid(n: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Return meshgrid arrays ``X, Y`` of shape (n, n) and spacing ``h``.

    ``X[j, i] = x_i`` and ``Y[j, i] = y_j`` (rows are y, columns are x).
    """
    if n < 3:
        raise ValueError("need at least 3 nodes per side (1 interior node)")
    x = np.linspace(0.0, 1.0, n)
    X, Y = np.meshgrid(x, x, indexing="xy")
    return X, Y, 1.0 / (n - 1)


def laplacian_matrix(n: int, k: float = 1.0) -> sp.csc_matrix:
    """Sparse matrix for ``-k * laplacian`` on the (n-2)^2 interior nodes."""
    m = n - 2
    h = 1.0 / (n - 1)
    main = 2.0 * np.ones(m)
    off = -1.0 * np.ones(m - 1)
    D = sp.diags([off, main, off], [-1, 0, 1], format="csr") / h**2
    I = sp.identity(m, format="csr")
    # Row-major ordering (index = j*m + i): x-direction couples neighbours
    # within a row -> kron(I, D); y-direction couples rows -> kron(D, I).
    return (k * (sp.kron(I, D) + sp.kron(D, I))).tocsc()


# ----------------------------------------------------------------------------
# General Dirichlet-Poisson solve
# ----------------------------------------------------------------------------
def solve_poisson(
    n: int,
    k: float,
    q: np.ndarray | ArrayFn,
    g: np.ndarray | ArrayFn,
) -> np.ndarray:
    """Solve ``-k lap T = q`` with ``T = g`` on the boundary.

    Parameters
    ----------
    n : nodes per side including boundary.
    k : conductivity (> 0).
    q : source, either an (n, n) array or a function ``q(X, Y)``.
    g : boundary values, either an (n, n) array (only its edges are read)
        or a function ``g(X, Y)``.

    Returns
    -------
    T : (n, n) array, boundary nodes set from ``g``.
    """
    if k <= 0:
        raise ValueError("conductivity k must be positive")
    X, Y, h = make_grid(n)
    Q = q(X, Y) if callable(q) else np.asarray(q, dtype=float)
    G = g(X, Y) if callable(g) else np.asarray(g, dtype=float)
    if Q.shape != (n, n) or G.shape != (n, n):
        raise ValueError(f"q and g must have shape {(n, n)}")

    # Right-hand side: interior source + known boundary neighbours.
    b = Q[1:-1, 1:-1].copy()
    c = k / h**2
    b[:, 0] += c * G[1:-1, 0]     # left edge   (x = 0)
    b[:, -1] += c * G[1:-1, -1]   # right edge  (x = 1)
    b[0, :] += c * G[0, 1:-1]     # bottom edge (y = 0)
    b[-1, :] += c * G[-1, 1:-1]   # top edge    (y = 1)

    A = laplacian_matrix(n, k)
    t_int = spla.spsolve(A, b.ravel())

    T = G.copy()
    T[1:-1, 1:-1] = t_int.reshape(n - 2, n - 2)
    return T


# ----------------------------------------------------------------------------
# The parametrised plate problem used to generate surrogate training data
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class PlateParams:
    """Physical parameters of one plate simulation (9 inputs to the surrogate)."""

    k: float = 1.0          # conductivity
    amp: float = 50.0       # Gaussian source amplitude
    x0: float = 0.5         # source centre x
    y0: float = 0.5         # source centre y
    sigma: float = 0.1      # source width
    t_left: float = 0.0     # Dirichlet temperature, x = 0
    t_right: float = 0.0    # x = 1
    t_bottom: float = 0.0   # y = 0
    t_top: float = 0.0      # y = 1

    def as_array(self) -> np.ndarray:
        return np.array(list(asdict(self).values()), dtype=float)

    @classmethod
    def names(cls) -> list[str]:
        return list(cls.__dataclass_fields__)


def gaussian_source(p: PlateParams) -> ArrayFn:
    def q(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        r2 = (X - p.x0) ** 2 + (Y - p.y0) ** 2
        return p.amp * np.exp(-r2 / (2.0 * p.sigma**2))
    return q


def edge_values(p: PlateParams, n: int) -> np.ndarray:
    """(n, n) array whose edges hold the four Dirichlet temperatures.

    Corners do not enter the 5-point stencil; they are set to the mean of the
    two edges that meet there so the output field has no artificial spike.
    """
    G = np.zeros((n, n))
    G[:, 0] = p.t_left
    G[:, -1] = p.t_right
    G[0, :] = p.t_bottom
    G[-1, :] = p.t_top
    G[0, 0] = 0.5 * (p.t_left + p.t_bottom)
    G[0, -1] = 0.5 * (p.t_right + p.t_bottom)
    G[-1, 0] = 0.5 * (p.t_left + p.t_top)
    G[-1, -1] = 0.5 * (p.t_right + p.t_top)
    return G


def solve_plate(p: PlateParams, n: int = 64) -> np.ndarray:
    """Temperature field (n, n) for one set of plate parameters."""
    return solve_poisson(n, p.k, gaussian_source(p), edge_values(p, n))


# ----------------------------------------------------------------------------
# Optimized batch solver: factorize once, reuse for every plate
# ----------------------------------------------------------------------------
class PlateBatchSolver:
    """Solve many plates on the same grid with ONE sparse LU factorization.

    The operator is k * L with L independent of the parameters, so

        k L t = b   <=>   L t = b / k.

    L is factorized once (scipy's SuperLU); each plate then costs only two
    sparse triangular solves, and many right-hand sides go in one call.
    """

    def __init__(self, n: int = 64):
        self.n = n
        self.X, self.Y, self.h = make_grid(n)
        self.lu = spla.splu(laplacian_matrix(n, k=1.0))

    def solve(self, params: np.ndarray) -> np.ndarray:
        """(N, 9) parameter rows -> (N, n, n) fields."""
        n, m = self.n, self.n - 2
        B = np.empty((m * m, len(params)))
        G_all = []
        for j, row in enumerate(params):
            p = PlateParams(*row)
            Q = gaussian_source(p)(self.X, self.Y)
            G = edge_values(p, n)
            b = Q[1:-1, 1:-1].copy()
            c = p.k / self.h**2
            b[:, 0] += c * G[1:-1, 0]
            b[:, -1] += c * G[1:-1, -1]
            b[0, :] += c * G[0, 1:-1]
            b[-1, :] += c * G[-1, 1:-1]
            B[:, j] = b.ravel() / p.k                    # divide by k: solve with L, not kL
            G_all.append(G)
        Tint = self.lu.solve(B)                          # all right-hand sides at once
        out = np.array(G_all)
        out[:, 1:-1, 1:-1] = Tint.T.reshape(len(params), m, m)
        return out
