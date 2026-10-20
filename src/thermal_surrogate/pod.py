"""POD + ridge regression: the classical reduced-order-model baseline.

1. POD   - SVD of the centred training fields gives orthonormal spatial modes
           phi_r and their energies lambda_r = s_r^2 / N.
2. Ridge - polynomial features of the 9 parameters -> the first R POD
           coefficients, fitted in closed form  W = (F^T F + alpha I)^-1 F^T A.
3. Predict - T(p) = mean + sum_r a_r(p) phi_r.
"""
from __future__ import annotations

from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------------
# Proper Orthogonal Decomposition
# ----------------------------------------------------------------------------
class POD:
    """Modes of a set of fields. Fields are flattened row-major to length M = n*n."""

    def fit(self, T: np.ndarray) -> "POD":
        N = len(T)
        self.shape = T.shape[1:]                         # (n, n), to un-flatten later
        F = T.reshape(N, -1).astype(np.float64)          # (N, M): one field per ROW
        self.mean = F.mean(axis=0)                       # (M,)
        Y = (F - self.mean).T                            # (M, N): one field per COLUMN
        U, s, _ = np.linalg.svd(Y, full_matrices=False)  # U (M, N), s (N,)
        self.modes = U                                   # columns are phi_r
        self.energy = s**2 / N                           # lambda_r
        return self

    def energy_fraction(self) -> np.ndarray:
        """Cumulative fraction of variance captured by the first 1, 2, ... modes."""
        return np.cumsum(self.energy) / self.energy.sum()

    def n_modes_for(self, fraction: float) -> int:
        """Smallest R whose modes capture at least `fraction` of the variance."""
        return int(np.searchsorted(self.energy_fraction(), fraction) + 1)

    def project(self, T: np.ndarray, R: int) -> np.ndarray:
        """Coefficients a_r = phi_r . (t - mean) for the first R modes -> (N, R)."""
        Y = T.reshape(len(T), -1) - self.mean
        return Y @ self.modes[:, :R]

    def reconstruct(self, a: np.ndarray) -> np.ndarray:
        """Fields from coefficients: mean + sum_r a_r phi_r -> (N, n, n)."""
        R = a.shape[1]
        F = self.mean + a @ self.modes[:, :R].T
        return F.reshape(len(a), *self.shape)


# ----------------------------------------------------------------------------
# Features and ridge regression
# ----------------------------------------------------------------------------
def polynomial_features(Z: np.ndarray, degree: int) -> np.ndarray:
    """All monomials of the columns of Z up to `degree` (no constant term).

    For 9 inputs: degree 1 -> 9 features, 2 -> 54, 3 -> 219, 4 -> 714.
    """
    cols = []
    for d in range(1, degree + 1):
        for idx in combinations_with_replacement(range(Z.shape[1]), d):
            cols.append(np.prod(Z[:, idx], axis=1))
    return np.stack(cols, axis=1)


def physics_features(X: np.ndarray) -> np.ndarray:
    """Append A/k and 1/k to the 9 raw parameters -> (N, 11).

    With zero edges the solution is exactly (A/k) * u(x0, y0, sigma) (drill Q4),
    so these two combinations carry the source's scaling directly; a polynomial
    in the raw k and A can only approximate 1/k.
    """
    k, amp = X[:, 0], X[:, 1]
    return np.column_stack([X, amp / k, 1.0 / k])


class Standardizer:
    """Shift and scale each column to mean 0, std 1 using TRAINING statistics."""

    def fit(self, X: np.ndarray) -> "Standardizer":
        self.mu, self.sd = X.mean(axis=0), X.std(axis=0)
        self.sd[self.sd == 0] = 1.0
        return self

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mu) / self.sd


def ridge_fit(F: np.ndarray, A: np.ndarray, alpha: float) -> np.ndarray:
    """Closed-form ridge, W = (F^T F + alpha I)^-1 F^T A. F and A must be centred."""
    k = F.shape[1]
    return np.linalg.solve(F.T @ F + alpha * np.eye(k), F.T @ A)


# ----------------------------------------------------------------------------
# The full surrogate
# ----------------------------------------------------------------------------
class PODRidge:
    """Parameters (N, 9) -> temperature fields (N, n, n)."""

    def __init__(self, n_modes: int, degree: int, alpha: float, physics: bool = False):
        self.n_modes, self.degree, self.alpha = n_modes, degree, alpha
        self.physics = physics

    def _inputs(self, X: np.ndarray) -> np.ndarray:
        return physics_features(X) if self.physics else X

    def fit(self, X: np.ndarray, T: np.ndarray, pod: POD | None = None) -> "PODRidge":
        self.pod = pod if pod is not None else POD().fit(T)
        A = self.pod.project(T, self.n_modes)              # (N, R), mean 0 by construction
        Z = self._inputs(X)
        self.x_scaler = Standardizer().fit(Z)
        F = polynomial_features(self.x_scaler(Z), self.degree)
        self.f_scaler = Standardizer().fit(F)              # centres the features too
        self.W = ridge_fit(self.f_scaler(F), A, self.alpha)
        return self

    def predict_coeffs(self, X: np.ndarray) -> np.ndarray:
        F = polynomial_features(self.x_scaler(self._inputs(X)), self.degree)
        return self.f_scaler(F) @ self.W

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.pod.reconstruct(self.predict_coeffs(X))

    # -- persistence ---------------------------------------------------------
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, n_modes=self.n_modes, degree=self.degree, alpha=self.alpha,
            physics=self.physics,
            mean=self.pod.mean, modes=self.pod.modes[:, : self.n_modes],
            energy=self.pod.energy, shape=np.array(self.pod.shape), W=self.W,
            x_mu=self.x_scaler.mu, x_sd=self.x_scaler.sd,
            f_mu=self.f_scaler.mu, f_sd=self.f_scaler.sd)

    @classmethod
    def load(cls, path: str | Path) -> "PODRidge":
        z = np.load(path)
        m = cls(int(z["n_modes"]), int(z["degree"]), float(z["alpha"]), bool(z["physics"]))
        m.pod = POD()
        m.pod.mean, m.pod.modes, m.pod.energy = z["mean"], z["modes"], z["energy"]
        m.pod.shape = tuple(int(v) for v in z["shape"])
        m.W = z["W"]
        m.x_scaler, m.f_scaler = Standardizer(), Standardizer()
        m.x_scaler.mu, m.x_scaler.sd = z["x_mu"], z["x_sd"]
        m.f_scaler.mu, m.f_scaler.sd = z["f_mu"], z["f_sd"]
        return m
