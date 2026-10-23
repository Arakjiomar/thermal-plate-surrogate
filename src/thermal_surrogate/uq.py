"""Uncertainty quantification with a deep ensemble.

M networks, identical except for their random seed, are trained on the same
data. Where the data pins the answer down they agree; away from the data they
disagree. Their spread is used as an estimate of the prediction's error:

    mean field   mu(x)    = (1/M) sum_m f_m(x)
    spread field sigma(x) = sqrt( (1/(M-1)) sum_m (f_m(x) - mu(x))^2 )

The raw spread is usually too small (M members share the same data and
architecture), so it is rescaled by one factor s fitted on the validation set:
sigma_cal = s * sigma, with s chosen so that the standardized errors
z = (T - mu) / (s sigma) have mean square 1 on validation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.special import ndtri

from .mlp import FieldSurrogate, TrainConfig


class Ensemble:
    def __init__(self, members: list[FieldSurrogate]):
        if len(members) < 2:
            raise ValueError("an ensemble needs at least 2 members")
        self.members = members
        self.scale = 1.0                    # calibration factor s (1 = uncalibrated)

    @classmethod
    def train(cls, X, T, X_val, T_val, cfg: TrainConfig, seeds, kind="pod", verbose=False):
        members = []
        for s in seeds:
            c = TrainConfig(**{**cfg.__dict__, "seed": s})
            members.append(FieldSurrogate(kind, physics=True).fit(X, T, X_val, T_val, c))
            if verbose:
                print(f"  member seed {s} done (best epoch {members[-1].best_epoch})")
        return cls(members)

    def predict_all(self, X: np.ndarray) -> np.ndarray:
        """Every member's prediction -> (M, N, n, n)."""
        return np.stack([m.predict(X) for m in self.members])

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Ensemble mean and (calibrated) spread fields, each (N, n, n)."""
        P = self.predict_all(X)
        return P.mean(axis=0), self.scale * P.std(axis=0, ddof=1)

    def calibrate(self, X_val: np.ndarray, T_val: np.ndarray) -> float:
        """Fit the single scale factor s on validation data (interior pixels only)."""
        self.scale = 1.0
        mu, sigma = self.predict(X_val)
        z = interior(T_val - mu) / interior(sigma)
        self.scale = float(np.sqrt(np.mean(z**2)))
        return self.scale

    def save(self, folder: str | Path) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        for i, m in enumerate(self.members):
            m.save(folder / f"member_{i}.pt")
        (folder / "ensemble.json").write_text(
            json.dumps({"n_members": len(self.members), "scale": self.scale}))

    @classmethod
    def load(cls, folder: str | Path) -> "Ensemble":
        folder = Path(folder)
        meta = json.loads((folder / "ensemble.json").read_text())
        e = cls([FieldSurrogate.load(folder / f"member_{i}.pt") for i in range(meta["n_members"])])
        e.scale = meta["scale"]
        return e


# ----------------------------------------------------------------------------
# Evaluation helpers
# ----------------------------------------------------------------------------
def interior(F: np.ndarray) -> np.ndarray:
    """Drop the boundary ring before scoring uncertainty.

    The surrogate does predict the edges (and can miss them out of distribution),
    but the true edge temperatures are INPUTS: a deployed surrogate should simply
    overwrite them, so only interior predictions carry real uncertainty. Corner
    pixels can also have ~0 spread, which would make z = error / spread blow up.
    """
    return F[..., 1:-1, 1:-1]


def coverage(T, mu, sigma, levels=(0.5, 0.8, 0.9, 0.95, 0.99)) -> dict[float, float]:
    """Fraction of interior pixels inside the central Gaussian interval mu +- k(p) sigma.

    For a well-calibrated Gaussian predictive distribution, the fraction
    inside the p-interval equals p.
    """
    z = np.abs(interior(T - mu) / interior(sigma))
    out = {}
    for p in levels:
        k = ndtri((1 + p) / 2)                       # inverse normal CDF: p = 0.95 -> k = 1.96
        out[p] = float(np.mean(z <= k))
    return out


def per_sample_spread(sigma: np.ndarray) -> np.ndarray:
    """One number per field: RMS of the spread over interior pixels -> (N,)."""
    s = interior(sigma)
    return np.sqrt((s**2).reshape(len(s), -1).mean(axis=1))


def per_sample_rmse(T: np.ndarray, mu: np.ndarray) -> np.ndarray:
    e = interior(T - mu)
    return np.sqrt((e**2).reshape(len(e), -1).mean(axis=1))


def auroc(scores_neg: np.ndarray, scores_pos: np.ndarray) -> float:
    """Probability that a random positive scores higher than a random negative
    (ties count half). 0.5 = no better than chance, 1.0 = perfect separation."""
    neg, pos = np.asarray(scores_neg), np.asarray(scores_pos)
    greater = (pos[:, None] > neg[None, :]).mean()
    ties = (pos[:, None] == neg[None, :]).mean()
    return float(greater + 0.5 * ties)
