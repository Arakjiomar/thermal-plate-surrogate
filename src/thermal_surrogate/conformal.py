"""Split conformal prediction for field surrogates.

The recipe (split conformal):
  1. A model gives a prediction mu and an uncertainty sigma for every pixel.
  2. On a CALIBRATION set (plates the model never trained on), compute a
     nonconformity score s_i per plate: how wrong the model was, in units of
     its own uncertainty.
  3. q_hat = the ceil((n + 1)(1 - alpha))-th smallest of the n scores.
  4. For a new plate, the claim "score <= q_hat" holds with probability
     at least 1 - alpha, and at most 1 - alpha + 1/(n + 1) for continuous scores.

Why it works (no assumptions about the model or the error distribution): if
the new plate is EXCHANGEABLE with the calibration plates (drawn the same way),
its score is equally likely to land at any of the n + 1 positions when all
n + 1 scores are sorted. So P(its rank <= k) = k / (n + 1) >= 1 - alpha.
The assumption is the whole story: a query drawn differently (e.g. outside the
training range when calibration was inside it) has no guarantee.

Scores (one per PLATE, because plates are the exchangeable units):
  "gate"   RMSE / spread over interior pixels.   q_hat * spread is an upper
           bound on the plate's RMSE, valid with prob >= 1 - alpha: the CLI
           defers when that bound exceeds its tolerance.
  "band"   max over interior pixels of |T - mu| / sigma.  mu +- q_hat * sigma
           contains the WHOLE field (every pixel at once) with prob >= 1 - alpha.
  "pixel"  every interior pixel's |T - mu| / sigma, pooled.  Coverage of a
           random pixel ~ 1 - alpha; only approximate, because pixels within a
           plate are correlated (they are not exchangeable units).

Mondrian (group-conditional) conformal: split the calibration plates into
groups (here: bins of OOD distance) and compute q_hat per group. Coverage then
holds within each group, not just on average.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .uq import interior

KINDS = ("gate", "band", "pixel")


# ----------------------------------------------------------------------------
# The core: one quantile
# ----------------------------------------------------------------------------
def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Split-conformal threshold: the ceil((n + 1)(1 - alpha))-th smallest score.

    Returns inf if that rank exceeds n (too few calibration points to certify
    1 - alpha: e.g. n = 8 can't certify 90%, because ceil(9 * 0.9) = 9 > 8).
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    s = np.sort(np.asarray(scores, dtype=float).ravel())
    n = len(s)
    k = math.ceil((n + 1) * (1 - alpha))
    return float(s[k - 1]) if k <= n else math.inf


# ----------------------------------------------------------------------------
# Nonconformity scores
# ----------------------------------------------------------------------------
def scores(kind: str, T: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """(N, n, n) fields -> scores: (N,) for 'gate' and 'band', (N * pixels,) for 'pixel'."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    e, s = interior(T - mu), interior(sigma)
    N = len(e)
    if kind == "gate":
        rmse = np.sqrt((e**2).reshape(N, -1).mean(axis=1))
        spread = np.sqrt((s**2).reshape(N, -1).mean(axis=1))
        return rmse / spread
    z = np.abs(e) / s
    return z.reshape(N, -1).max(axis=1) if kind == "band" else z.ravel()


def per_plate_spread(sigma: np.ndarray) -> np.ndarray:
    """RMS of sigma over interior pixels, one number per plate (same as uq.per_sample_spread)."""
    s = interior(sigma)
    return np.sqrt((s**2).reshape(len(s), -1).mean(axis=1))


# ----------------------------------------------------------------------------
# Calibrator (optionally Mondrian over OOD-distance bins)
# ----------------------------------------------------------------------------
@dataclass
class ConformalCalibrator:
    """Fit q_hat on calibration fields; apply it to new predictions.

    bins: increasing OOD-distance edges for Mondrian conformal, e.g. (0.0, 0.1)
          gives groups  d <= 0  |  0 < d <= 0.1  |  d > 0.1.  None = one group.
    """
    alpha: float = 0.1
    kind: str = "gate"
    bins: tuple | None = None
    q: list = field(default_factory=list)          # one q_hat per group
    n: list = field(default_factory=list)          # calibration plates per group

    def group(self, dist) -> np.ndarray:
        """Group index for each query (0 if not Mondrian)."""
        if self.bins is None:
            return np.zeros(len(np.atleast_1d(dist)) if dist is not None else 1, dtype=int)
        # d <= edge_0 -> 0, edge_0 < d <= edge_1 -> 1, ...
        return np.searchsorted(np.asarray(self.bins), np.asarray(dist), side="left")

    def fit(self, T, mu, sigma, dist=None) -> "ConformalCalibrator":
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        if self.bins is not None and dist is None:
            raise ValueError("Mondrian calibration needs the OOD distance of each plate")
        g = self.group(dist) if self.bins is not None else np.zeros(len(T), dtype=int)
        n_groups = 1 if self.bins is None else len(self.bins) + 1
        self.q, self.n = [], []
        for j in range(n_groups):
            idx = np.flatnonzero(g == j)
            self.n.append(int(len(idx)))
            # An empty group can't be certified: q = inf, i.e. "always defer / infinite interval".
            self.q.append(conformal_quantile(scores(self.kind, T[idx], mu[idx], sigma[idx]), self.alpha)
                          if len(idx) else math.inf)
        return self

    def q_for(self, dist=None, n_queries: int | None = None) -> np.ndarray:
        """q_hat for each query, (N,)."""
        q = np.asarray(self.q)
        if self.bins is None:
            return np.full(n_queries if n_queries is not None else len(np.atleast_1d(dist)), q[0])
        return q[self.group(dist)]

    def interval(self, mu, sigma, dist=None):
        """mu -+ q_hat * sigma per plate -> (lower, upper), each (N, n, n)."""
        q = self.q_for(dist, n_queries=len(mu))[:, None, None]
        return mu - q * sigma, mu + q * sigma

    def rmse_bound(self, sigma, dist=None) -> np.ndarray:
        """'gate' only: upper bound on each plate's interior RMSE, valid w.p. >= 1 - alpha."""
        if self.kind != "gate":
            raise ValueError("rmse_bound needs kind='gate'")
        return self.q_for(dist, n_queries=len(sigma)) * per_plate_spread(sigma)

    def to_dict(self) -> dict:
        return {"alpha": self.alpha, "kind": self.kind,
                "bins": None if self.bins is None else list(self.bins),
                "q": [None if math.isinf(v) else v for v in self.q], "n": self.n}

    @classmethod
    def from_dict(cls, d: dict) -> "ConformalCalibrator":
        c = cls(d["alpha"], d["kind"], None if d["bins"] is None else tuple(d["bins"]))
        c.q = [math.inf if v is None else v for v in d["q"]]
        c.n = list(d["n"])
        return c


def coverage_of(cal: ConformalCalibrator, T, mu, sigma, dist=None) -> float:
    """Fraction of queries satisfying the calibrator's claim (the thing it guarantees).

    gate: plate RMSE <= q * spread;  band: every interior pixel inside;  pixel: pixel inside.
    """
    q = cal.q_for(dist, n_queries=len(T))
    if cal.kind == "pixel":
        z = np.abs(interior(T - mu)) / interior(sigma)
        return float(np.mean(z <= q[:, None, None]))
    return float(np.mean(scores(cal.kind, T, mu, sigma) <= q))
