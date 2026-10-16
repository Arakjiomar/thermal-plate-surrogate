"""Dataset generation: sample plate parameters and solve for temperature fields.

Sampling
--------
* In-distribution: one Latin hypercube over the 9-D parameter box, randomly
  split into train / val / test.
* Out-of-distribution (OOD): a Latin hypercube over a *widened* box, keeping
  only points that leave the original box in at least one coordinate. These
  test whether the surrogate extrapolates (it shouldn't be trusted to).

Every sample is stored with its normalised "OOD distance": how far, in units
of the original range width, its furthest coordinate sits outside the box
(0 for in-distribution points). This is the x-axis for the later
uncertainty-vs-distance plots.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .solver import PlateParams, solve_plate

PARAM_NAMES = PlateParams.names()


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
@dataclass
class DatasetConfig:
    n: int
    lows: np.ndarray        # (9,) in-distribution box
    highs: np.ndarray
    n_train: int
    n_val: int
    n_test: int
    n_ood: int
    ood_expand: float
    phys_lows: np.ndarray   # (9,) hard physical limits (-inf/inf if none)
    phys_highs: np.ndarray
    seed: int

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DatasetConfig":
        raw = yaml.safe_load(Path(path).read_text())
        box = raw["params"]
        if list(box) != PARAM_NAMES:
            raise ValueError(f"config params must be, in order: {PARAM_NAMES}")
        lows = np.array([box[k][0] for k in PARAM_NAMES], dtype=float)
        highs = np.array([box[k][1] for k in PARAM_NAMES], dtype=float)
        lim = raw.get("physical_limits", {}) or {}
        plo = np.array([_or(lim.get(k, [None, None])[0], -np.inf) for k in PARAM_NAMES])
        phi = np.array([_or(lim.get(k, [None, None])[1], np.inf) for k in PARAM_NAMES])
        return cls(raw["n"], lows, highs, raw["n_train"], raw["n_val"], raw["n_test"],
                   raw["n_ood"], raw["ood_expand"], plo, phi, raw["seed"])


def _or(v, default):
    return default if v is None else float(v)


# ----------------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------------
def latin_hypercube(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """n points in [0, 1)^d with exactly one point in each of the n equal-width
    strata along every axis.

    For each dimension: split [0, 1) into n slices, put one uniform point in
    each slice, then shuffle which sample gets which slice (independently per
    dimension, so the coordinates are not correlated).
    """
    u = (np.arange(n)[:, None] + rng.random((n, d))) / n   # one point per slice
    for j in range(d):
        u[:, j] = u[rng.permutation(n), j]                  # decouple the axes
    return u


def scale(u: np.ndarray, lows: np.ndarray, highs: np.ndarray) -> np.ndarray:
    return lows + u * (highs - lows)


def ood_distance(X: np.ndarray, lows: np.ndarray, highs: np.ndarray) -> np.ndarray:
    """Largest excursion outside the box, in units of each range's width (0 inside)."""
    width = highs - lows
    below = (lows - X) / width
    above = (X - highs) / width
    return np.maximum(0.0, np.maximum(below, above)).max(axis=1)


def sample_in_distribution(cfg: DatasetConfig, rng) -> dict[str, np.ndarray]:
    total = cfg.n_train + cfg.n_val + cfg.n_test
    X = scale(latin_hypercube(total, len(PARAM_NAMES), rng), cfg.lows, cfg.highs)
    idx = rng.permutation(total)
    a, b = cfg.n_train, cfg.n_train + cfg.n_val
    return {"train": X[idx[:a]], "val": X[idx[a:b]], "test": X[idx[b:]]}


def sample_ood(cfg: DatasetConfig, rng, max_rounds: int = 50) -> np.ndarray:
    """Latin hypercube on the widened box; reject points inside the original box."""
    width = cfg.highs - cfg.lows
    wlo = np.maximum(cfg.lows - cfg.ood_expand * width, cfg.phys_lows)
    whi = np.minimum(cfg.highs + cfg.ood_expand * width, cfg.phys_highs)
    kept = []
    for _ in range(max_rounds):
        X = scale(latin_hypercube(cfg.n_ood, len(PARAM_NAMES), rng), wlo, whi)
        kept.append(X[ood_distance(X, cfg.lows, cfg.highs) > 0])
        if sum(len(k) for k in kept) >= cfg.n_ood:
            return np.concatenate(kept)[: cfg.n_ood]
    raise RuntimeError("could not draw enough OOD samples; increase ood_expand")


# ----------------------------------------------------------------------------
# Solving and saving
# ----------------------------------------------------------------------------
def solve_batch(X: np.ndarray, n: int) -> np.ndarray:
    """(N, 9) parameters -> (N, n, n) float32 temperature fields."""
    out = np.empty((len(X), n, n), dtype=np.float32)
    for i, row in enumerate(X):
        out[i] = solve_plate(PlateParams(*row), n)
    return out


def generate(cfg: DatasetConfig, verbose: bool = False) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    splits = sample_in_distribution(cfg, rng)
    splits["ood"] = sample_ood(cfg, rng)
    data: dict[str, np.ndarray] = {}
    for name, X in splits.items():
        if verbose:
            print(f"solving {name:5s}: {len(X)} samples")
        data[f"X_{name}"] = X
        data[f"T_{name}"] = solve_batch(X, cfg.n)
        data[f"ood_dist_{name}"] = ood_distance(X, cfg.lows, cfg.highs)
    data["param_names"] = np.array(PARAM_NAMES)
    data["box_lows"], data["box_highs"] = cfg.lows, cfg.highs
    return data


def fingerprint(data: dict[str, np.ndarray]) -> str:
    """Short SHA-256 of all arrays, so two runs can be checked as identical."""
    h = hashlib.sha256()
    for key in sorted(data):
        h.update(key.encode())
        h.update(np.ascontiguousarray(data[key]).tobytes())
    return h.hexdigest()[:16]


def save(data: dict[str, np.ndarray], path: str | Path, meta: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data, meta=np.array(json.dumps(meta)))


def load(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}
