"""Active learning: let the surrogate choose which simulations to run next.

Pool-based, batch-mode loop:

    labelled set L  (parameters + solved fields)
    pool U          (parameters only: free to generate, expensive to label)

    repeat until the simulation budget is spent:
        1. train a deep ensemble on L
        2. score every candidate in U        (acquisition function)
        3. pick a batch of B candidates      (selection rule)
        4. run the solver on them, move them from U to L

Three selection rules are compared:
    random   : B candidates uniformly at random (the baseline)
    spread   : the B candidates with the largest ensemble spread (greedy)
    diverse  : the top `factor * B` by spread, then B of those chosen to be
               far apart in parameter space (farthest-point selection)

Greedy top-B tends to pick near-duplicates: if one region is uncertain, every
point in it scores high, and the whole batch lands there. The diverse rule
keeps the "go where the model is unsure" signal but spreads the batch out.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .dataset import solve_batch
from .metrics import rel_l2
from .mlp import TrainConfig
from .uq import Ensemble, coverage, per_sample_rmse, per_sample_spread

STRATEGIES = ("random", "spread", "diverse")


# ----------------------------------------------------------------------------
# Acquisition: one score per candidate
# ----------------------------------------------------------------------------
def pool_spread(ens: Ensemble, X: np.ndarray, chunk: int = 500) -> np.ndarray:
    """RMS interior ensemble spread (deg C) for every row of X -> (N,).

    Computed in chunks: 20,000 candidates x 5 members x 4096 pixels in float64
    would be 3.3 GB at once. The calibration factor is a single constant, so it
    doesn't change the ranking and is left out (raw spread).
    """
    out = np.empty(len(X))
    for i in range(0, len(X), chunk):
        P = ens.predict_all(X[i : i + chunk])                 # (M, c, n, n)
        out[i : i + chunk] = per_sample_spread(P.std(axis=0, ddof=1))
    return out


# ----------------------------------------------------------------------------
# Selection rules: scores -> indices of the batch
# ----------------------------------------------------------------------------
def select_random(n_pool: int, B: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(n_pool, size=B, replace=False)


def select_top(scores: np.ndarray, B: int) -> np.ndarray:
    """Indices of the B largest scores, largest first."""
    return np.argsort(scores)[::-1][:B]


def select_diverse(scores: np.ndarray, Z: np.ndarray, B: int, factor: int = 10) -> np.ndarray:
    """Spread-then-diversity batch selection.

    1. Shortlist the top `factor * B` candidates by score.
    2. Start the batch with the highest-scoring one.
    3. Repeatedly add the shortlisted candidate whose distance to its NEAREST
       already-chosen point is largest (greedy farthest-point / maximin).

    Z must be in comparable units on every axis (here: the parameter box mapped
    to the unit cube), otherwise the axis with the largest numbers dominates.
    """
    short = select_top(scores, min(factor * B, len(scores)))
    Zs = Z[short]
    chosen = [0]                                            # position in `short`
    d_min = np.linalg.norm(Zs - Zs[0], axis=1)              # distance to nearest chosen
    for _ in range(B - 1):
        nxt = int(np.argmax(d_min))
        chosen.append(nxt)
        d_min = np.minimum(d_min, np.linalg.norm(Zs - Zs[nxt], axis=1))
    return short[np.array(chosen)]


def unit_cube(X: np.ndarray, lows: np.ndarray, highs: np.ndarray) -> np.ndarray:
    """Parameters -> box coordinates in [0, 1] (outside the box: <0 or >1)."""
    return (X - lows) / (highs - lows)


# ----------------------------------------------------------------------------
# Summarising a learning curve
# ----------------------------------------------------------------------------
def fit_power_law(n: np.ndarray, err: np.ndarray) -> tuple[float, float]:
    """Least-squares fit of err = c * n^(-alpha) in log-log space -> (c, alpha)."""
    slope, intercept = np.polyfit(np.log(n), np.log(err), 1)
    return float(np.exp(intercept)), float(-slope)


def equivalent_n(err, c: float, alpha: float):
    """How many samples the fitted curve needs to reach `err` (inverse of the fit)."""
    return (c / np.asarray(err)) ** (1.0 / alpha)


# ----------------------------------------------------------------------------
# The loop
# ----------------------------------------------------------------------------
@dataclass
class ALConfig:
    strategy: str = "random"
    batch: int = 50                 # simulations per round
    budget: int = 400               # total labelled training points at the end
    members: int = 5                # ensemble size
    factor: int = 10                # diverse rule: shortlist size = factor * batch
    seed: int = 0
    train: TrainConfig = field(default_factory=TrainConfig)


def _evaluate(ens: Ensemble, evals: dict) -> dict:
    out = {}
    for name, (X, T) in evals.items():
        mu, sigma = ens.predict(X)
        out[name] = {
            "rel_l2": float(rel_l2(mu, T).mean()),
            "rmse": float(per_sample_rmse(T, mu).mean()),
            "cov90": coverage(T, mu, sigma, levels=(0.9,))[0.9],
        }
    return out


def run_loop(X_lab, T_lab, X_pool, X_val, T_val, evals: dict, cfg: ALConfig,
             lows, highs, n_grid: int = 64, solve=None, verbose=False) -> dict:
    """Run one active-learning experiment; return its round-by-round history.

    X_lab, T_lab : initial labelled set (already solved)
    X_pool       : unlabelled candidates (parameters only)
    X_val, T_val : early stopping + calibration; NEVER acquired, NEVER trained on
    evals        : {"test": (X, T), "ood": (X, T), ...}, scored after every round
    lows, highs  : box used to map parameters to the unit cube (diverse rule)
    solve        : X -> T; defaults to the finite-difference solver
    """
    if cfg.strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}")
    if cfg.budget < len(X_lab):
        raise ValueError("budget is smaller than the initial labelled set")
    solve = solve or (lambda X: solve_batch(X, n_grid))
    rng = np.random.default_rng(cfg.seed)
    member_seeds = [1000 * cfg.seed + j for j in range(cfg.members)]

    X_lab, T_lab = X_lab.copy(), T_lab.copy()
    available = np.ones(len(X_pool), dtype=bool)            # still in the pool?
    history = {"strategy": cfg.strategy, "seed": cfg.seed, "rounds": [], "picks": []}

    while True:
        t0 = time.perf_counter()
        ens = Ensemble.train(X_lab, T_lab, X_val, T_val, cfg.train, member_seeds)
        scale = ens.calibrate(X_val, T_val)
        rec = {"n": len(X_lab), "scale": scale, **_evaluate(ens, evals)}

        B = min(cfg.batch, cfg.budget - len(X_lab))
        if B > 0:
            idx_av = np.flatnonzero(available)
            if cfg.strategy == "random":
                pick = select_random(len(idx_av), B, rng)
                rec["pick_spread"] = None
            else:
                s = pool_spread(ens, X_pool[idx_av])
                if cfg.strategy == "spread":
                    pick = select_top(s, B)
                else:
                    pick = select_diverse(s, unit_cube(X_pool[idx_av], lows, highs), B, cfg.factor)
                rec["pick_spread"] = float(s[pick].mean())
                rec["pool_spread_median"] = float(np.median(s))
            chosen = idx_av[pick]
            available[chosen] = False
            X_new = X_pool[chosen]
            X_lab = np.concatenate([X_lab, X_new])
            T_lab = np.concatenate([T_lab, solve(X_new)])
            history["picks"].append(X_new.tolist())

        rec["seconds"] = time.perf_counter() - t0
        history["rounds"].append(rec)
        if verbose:
            parts = "  ".join(f"{k} {v['rel_l2']:.2%}" for k, v in rec.items()
                              if isinstance(v, dict))
            print(f"  [{cfg.strategy:7s} seed {cfg.seed}] n={rec['n']:4d}  {parts}"
                  f"  s={scale:.2f}  ({rec['seconds']:.0f} s)", flush=True)
        if B <= 0:
            return history
