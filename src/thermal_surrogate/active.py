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
    cluster  : split the WHOLE pool into B clusters (k-means), then take the
               most uncertain candidate in each cluster (representative + uncertain)

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

STRATEGIES = ("random", "spread", "diverse", "cluster")


# ----------------------------------------------------------------------------
# Acquisition: one score per candidate
# ----------------------------------------------------------------------------
def relative_spread(P: np.ndarray) -> np.ndarray:
    """Spread relative to the field's own size, from member predictions (M, N, n, n) -> (N,).

    RMS interior spread / RMS interior predicted temperature. Dimensionless, so a
    plate that is 5x hotter (and therefore has ~5x larger disagreement in deg C)
    doesn't automatically outrank a cooler one the model is equally unsure about.
    """
    return per_sample_spread(P.std(axis=0, ddof=1)) / per_sample_spread(P.mean(axis=0))


def pool_spread(ens: Ensemble, X: np.ndarray, chunk: int = 500, relative: bool = False) -> np.ndarray:
    """Acquisition score for every row of X -> (N,).

    relative=False: RMS interior ensemble spread in deg C (absolute).
    relative=True : that spread divided by the RMS interior predicted field (see relative_spread).

    Computed in chunks: 20,000 candidates x 5 members x 4096 pixels in float64
    would be 3.3 GB at once. The calibration factor is a single constant, so it
    doesn't change the ranking and is left out (raw spread).
    """
    out = np.empty(len(X))
    for i in range(0, len(X), chunk):
        P = ens.predict_all(X[i : i + chunk])                 # (M, c, n, n)
        out[i : i + chunk] = (relative_spread(P) if relative
                              else per_sample_spread(P.std(axis=0, ddof=1)))
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


def select_cluster(scores: np.ndarray, Z: np.ndarray, B: int, rng: np.random.Generator,
                   iters: int = 25) -> np.ndarray:
    """Representative-then-uncertain batch selection.

    1. k-means the whole pool into B clusters (in unit-cube coordinates), so the
       clusters follow where the candidates actually are (the query distribution).
    2. From each cluster take its highest-scoring candidate.

    Uncertainty decides WHICH point inside a region; the clustering decides which
    REGIONS get a point. Top-score shortlists (spread / diverse) let a few extreme
    corners of the box claim the whole batch; this can't, because every cluster
    gets exactly one pick.
    """
    from scipy.cluster.vq import kmeans2
    _, label = kmeans2(Z, B, iter=iters, minit="++", seed=rng)
    picks = []
    for c in range(B):
        members = np.flatnonzero(label == c)
        if len(members):
            picks.append(members[np.argmax(scores[members])])
    picks = np.array(picks)
    if len(picks) < B:                                   # empty clusters: top up by score
        rest = [i for i in select_top(scores, len(scores)) if i not in set(picks.tolist())]
        picks = np.concatenate([picks, np.array(rest[: B - len(picks)], dtype=int)])
    return picks


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
    tol: float = 1.0                # deg C; the CLI defers to the solver above this spread
    score: str = "abs"              # acquisition score: "abs" (deg C) or "rel" (spread / field size)
    seed: int = 0
    train: TrainConfig = field(default_factory=TrainConfig)


def member_seeds_for(cfg: ALConfig) -> list[int]:
    """Seeds of the ensemble members for one run (distinct across runs)."""
    return [1000 * cfg.seed + j for j in range(cfg.members)]


def _evaluate(ens: Ensemble, evals: dict, tol: float) -> dict:
    out = {}
    for name, (X, T) in evals.items():
        mu, sigma = ens.predict(X)                      # sigma is calibrated (scale s)
        out[name] = {
            "rel_l2": float(rel_l2(mu, T).mean()),
            "rmse": float(per_sample_rmse(T, mu).mean()),
            "cov90": coverage(T, mu, sigma, levels=(0.9,))[0.9],
            # share of queries the CLI would send back to the solver (same rule as `predict`)
            "defer": float(np.mean(per_sample_spread(sigma) > tol)),
        }
    return out


def run_loop(X_lab, T_lab, X_pool, X_val, T_val, evals: dict, cfg: ALConfig,
             lows, highs, n_grid: int = 64, solve=None, verbose=False,
             first_ensemble: Ensemble | None = None) -> dict:
    """Run one active-learning experiment; return its round-by-round history.

    X_lab, T_lab : initial labelled set (already solved)
    X_pool       : unlabelled candidates (parameters only)
    X_val, T_val : early stopping + calibration; NEVER acquired, NEVER trained on
    evals        : {"test": (X, T), "ood": (X, T), ...}, scored after every round
    lows, highs  : box used to map parameters to the unit cube (diverse rule)
    solve        : X -> T; defaults to the finite-difference solver
    first_ensemble: optional ensemble already trained on (X_lab, T_lab) with this
                   config's member seeds; reused for round 1 instead of retraining
                   (lets several strategies share an identical, expensive first round)
    """
    if cfg.strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}")
    if cfg.score not in ("abs", "rel"):
        raise ValueError("score must be 'abs' or 'rel'")
    if cfg.budget < len(X_lab):
        raise ValueError("budget is smaller than the initial labelled set")
    solve = solve or (lambda X: solve_batch(X, n_grid))
    rng = np.random.default_rng(cfg.seed)
    member_seeds = member_seeds_for(cfg)

    X_lab, T_lab = X_lab.copy(), T_lab.copy()
    available = np.ones(len(X_pool), dtype=bool)            # still in the pool?
    history = {"strategy": cfg.strategy, "seed": cfg.seed, "rounds": [], "picks": []}

    while True:
        t0 = time.perf_counter()
        if first_ensemble is not None and len(history["rounds"]) == 0:
            ens = first_ensemble
        else:
            ens = Ensemble.train(X_lab, T_lab, X_val, T_val, cfg.train, member_seeds)
        scale = ens.calibrate(X_val, T_val)
        rec = {"n": len(X_lab), "scale": scale, **_evaluate(ens, evals, cfg.tol)}

        B = min(cfg.batch, cfg.budget - len(X_lab))
        if B > 0:
            idx_av = np.flatnonzero(available)
            if cfg.strategy == "random":
                pick = select_random(len(idx_av), B, rng)
                rec["pick_spread"] = None
            else:
                s = pool_spread(ens, X_pool[idx_av], relative=cfg.score == "rel")
                Z = unit_cube(X_pool[idx_av], lows, highs)
                if cfg.strategy == "spread":
                    pick = select_top(s, B)
                elif cfg.strategy == "diverse":
                    pick = select_diverse(s, Z, B, cfg.factor)
                else:
                    pick = select_cluster(s, Z, B, rng)
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
