"""Active learning: does letting the ensemble choose the simulations save solves?

Experiment A (inside the training box)
    pool    : 20,000 Latin-hypercube candidates from the training box (parameters only)
    start   : 50 random pool points, solved
    loop    : train a 5-member ensemble -> pick 50 -> solve -> repeat, up to 400 solves
    arms    : random | spread (greedy top-50) | diverse (top-500 by spread, then
              50 far apart), 3 seeds each
    scored  : the fixed test and OOD sets after every round (never acquired)

Headline: fit error = c * n^-alpha to the random arm, then convert each arm's
errors into "equivalent random solves" n_eq; n_eq / n is the saving.

Usage:  python scripts/active_learning.py            (~70 min on 2 cores)
        python scripts/active_learning.py --fast     (3 members, seed 0 only, ~15 min)
        python scripts/active_learning.py --plot     (re-plot from saved runs)
Writes: runs/active/A_*.json (one per run, so an interrupted job resumes),
        results/active_learning.json, figures/active_learning.png
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, NullFormatter

from thermal_surrogate.active import (ALConfig, STRATEGIES, equivalent_n, fit_power_law,
                                      run_loop, unit_cube)
from thermal_surrogate.dataset import latin_hypercube, load, scale, solve_batch
from thermal_surrogate.mlp import TrainConfig

ROOT = Path(__file__).resolve().parents[1]
FAST = "--fast" in sys.argv
RUNS = ROOT / "runs" / ("active_fast" if FAST else "active")
SEEDS = [0] if FAST else [0, 1, 2]
MEMBERS = 3 if FAST else 5
N_POOL, N_INIT, BATCH, BUDGET = 20_000, 50, 50, 400
POOL_SEED = 12345
COLORS = {"random": "#7f7f7f", "spread": "#d62728", "diverse": "#1f77b4"}
LABELS = {"random": "random (baseline)", "spread": "greedy top spread", "diverse": "spread + diversity"}

d = load(ROOT / "data" / "plate_dataset.npz")
lows, highs, names = d["box_lows"], d["box_highs"], [str(s) for s in d["param_names"]]
evals = {"test": (d["X_test"], d["T_test"]), "ood": (d["X_ood"], d["T_ood"])}
best = json.loads((ROOT / "results" / "regularization.json").read_text())["best"]
train_cfg = TrainConfig(width=best["width"], depth=2, weight_decay=best["weight_decay"])


# ---------------------------------------------------------------- 1. run (or resume)
def run_one(strategy: str, seed: int) -> dict:
    path = RUNS / f"A_{strategy}_s{seed}.json"
    if path.exists():
        return json.loads(path.read_text())
    pool = scale(latin_hypercube(N_POOL, 9, np.random.default_rng(POOL_SEED)), lows, highs)
    # Same starting 50 for every arm with this seed: only the selection rule differs.
    init = np.random.default_rng(seed).choice(N_POOL, size=N_INIT, replace=False)
    keep = np.setdiff1d(np.arange(N_POOL), init)
    X0 = pool[init]
    cfg = ALConfig(strategy=strategy, batch=BATCH, budget=BUDGET, members=MEMBERS,
                   seed=seed, train=train_cfg)
    h = run_loop(X0, solve_batch(X0, 64), pool[keep], d["X_val"], d["T_val"], evals, cfg,
                 lows, highs, verbose=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(h))
    return h


if "--plot" not in sys.argv:
    for seed in SEEDS:                          # seed-major: a partial job still has full arm sets
        for strategy in STRATEGIES:
            run_one(strategy, seed)

hist = {s: [json.loads(p.read_text()) for p in sorted(RUNS.glob(f"A_{s}_s*.json"))]
        for s in STRATEGIES}


# ---------------------------------------------------------------- 2. aggregate
def curve(runs, split, metric):
    n = np.array([r["n"] for r in runs[0]["rounds"]])
    v = np.array([[r[split][metric] for r in h["rounds"]] for h in runs])   # (seeds, rounds)
    return n, v.mean(axis=0), (v.std(axis=0, ddof=1) if len(v) > 1 else np.zeros(len(n)))


res = {"config": {"pool": N_POOL, "init": N_INIT, "batch": BATCH, "budget": BUDGET,
                  "members": MEMBERS, "seeds": SEEDS, "fast": FAST},
       "arms": {}}
fits = {}
for split in ("test", "ood"):
    for metric in ("rel_l2", "rmse"):
        n, m, _ = curve(hist["random"], split, metric)
        fits[(split, metric)] = fit_power_law(n, m)

for s in STRATEGIES:
    arm = {"n_runs": len(hist[s])}
    for split in ("test", "ood"):
        for metric in ("rel_l2", "rmse"):
            n, m, sd = curve(hist[s], split, metric)
            c, a = fits[(split, metric)]
            n_eq = equivalent_n(m, c, a)
            arm[f"{split}_{metric}"] = {"n": n.tolist(), "mean": m.tolist(), "std": sd.tolist(),
                                        "n_eq": n_eq.tolist(), "multiplier": (n_eq / n).tolist()}
        _, cov, _ = curve(hist[s], split, "cov90")
        arm[f"{split}_cov90"] = cov.tolist()
    arm["scale"] = np.mean([[r["scale"] for r in h["rounds"]] for h in hist[s]], axis=0).tolist()
    picks = np.concatenate([np.concatenate([np.array(p) for p in h["picks"]]) for h in hist[s]])
    U = unit_cube(picks, lows, highs)
    arm["pick_mean_unit"] = dict(zip(names, U.mean(axis=0).round(3).tolist()))
    # how close to the box boundary the picks are: distance of the nearest face, in range widths
    arm["pick_face_distance_median"] = float(np.median(np.minimum(U, 1 - U).min(axis=1)))
    res["arms"][s] = arm
res["random_fit"] = {f"{k[0]}_{k[1]}": {"c": v[0], "alpha": v[1]} for k, v in fits.items()}

# Reference: how close a random 9-D point is to its nearest face (for the pick-location claim)
U_ref = latin_hypercube(20_000, 9, np.random.default_rng(1))
res["reference_face_distance_median"] = float(np.median(np.minimum(U_ref, 1 - U_ref).min(axis=1)))

out = ROOT / "results" / ("active_learning_fast.json" if FAST else "active_learning.json")
out.write_text(json.dumps(res, indent=2))

print(f"\nrandom-arm fit, test rel L2: error = {fits[('test', 'rel_l2')][0]:.3f} * n^-"
      f"{fits[('test', 'rel_l2')][1]:.2f}")
print(f"{'arm':10s} {'n':>4s} {'test relL2':>11s} {'test RMSE':>10s} {'OOD relL2':>10s}"
      f" {'x (relL2)':>9s} {'x (RMSE)':>9s}")
for s in STRATEGIES:
    a = res["arms"][s]
    for i in (3, 7):                                   # n = 200 and n = 400
        print(f"{s:10s} {a['test_rel_l2']['n'][i]:4d} {a['test_rel_l2']['mean'][i]:10.2%}"
              f" {a['test_rmse']['mean'][i]:9.2f}° {a['ood_rel_l2']['mean'][i]:9.2%}"
              f" {a['test_rel_l2']['multiplier'][i]:8.2f}x {a['test_rmse']['multiplier'][i]:8.2f}x")


# ---------------------------------------------------------------- 3. figure
fig, ax = plt.subplots(2, 2, figsize=(12, 9))
panels = [(ax[0, 0], "test", "rel_l2", "test relative L2 error"),
          (ax[0, 1], "test", "rmse", "test RMSE (°C)"),
          (ax[1, 0], "ood", "rel_l2", "OOD relative L2 error")]
for a_, split, metric, title in panels:
    for s in STRATEGIES:
        n, m, sd = curve(hist[s], split, metric)
        a_.errorbar(n, m, yerr=sd, marker="o", ms=4, capsize=3, color=COLORS[s], label=LABELS[s])
    c, al = fits[(split, metric)]
    nn = np.linspace(50, 400, 100)
    a_.plot(nn, c * nn ** -al, ":", color=COLORS["random"], lw=1,
            label=f"random fit  ∝ n^-{al:.2f}")
    a_.set_xscale("log"); a_.set_yscale("log")
    a_.set_xticks([50, 100, 200, 400]); a_.set_xticklabels(["50", "100", "200", "400"])
    a_.xaxis.set_minor_formatter(NullFormatter())
    fmt = (lambda v, _: f"{v:.0%}" if v >= 0.1 else f"{v:.1%}") if metric == "rel_l2" else \
          (lambda v, _: f"{v:g}°")
    a_.yaxis.set_major_formatter(FuncFormatter(fmt))
    a_.yaxis.set_minor_formatter(FuncFormatter(fmt))
    a_.tick_params(axis="y", which="minor", labelsize=7)
    a_.set_xlabel("simulations used (training-set size)"); a_.set_title(title)
    a_.grid(alpha=0.3, which="both"); a_.legend(fontsize=8)

# where the picks land: mean unit-cube coordinate per parameter
a_ = ax[1, 1]
xpos = np.arange(9)
for j, s in enumerate(STRATEGIES):
    a_.bar(xpos + (j - 1) * 0.27, list(res["arms"][s]["pick_mean_unit"].values()), 0.27,
           color=COLORS[s], label=LABELS[s])
a_.axhline(0.5, color="k", lw=0.8, ls="--")
a_.set_xticks(xpos); a_.set_xticklabels(names, rotation=30)
a_.set_ylim(0, 1); a_.set_ylabel("mean position of picks in the box (0 = low end, 1 = high end)")
a_.set_title("where each rule sends the simulations"); a_.legend(fontsize=8)
fig.suptitle(f"Active learning: {N_INIT} → {BUDGET} solves, batches of {BATCH}, "
             f"{len(SEEDS)} seed(s), {MEMBERS}-member ensemble", fontsize=12)
fig.tight_layout()
figpath = ROOT / "figures" / ("active_learning_fast.png" if FAST else "active_learning.png")
fig.savefig(figpath, dpi=130)
print(f"wrote {out.relative_to(ROOT)} and {figpath.relative_to(ROOT)}")
