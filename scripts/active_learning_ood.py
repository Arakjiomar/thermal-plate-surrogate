"""Active learning, experiment B: extend the surrogate into the OOD region.

The shipped model is trained on 800 plates inside the box and is weak just
outside it (2.7% vs 0.55% rel. L2). Here we spend 200 extra simulations and ask
where they should go:

    inbox_random : 200 more random plates INSIDE the original box   ("just get more data")
    wide_random  : 200 random plates from the widened box            (cover the new region evenly)
    wide_diverse : 200 plates from the widened box chosen by ensemble
                   spread (deg C) + diversity                        (let the model choose)
    wide_diverse_rel : same, but scored by RELATIVE spread (spread / field size), added
                   after wide_diverse chased the hottest plates (low-k corner)
    wide_cluster : k-means the widened pool into 100 clusters, take the most uncertain
                   plate in each (representative + uncertain), added after both
                   top-spread rules turned out to pick extreme corner plates

Start: the 800 training plates.  Batches of 100, so trainings at n = 800, 900, 1000.
Scored after every round on the fixed test (in-box) and OOD sets, including the
share of queries the CLI would defer to the solver at its 1 C tolerance.

Usage:  python scripts/active_learning_ood.py                 (~80 min on 2 cores)
        python scripts/active_learning_ood.py --seeds 0 2     (run a subset, e.g. in parallel)
        python scripts/active_learning_ood.py --seeds 2 --arms wide_cluster   (one arm only)
        python scripts/active_learning_ood.py --fast          (3 members, seed 0, ~15 min)
        python scripts/active_learning_ood.py --plot          (re-plot from saved runs)
Writes: runs/active/B_*.json, results/active_learning_ood.json, figures/active_learning_ood.png
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from thermal_surrogate.active import ALConfig, member_seeds_for, run_loop, unit_cube
from thermal_surrogate.dataset import (DatasetConfig, latin_hypercube, load, ood_distance,
                                       scale, solve_batch, widened_box)
from thermal_surrogate.pod import POD
from thermal_surrogate.mlp import TrainConfig
from thermal_surrogate.uq import Ensemble

ROOT = Path(__file__).resolve().parents[1]
FAST = "--fast" in sys.argv
RUNS = ROOT / "runs" / ("active_fast" if FAST else "active")
if "--seeds" in sys.argv:
    SEEDS = []
    for a in sys.argv[sys.argv.index("--seeds") + 1:]:
        if not a.isdigit():
            break
        SEEDS.append(int(a))
else:
    SEEDS = [0] if FAST else [0, 1, 2]
MEMBERS = 3 if FAST else 5
N_POOL, BATCH, EXTRA = 20_000, 100, 200
POOL_SEED = 54321
ARMS = ("inbox_random", "wide_random", "wide_diverse", "wide_diverse_rel", "wide_cluster")
COLORS = {"inbox_random": "#bcbd22", "wide_random": "#7f7f7f", "wide_diverse": "#1f77b4",
          "wide_diverse_rel": "#2ca02c", "wide_cluster": "#9467bd"}
LABELS = {"inbox_random": "+200 random, inside box", "wide_random": "+200 random, widened box",
          "wide_diverse": "+200 spread (°C) + diversity, widened box",
          "wide_diverse_rel": "+200 RELATIVE spread + diversity, widened box",
          "wide_cluster": "+200 most uncertain per k-means cluster, widened box"}

cfg_data = DatasetConfig.from_yaml(ROOT / "configs" / "default.yaml")
d = load(ROOT / "data" / "plate_dataset.npz")
lows, highs = d["box_lows"], d["box_highs"]
wlo, whi = widened_box(cfg_data)
names = [str(s) for s in d["param_names"]]
evals = {"test": (d["X_test"], d["T_test"]), "ood": (d["X_ood"], d["T_ood"])}
best = json.loads((ROOT / "results" / "regularization.json").read_text())["best"]
train_cfg = TrainConfig(width=best["width"], depth=2, weight_decay=best["weight_decay"])


def pool_for(arm: str) -> np.ndarray:
    rng = np.random.default_rng(POOL_SEED)
    if arm == "inbox_random":
        return scale(latin_hypercube(N_POOL, 9, rng), lows, highs)
    return scale(latin_hypercube(N_POOL, 9, rng), wlo, whi)


# ---------------------------------------------------------------- 1. run (or resume)
_first = {}                                       # seed -> ensemble trained on the 800 plates


def first_round(cfg: ALConfig) -> Ensemble:
    """Round 1 is identical for every arm (same 800 plates, same member seeds): train it once."""
    if cfg.seed not in _first:
        _first[cfg.seed] = Ensemble.train(d["X_train"], d["T_train"], d["X_val"], d["T_val"],
                                          cfg.train, member_seeds_for(cfg))
    return _first[cfg.seed]


def run_one(arm: str, seed: int) -> dict:
    path = RUNS / f"B_{arm}_s{seed}.json"
    if path.exists():
        return json.loads(path.read_text())
    strategy = ("diverse" if arm.startswith("wide_diverse") else
                "cluster" if arm == "wide_cluster" else "random")
    cfg = ALConfig(strategy=strategy,
                   score="rel" if arm == "wide_diverse_rel" else "abs",
                   batch=BATCH, budget=len(d["X_train"]) + EXTRA, members=MEMBERS,
                   seed=seed, train=train_cfg)
    # Diversity distances are measured in the WIDENED box's unit cube: that's the region searched.
    h = run_loop(d["X_train"], d["T_train"], pool_for(arm), d["X_val"], d["T_val"], evals, cfg,
                 wlo, whi, verbose=True, first_ensemble=first_round(cfg))
    h["arm"] = arm
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(h))
    return h


if "--plot" not in sys.argv:
    only = sys.argv[sys.argv.index("--arms") + 1].split(",") if "--arms" in sys.argv else ARMS
    for seed in SEEDS:
        for arm in ARMS:
            if arm in only:
                run_one(arm, seed)
    if "--seeds" in sys.argv:
        sys.exit(0)                               # partial job: aggregate later with --plot

hist = {a: [json.loads(p.read_text()) for p in sorted(RUNS.glob(f"B_{a}_s*.json"))] for a in ARMS}


# ---------------------------------------------------------------- 2. aggregate
def curve(runs, split, metric):
    n = np.array([r["n"] for r in runs[0]["rounds"]])
    v = np.array([[r[split][metric] for r in h["rounds"]] for h in runs])
    return n, v.mean(axis=0), (v.std(axis=0, ddof=1) if len(v) > 1 else np.zeros(len(n)))


res = {"config": {"start": len(d["X_train"]), "extra": EXTRA, "batch": BATCH, "pool": N_POOL,
                  "members": MEMBERS, "seeds": SEEDS, "fast": FAST,
                  "widened_box": {"low": wlo.tolist(), "high": whi.tolist()}},
       "arms": {}}
for a in ARMS:
    arm = {"n_runs": len(hist[a])}
    for split in ("test", "ood"):
        for metric in ("rel_l2", "rmse", "cov90", "defer"):
            n, m, sd = curve(hist[a], split, metric)
            arm[f"{split}_{metric}"] = {"n": n.tolist(), "mean": m.tolist(), "std": sd.tolist()}
    picks = np.concatenate([np.concatenate([np.array(p) for p in h["picks"]]) for h in hist[a]])
    dist = ood_distance(picks, lows, highs)
    arm["picks_outside_box"] = float(np.mean(dist > 0))
    arm["picks_ood_distance_median"] = float(np.median(dist))
    arm["pick_mean_unit_widened"] = dict(zip(names, unit_cube(picks, wlo, whi).mean(0).round(3).tolist()))
    # How extreme is each pick? Number of the 9 parameters outside the training box.
    k_out = ((picks < lows) | (picks > highs)).sum(axis=1)
    arm["params_outside_hist"] = (np.bincount(k_out, minlength=10) / len(k_out)).tolist()
    # Global target scale the network trains on (POD-coefficient std), first seed's final set.
    # A few extreme plates inflate it, and every other plate's precision is measured in its units.
    P0 = np.concatenate([np.array(p) for p in hist[a][0]["picks"]])
    T0 = np.concatenate([d["T_train"], solve_batch(P0, 64)])
    arm["target_scale_final"] = float(POD().fit(T0).project(T0, 17).std())
    arm["picks_peak_T_max"] = float(T0[len(d["T_train"]):].max())
    res["arms"][a] = arm
# For reference: how much of the widened box lies outside the training box.
U = scale(latin_hypercube(20_000, 9, np.random.default_rng(1)), wlo, whi)
res["widened_box_fraction_outside"] = float(np.mean(ood_distance(U, lows, highs) > 0))
res["ood_set_distance_median"] = float(np.median(d["ood_dist_ood"]))
k_ood = ((d["X_ood"] < lows) | (d["X_ood"] > highs)).sum(axis=1)
res["ood_set_params_outside_hist"] = (np.bincount(k_ood, minlength=10) / len(k_ood)).tolist()
res["target_scale_800"] = float(POD().fit(d["T_train"]).project(d["T_train"], 17).std())

out = ROOT / "results" / ("active_learning_ood_fast.json" if FAST else "active_learning_ood.json")
out.write_text(json.dumps(res, indent=2))

print(f"\n{'arm':14s} {'n':>5s} {'OOD relL2':>10s} {'OOD RMSE':>9s} {'OOD defer':>9s}"
      f" {'test relL2':>10s} {'test defer':>10s} {'outside':>8s}")
for a in ARMS:
    r = res["arms"][a]
    for i in range(len(r["ood_rel_l2"]["n"])):
        print(f"{a:14s} {r['ood_rel_l2']['n'][i]:5d} {r['ood_rel_l2']['mean'][i]:9.2%}"
              f" {r['ood_rmse']['mean'][i]:8.2f}° {r['ood_defer']['mean'][i]:8.0%}"
              f" {r['test_rel_l2']['mean'][i]:9.2%} {r['test_defer']['mean'][i]:9.0%}"
              f" {r['picks_outside_box']:7.0%}")


# ---------------------------------------------------------------- 3. figure
fig, ax = plt.subplots(2, 2, figsize=(12, 9))
panels = [(ax[0, 0], "ood", "rel_l2", "OOD relative L2 error", "{:.1%}"),
          (ax[0, 1], "ood", "defer", "OOD queries the CLI defers to the solver (1 °C tolerance)", "{:.0%}"),
          (ax[1, 0], "test", "rel_l2", "in-box test relative L2 error (must not get worse)", "{:.2%}")]
for a_, split, metric, title, fmt in panels:
    for a in ARMS:
        n, m, sd = curve(hist[a], split, metric)
        a_.errorbar(n, m, yerr=sd, marker="o", ms=5, capsize=3, color=COLORS[a], label=LABELS[a])
    a_.set_xticks(n); a_.set_xlabel("simulations used (training-set size)")
    a_.yaxis.set_major_formatter(FuncFormatter(lambda v, _, f=fmt: f.format(v)))
    a_.set_title(title, fontsize=10); a_.grid(alpha=0.3); a_.legend(fontsize=7, loc="best")

a_ = ax[1, 1]
kk = np.arange(10)
a_.plot(kk, res["ood_set_params_outside_hist"], "k--", marker="s", ms=4, lw=1.5,
        label="OOD test set (what the model will be asked)")
for a in ARMS[1:]:
    r = res["arms"][a]
    a_.plot(kk, r["params_outside_hist"], marker="o", ms=4, color=COLORS[a],
            label=f"{LABELS[a].replace('+200 ', '').replace(', widened box', '')}"
                  f"  (target scale {r['target_scale_final']:.0f})")
a_.set_xticks(kk); a_.set_xlim(-0.3, 9.3)
a_.set_xlabel("number of the 9 parameters outside the training box (per plate)")
a_.set_ylabel("fraction of plates")
a_.set_title(f"how extreme the extra plates are  (training-only target scale "
             f"{res['target_scale_800']:.0f})", fontsize=10)
a_.grid(alpha=0.3); a_.legend(fontsize=7)
fig.suptitle(f"Extending the surrogate: 800 in-box plates + {EXTRA} more, batches of {BATCH}, "
             f"{len(SEEDS)} seed(s), {MEMBERS}-member ensemble", fontsize=12)
fig.tight_layout()
figpath = ROOT / "figures" / ("active_learning_ood_fast.png" if FAST else "active_learning_ood.png")
fig.savefig(figpath, dpi=130)
print(f"wrote {out.relative_to(ROOT)} and {figpath.relative_to(ROOT)}")
