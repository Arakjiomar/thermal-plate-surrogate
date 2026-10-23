"""Deep-ensemble uncertainty for the POD + MLP surrogate.

Trains 5 members (best milestone-6 config, seeds 0-4), calibrates the spread on
the validation set, and checks whether the uncertainty is trustworthy:
  * accuracy: ensemble mean vs. single members
  * ranking: does a larger spread mean a larger error? (Spearman correlation)
  * calibration: do 90% intervals contain 90% of the truth? (in- and out-of-distribution)
  * OOD detection: can the spread alone tell OOD inputs from test inputs? (AUROC)
  * selective prediction: error when the most uncertain cases are sent back to the solver

Usage:  python scripts/train_ensemble.py           (~5 min)
        python scripts/train_ensemble.py --reuse   (skip training, load runs/ensemble/)
Writes: results/ensemble.json, figures/ensemble_uq.png, runs/ensemble/ (gitignored)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from thermal_surrogate.dataset import load
from thermal_surrogate.metrics import rel_l2
from thermal_surrogate.mlp import TrainConfig
from thermal_surrogate.uq import (Ensemble, auroc, coverage, per_sample_rmse,
                                  per_sample_spread)

ROOT = Path(__file__).resolve().parents[1]
d = load(ROOT / "data" / "plate_dataset.npz")
FOLDER = ROOT / "runs" / "ensemble"
SEEDS = [0, 1, 2, 3, 4]
LEVELS = (0.5, 0.8, 0.9, 0.95, 0.99)

# ---------------------------------------------------------------- 1. train (or reload) and calibrate
if "--reuse" in sys.argv and (FOLDER / "ensemble.json").exists():
    ens = Ensemble.load(FOLDER)
else:
    best = json.loads((ROOT / "results" / "regularization.json").read_text())["best"]
    cfg = TrainConfig(width=best["width"], depth=2, weight_decay=best["weight_decay"])
    print(f"training {len(SEEDS)} members: width {cfg.width}, weight decay {cfg.weight_decay}")
    ens = Ensemble.train(d["X_train"], d["T_train"], d["X_val"], d["T_val"], cfg, SEEDS, verbose=True)
ens.scale = 1.0                                              # raw (uncalibrated) spread first
raw_cov = {s: coverage(d[f"T_{s}"], *ens.predict(d[f"X_{s}"])) for s in ("test", "ood")}
scale = ens.calibrate(d["X_val"], d["T_val"])
ens.save(FOLDER)
print(f"calibration scale s = {scale:.2f}  (raw spread is {scale:.1f}x too small)")

# ---------------------------------------------------------------- 2. evaluate
res = {"n_members": len(ens.members), "scale": scale, "splits": {}}
cache = {}
for split in ("test", "ood"):
    X, T = d[f"X_{split}"], d[f"T_{split}"]
    P = ens.predict_all(X)
    mu, sigma = ens.predict(X)
    err = per_sample_rmse(T, mu)
    spread = per_sample_spread(sigma)
    rho = spearmanr(spread, err).statistic
    cache[split] = dict(err=err, spread=spread, mu=mu, sigma=sigma, T=T)
    res["splits"][split] = {
        "ensemble_rel_l2": float(rel_l2(mu, T).mean()),
        "member_rel_l2": [float(rel_l2(p, T).mean()) for p in P],
        "spearman_spread_vs_error": float(rho),
        "coverage_raw": {str(k): v for k, v in raw_cov[split].items()},
        "coverage_calibrated": {str(k): v for k, v in coverage(T, mu, sigma, LEVELS).items()},
        "mean_spread_deg": float(spread.mean()),
        "mean_rmse_deg": float(err.mean()),
    }
    r = res["splits"][split]
    print(f"{split:4s}: ensemble rel L2 {r['ensemble_rel_l2']*100:.2f}%  "
          f"(members {min(r['member_rel_l2'])*100:.2f}-{max(r['member_rel_l2'])*100:.2f}%)  "
          f"Spearman {rho:.2f}  90% coverage raw {r['coverage_raw']['0.9']:.2f} "
          f"-> calibrated {r['coverage_calibrated']['0.9']:.2f}")

# OOD detection from the spread alone: test = negative, OOD = positive.
res["ood_auroc"] = auroc(cache["test"]["spread"], cache["ood"]["spread"])
print(f"OOD detection AUROC from spread: {res['ood_auroc']:.2f}")

# Selective prediction on the pooled test + OOD queries: hand the most uncertain
# fraction back to the solver (error 0) and measure the error that remains.
err_all = np.concatenate([cache[s]["err"] for s in ("test", "ood")])
spr_all = np.concatenate([cache[s]["spread"] for s in ("test", "ood")])
fractions = np.linspace(0, 0.5, 11)
def remaining_error(order):
    out = []
    for f in fractions:
        keep = order[: len(order) - int(round(f * len(order)))]
        e = err_all.copy(); e[np.setdiff1d(np.arange(len(e)), keep)] = 0.0
        out.append(float(e.mean()))
    return out
res["selective"] = {
    "fraction_to_solver": fractions.tolist(),
    "by_spread": remaining_error(np.argsort(spr_all)),        # defer the most uncertain
    "oracle": remaining_error(np.argsort(err_all)),           # defer the truly worst
    "random": remaining_error(np.random.default_rng(0).permutation(len(err_all))),
}
i20 = int(np.argmin(np.abs(fractions - 0.2)))
print(f"deferring the 20% most uncertain cuts mean RMSE from {res['selective']['by_spread'][0]:.2f} "
      f"to {res['selective']['by_spread'][i20]:.2f} deg (oracle {res['selective']['oracle'][i20]:.2f}, "
      f"random {res['selective']['random'][i20]:.2f})")
(ROOT / "results" / "ensemble.json").write_text(json.dumps(res, indent=2))

# ---------------------------------------------------------------- 3. figure
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "0.2", "0.45"

def tidy(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(alpha=0.2, lw=0.5)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# (a) spread vs. error
ax = axes[0, 0]
for split, color, label in [("test", BLUE, "test"), ("ood", ORANGE, "OOD")]:
    c = cache[split]
    ax.scatter(c["spread"], c["err"], s=14, color=color, alpha=0.75, lw=0,
               label=f"{label} (Spearman {res['splits'][split]['spearman_spread_vs_error']:.2f})")
lim = [min(err_all.min(), spr_all.min()) * 0.8, max(err_all.max(), spr_all.max()) * 1.2]
ax.plot(lim, lim, "--", color=MUTED, lw=1, label="error = spread")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("predicted uncertainty: calibrated spread (°C RMS)")
ax.set_ylabel("actual error: RMSE of ensemble mean (°C)")
ax.set_title("Does a larger spread mean a larger error?", fontsize=10, color=INK)
ax.legend(frameon=False, fontsize=8.5); tidy(ax)

# (b) spread vs. OOD distance
ax = axes[0, 1]
ax.scatter(np.random.default_rng(0).uniform(-0.006, 0.006, len(cache["test"]["spread"])),
           cache["test"]["spread"], s=12, color=BLUE, alpha=0.6, lw=0, label="test")
ax.scatter(d["ood_dist_ood"], cache["ood"]["spread"], s=14, color=ORANGE, alpha=0.8, lw=0, label="OOD")
ax.set_yscale("log"); ax.set_xlabel(r"OOD distance $d_{OOD}$")
ax.set_ylabel("calibrated spread (°C RMS)")
ax.set_title(f"Uncertainty grows outside the training box (AUROC {res['ood_auroc']:.2f})",
             fontsize=10, color=INK)
ax.legend(frameon=False, fontsize=8.5); tidy(ax)

# (c) calibration curve
ax = axes[0, 2]
ax.plot([0, 1], [0, 1], "--", color=MUTED, lw=1, label="perfect calibration")
for split, color in [("test", BLUE), ("ood", ORANGE)]:
    r = res["splits"][split]
    ax.plot(LEVELS, [r["coverage_raw"][str(p)] for p in LEVELS], ":o", color=color, ms=4, lw=1.2,
            alpha=0.6, label=f"{split}, raw spread")
    ax.plot(LEVELS, [r["coverage_calibrated"][str(p)] for p in LEVELS], "-o", color=color, ms=5,
            lw=1.8, label=f"{split}, calibrated on val")
ax.set_xlim(0.45, 1.0); ax.set_ylim(0, 1.0)
ax.set_xlabel("nominal interval probability"); ax.set_ylabel("fraction of true pixels inside")
ax.set_title("Calibration of mu ± k sigma intervals", fontsize=10, color=INK)
ax.legend(frameon=False, fontsize=8, loc="upper left"); tidy(ax)

# (d) selective prediction
ax = axes[1, 0]
s = res["selective"]
for key, color, ls, label in [("random", MUTED, ":", "defer at random"),
                              ("by_spread", BLUE, "-", "defer most uncertain"),
                              ("oracle", "0.1", "--", "defer truly worst (oracle)")]:
    ax.plot(np.array(s["fraction_to_solver"]) * 100, s[key], ls, color=color, lw=1.8, label=label)
ax.set_xlabel("% of queries sent back to the solver (test + OOD pooled)")
ax.set_ylabel("mean RMSE over all queries (°C)")
ax.set_title("Using the uncertainty: defer the risky cases", fontsize=10, color=INK)
ax.legend(frameon=False, fontsize=8.5); tidy(ax)

# (e, f) one OOD case: error map vs. predicted spread
i = int(np.argsort(cache["ood"]["err"])[int(0.9 * len(cache["ood"]["err"]))])
c = cache["ood"]
err_map, sig_map = c["mu"][i] - c["T"][i], c["sigma"][i]
v = max(np.abs(err_map).max(), sig_map.max())
for ax, img, title, cmap, lims in [
        (axes[1, 1], err_map, f"OOD case: actual error (RMSE {c['err'][i]:.1f} °C)", "RdBu_r", (-v, v)),
        (axes[1, 2], sig_map, f"predicted spread (RMS {c['spread'][i]:.1f} °C)", "magma", (0, v))]:
    im = ax.imshow(img, origin="lower", extent=[0, 1, 0, 1], cmap=cmap, vmin=lims[0], vmax=lims[1])
    ax.set_title(title, fontsize=10, color=INK); ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="°C")
fig.tight_layout()
fig.savefig(ROOT / "figures" / "ensemble_uq.png", dpi=150, bbox_inches="tight")
print("saved results/ensemble.json, figures/ensemble_uq.png")
