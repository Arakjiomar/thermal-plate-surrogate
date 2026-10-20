"""Fit the POD + ridge baseline, select hyperparameters on the validation set,
and evaluate on test and OOD.

Usage:  python scripts/fit_pod.py
Writes: results/pod_ridge.json, figures/pod_modes.png, figures/pod_ridge_results.png,
        runs/pod_ridge_{generic,physics}.npz (gitignored)
"""
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from thermal_surrogate.dataset import load
from thermal_surrogate.metrics import rel_l2, summarize
from thermal_surrogate.pod import POD, PODRidge

ROOT = Path(__file__).resolve().parents[1]
d = load(ROOT / "data" / "plate_dataset.npz")
Xtr, Ttr = d["X_train"], d["T_train"]

# ---------------------------------------------------------------- 1. POD
pod = POD().fit(Ttr)
R = pod.n_modes_for(0.9999)
print(f"modes for 99% / 99.9% / 99.99% energy: "
      f"{pod.n_modes_for(0.99)} / {pod.n_modes_for(0.999)} / {R}  -> using R = {R}")

# ---------------------------------------------------------------- 2. model selection on val
DEGREES, ALPHAS = [1, 2, 3, 4], np.logspace(-6, 4, 11)
grid, best = [], {}
for physics in (False, True):
    for deg in DEGREES:
        for alpha in ALPHAS:
            m = PODRidge(R, deg, alpha, physics).fit(Xtr, Ttr, pod)
            err = float(rel_l2(m.predict(d["X_val"]), d["T_val"]).mean())
            grid.append({"physics": physics, "degree": deg, "alpha": float(alpha), "val_rel_l2": err})
            key = "physics" if physics else "generic"
            if key not in best or err < best[key]["val_rel_l2"]:
                best[key] = grid[-1]

# ---------------------------------------------------------------- 3. evaluate the winners
results = {"n_modes": R, "energy_fraction": float(pod.energy_fraction()[R - 1]), "models": {}}
preds = {}
for key, cfg in best.items():
    m = PODRidge(R, cfg["degree"], cfg["alpha"], cfg["physics"]).fit(Xtr, Ttr, pod)
    m.save(ROOT / "runs" / f"pod_ridge_{key}.npz")
    t0 = time.perf_counter()
    P_test = m.predict(d["X_test"])
    ms_per_field = 1e3 * (time.perf_counter() - t0) / len(P_test)
    P_ood = m.predict(d["X_ood"])
    preds[key] = {"test": P_test, "ood": P_ood}
    results["models"][key] = {
        "degree": cfg["degree"], "alpha": cfg["alpha"], "val_rel_l2": cfg["val_rel_l2"],
        "test": summarize(P_test, d["T_test"]), "ood": summarize(P_ood, d["T_ood"]),
        "ms_per_field": ms_per_field}

# "Oracle": the best any coefficient predictor could do with R modes (pure projection).
results["oracle"] = {s: float(rel_l2(pod.reconstruct(pod.project(d[f"T_{s}"], R)), d[f"T_{s}"]).mean())
                     for s in ("test", "ood")}
results["grid"] = grid

out = ROOT / "results" / "pod_ridge.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(results, indent=2))

print(f"\n{'model':8s} {'deg':>3s} {'alpha':>7s} {'val':>7s} {'test':>7s} {'OOD':>7s}  (mean relative L2)")
for key, r in results["models"].items():
    print(f"{key:8s} {r['degree']:3d} {r['alpha']:7.0e} {r['val_rel_l2']:7.4f} "
          f"{r['test']['rel_l2_mean']:7.4f} {r['ood']['rel_l2_mean']:7.4f}")
print(f"{'oracle':8s} {'':3s} {'':7s} {'':7s} {results['oracle']['test']:7.4f} {results['oracle']['ood']:7.4f}")
print(f"prediction cost: {results['models']['physics']['ms_per_field']:.3f} ms per field")
print(f"saved {out}")

# ---------------------------------------------------------------- 4. figures
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "0.2", "0.45"

def tidy(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(alpha=0.2, lw=0.5)

# (a) spectrum + first modes
fig = plt.figure(figsize=(13, 5.2))
gs = fig.add_gridspec(2, 5, width_ratios=[1.6, 0.15, 1, 1, 1], wspace=0.3, hspace=0.35)
ax = fig.add_subplot(gs[:, 0])
r = np.arange(1, 61)
ax.semilogy(r, pod.energy[:60] / pod.energy.sum(), "o-", color=BLUE, ms=3.5, lw=1.5)
ax.axvline(R, color=MUTED, lw=1, ls="--")
ax.text(R + 1, 0.2, f"R = {R}\n99.99% energy", fontsize=9, color=INK)
ax.set_xlabel("mode r"); ax.set_ylabel(r"energy fraction $\lambda_r / \sum\lambda$")
ax.set_title("POD spectrum of 800 training fields", fontsize=10, color=INK)
tidy(ax)
for k in range(6):
    a = fig.add_subplot(gs[k // 3, 2 + k % 3])
    phi = pod.modes[:, k].reshape(pod.shape)
    v = np.abs(phi).max()
    a.imshow(phi, origin="lower", extent=[0, 1, 0, 1], cmap="RdBu_r", vmin=-v, vmax=v)
    a.set_title(f"mode {k+1}  ({100*pod.energy[k]/pod.energy.sum():.1f}%)", fontsize=9, color=INK)
    a.set_xticks([]); a.set_yticks([])
fig.savefig(ROOT / "figures" / "pod_modes.png", dpi=150, bbox_inches="tight")

# (b) example prediction + error vs OOD distance
fig = plt.figure(figsize=(13, 4.4))
gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 0.2, 1.6], wspace=0.3)
e_test = rel_l2(preds["physics"]["test"], d["T_test"])
i = int(np.argsort(e_test)[len(e_test) // 2])                 # median-error test case
true, pred = d["T_test"][i], preds["physics"]["test"][i]
lo, hi = min(true.min(), pred.min()), max(true.max(), pred.max())
for c, (img, title, cmap, lims) in enumerate([
        (true, "solver", "inferno", (lo, hi)),
        (pred, "POD + ridge (physics)", "inferno", (lo, hi)),
        (pred - true, f"error (rel L2 {e_test[i]:.3f})", "RdBu_r", None)]):
    a = fig.add_subplot(gs[0, c])
    if lims is None:
        v = np.abs(img).max(); lims = (-v, v)
    im = a.imshow(img, origin="lower", extent=[0, 1, 0, 1], cmap=cmap, vmin=lims[0], vmax=lims[1])
    a.set_title(title, fontsize=9, color=INK); a.set_xticks([]); a.set_yticks([])
    cb = fig.colorbar(im, ax=a, fraction=0.046, pad=0.03); cb.ax.tick_params(labelsize=7)
ax = fig.add_subplot(gs[0, 4])
for key, color, label in [("generic", ORANGE, "generic features"), ("physics", BLUE, "+ A/k, 1/k")]:
    e_t = rel_l2(preds[key]["test"], d["T_test"]); e_o = rel_l2(preds[key]["ood"], d["T_ood"])
    ax.scatter(np.zeros_like(e_t) + np.random.default_rng(0).uniform(-0.006, 0.006, len(e_t)),
               e_t, s=9, color=color, alpha=0.5, lw=0)
    ax.scatter(d["ood_dist_ood"], e_o, s=14, color=color, alpha=0.8, lw=0, label=label)
ax.set_yscale("log")
ax.set_xlabel(r"OOD distance $d_{OOD}$ (0 = test set)"); ax.set_ylabel("relative L2 error")
ax.set_title("Error grows outside the training box", fontsize=10, color=INK)
ax.legend(frameon=False, fontsize=8, loc="upper left"); tidy(ax)
fig.savefig(ROOT / "figures" / "pod_ridge_results.png", dpi=150, bbox_inches="tight")
print("saved figures/pod_modes.png, figures/pod_ridge_results.png")
