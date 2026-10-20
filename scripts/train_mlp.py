"""Train the neural field surrogates and compare them with the POD + ridge baseline.

Usage:  python scripts/train_mlp.py
Writes: results/mlp.json, figures/mlp_results.png, runs/mlp_{pod,direct}.pt (gitignored)
Needs:  results/pod_ridge.json (run scripts/fit_pod.py first)
"""
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from thermal_surrogate.dataset import load
from thermal_surrogate.metrics import rel_l2, summarize
from thermal_surrogate.mlp import FieldSurrogate, TrainConfig
from thermal_surrogate.pod import PODRidge

torch.set_num_threads(max(1, torch.get_num_threads()))
ROOT = Path(__file__).resolve().parents[1]
d = load(ROOT / "data" / "plate_dataset.npz")
args = (d["X_train"], d["T_train"], d["X_val"], d["T_val"])

# ---------------------------------------------------------------- 1. small model selection on val
CANDIDATES = {
    "pod":    [dict(width=w, depth=k) for w in (128, 256) for k in (2, 3, 4)],
    "direct": [dict(width=w, depth=3) for w in (256, 512)],
}
grid, best = [], {}
for kind, cands in CANDIDATES.items():
    for c in cands:
        t0 = time.perf_counter()
        m = FieldSurrogate(kind, physics=True).fit(*args, TrainConfig(**c))
        err = float(rel_l2(m.predict(d["X_val"]), d["T_val"]).mean())
        row = {"kind": kind, **c, "val_rel_l2": err, "best_epoch": m.best_epoch,
               "train_seconds": round(time.perf_counter() - t0, 1)}
        grid.append(row)
        print(f"{kind:6s} width {c['width']:3d} depth {c['depth']}  val rel L2 {err:.4f}  "
              f"(best epoch {m.best_epoch}, {row['train_seconds']} s)")
        if kind not in best or err < best[kind][0]:
            best[kind] = (err, m, c)

# Does the network still need the physics features? Same architecture, raw inputs.
c = best["pod"][2]
m_raw = FieldSurrogate("pod", physics=False).fit(*args, TrainConfig(**c))
best["pod_raw"] = (float(rel_l2(m_raw.predict(d["X_val"]), d["T_val"]).mean()), m_raw, c)

# ---------------------------------------------------------------- 2. evaluate on test and OOD
baseline = PODRidge.load(ROOT / "runs" / "pod_ridge_physics.npz")
models = {"pod_ridge": baseline, "pod_mlp": best["pod"][1],
          "pod_mlp_raw_inputs": best["pod_raw"][1], "direct_mlp": best["direct"][1]}
results, preds = {"grid": grid, "models": {}}, {}
for name, m in models.items():
    t0 = time.perf_counter()
    P_test = m.predict(d["X_test"])
    ms = 1e3 * (time.perf_counter() - t0) / len(P_test)
    P_ood = m.predict(d["X_ood"])
    preds[name] = (P_test, P_ood)
    entry = {"test": summarize(P_test, d["T_test"]), "ood": summarize(P_ood, d["T_ood"]),
             "ms_per_field": ms}
    if name != "pod_ridge":
        entry["config"] = {**m.cfg.__dict__, "kind": m.kind, "physics": m.physics}
        entry["best_epoch"] = m.best_epoch
        m.save(ROOT / "runs" / f"mlp_{name}.pt")
    results["models"][name] = entry

out = ROOT / "results" / "mlp.json"
out.write_text(json.dumps(results, indent=2))
print(f"\n{'model':20s} {'test relL2':>10s} {'test RMSE':>9s} {'peak err':>8s} {'OOD relL2':>9s}")
for name, r in results["models"].items():
    t = r["test"]
    print(f"{name:20s} {t['rel_l2_mean']:10.4f} {t['rmse_mean']:9.2f} {t['max_t_err_mean']:8.2f} "
          f"{r['ood']['rel_l2_mean']:9.4f}")
print(f"saved {out}")

# ---------------------------------------------------------------- 3. figure
BLUE, ORANGE, AQUA, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "0.2", "0.45"
STYLE = {"pod_ridge": (ORANGE, "POD + ridge"), "pod_mlp": (BLUE, "POD + MLP"),
         "direct_mlp": (AQUA, "direct MLP")}

def tidy(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(alpha=0.2, lw=0.5)

fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.05], hspace=0.38, wspace=0.32)

# (a) learning curves
ax = fig.add_subplot(gs[0, :2])
for name in ("pod_mlp", "direct_mlp"):
    m, (color, label) = models[name], STYLE[name]
    ax.semilogy(m.history["epoch"], m.history["train"], color=color, lw=1, alpha=0.6, ls="--")
    ax.semilogy(m.history["epoch"], m.history["val"], color=color, lw=1.8, label=f"{label} (val)")
    ax.axvline(m.best_epoch, color=color, lw=0.8, ls=":")
ax.set_xlabel("epoch"); ax.set_ylabel("MSE (scaled targets)")
ax.set_title("Learning curves: dashed = train, solid = validation, dotted = kept epoch",
             fontsize=10, color=INK)
ax.legend(frameon=False, fontsize=9); tidy(ax)

# (b) error vs OOD distance
ax = fig.add_subplot(gs[0, 2:])
jit = np.random.default_rng(0).uniform(-0.006, 0.006, len(d["X_test"]))
for name, (color, label) in STYLE.items():
    P_test, P_ood = preds[name]
    ax.scatter(jit, rel_l2(P_test, d["T_test"]), s=8, color=color, alpha=0.45, lw=0)
    ax.scatter(d["ood_dist_ood"], rel_l2(P_ood, d["T_ood"]), s=14, color=color, alpha=0.8,
               lw=0, label=label)
ax.set_yscale("log"); ax.set_xlabel(r"OOD distance $d_{OOD}$ (0 = test set)")
ax.set_ylabel("relative L2 error"); ax.set_title("Error vs. distance from training data",
                                                  fontsize=10, color=INK)
ax.legend(frameon=False, fontsize=9, loc="upper left"); tidy(ax)

# (c) one hard test case: truth and the three error maps on a shared scale
e = rel_l2(preds["pod_ridge"][0], d["T_test"])
i = int(np.argsort(e)[int(0.9 * len(e))])                     # 90th-percentile case for ridge
true = d["T_test"][i]
errs = {n: preds[n][0][i] - true for n in STYLE}
v = max(np.abs(x).max() for x in errs.values())
a = fig.add_subplot(gs[1, 0])
im = a.imshow(true, origin="lower", extent=[0, 1, 0, 1], cmap="inferno")
a.set_title("solver (a hard test case)", fontsize=9, color=INK); a.set_xticks([]); a.set_yticks([])
fig.colorbar(im, ax=a, fraction=0.046, pad=0.03).ax.tick_params(labelsize=7)
for c, (n, (color, label)) in enumerate(STYLE.items()):
    a = fig.add_subplot(gs[1, c + 1])
    im = a.imshow(errs[n], origin="lower", extent=[0, 1, 0, 1], cmap="RdBu_r", vmin=-v, vmax=v)
    a.set_title(f"{label} error (rel L2 {rel_l2(preds[n][0][i:i+1], true[None])[0]:.3f})",
                fontsize=9, color=INK)
    a.set_xticks([]); a.set_yticks([])
    fig.colorbar(im, ax=a, fraction=0.046, pad=0.03).ax.tick_params(labelsize=7)
fig.savefig(ROOT / "figures" / "mlp_results.png", dpi=150, bbox_inches="tight")
print("saved figures/mlp_results.png")
