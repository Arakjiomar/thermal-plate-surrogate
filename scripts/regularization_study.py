"""Regularization and data-size study for the POD + MLP surrogate, over 5 seeds.

Part A: weight decay {0, 1e-3, 1e-2, 1e-1} x width {64, 256}  (depth 2, all 800 samples)
Part B: training-set size {100, 200, 400, 800}                  (best Part-A config)

Usage:  python scripts/regularization_study.py              (~20-40 min on a laptop CPU)
        python scripts/regularization_study.py --plot-only  (redraw the figure from the JSON)
Writes: results/regularization.json, figures/regularization.png
"""
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from thermal_surrogate.dataset import load
from thermal_surrogate.mlp import TrainConfig
from thermal_surrogate.study import run_once, seed_stats, subsample

ROOT = Path(__file__).resolve().parents[1]
d = load(ROOT / "data" / "plate_dataset.npz")
held_out = (d["X_val"], d["T_val"], d["X_test"], d["T_test"], d["X_ood"], d["T_ood"])
SEEDS = [0, 1, 2, 3, 4]
WEIGHT_DECAYS = [0.0, 1e-3, 1e-2, 1e-1]
WIDTHS = [64, 256]
SIZES = [100, 200, 400, 800]
KEYS = ("val_rel_l2", "test_rel_l2", "ood_rel_l2", "gap", "best_epoch")
t_start = time.perf_counter()

def summarize(runs):
    return {k: seed_stats(runs, k) for k in KEYS}

RESULTS = ROOT / "results" / "regularization.json"


def train_everything():
    """Parts A and B; returns the results dict (also written to RESULTS)."""
    part_a = []
    for width in WIDTHS:
        for wd in WEIGHT_DECAYS:
            runs = [run_once(d["X_train"], d["T_train"], *held_out,
                             TrainConfig(width=width, depth=2, weight_decay=wd, seed=s))
                    for s in SEEDS]
            entry = {"width": width, "weight_decay": wd, "runs": runs, **summarize(runs)}
            part_a.append(entry)
            print(f"A  width {width:3d}  wd {wd:6.0e}  val {entry['val_rel_l2']['mean']*100:.3f}"
                  f" ± {entry['val_rel_l2']['std']*100:.3f}%  test {entry['test_rel_l2']['mean']*100:.3f}%"
                  f"  gap {entry['gap']['mean']:6.1f}x  ({time.perf_counter() - t_start:.0f} s)")

    # Select by mean VALIDATION error across seeds (never by test).
    best = min(part_a, key=lambda e: e["val_rel_l2"]["mean"])
    print(f"best: width {best['width']}, weight decay {best['weight_decay']:g}")

    # ---------------------------------------------------------------- Part B
    part_b = []
    for n in SIZES:
        runs = []
        for s in SEEDS:
            X, T = subsample(d["X_train"], d["T_train"], n, seed=100 + s)
            runs.append(run_once(X, T, *held_out, TrainConfig(width=best["width"], depth=2,
                                 weight_decay=best["weight_decay"], seed=s)))
        entry = {"n_train": n, "runs": runs, **summarize(runs)}
        part_b.append(entry)
        print(f"B  n {n:4d}  test {entry['test_rel_l2']['mean']*100:.3f} ± "
              f"{entry['test_rel_l2']['std']*100:.3f}%  gap {entry['gap']['mean']:6.1f}x"
              f"  ({time.perf_counter() - t_start:.0f} s)")

    # Power-law fit  error ~ C * n^(-beta)  on the mean test errors
    ns = np.array(SIZES, float)
    errs = np.array([e["test_rel_l2"]["mean"] for e in part_b])
    beta = -np.polyfit(np.log(ns), np.log(errs), 1)[0]
    print(f"data scaling: test error ~ n^(-{beta:.2f})")

    baseline = json.loads((ROOT / "results" / "pod_ridge.json").read_text())["models"]["physics"]
    out = {"seeds": SEEDS, "part_a": part_a,
           "best": {"width": best["width"], "weight_decay": best["weight_decay"]},
           "part_b": part_b, "data_scaling_exponent": float(beta),
           "pod_ridge_test_rel_l2": baseline["test"]["rel_l2_mean"],
           "minutes": round((time.perf_counter() - t_start) / 60, 1)}
    RESULTS.write_text(json.dumps(out, indent=2))
    return out

out = json.loads(RESULTS.read_text()) if "--plot-only" in sys.argv else train_everything()
part_a, part_b, beta = out["part_a"], out["part_b"], out["data_scaling_exponent"]
best = out["best"]
ns = np.array([e["n_train"] for e in part_b], float)
errs = np.array([e["test_rel_l2"]["mean"] for e in part_b])

# ---------------------------------------------------------------- figure
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "0.2", "0.45"
COLORS = {64: ORANGE, 256: BLUE}

def tidy(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(alpha=0.2, lw=0.5)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
xpos = np.arange(len(WEIGHT_DECAYS))
labels = ["0", "1e-3", "1e-2", "1e-1"]
for ax, key, title in [(axes[0], "test_rel_l2", "Test error vs. weight decay"),
                       (axes[1], "gap", "Overfitting: val MSE / train MSE")]:
    for k, width in enumerate(WIDTHS):
        rows = [e for e in part_a if e["width"] == width]
        scale = 100 if key == "test_rel_l2" else 1
        mean = np.array([e[key]["mean"] for e in rows]) * scale
        if key == "gap":   # spread exceeds the mean on a log axis -> show the seed range
            vals = [np.array([r[key] for r in e["runs"]]) for e in rows]
            yerr = [mean - [v.min() for v in vals], [v.max() for v in vals] - mean]
        else:              # mean +- one standard deviation across seeds
            yerr = np.array([e[key]["std"] for e in rows]) * scale
        x = xpos + (k - 0.5) * 0.12
        ax.errorbar(x, mean, yerr=yerr, fmt="o-", color=COLORS[width], ms=6,
                    capsize=4, lw=1.6, label=f"width {width}")
        for e, xi in zip(rows, x):                        # individual seeds, faint
            ax.scatter([xi] * len(e["runs"]), [r[key] * scale for r in e["runs"]],
                       s=8, color=COLORS[width], alpha=0.3, lw=0)
    ax.set_xticks(xpos, labels); ax.set_xlabel("AdamW weight decay")
    ax.set_title(title, fontsize=10, color=INK); tidy(ax)
axes[0].set_ylabel("test relative L2 (%)"); axes[0].legend(frameon=False, fontsize=9)
axes[1].set_yscale("log"); axes[1].set_ylabel("ratio at kept epoch (bars: seed range)")

ax = axes[2]
mean = np.array([e["test_rel_l2"]["mean"] for e in part_b]) * 100
sd = np.array([e["test_rel_l2"]["std"] for e in part_b]) * 100
ax.errorbar(ns, mean, yerr=sd, fmt="o-", color=BLUE, ms=6, capsize=4, lw=1.6,
            label="POD + MLP (mean ± sd, 5 seeds)")
ax.plot(ns, errs[-1] * 100 * (ns / ns[-1]) ** (-beta), "--", color=MUTED, lw=1,
        label=rf"fit $\propto n^{{-{beta:.2f}}}$")
ax.axhline(out["pod_ridge_test_rel_l2"] * 100, color=ORANGE, lw=1.2, ls=":",
           label="POD + ridge, n = 800")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks(ns, [str(int(v)) for v in ns])
ax.minorticks_off()
yt = [0.6, 1, 2, 3, 5]
ax.set_yticks(yt, [f"{v:g}" for v in yt])
ax.set_xlabel("training simulations n"); ax.set_ylabel("test relative L2 (%)")
ax.set_title(f"How many simulations? (width {best['width']}, wd {best['weight_decay']:g})",
             fontsize=10, color=INK)
ax.legend(frameon=False, fontsize=9); tidy(ax)
fig.tight_layout()
fig.savefig(ROOT / "figures" / "regularization.png", dpi=150, bbox_inches="tight")
print("saved figures/regularization.png")
