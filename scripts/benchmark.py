"""Speed benchmark: solvers vs. surrogates, on this machine's CPU.

Usage:  python scripts/benchmark.py        (~2 min)
Needs:  data/plate_dataset.npz, runs/ from fit_pod.py and train_mlp.py
Writes: results/benchmark.json, figures/benchmark.png
"""
import json
import platform
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from thermal_surrogate.dataset import load
from thermal_surrogate.mlp import FieldSurrogate
from thermal_surrogate.pod import PODRidge
from thermal_surrogate.solver import PlateBatchSolver, PlateParams, solve_plate

ROOT = Path(__file__).resolve().parents[1]
d = load(ROOT / "data" / "plate_dataset.npz")
X_pool = np.concatenate([d["X_test"], d["X_val"]])        # 200 realistic parameter rows


def median_seconds(fn, repeats):
    """Median wall-clock time of fn() over `repeats` calls, after one warm-up call."""
    fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def batch(n):
    return X_pool[np.arange(n) % len(X_pool)]


# ---------------------------------------------------------------- 1. per-field cost vs batch size
BATCHES = [1, 10, 100, 1000]
batch_solver = PlateBatchSolver(64)
methods = {
    "solver (naive)":        lambda X: [solve_plate(PlateParams(*r), 64) for r in X],
    "solver (prefactorized)": batch_solver.solve,
    "POD + ridge":           PODRidge.load(ROOT / "runs" / "pod_ridge_physics.npz").predict,
    "POD + MLP":             FieldSurrogate.load(ROOT / "runs" / "mlp_pod_mlp.pt").predict,
    "direct MLP":            FieldSurrogate.load(ROOT / "runs" / "mlp_direct_mlp.pt").predict,
}
per_field = {}
for name, fn in methods.items():
    per_field[name] = {}
    for b in BATCHES:
        if name == "solver (naive)" and b > 100:
            per_field[name][b] = per_field[name][100]       # cost per field is flat; skip 17 s run
            continue
        X = batch(b)
        reps = 100 if b <= 10 else 15
        per_field[name][b] = median_seconds(lambda: fn(X), reps) / b * 1e3   # ms per field
    print(f"{name:24s} " + "  ".join(f"b={b}: {per_field[name][b]:8.4f} ms" for b in BATCHES))

# ---------------------------------------------------------------- 2. solver cost vs grid size
GRIDS = [32, 64, 128, 256]
grid_cost = {}
p = PlateParams()
for n in GRIDS:
    naive = median_seconds(lambda: solve_plate(p, n), 5) * 1e3
    factor = median_seconds(lambda: PlateBatchSolver(n), 3) * 1e3
    s = PlateBatchSolver(n)
    reuse = median_seconds(lambda: s.solve(batch(100)), 3) / 100 * 1e3
    grid_cost[n] = {"naive_ms": naive, "factorize_ms": factor, "prefactorized_per_field_ms": reuse}
    print(f"grid {n:3d}: naive {naive:8.2f} ms   factorize {factor:8.2f} ms   "
          f"prefactorized {reuse:7.3f} ms/field")
logn = np.log(np.array(GRIDS, float))
naive_exp = np.polyfit(logn, np.log([grid_cost[n]["naive_ms"] for n in GRIDS]), 1)[0]

# ---------------------------------------------------------------- 3. break-even
# One-off cost of the surrogate: generate the 1000 training/val/test solves + train once.
mlp_json = json.loads((ROOT / "results" / "mlp.json").read_text())
cfg = mlp_json["models"]["pod_mlp"]["config"]
train_s = next(g["train_seconds"] for g in mlp_json["grid"]
               if g["kind"] == "pod" and g["width"] == cfg["width"] and g["depth"] == cfg["depth"])
data_s = 1000 * per_field["solver (naive)"][100] / 1e3
upfront_s = data_s + train_s
# Throughput = cost per field at each method's BEST batch size. Very large batches can be
# slower per field once the output (1000 x 4096 float64 = 33 MB) no longer fits in cache.
best_ms = {name: min(row.values()) for name, row in per_field.items()}
surrogate_ms = best_ms["POD + MLP"]
break_even = {
    name: upfront_s * 1e3 / (best_ms[name] - surrogate_ms)
    for name in ("solver (naive)", "solver (prefactorized)")
}
print("best throughput (ms/field): " + ", ".join(f"{k} {v:.4f}" for k, v in best_ms.items()))
print(f"upfront cost {upfront_s:.0f} s (data {data_s:.0f} s + training {train_s:.0f} s); "
      f"break-even after {break_even['solver (naive)']:,.0f} queries vs naive, "
      f"{break_even['solver (prefactorized)']:,.0f} vs prefactorized")

out = {
    "machine": {"platform": platform.platform(), "processor": platform.processor(),
                "python": platform.python_version(), "torch": torch.__version__,
                "torch_threads": torch.get_num_threads()},
    "ms_per_field": per_field, "grid_cost": grid_cost, "naive_solver_grid_exponent": float(naive_exp),
    "upfront_seconds": {"data": data_s, "training": train_s},
    "break_even_queries": break_even,
    "best_ms_per_field": best_ms,
    "speedup_best_throughput": {k: best_ms[k] / surrogate_ms
                                for k in ("solver (naive)", "solver (prefactorized)")},
}
(ROOT / "results" / "benchmark.json").write_text(json.dumps(out, indent=2))

# ---------------------------------------------------------------- figure
COLORS = {"solver (naive)": "0.45", "solver (prefactorized)": "0.15", "POD + ridge": "#eb6834",
          "POD + MLP": "#2a78d6", "direct MLP": "#1baf7a"}
INK = "0.2"

def tidy(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(alpha=0.2, lw=0.5)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.4))
for name, row in per_field.items():
    ls = "--" if name.startswith("solver") else "-"
    a1.loglog(BATCHES, [row[b] for b in BATCHES], ls, marker="o", ms=5, lw=1.8,
              color=COLORS[name], label=name)
a1.set_xticks(BATCHES, [str(b) for b in BATCHES]); a1.minorticks_off()
a1.set_xlabel("fields per call (batch size)"); a1.set_ylabel("ms per field")
a1.set_title("Cost per 64×64 field on one laptop CPU", fontsize=10, color=INK)
a1.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3)
tidy(a1)

a2.loglog(GRIDS, [grid_cost[n]["naive_ms"] for n in GRIDS], "--o", color=COLORS["solver (naive)"],
          lw=1.8, ms=5, label=rf"solver, naive ($\propto n^{{{naive_exp:.1f}}}$)")
a2.loglog(GRIDS, [grid_cost[n]["prefactorized_per_field_ms"] for n in GRIDS], "--o",
          color=COLORS["solver (prefactorized)"], lw=1.8, ms=5, label="solver, prefactorized (per field)")
a2.axhline(surrogate_ms, color=COLORS["POD + MLP"], lw=1.5,
           label="POD + MLP, best batch (trained at 64×64 only)")
a2.set_xticks(GRIDS, [f"{n}×{n}" for n in GRIDS]); a2.minorticks_off()
a2.set_xlabel("grid"); a2.set_ylabel("ms per field")
a2.set_title("Solver cost grows with resolution", fontsize=10, color=INK)
a2.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1)
tidy(a2)
fig.tight_layout()
fig.savefig(ROOT / "figures" / "benchmark.png", dpi=150, bbox_inches="tight")
print("saved results/benchmark.json, figures/benchmark.png")
