"""Field-to-field U-Net vs. the MLP surrogates, same data, same seeds, same metrics.

Arms (3 seeds each):
    unet          U-Net on image inputs, exact edges, exact scale equivariance (scale_norm)
    unet_noscale  same U-Net without the per-plate scale (ablation)
    pod_mlp       POD + MLP, best milestone-6 config (width 256, depth 2, weight decay 0.01)
    direct_mlp    MLP predicting all 4096 pixels (milestone 5: width 256, depth 3)

Usage:  python scripts/train_unet.py                          (~3.5 h on 2 cores, sequential)
        python scripts/train_unet.py --arms unet --seeds 0 1 2 (subset; run jobs side by side)
        python scripts/train_unet.py --plot                   (aggregate saved runs, draw figure)
Writes: runs/unet/<arm>_s<seed>.{json,pt}, results/unet.json, figures/unet_results.png
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
from thermal_surrogate.metrics import rel_l2, summarize
from thermal_surrogate.mlp import FieldSurrogate, TrainConfig
from thermal_surrogate.unet import FieldUNet, UNetConfig

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "unet"
ARMS = ("unet", "unet_noscale", "pod_mlp", "direct_mlp")
LABELS = {"unet": "U-Net (scale-equivariant)", "unet_noscale": "U-Net, no scale norm",
          "pod_mlp": "POD + MLP", "direct_mlp": "direct MLP"}
COLORS = {"unet": "#1f77b4", "unet_noscale": "#9ecae1", "pod_mlp": "#d62728", "direct_mlp": "#ff9896"}


def arg_list(flag, cast=str):
    if flag not in sys.argv:
        return None
    out = []
    for a in sys.argv[sys.argv.index(flag) + 1:]:
        if a.startswith("--"):
            break
        out.append(cast(a))
    return out


SEEDS = arg_list("--seeds", int) or [0, 1, 2]
ONLY = arg_list("--arms") or list(ARMS)
d = load(ROOT / "data" / "plate_dataset.npz")


def make_model(arm: str, seed: int):
    if arm.startswith("unet"):
        return FieldUNet(UNetConfig(scale_norm=(arm == "unet"), seed=seed))
    if arm == "pod_mlp":
        return FieldSurrogate("pod", physics=True), TrainConfig(width=256, depth=2,
                                                                weight_decay=0.01, seed=seed)
    return FieldSurrogate("direct", physics=True), TrainConfig(width=256, depth=3, seed=seed)


def run_one(arm: str, seed: int) -> dict:
    path = RUNS / f"{arm}_s{seed}.json"
    if path.exists():
        return json.loads(path.read_text())
    print(f"[{arm} seed {seed}] training", flush=True)
    t0 = time.perf_counter()
    if arm.startswith("unet"):
        m = make_model(arm, seed).fit(d["X_train"], d["T_train"], d["X_val"], d["T_val"],
                                      verbose=True)
        history = m.history
    else:
        m, cfg = make_model(arm, seed)
        m.fit(d["X_train"], d["T_train"], d["X_val"], d["T_val"], cfg)
        history = None
    train_s = time.perf_counter() - t0
    res = {"arm": arm, "seed": seed, "train_seconds": train_s, "best_epoch": m.best_epoch,
           "history": history}
    for split in ("val", "test", "ood"):
        res[split] = summarize(m.predict(d[f"X_{split}"]), d[f"T_{split}"])
    X1 = d["X_test"][:1]                                   # single-query latency (warm)
    m.predict(X1)
    t0 = time.perf_counter()
    for _ in range(20):
        m.predict(X1)
    res["ms_single_query"] = (time.perf_counter() - t0) / 20 * 1e3
    t0 = time.perf_counter()
    m.predict(d["X_test"])
    res["ms_per_field_batch100"] = (time.perf_counter() - t0) / len(d["X_test"]) * 1e3
    RUNS.mkdir(parents=True, exist_ok=True)
    m.save(RUNS / f"{arm}_s{seed}.pt")
    path.write_text(json.dumps(res))
    print(f"[{arm} seed {seed}] test {res['test']['rel_l2_mean']:.3%}  ood {res['ood']['rel_l2_mean']:.3%}"
          f"  ({train_s / 60:.0f} min)", flush=True)
    return res


if "--plot" not in sys.argv:
    for seed in SEEDS:
        for arm in ARMS:
            if arm in ONLY:
                run_one(arm, seed)
    sys.exit(0)

# ---------------------------------------------------------------- aggregate
# Aggregate only the requested seeds (default 0-2), so extra runs trained for other
# purposes (e.g. seeds 3-4 for the 5-member ensemble) don't change this comparison.
runs = {a: [r for r in (json.loads(p.read_text()) for p in sorted(RUNS.glob(f"{a}_s*.json")))
            if r["seed"] in SEEDS] for a in ARMS}
runs = {a: r for a, r in runs.items() if r}
METRICS = ("rel_l2_mean", "rel_l2_p95", "rmse_mean", "max_t_err_mean", "hotspot_err_mean")
res = {"arms": {}}
for a, rs in runs.items():
    arm = {"n_seeds": len(rs), "seeds": [r["seed"] for r in rs]}
    for split in ("test", "ood"):
        for k in METRICS:
            v = np.array([r[split][k] for r in rs])
            arm[f"{split}_{k}"] = {"mean": float(v.mean()),
                                   "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0}
    for k in ("train_seconds", "ms_single_query", "ms_per_field_batch100"):
        arm[k] = float(np.mean([r[k] for r in rs]))
    arm["n_params"] = None
    res["arms"][a] = arm
# Latency, re-measured here on an idle machine (the per-run numbers above were taken
# while two training jobs shared the cores). Seed-0 model of each arm, warm.
def load_model(a):
    return (FieldUNet.load if a.startswith("unet") else FieldSurrogate.load)(RUNS / f"{a}_s0.pt")


for a in res["arms"]:
    m, X1 = load_model(a), d["X_test"][:1]
    m.predict(X1)
    t0 = time.perf_counter()
    for _ in range(50):
        m.predict(X1)
    single = (time.perf_counter() - t0) / 50 * 1e3
    t0 = time.perf_counter()
    for _ in range(3):
        m.predict(d["X_test"])
    batch = (time.perf_counter() - t0) / (3 * len(d["X_test"])) * 1e3
    res["arms"][a]["latency_idle"] = {"ms_single_query": single, "ms_per_field_batch100": batch}
    if a.startswith("unet"):
        res["arms"][a]["n_params"] = int(sum(p.numel() for p in m.net.parameters()))
    else:
        res["arms"][a]["n_params"] = int(sum(p.numel() for p in m.net.parameters()))
(ROOT / "results" / "unet.json").write_text(json.dumps(res, indent=2))

print(f"\n{'model':26s} {'test relL2':>15s} {'test p95':>9s} {'RMSE':>7s} {'peakT':>7s}"
      f" {'hotspot':>8s} {'OOD relL2':>15s} {'OOD RMSE':>8s} {'ms/query':>8s} {'train':>6s}")
for a, r in res["arms"].items():
    g = lambda k: r[k]["mean"]
    print(f"{LABELS[a]:26s} {g('test_rel_l2_mean'):7.2%} ± {r['test_rel_l2_mean']['std']:.2%}"
          f" {g('test_rel_l2_p95'):8.2%} {g('test_rmse_mean'):6.2f}° {g('test_max_t_err_mean'):6.2f}°"
          f" {g('test_hotspot_err_mean'):8.3f} {g('ood_rel_l2_mean'):7.2%} ± {r['ood_rel_l2_mean']['std']:.2%}"
          f" {g('ood_rmse_mean'):7.2f}° {r['latency_idle']['ms_single_query']:7.2f} {r['train_seconds'] / 60:5.0f}m"
          f"  ({r['latency_idle']['ms_per_field_batch100']:.2f} ms/field in batches, {r['n_params']:,} params)")

# ---------------------------------------------------------------- figure
from thermal_surrogate.unet import FieldUNet as _U          # noqa: E402
unet0 = _U.load(RUNS / "unet_s0.pt")
pod0 = FieldSurrogate.load(RUNS / "pod_mlp_s0.pt")
cases = []
for split in ("test", "ood"):
    X, T = d[f"X_{split}"], d[f"T_{split}"]
    e_pod = rel_l2(pod0.predict(X), T)
    i_med = int(np.argsort(e_pod)[len(e_pod) // 2])          # a typical plate
    i_bad = int(np.argmax(e_pod))                           # POD + MLP's worst plate
    cases += [(split, i_med, "typical"), (split, i_bad, "POD + MLP's worst")]

fig = plt.figure(figsize=(15, 13))
gs = fig.add_gridspec(len(cases) + 1, 4, height_ratios=[1.3] + [1] * len(cases))
ax = fig.add_subplot(gs[0, :2])
names = [a for a in ARMS if a in res["arms"]]
x = np.arange(len(names))
for j, (split, off) in enumerate((("test", -0.2), ("ood", 0.2))):
    m = [res["arms"][a][f"{split}_rel_l2_mean"]["mean"] for a in names]
    s = [res["arms"][a][f"{split}_rel_l2_mean"]["std"] for a in names]
    ax.bar(x + off, m, 0.38, yerr=s, capsize=3, color=[COLORS[a] for a in names],
           hatch=None if split == "test" else "//", edgecolor="k", lw=0.5,
           label="test (in distribution)" if split == "test" else "OOD")
ax.set_xticks(x); ax.set_xticklabels([LABELS[a] for a in names], fontsize=8)
ax.set_yscale("log"); ax.set_ylabel("relative L2 error (mean ± sd over seeds)")
pct = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.1%}")
ax.yaxis.set_major_formatter(pct); ax.yaxis.set_minor_formatter(pct)
ax.tick_params(axis="y", which="minor", labelsize=7)
ax.legend(fontsize=8); ax.set_title("accuracy", fontsize=10); ax.grid(alpha=0.3, axis="y")
ax = fig.add_subplot(gs[0, 2:])
for r in runs.get("unet", []) + runs.get("unet_noscale", []):
    h = r["history"]
    ax.semilogy(h["epoch"], h["val"], color=COLORS[r["arm"]], lw=1,
                label=f"{LABELS[r['arm']]}" if r["seed"] == 0 else None)
ax.set_xlabel("epoch"); ax.set_ylabel("validation loss (each model's own units)")
ax.set_title("U-Net training curves (3 seeds each)", fontsize=10); ax.legend(fontsize=8)
ax.grid(alpha=0.3)
for row, (split, i, what) in enumerate(cases, start=1):
    X, T = d[f"X_{split}"][i:i + 1], d[f"T_{split}"][i]
    Ep, Eu = pod0.predict(X)[0] - T, unet0.predict(X)[0] - T
    lim = max(np.abs(Ep).max(), np.abs(Eu).max())
    panels = [(T, "magma", None, f"{split}: {what} plate, true T"),
              (Ep, "RdBu_r", lim, f"POD + MLP error ({rel_l2(Ep[None] + T[None], T[None])[0]:.2%})"),
              (Eu, "RdBu_r", lim, f"U-Net error ({rel_l2(Eu[None] + T[None], T[None])[0]:.2%})")]
    for col, (F, cmap, L, title) in enumerate(panels):
        a = fig.add_subplot(gs[row, col])
        im = a.imshow(F, origin="lower", cmap=cmap, vmin=-L if L else None, vmax=L)
        a.set_title(title, fontsize=8); a.set_xticks([]); a.set_yticks([])
        fig.colorbar(im, ax=a, fraction=0.046, label="°C")
    a = fig.add_subplot(gs[row, 3]); a.axis("off")
    p = d[f"X_{split}"][i]
    a.text(0, 0.5, "\n".join(f"{n} = {v:.3g}" for n, v in zip([str(s) for s in d["param_names"]], p)),
           fontsize=8, va="center", family="monospace")
fig.suptitle("Field-to-field U-Net vs. MLP surrogates (same data, 3 seeds each; error maps from seed 0)",
             fontsize=12)
fig.tight_layout()
fig.savefig(ROOT / "figures" / "unet_results.png", dpi=110)
print("wrote results/unet.json and figures/unet_results.png")
