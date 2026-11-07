"""Stress tests for the field-to-field U-Net vs. the MLP surrogates.

1. Single-parameter OOD sweeps (no training): start from the centre of the box and
   push ONE thing past the training range, everything else held at the centre.
   Every sweep point is solved exactly, so the error curve shows which input a
   model can extrapolate in, and where it breaks.
       x0        source slides toward the left wall (box: 0.2-0.8)
       k         conductivity drops, so A/k grows (box: 0.5-5)
       sigma     source narrows (box: 0.05-0.2)
       t_left    one edge gets hotter (box: 0-100)
       scale c   source amplitude AND all four edges multiplied by c
                 (the U-Net is exactly equivariant to this; the MLPs aren't)
   Uses the 3 seeds of each model trained by train_unet.py.

2. Data efficiency: retrain the U-Net and POD + MLP on random subsets of
   n = 100 / 200 / 400 training plates (3 seeds; n = 800 comes from train_unet.py).
   Same epochs as at n = 800, as in milestone 6.

Usage:  python scripts/stress_tests.py --sweeps
        python scripts/stress_tests.py --datasize [--arms unet pod_mlp] [--seeds 0 1 2] [--sizes 100 200 400]
        python scripts/stress_tests.py --plot
Writes: runs/unet/size<n>_<arm>_s<seed>.{json,pt}, results/stress_tests.json,
        figures/unet_sweeps.png, figures/unet_datasize.png
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, NullFormatter

from thermal_surrogate.dataset import load
from thermal_surrogate.metrics import rel_l2, summarize
from thermal_surrogate.mlp import FieldSurrogate, TrainConfig
from thermal_surrogate.solver import PlateParams, solve_plate
from thermal_surrogate.study import subsample
from thermal_surrogate.unet import FieldUNet, UNetConfig

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "unet"
OUT = ROOT / "results" / "stress_tests.json"
MODELS = ("unet", "unet_noscale", "pod_mlp", "direct_mlp")
LABELS = {"unet": "U-Net (scale-equivariant)", "unet_noscale": "U-Net, no scale norm",
          "pod_mlp": "POD + MLP", "direct_mlp": "direct MLP"}
COLORS = {"unet": "#1f77b4", "unet_noscale": "#9ecae1", "pod_mlp": "#d62728", "direct_mlp": "#ff9896"}
pct = FuncFormatter(lambda v, _: f"{v:.2%}" if v < 0.01 else f"{v:.1%}" if v < 0.1 else f"{v:.0%}")

d = load(ROOT / "data" / "plate_dataset.npz")
lows, highs = d["box_lows"], d["box_highs"]
names = [str(s) for s in d["param_names"]]
centre = (lows + highs) / 2


def arg_list(flag, cast=str):
    if flag not in sys.argv:
        return None
    out = []
    for a in sys.argv[sys.argv.index(flag) + 1:]:
        if a.startswith("--"):
            break
        out.append(cast(a))
    return out


def load_model(arm, seed, prefix=""):
    path = RUNS / f"{prefix}{arm}_s{seed}.pt"
    return (FieldUNet.load if arm.startswith("unet") else FieldSurrogate.load)(path)


def results() -> dict:
    return json.loads(OUT.read_text()) if OUT.exists() else {}


def save_results(key, value):
    r = results()
    r[key] = value
    OUT.write_text(json.dumps(r, indent=2))


# ---------------------------------------------------------------- 1. sweeps
def sweep_inputs():
    """name -> (values, X (m, 9), box interval in the swept quantity)."""
    sw = {}
    def one(i, values):
        X = np.tile(centre, (len(values), 1))
        X[:, i] = values
        return X
    sw["x0"] = (np.linspace(0.5, 0.05, 19), one(2, np.linspace(0.5, 0.05, 19)), (0.2, 0.8))
    k = np.geomspace(2.75, 0.2, 19)
    sw["k"] = (k, one(0, k), (0.5, 5.0))
    s = np.linspace(0.125, 0.03, 20)
    sw["sigma"] = (s, one(4, s), (0.05, 0.20))
    t = np.linspace(50, 200, 16)
    sw["t_left"] = (t, one(5, t), (0.0, 100.0))
    c = np.geomspace(0.5, 4.0, 16)                     # scale A and all 4 edges together
    X = np.tile(centre, (len(c), 1))
    X[:, 1] *= c
    X[:, 5:9] *= c[:, None]
    # In range while A = c * 2750 stays in [500, 5000] and the edges c * 50 stay <= 100:
    sw["scale"] = (c, X, (500 / centre[1], highs[1] / centre[1]))
    return sw


def run_sweeps():
    sw = sweep_inputs()
    out = {}
    for name, (vals, X, box) in sw.items():
        T = np.stack([solve_plate(PlateParams(*row), 64) for row in X])
        entry = {"values": vals.tolist(), "box": list(box), "models": {}}
        for arm in MODELS:
            errs = np.array([rel_l2(load_model(arm, s).predict(X), T) for s in (0, 1, 2)])
            entry["models"][arm] = {"mean": errs.mean(0).tolist(), "min": errs.min(0).tolist(),
                                    "max": errs.max(0).tolist()}
        out[name] = entry
        print(f"sweep {name:7s} done: last point " + "  ".join(
            f"{a} {entry['models'][a]['mean'][-1]:.2%}" for a in MODELS), flush=True)
    save_results("sweeps", out)


# ---------------------------------------------------------------- 2. data size
def run_datasize(arms, seeds, sizes):
    for seed in seeds:
        for n in sizes:
            for arm in arms:
                path = RUNS / f"size{n}_{arm}_s{seed}.json"
                if path.exists():
                    continue
                X, T = subsample(d["X_train"], d["T_train"], n, seed)
                if arm == "unet":
                    m = FieldUNet(UNetConfig(scale_norm=True, seed=seed)).fit(X, T, d["X_val"], d["T_val"])
                else:
                    m = FieldSurrogate("pod", physics=True)
                    m.fit(X, T, d["X_val"], d["T_val"],
                          TrainConfig(width=256, depth=2, weight_decay=0.01, seed=seed))
                res = {"arm": arm, "seed": seed, "n": n,
                       "test": summarize(m.predict(d["X_test"]), d["T_test"]),
                       "ood": summarize(m.predict(d["X_ood"]), d["T_ood"])}
                m.save(RUNS / f"size{n}_{arm}_s{seed}.pt")
                path.write_text(json.dumps(res))
                print(f"[{arm} n={n} seed {seed}] test {res['test']['rel_l2_mean']:.3%}"
                      f"  ood {res['ood']['rel_l2_mean']:.3%}", flush=True)


def collect_datasize():
    out = {}
    for arm in ("unet", "pod_mlp"):
        rows = []
        for n in (100, 200, 400, 800):
            files = (sorted(RUNS.glob(f"size{n}_{arm}_s*.json")) if n < 800
                     else sorted(RUNS.glob(f"{arm}_s*.json")))
            vals = {s: np.array([json.loads(f.read_text())[s]["rel_l2_mean"] for f in files])
                    for s in ("test", "ood")}
            if len(files):
                rows.append({"n": n, "seeds": len(files),
                             **{f"{s}_mean": float(v.mean()) for s, v in vals.items()},
                             **{f"{s}_std": float(v.std(ddof=1)) if len(v) > 1 else 0.0
                                for s, v in vals.items()}})
        if len(rows) >= 2:
            n = np.array([r["n"] for r in rows]); e = np.array([r["test_mean"] for r in rows])
            slope = np.polyfit(np.log(n), np.log(e), 1)[0]
        else:
            slope = None
        out[arm] = {"rows": rows, "test_exponent": None if slope is None else float(-slope)}
    save_results("datasize", out)
    return out


# ---------------------------------------------------------------- 3. figures
def plot():
    r = results()
    if "sweeps" in r:
        sw = r["sweeps"]
        titles = {"x0": ("source centre $x_0$ (→ left wall)", True),
                  "k": ("conductivity $k$ (→ larger A/k)", True),
                  "sigma": ("source width $\\sigma$ (→ narrower)", True),
                  "t_left": ("left-edge temperature (°C)", False),
                  "scale": ("scale factor c on A and all edges", False)}
        fig, axes = plt.subplots(1, 5, figsize=(22, 4.6))
        for ax, (name, e) in zip(axes, sw.items()):
            v = np.array(e["values"])
            lo, hi = e["box"]
            ax.axvspan(max(lo, v.min()), min(hi, v.max()), color="0.92", label="training range")
            for arm, m in e["models"].items():
                ax.plot(v, m["mean"], color=COLORS[arm], lw=2, label=LABELS[arm])
                ax.fill_between(v, m["min"], m["max"], color=COLORS[arm], alpha=0.15)
            ax.set_yscale("log"); ax.yaxis.set_major_formatter(pct)
            ax.yaxis.set_minor_formatter(NullFormatter())
            label, flip = titles[name]
            ax.set_xlabel(label)
            if flip:
                ax.invert_xaxis()
            if name == "k":
                ax.set_xscale("log")
                ax.set_xticks([2.75, 2, 1, 0.5, 0.3, 0.2])
                ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
                ax.xaxis.set_minor_formatter(NullFormatter())
            ax.grid(alpha=0.3, which="both")
        axes[0].set_ylabel("relative L2 error (mean of 3 seeds; band = min–max)")
        axes[0].legend(fontsize=7, loc="upper left")
        fig.suptitle("Push one input past the training range (everything else at the box centre). "
                     "Grey = inside the training range.", fontsize=11)
        fig.tight_layout()
        fig.savefig(ROOT / "figures" / "unet_sweeps.png", dpi=120)
    if "datasize" in r:
        ds = r["datasize"]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
        for ax, split in zip(axes, ("test", "ood")):
            for arm, e in ds.items():
                n = [x["n"] for x in e["rows"]]
                m = [x[f"{split}_mean"] for x in e["rows"]]
                s = [x[f"{split}_std"] for x in e["rows"]]
                lab = LABELS[arm] + (f"  (∝ n^-{e['test_exponent']:.2f})" if split == "test"
                                     and e["test_exponent"] else "")
                ax.errorbar(n, m, yerr=s, marker="o", capsize=3, color=COLORS[arm], label=lab)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xticks([100, 200, 400, 800]); ax.set_xticklabels(["100", "200", "400", "800"])
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.yaxis.set_major_formatter(pct); ax.yaxis.set_minor_formatter(pct)
            ax.tick_params(axis="y", which="minor", labelsize=7)
            ax.set_xlabel("training plates (simulations)")
            ax.set_title(f"{'test (in range)' if split == 'test' else 'OOD'} relative L2 error",
                         fontsize=10)
            ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
        fig.suptitle("How many simulations does each model need? (3 seeds, mean ± sd)", fontsize=11)
        fig.tight_layout()
        fig.savefig(ROOT / "figures" / "unet_datasize.png", dpi=120)


if "--sweeps" in sys.argv:
    run_sweeps()
if "--datasize" in sys.argv:
    run_datasize(arg_list("--arms") or ["unet", "pod_mlp"], arg_list("--seeds", int) or [0, 1, 2],
                 arg_list("--sizes", int) or [100, 200, 400])
if "--plot" in sys.argv:
    ds = collect_datasize()
    for arm, e in ds.items():
        print(arm, [(x["n"], f"{x['test_mean']:.3%}", f"{x['ood_mean']:.3%}") for x in e["rows"]],
              "exponent", e["test_exponent"])
    plot()
    print("wrote results/stress_tests.json and figures")
