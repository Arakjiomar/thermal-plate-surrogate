"""Conformal prediction for the surrogate ensembles: which calibration makes the
uncertainty honest in range AND out of range?

Fresh calibration plates (new seeds; never used for training, early stopping or
testing):
    cal_in   200 Latin-hypercube plates inside the training box
    cal_ood  300 plates from the widened box, outside the training box
             (same sampler as the OOD test set, different seed)

Methods, applied to both 5-member ensembles (POD + MLP shipped, U-Net):
    M1 ensemble    one calibration factor fitted on the validation set (status quo)
    M2 in-range    split conformal on cal_in only
    M3 query-like  split conformal on cal_in + cal_ood pooled (one q)
    M4 Mondrian    split conformal on cal_in + cal_ood, one q per OOD-distance bin

Scores: 'gate' (plate RMSE / spread; drives the CLI) and 'pixel' (pooled pixels).
Evaluated on the held-out test (in range) and OOD sets (100 each), and on larger
FRESH evaluation sets (500 in range + 500 OOD, another new seed) so coverage is
measured to about +-1.3% instead of +-3%: coverage, coverage vs.
OOD distance, error-bar width, and the CLI gate at 1 C (deferred / missed).
Calibration variability: each conformal method is refit on 200 random halves of
its calibration pool; coverage is reported as mean +- sd over those refits.

Usage:  python scripts/conformal_eval.py
Writes: data/calibration.npz (regenerated, not committed), results/conformal.json,
        figures/conformal.png
"""
import json
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from thermal_surrogate.conformal import ConformalCalibrator, coverage_of, per_plate_spread
from thermal_surrogate.dataset import (DatasetConfig, latin_hypercube, load, ood_distance,
                                       sample_ood, scale, solve_batch)
from thermal_surrogate.unet import FieldUNet
from thermal_surrogate.uq import Ensemble, coverage, per_sample_rmse

ROOT = Path(__file__).resolve().parents[1]
ALPHA, TOL, BINS = 0.1, 1.0, (0.0, 0.1, 0.2)
N_IN, N_OOD, SEED, REFITS = 200, 300, 2026, 200
N_EVAL, EVAL_SEED = 500, 2027
SPLITS = ("test", "ood", "big_in", "big_ood")
cfg = DatasetConfig.from_yaml(ROOT / "configs" / "default.yaml")
d = load(ROOT / "data" / "plate_dataset.npz")
lows, highs = d["box_lows"], d["box_highs"]

# ---------------------------------------------------------------- 1. calibration plates
cal_path = ROOT / "data" / "calibration.npz"
if cal_path.exists():
    c = dict(np.load(cal_path))
else:
    rng = np.random.default_rng(SEED)
    X_in = scale(latin_hypercube(N_IN, 9, rng), lows, highs)
    X_ood = sample_ood(replace(cfg, n_ood=N_OOD), rng)
    c = {"X_in": X_in, "T_in": solve_batch(X_in, cfg.n), "X_ood": X_ood, "T_ood": solve_batch(X_ood, cfg.n)}
    np.savez_compressed(cal_path, **c)
cal = {"X": np.concatenate([c["X_in"], c["X_ood"]]), "T": np.concatenate([c["T_in"], c["T_ood"]])}
cal["dist"] = ood_distance(cal["X"], lows, highs)
is_in = np.r_[np.ones(N_IN, bool), np.zeros(N_OOD, bool)]
evals = {s: {"X": d[f"X_{s}"], "T": d[f"T_{s}"]} for s in ("test", "ood")}
eval_path = ROOT / "data" / "conformal_eval.npz"
if eval_path.exists():
    b = dict(np.load(eval_path))
else:
    rng = np.random.default_rng(EVAL_SEED)
    Xi = scale(latin_hypercube(N_EVAL, 9, rng), lows, highs)
    Xo = sample_ood(replace(cfg, n_ood=N_EVAL), rng)
    b = {"X_in": Xi, "T_in": solve_batch(Xi, cfg.n), "X_ood": Xo, "T_ood": solve_batch(Xo, cfg.n)}
    np.savez_compressed(eval_path, **b)
evals["big_in"] = {"X": b["X_in"], "T": b["T_in"]}
evals["big_ood"] = {"X": b["X_ood"], "T": b["T_ood"]}
for e_ in evals.values():
    e_["dist"] = ood_distance(e_["X"], lows, highs)

# ---------------------------------------------------------------- 2. predictions (raw spread)
ens = {"pod_mlp": Ensemble.load(ROOT / "model"),
       "unet": Ensemble([FieldUNet.load(ROOT / "runs" / "unet" / f"unet_s{s}.pt") for s in range(5)])}
pred = {}
for name, e in ens.items():
    e.scale = 1.0
    pred[name] = {"cal": e.predict(cal["X"]), **{s: e.predict(evals[s]["X"]) for s in evals}}
    e.calibrate(d["X_val"], d["T_val"])
    pred[name]["m1_scale"] = e.scale

# ---------------------------------------------------------------- 3. methods
def fit(method, kind, name, idx=None):
    """Fit a calibrator for one method on (a subset of) its calibration pool."""
    mu, sg = pred[name]["cal"]
    pool = np.flatnonzero(is_in) if method == "M2" else np.arange(len(cal["X"]))
    if idx is not None:
        pool = pool[idx]
    bins = BINS if method == "M4" else None
    return ConformalCalibrator(ALPHA, kind, bins).fit(cal["T"][pool], mu[pool], sg[pool],
                                                       cal["dist"][pool] if bins else None)


def gate_stats(bound, err):
    """CLI gate at TOL: defer when the error bound exceeds the tolerance."""
    defer = bound > TOL
    trusted = ~defer
    return {"deferred": float(defer.mean()), "missed": float(np.mean((err > TOL) & trusted)),
            "worst_trusted": float(err[trusted].max()) if trusted.any() else 0.0,
            "truly_bad": float(np.mean(err > TOL))}


res = {"alpha": ALPHA, "tolerance_deg": TOL, "bins": BINS,
       "calibration": {"n_in": N_IN, "n_ood": N_OOD, "seed": SEED,
                       "n_per_bin": np.bincount(np.searchsorted(BINS, cal["dist"]), minlength=4).tolist()},
       "ensembles": {}}
rng = np.random.default_rng(0)
for name in ens:
    r = {"m1_scale": pred[name]["m1_scale"], "methods": {}}
    for method in ("M1", "M2", "M3", "M4"):
        m = {}
        for kind in ("gate", "pixel"):
            k = {}
            if method != "M1":
                calib = fit(method, kind, name)
                k["q"] = calib.q
                k["n_per_group"] = calib.n
            for s in SPLITS:
                mu, sg = pred[name][s]
                T, dist = evals[s]["T"], evals[s]["dist"]
                if method == "M1":                            # status quo: Gaussian 90% from the val factor
                    sgc = sg * pred[name]["m1_scale"]
                    if kind == "pixel":
                        cov = coverage(T, mu, sgc, (1 - ALPHA,))[1 - ALPHA]
                    else:                                     # claim RMSE <= z_0.95 * spread (Gaussian analogue)
                        cov = float(np.mean(per_sample_rmse(T, mu) <= 1.645 * per_plate_spread(sgc)))
                    k[s] = {"coverage": cov, "coverage_sd": 0.0,
                            "eval_sd": float(np.sqrt(cov * (1 - cov) / len(T)))}
                    if kind == "gate":
                        k[s]["gate"] = gate_stats(per_plate_spread(sgc), per_sample_rmse(T, mu))
                        k[s]["mean_bound_deg"] = float(per_plate_spread(sgc).mean())
                    continue
                cv = coverage_of(calib, T, mu, sg, dist)
                k[s] = {"coverage": cv, "eval_sd": float(np.sqrt(cv * (1 - cv) / len(T)))}
                # variability over calibration draws: refit on random halves of the pool
                pool_n = N_IN if method == "M2" else N_IN + N_OOD
                covs = [coverage_of(fit(method, kind, name, rng.choice(pool_n, pool_n // 2, replace=False)),
                                    T, mu, sg, dist) for _ in range(REFITS)]
                k[s]["coverage_halfcal_mean"] = float(np.mean(covs))
                k[s]["coverage_sd"] = float(np.std(covs))
                if kind == "gate":
                    bound = calib.rmse_bound(sg, dist)
                    k[s]["gate"] = gate_stats(bound, per_sample_rmse(T, mu))
                    k[s]["mean_bound_deg"] = float(np.mean(bound[np.isfinite(bound)])) if np.isfinite(bound).any() else None
                    # coverage per OOD-distance bin of the evaluation queries
            m[kind] = k
        r["methods"][method] = m
    # gate coverage vs OOD distance, pooled test + OOD queries
    BIG = ("big_in", "big_ood")
    allT = np.concatenate([evals[s]["T"] for s in BIG]); alld = np.concatenate([evals[s]["dist"] for s in BIG])
    allmu = np.concatenate([pred[name][s][0] for s in BIG]); allsg = np.concatenate([pred[name][s][1] for s in BIG])
    err_all, spr_all = per_sample_rmse(allT, allmu), per_plate_spread(allsg)
    edges = [-1, 0.0, 0.1, 0.2, 10]                          # widened box reaches ~0.25
    vs = {"bin_edges": edges[1:-1], "n": [], "M1": [], "M2": [], "M3": [], "M4": []}
    g_all = np.digitize(alld, edges[1:-1], right=True)
    for b in range(len(edges) - 1):
        sel = g_all == b
        vs["n"].append(int(sel.sum()))
        if not sel.any():                                     # no queries this far out
            for method in ("M1", "M2", "M3", "M4"):
                vs[method].append(float("nan"))
            continue
        vs["M1"].append(float(np.mean(err_all[sel] <= 1.645 * spr_all[sel] * pred[name]["m1_scale"])))
        for method in ("M2", "M3", "M4"):
            calib = fit(method, "gate", name)
            vs[method].append(coverage_of(calib, allT[sel], allmu[sel], allsg[sel], alld[sel]))
    r["gate_coverage_vs_distance"] = vs
    res["ensembles"][name] = r

(ROOT / "results" / "conformal.json").write_text(json.dumps(res, indent=2, default=float))

# ---------------------------------------------------------------- 4. print
NAMES = {"M1": "M1 ensemble factor", "M2": "M2 conformal, in-range cal", "M3": "M3 conformal, query-like cal",
         "M4": "M4 Mondrian by distance"}
print("calibration plates per Mondrian bin (in box | 0-0.1 | 0.1-0.2 | >0.2):", res["calibration"]["n_per_bin"])
for name, r in res["ensembles"].items():
    print(f"\n=== {name}  (M1 factor {r['m1_scale']:.2f})")
    print(f"{'method (fresh 500 + 500)':30s} {'gate cov in':>14s} {'gate cov OOD':>14s} {'pixel in':>9s} {'pixel OOD':>10s}"
          f" | OOD gate: defer  missed  worst   bound      [orig 100+100: gate in / OOD]")
    for method, m in r["methods"].items():
        g, p = m["gate"], m["pixel"]
        gs = g["big_ood"]["gate"]
        tot = lambda x: np.hypot(x["coverage_sd"], x["eval_sd"])
        print(f"{NAMES[method]:30s} {g['big_in']['coverage']:7.3f} ±{tot(g['big_in']):.3f}"
              f" {g['big_ood']['coverage']:7.3f} ±{tot(g['big_ood']):.3f}"
              f" {p['big_in']['coverage']:8.2f} {p['big_ood']['coverage']:10.2f}"
              f" | {gs['deferred']:5.0%} {gs['missed']:6.1%} {gs['worst_trusted']:6.2f}° "
              f"{(g['big_ood']['mean_bound_deg'] or float('nan')):6.2f}°"
              f"   [{g['test']['coverage']:.2f} / {g['ood']['coverage']:.2f}]"
              + (f"  q={['%.2f' % v for v in g['q']]}" if 'q' in g else ""))
        tg = g["big_in"]["gate"]
        print(f"{'':30s}   in-range gate: defer {tg['deferred']:.0%} missed {tg['missed']:.1%} worst {tg['worst_trusted']:.2f}°")

# ---------------------------------------------------------------- 5. figure
COL = {"M1": "0.55", "M2": "#ff7f0e", "M3": "#2ca02c", "M4": "#1f77b4"}
fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))
for row, name in enumerate(("unet", "pod_mlp")):
    r = res["ensembles"][name]
    title = "U-Net ensemble" if name == "unet" else "POD + MLP ensemble (shipped)"
    a = ax[row, 0]
    vs = r["gate_coverage_vs_distance"]
    labels = ["in box"] + [f"{lo:g}–{hi:g}" for lo, hi in zip(vs["bin_edges"][:-1], vs["bin_edges"][1:])] + [f">{vs['bin_edges'][-1]:g}"]
    x = np.arange(len(labels))
    for method in ("M1", "M2", "M3", "M4"):
        a.plot(x, vs[method], "-o", color=COL[method], label=NAMES[method], lw=2 if method == "M4" else 1.4)
    a.axhline(1 - ALPHA, color="k", ls="--", lw=1)
    a.set_xticks(x); a.set_xticklabels([f"{l}\n(n={n})" for l, n in zip(labels, vs["n"])], fontsize=8)
    a.set_ylim(0, 1.02); a.set_xlabel("OOD distance of the query (range widths outside the box)")
    a.set_ylabel("gate claim holds (plate RMSE ≤ bound)")
    a.set_title(f"{title}: coverage vs. distance", fontsize=10); a.legend(fontsize=7, loc="lower left")
    a.grid(alpha=0.3)
    a = ax[row, 1]
    w = 0.2
    for j, method in enumerate(("M1", "M2", "M3", "M4")):
        vals = [r["methods"][method]["gate"][s]["coverage"] for s in ("big_in", "big_ood")]
        sds = [float(np.hypot(r["methods"][method]["gate"][s]["coverage_sd"], r["methods"][method]["gate"][s]["eval_sd"]))
               for s in ("big_in", "big_ood")]
        b = a.bar(np.arange(2) + (j - 1.5) * w, vals, w, yerr=sds, capsize=2, color=COL[method], label=NAMES[method])
        a.bar_label(b, labels=[f"{v:.0%}" for v in vals], fontsize=7)
    a.axhline(1 - ALPHA, color="k", ls="--", lw=1)
    a.set_xticks([0, 1]); a.set_xticklabels([f"in range (fresh {N_EVAL})", f"OOD (fresh {N_EVAL})"]); a.set_ylim(0, 1.1)
    a.set_ylabel("gate coverage (± sd: calibration draws + evaluation size)")
    a.set_title("Does the 90% promise hold?", fontsize=10); a.legend(fontsize=7, loc="lower left")
    a = ax[row, 2]
    for j, method in enumerate(("M1", "M2", "M3", "M4")):
        gs = r["methods"][method]["gate"]["big_ood"]["gate"]
        vals = [gs["deferred"], gs["missed"]]
        b = a.bar(np.arange(2) + (j - 1.5) * w, vals, w, color=COL[method], label=NAMES[method])
        a.bar_label(b, labels=[f"{v:.0%}" for v in vals], fontsize=7)
    tb = r["methods"]["M1"]["gate"]["big_ood"]["gate"]["truly_bad"]
    a.axhline(tb, color="k", ls=":", lw=1, label=f"actually worse than {TOL:g} °C ({tb:.0%})")
    a.set_xticks([0, 1]); a.set_xticklabels([f"OOD deferred to solver", "OOD missed (bad but trusted)"])
    a.set_ylim(0, 1.05); a.set_title(f"CLI gate on OOD queries (tolerance {TOL:g} °C)", fontsize=10)
    a.legend(fontsize=7, loc="upper right")
fig.suptitle("Conformal calibration: making the 90% promise hold in and out of range (fresh calibration plates)",
             fontsize=12)
fig.tight_layout()
fig.savefig(ROOT / "figures" / "conformal.png", dpi=120)
print("\nwrote results/conformal.json and figures/conformal.png")
