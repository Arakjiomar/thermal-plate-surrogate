"""Deep-ensemble uncertainty for the U-Net, side by side with the shipped POD + MLP
ensemble, plus an honest speed benchmark.

Both ensembles: 5 members (seeds 0-4), one calibration factor fitted on the
validation set, scored on interior pixels with the milestone-8 code:
  accuracy of the mean, spread-vs-error ranking (Spearman), 90% coverage raw and
  calibrated, OOD detection from the spread (AUROC), the CLI's deferral rate at
  1 C, and selective prediction (send the most uncertain 20% to the solver).

Speed: per-field time for the U-Net split into input encoding and network, for
one query and for batches, next to the solver baselines from benchmark.py.

Needs runs/unet/unet_s{0..4}.pt (train_unet.py --arms unet --seeds 0 1 2 3 4)
and model/ (export_model.py).
Usage:  python scripts/unet_ensemble.py
Writes: results/unet_ensemble.json, figures/unet_ensemble.png
"""
import json
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

from thermal_surrogate.dataset import load
from thermal_surrogate.fields import encode, plate_scale
from thermal_surrogate.metrics import rel_l2
from thermal_surrogate.unet import FieldUNet
from thermal_surrogate.uq import Ensemble, auroc, coverage, per_sample_rmse, per_sample_spread

ROOT = Path(__file__).resolve().parents[1]
d = load(ROOT / "data" / "plate_dataset.npz")
LEVELS = (0.5, 0.8, 0.9, 0.95, 0.99)
TOL = 1.0                                                   # deg C, the CLI's default
COLORS = {"pod_mlp": "#d62728", "unet": "#1f77b4"}
LABELS = {"pod_mlp": "POD + MLP ensemble (shipped)", "unet": "U-Net ensemble"}

ens = {"pod_mlp": Ensemble.load(ROOT / "model"),
       "unet": Ensemble([FieldUNet.load(ROOT / "runs" / "unet" / f"unet_s{s}.pt")
                         for s in range(int(os.environ.get("UNET_MEMBERS", 5)))])}

# ---------------------------------------------------------------- 1. uncertainty
res, cache = {"tolerance_deg": TOL, "ensembles": {}}, {}
for name, e in ens.items():
    e.scale = 1.0
    raw = {s: coverage(d[f"T_{s}"], *e.predict(d[f"X_{s}"]), LEVELS) for s in ("test", "ood")}
    scale = e.calibrate(d["X_val"], d["T_val"])
    r = {"scale": scale, "splits": {}}
    cache[name] = {}
    for split in ("test", "ood"):
        X, T = d[f"X_{split}"], d[f"T_{split}"]
        P = e.predict_all(X)
        mu, sigma = e.predict(X)
        err, spread = per_sample_rmse(T, mu), per_sample_spread(sigma)
        cache[name][split] = dict(err=err, spread=spread)
        r["splits"][split] = {
            "ensemble_rel_l2": float(rel_l2(mu, T).mean()),
            "member_rel_l2": [float(rel_l2(p, T).mean()) for p in P],
            "mean_rmse_deg": float(err.mean()),
            "spearman_spread_vs_error": float(spearmanr(spread, err).statistic),
            "coverage_raw": {str(k): v for k, v in raw[split].items()},
            "coverage_calibrated": {str(k): v for k, v in coverage(T, mu, sigma, LEVELS).items()},
            "deferral_rate": float(np.mean(spread > TOL)),
            # Is the gate right? "bad" = actual RMSE above the tolerance.
            "truly_bad_rate": float(np.mean(err > TOL)),
            "missed_rate": float(np.mean((err > TOL) & (spread <= TOL))),   # bad but trusted
            "needless_rate": float(np.mean((err <= TOL) & (spread > TOL))), # fine but deferred
            "max_error_trusted_deg": float(err[spread <= TOL].max()) if np.any(spread <= TOL) else 0.0,
            "median_error_over_spread": float(np.median(err / spread)),
            "member_error_correlation": float(np.mean(np.corrcoef(
                (P - T)[:, :, 1:-1, 1:-1].reshape(len(P), -1))[np.triu_indices(len(P), 1)])),
            "mean_spread_deg": float(spread.mean()),
        }
    r["ood_auroc"] = auroc(cache[name]["test"]["spread"], cache[name]["ood"]["spread"])
    err_all = np.concatenate([cache[name][s]["err"] for s in ("test", "ood")])
    spr_all = np.concatenate([cache[name][s]["spread"] for s in ("test", "ood")])
    fr = np.linspace(0, 0.5, 11)

    def remaining(order):                                   # deferred queries get error 0
        out = []
        for f in fr:
            e_ = err_all.copy()
            k = int(round(f * len(order)))
            if k:
                e_[order[-k:]] = 0.0
            out.append(float(e_.mean()))
        return out
    r["selective"] = {"fraction_to_solver": fr.tolist(),
                      "by_spread": remaining(np.argsort(spr_all)),
                      "oracle": remaining(np.argsort(err_all)),
                      "random": remaining(np.random.default_rng(0).permutation(len(err_all)))}
    res["ensembles"][name] = r

# ---------------------------------------------------------------- 2. speed
torch.set_num_threads(2)
def timed(fn, reps):
    fn()                                                      # warm-up
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps

X1, X100 = d["X_test"][:1], d["X_test"]
u0 = ens["unet"].members[0]
E100 = torch.as_tensor(encode(X100), dtype=torch.float32)
speed = {
    "unet_encode_ms_per_field_batch100": timed(lambda: (lambda E: plate_scale(X100, 64, E[:, 1]))(encode(X100)), 5) / 100 * 1e3,
    "unet_network_ms_per_field_batch100": timed(lambda: u0.net.eval()(E100), 5) / 100 * 1e3,
    "unet_single_ms_single_query": timed(lambda: u0.predict(X1), 30) * 1e3,
    "unet_single_ms_per_field_batch100": timed(lambda: u0.predict(X100), 5) / 100 * 1e3,
    "unet_ensemble_ms_single_query": timed(lambda: ens["unet"].predict(X1), 10) * 1e3,
    "unet_ensemble_ms_per_field_batch100": timed(lambda: ens["unet"].predict(X100), 3) / 100 * 1e3,
    "pod_ensemble_ms_single_query": timed(lambda: ens["pod_mlp"].predict(X1), 30) * 1e3,
    "pod_ensemble_ms_per_field_batch100": timed(lambda: ens["pod_mlp"].predict(X100), 10) / 100 * 1e3,
}
with torch.no_grad():
    speed["unet_network_ms_per_field_batch100"] = timed(lambda: u0.net.eval()(E100), 5) / 100 * 1e3
bench = json.loads((ROOT / "results" / "benchmark.json").read_text())
res["speed"] = speed
mpf = bench["ms_per_field"]
res["solver_reference"] = {                                 # from benchmark.py (milestone 7)
    "solver_naive_ms_single_query": mpf["solver (naive)"]["1"],
    "solver_prefactorized_ms_single_query": mpf["solver (prefactorized)"]["1"],
    "solver_prefactorized_best_ms_per_field": min(mpf["solver (prefactorized)"].values())}
(ROOT / "results" / "unet_ensemble.json").write_text(json.dumps(res, indent=2))

for name, r in res["ensembles"].items():
    print(f"\n{LABELS[name]}  (calibration scale {r['scale']:.2f}, OOD AUROC {r['ood_auroc']:.2f})")
    for split in ("test", "ood"):
        s = r["splits"][split]
        print(f"  {split:4s} mean {s['ensemble_rel_l2']:.3%} (members {min(s['member_rel_l2']):.3%}-"
              f"{max(s['member_rel_l2']):.3%})  RMSE {s['mean_rmse_deg']:.2f}°  Spearman "
              f"{s['spearman_spread_vs_error']:.2f}  90% cov {s['coverage_raw']['0.9']:.2f} -> "
              f"{s['coverage_calibrated']['0.9']:.2f}  deferred {s['deferral_rate']:.0%}  truly >tol "
              f"{s['truly_bad_rate']:.0%}  missed {s['missed_rate']:.0%}  worst trusted "
              f"{s['max_error_trusted_deg']:.2f}°  err/spread {s['median_error_over_spread']:.2f}  "
              f"member corr {s['member_error_correlation']:.2f}")
    sel = r["selective"]
    print(f"  defer 20% most uncertain: mean RMSE {sel['by_spread'][0]:.2f} -> {sel['by_spread'][4]:.2f}°"
          f" (oracle {sel['oracle'][4]:.2f}, random {sel['random'][4]:.2f})")
print("\nspeed (ms):", {k: round(v, 3) for k, v in speed.items()})

# ---------------------------------------------------------------- 3. figure
fig, ax = plt.subplots(2, 3, figsize=(16, 9.5))
a = ax[0, 0]
for name in ens:
    for split, mk in (("test", "o"), ("ood", "^")):
        c = cache[name][split]
        a.scatter(c["spread"], c["err"], s=12, marker=mk, color=COLORS[name], alpha=0.55, lw=0,
                  label=f"{LABELS[name].split(' (')[0]}, {split}")
lo = min(min(cache[n][s][k].min() for n in ens for s in ("test", "ood") for k in ("err", "spread")), 1e-2)
hi = max(cache[n][s][k].max() for n in ens for s in ("test", "ood") for k in ("err", "spread"))
a.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="error = spread")
a.set_xscale("log"); a.set_yscale("log")
a.set_xlabel("calibrated spread (°C RMS)"); a.set_ylabel("actual RMSE of ensemble mean (°C)")
a.set_title("Does a larger spread mean a larger error?", fontsize=10); a.legend(fontsize=7)
a = ax[0, 1]
a.plot([0.45, 1], [0.45, 1], "k--", lw=0.8)
for name in ens:
    for split, ls in (("test", "-"), ("ood", ":")):
        cov = res["ensembles"][name]["splits"][split]["coverage_calibrated"]
        a.plot(LEVELS, [cov[str(p)] for p in LEVELS], ls, marker="o", ms=4, color=COLORS[name],
               label=f"{LABELS[name].split(' (')[0]}, {split}")
a.set_xlabel("nominal interval probability"); a.set_ylabel("fraction of true pixels inside")
a.set_title("Calibration (one factor fitted on validation)", fontsize=10); a.legend(fontsize=7)
a = ax[0, 2]
for name in ens:
    sel = res["ensembles"][name]["selective"]
    f = np.array(sel["fraction_to_solver"]) * 100
    a.plot(f, sel["by_spread"], "-", color=COLORS[name], lw=2, label=f"{LABELS[name].split(' (')[0]}: defer most uncertain")
    a.plot(f, sel["oracle"], "--", color=COLORS[name], lw=1, label="  oracle")
    a.plot(f, sel["random"], ":", color=COLORS[name], lw=1, label="  random")
a.set_yscale("log"); a.set_xlabel("% of queries sent to the solver (test + OOD)")
a.set_ylabel("mean RMSE over all queries (°C)")
a.set_title("Selective prediction", fontsize=10); a.legend(fontsize=7)
a = ax[1, 0]
x = np.arange(2)
kinds = [("truly_bad_rate", "actually worse than tolerance", None, 0.35),
         ("deferral_rate", "sent to the solver", None, 1.0),
         ("missed_rate", "MISSED: bad but trusted", "///", 1.0)]
w = 0.13
for j, name in enumerate(ens):
    for q, (key, lab, hatch, alpha) in enumerate(kinds):
        vals = [res["ensembles"][name]["splits"][sp][key] for sp in ("test", "ood")]
        off = (j * 3 + q - 2.5) * w
        b = a.bar(x + off, vals, w, color=COLORS[name], alpha=alpha, hatch=hatch, edgecolor="k", lw=0.4,
                  label=f"{LABELS[name].split(' (')[0]}: {lab}")
        a.bar_label(b, labels=[f"{v:.0%}" for v in vals], fontsize=6.5)
a.set_xticks(x); a.set_xticklabels(["test (in range)", "OOD"]); a.set_ylim(0, 0.95)
a.set_ylabel(f"fraction of queries (tolerance {TOL:g} °C)")
a.set_title("Is the trust/solver gate right?", fontsize=10); a.legend(fontsize=6.5, loc="upper left")
a = ax[1, 1]
for j, name in enumerate(ens):
    vals = [res["ensembles"][name]["splits"][s]["ensemble_rel_l2"] for s in ("test", "ood")]
    b = a.bar(x + (j - 0.5) * 0.38, vals, 0.38, color=COLORS[name], label=LABELS[name])
    a.bar_label(b, labels=[f"{v:.2%}" for v in vals], fontsize=8)
a.set_xticks(x); a.set_xticklabels(["test (in range)", "OOD"]); a.set_yscale("log")
a.set_ylabel("relative L2 error of the ensemble mean")
a.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.1%}"))
a.set_title("Accuracy of the ensemble mean", fontsize=10); a.legend(fontsize=8)
a = ax[1, 2]
ref = res["solver_reference"]
items = [("solver, naive", ref["solver_naive_ms_single_query"]),
         ("solver, prefactorized\n(best throughput)", ref["solver_prefactorized_best_ms_per_field"]),
         ("POD + MLP ensemble", speed["pod_ensemble_ms_per_field_batch100"]),
         ("U-Net (1 model)\nencoding", speed["unet_encode_ms_per_field_batch100"]),
         ("U-Net (1 model)\nnetwork", speed["unet_network_ms_per_field_batch100"]),
         ("U-Net ensemble", speed["unet_ensemble_ms_per_field_batch100"])]
b = a.barh([k for k, _ in items][::-1], [v for _, v in items][::-1],
           color=["0.45", "0.65", "#d62728", "#9ecae1", "#9ecae1", "#1f77b4"][::-1])
a.bar_label(b, labels=[f"{v:.3g} ms" for _, v in items][::-1], fontsize=8)
a.set_xscale("log"); a.set_xlabel("ms per field (batches of 100, 2 CPU threads)")
a.set_title("Cost per prediction", fontsize=10)
fig.suptitle("Uncertainty and cost: U-Net ensemble vs. the shipped POD + MLP ensemble (5 members each)",
             fontsize=12)
fig.tight_layout()
fig.savefig(ROOT / "figures" / "unet_ensemble.png", dpi=120)
print("wrote results/unet_ensemble.json and figures/unet_ensemble.png")
