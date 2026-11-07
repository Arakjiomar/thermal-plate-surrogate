"""plate-surrogate: command-line inference for the thermal plate surrogate.

    plate-surrogate info
    plate-surrogate predict --k 2 --amp 3000 --x0 0.4 --y0 0.6 --sigma 0.1 \\
                            --t-left 20 --t-right 80 --t-bottom 40 --t-top 10
    plate-surrogate predict --input cases.json --fallback --save fields.npz
    plate-surrogate solve   --k 2 --amp 3000 ...

`predict` returns the ensemble-mean field, its calibrated uncertainty, and a
verdict: trust the surrogate if the uncertainty is below a tolerance, otherwise
run the solver (automatically, with --fallback).

Two models ship:
    --model pod    (default) 5 POD + MLP networks; verdict from the calibrated spread
    --model unet   5 field-to-field U-Nets (more accurate) with a conformal gate:
                   a 90% upper bound on each plate's RMSE (Mondrian conformal by
                   distance from the training range); queries farther out than any
                   calibration plate are not certified and always go to the solver.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from .conformal import ConformalCalibrator
from .dataset import PARAM_NAMES, ood_distance
from .solver import PlateParams, solve_plate
from .uq import Ensemble, impose_edges, per_sample_spread

_REPO = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = Path(os.environ.get("PLATE_MODEL_DIR", _REPO / "model"))
DEFAULT_UNET_DIR = Path(os.environ.get("PLATE_UNET_DIR", _REPO / "model_unet"))
FLAGS = {name: "--" + name.replace("_", "-") for name in PARAM_NAMES}   # t_left -> --t-left
HELP = {"k": "conductivity", "amp": "source amplitude A", "x0": "source centre x (0-1)",
        "y0": "source centre y (0-1)", "sigma": "source width", "t_left": "edge temperature, x = 0",
        "t_right": "edge temperature, x = 1", "t_bottom": "edge temperature, y = 0",
        "t_top": "edge temperature, y = 1"}


# ----------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------
def read_cases(args) -> np.ndarray:
    """(N, 9) parameter rows from --input (JSON list of objects) or from the flags."""
    if args.input:
        cases = json.loads(Path(args.input).read_text())
        if isinstance(cases, dict):
            cases = [cases]
        missing = [n for c in cases for n in PARAM_NAMES if n not in c]
        if missing:
            raise SystemExit(f"error: input is missing parameters: {sorted(set(missing))}")
        X = np.array([[float(c[n]) for n in PARAM_NAMES] for c in cases])
    else:
        values = [getattr(args, n) for n in PARAM_NAMES]
        missing = [FLAGS[n] for n, v in zip(PARAM_NAMES, values) if v is None]
        if missing:
            raise SystemExit(f"error: missing {', '.join(missing)} (or use --input FILE)")
        X = np.array([values], dtype=float)
    bad = X[:, PARAM_NAMES.index("k")] <= 0
    bad |= X[:, PARAM_NAMES.index("sigma")] <= 0
    if bad.any():
        raise SystemExit("error: k and sigma must be positive")
    return X


def add_param_flags(p: argparse.ArgumentParser) -> None:
    for name in PARAM_NAMES:
        p.add_argument(FLAGS[name], dest=name, type=float, help=HELP[name])
    p.add_argument("--input", help="JSON file: one object or a list of objects with the 9 parameters")
    p.add_argument("--save", help="write the fields to this .npz file")
    p.add_argument("--plot", help="write a PNG of the (first) field to this path")


# ----------------------------------------------------------------------------
# Outputs
# ----------------------------------------------------------------------------
def summarize_field(T: np.ndarray) -> dict:
    """Peak over the whole plate (may sit on a hot edge), interior peak (the
    source hotspot), and mean temperature. Positions are (x, y) in plate units."""
    n = T.shape[-1]
    j, i = np.unravel_index(int(np.argmax(T)), T.shape)
    inner = T[1:-1, 1:-1]
    jj, ii = np.unravel_index(int(np.argmax(inner)), inner.shape)
    return {"peak_T": round(float(T.max()), 3),
            "peak_xy": [round(i / (n - 1), 4), round(j / (n - 1), 4)],
            "interior_peak_T": round(float(inner.max()), 3),
            "interior_peak_xy": [round((ii + 1) / (n - 1), 4), round((jj + 1) / (n - 1), 4)],
            "mean_T": round(float(T.mean()), 3)}


def save_plot(path, T, sigma=None, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ncols = 1 if sigma is None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5.2 * ncols, 4.3), squeeze=False)
    panels = [(T, "temperature (°C)", "inferno")]
    if sigma is not None:
        panels.append((sigma, "uncertainty, 1 sd (°C)", "magma"))
    for ax, (img, label, cmap) in zip(axes[0], panels):
        im = ax.imshow(img, origin="lower", extent=[0, 1, 0, 1], cmap=cmap)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, label=label)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")


# ----------------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------------
def resolve_model(value: str) -> Path:
    """'pod' / 'unet' name a shipped model; anything else is a model folder."""
    return {"pod": DEFAULT_MODEL_DIR, "unet": DEFAULT_UNET_DIR}.get(value, Path(value))


def load_model(folder: Path):
    folder = resolve_model(str(folder))
    card_path = folder / "model_card.json"
    if not card_path.exists():
        raise SystemExit(f"error: no model found in {folder} (set PLATE_MODEL_DIR or --model)")
    return Ensemble.load(folder), json.loads(card_path.read_text())


def cmd_info(args) -> dict:
    _, card = load_model(Path(args.model))
    return card


def cmd_predict(args) -> list[dict]:
    ens, card = load_model(Path(args.model))
    X = read_cases(args)
    tol = args.tol if args.tol is not None else card["default_tolerance_C"]
    t0 = time.perf_counter()
    mu, sigma = ens.predict(X)
    mu, sigma = impose_edges(mu, sigma, X)
    ms = (time.perf_counter() - t0) * 1e3 / len(X)
    spread = per_sample_spread(sigma)
    d_ood = ood_distance(X, np.array(card["training_box"]["low"]), np.array(card["training_box"]["high"]))

    # Conformal gate (U-Net model): a 90% upper bound on each plate's RMSE, from the
    # Mondrian calibrator; not certified beyond the farthest calibration plate.
    conf = card.get("conformal")
    if conf is not None:
        cal = ConformalCalibrator.from_dict(conf)
        bound = cal.rmse_bound(sigma, d_ood)
        certified = d_ood <= card["max_calibrated_ood_distance"]
        bound[~certified] = np.inf

    results, fields = [], mu.copy()
    for i, row in enumerate(X):
        r = {"params": dict(zip(PARAM_NAMES, row.tolist())),
             "uncertainty_rms_C": round(float(spread[i]), 3),
             "tolerance_C": tol,
             "ood_distance": round(float(d_ood[i]), 3),
             "inside_training_range": bool(d_ood[i] == 0)}
        if conf is None:
            trust = bool(spread[i] <= tol)
        else:
            trust = bool(certified[i] and bound[i] <= tol)
            r["error_bound_90_C"] = round(float(bound[i]), 3) if np.isfinite(bound[i]) else None
            r["certified"] = bool(certified[i])
            r["reason"] = (f"outside the calibrated range (OOD distance {d_ood[i]:.3f} > "
                           f"{card['max_calibrated_ood_distance']:.3f}): uncertainty not certified"
                           if not certified[i] else
                           f"90% error bound {bound[i]:.3f} °C {'<=' if trust else '>'} tolerance {tol:g} °C")
        r.update({"verdict": "trust surrogate" if trust else "run solver",
                  "source": "surrogate", "ms": round(ms, 3)})
        if not trust and args.fallback:
            t1 = time.perf_counter()
            fields[i] = solve_plate(PlateParams(*row), mu.shape[-1])
            r.update(source="solver", ms=round((time.perf_counter() - t1) * 1e3, 3))
        r.update(summarize_field(fields[i]))
        results.append(r)

    if args.save:
        np.savez_compressed(args.save, params=X, T=fields, sigma=sigma,
                            source=np.array([r["source"] for r in results]))
    if args.plot:
        r = results[0]
        save_plot(args.plot, fields[0], None if r["source"] == "solver" else sigma[0],
                  f"{r['source']}: peak {r['peak_T']:.1f} °C, uncertainty {r['uncertainty_rms_C']:.2f} °C"
                  f" -> {r['verdict']}")
    return results


def cmd_solve(args) -> list[dict]:
    X = read_cases(args)
    out, fields = [], []
    for row in X:
        t0 = time.perf_counter()
        T = solve_plate(PlateParams(*row), args.n)
        fields.append(T)
        out.append({"params": dict(zip(PARAM_NAMES, row.tolist())), "source": "solver",
                    "ms": round((time.perf_counter() - t0) * 1e3, 3), **summarize_field(T)})
    if args.save:
        np.savez_compressed(args.save, params=X, T=np.array(fields))
    if args.plot:
        save_plot(args.plot, fields[0], None, f"solver: peak {out[0]['peak_T']:.1f} °C")
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="plate-surrogate", description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="pod",
                    help="'pod' (default, POD + MLP ensemble), 'unet' (U-Net ensemble with a "
                         "conformal gate), or a model folder")
    sub = ap.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="show the model card (training range, accuracy, calibration)")

    p = sub.add_parser("predict", help="predict the temperature field with uncertainty")
    add_param_flags(p)
    p.add_argument("--tol", type=float, help="uncertainty tolerance in °C (default from model card)")
    p.add_argument("--fallback", action="store_true",
                   help="run the solver automatically when uncertainty exceeds the tolerance")

    s = sub.add_parser("solve", help="run the reference finite-difference solver")
    add_param_flags(s)
    s.add_argument("--n", type=int, default=64, help="grid nodes per side")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handler = {"info": cmd_info, "predict": cmd_predict, "solve": cmd_solve}[args.command]
    json.dump(handler(args), sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
