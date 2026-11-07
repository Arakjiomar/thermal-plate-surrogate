"""Package the U-Net ensemble with its conformal gate into model_unet/ for the CLI
and the Docker image (`plate-surrogate --model unet ...`).

The gate is Mondrian split conformal by OOD distance, fitted on the same fresh
calibration plates as scripts/conformal_eval.py (method M4): 200 in range + 300
outside, seed 2026. Queries farther out than the farthest calibration plate are
marked uncertified and always sent to the solver.

Usage:  python scripts/export_unet.py   (after train_unet.py seeds 0-4 and conformal_eval.py)
Writes: model_unet/member_*.pt, model_unet/ensemble.json, model_unet/model_card.json
"""
import json
import shutil
import subprocess
from datetime import date
from pathlib import Path

import numpy as np

from thermal_surrogate import __version__
from thermal_surrogate.conformal import ConformalCalibrator, calibration_plates
from thermal_surrogate.dataset import PARAM_NAMES, DatasetConfig
from thermal_surrogate.unet import FieldUNet
from thermal_surrogate.uq import Ensemble

ROOT = Path(__file__).resolve().parents[1]
DST = ROOT / "model_unet"
ALPHA, BINS, TOL = 0.1, (0.0, 0.1, 0.2), 1.0

cfg = DatasetConfig.from_yaml(ROOT / "configs" / "default.yaml")
ens = Ensemble([FieldUNet.load(ROOT / "runs" / "unet" / f"unet_s{s}.pt") for s in range(5)])
ens.scale = 1.0                                     # conformal works on the raw spread

cal_set = calibration_plates(cfg, n_in=200, n_ood=300, seed=2026)
mu, sigma = ens.predict(cal_set["X"])
cal = ConformalCalibrator(ALPHA, "gate", BINS).fit(cal_set["T"], mu, sigma, cal_set["dist"])

# Cross-check against the experiment that chose this method (same plates, same q).
conf_res = json.loads((ROOT / "results" / "conformal.json").read_text())
m4 = conf_res["ensembles"]["unet"]["methods"]["M4"]["gate"]
assert np.allclose(cal.q, m4["q"]), (cal.q, m4["q"])
uq = json.loads((ROOT / "results" / "unet_ensemble.json").read_text())["ensembles"]["unet"]

try:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip() or "unknown"
except OSError:
    commit = "unknown"

card = {
    "model": "thermal-plate-surrogate: 5-member ensemble of field-to-field U-Nets with a "
             "conformal trust/solver gate",
    "member_type": "unet", "version": __version__, "exported": str(date.today()), "git_commit": commit,
    "grid": cfg.n, "parameters": PARAM_NAMES,
    "training_box": {"low": cfg.lows.tolist(), "high": cfg.highs.tolist()},
    "n_members": len(ens.members), "default_tolerance_C": TOL,
    "conformal": cal.to_dict(),
    "max_calibrated_ood_distance": float(cal_set["dist"].max()),
    "calibration_set": {"n_in_box": 200, "n_ood": 300, "seed": 2026,
                        "description": "fresh plates, never used for training, early stopping or testing; "
                                       "OOD plates from the widened box (each range +25% of its width), "
                                       "the same sampler as the OOD test set",
                        "plates_per_distance_bin": cal.n},
    "guarantee": "For a query drawn like the calibration plates in its distance bin, the plate's interior "
                 "RMSE is <= error_bound_90_C with probability >= 90%. Queries beyond "
                 "max_calibrated_ood_distance are not certified and are always deferred.",
    "accuracy_rel_l2": {s: uq["splits"][s]["ensemble_rel_l2"] for s in ("test", "ood")},
    "gate_on_fresh_plates": {
        "coverage": {"in_range": m4["big_in"]["coverage"], "ood": m4["big_ood"]["coverage"]},
        "deferred": {"in_range": m4["big_in"]["gate"]["deferred"], "ood": m4["big_ood"]["gate"]["deferred"]},
        "missed": {"in_range": m4["big_in"]["gate"]["missed"], "ood": m4["big_ood"]["gate"]["missed"]},
        "worst_trusted_C": {"in_range": m4["big_in"]["gate"]["worst_trusted"],
                            "ood": m4["big_ood"]["gate"]["worst_trusted"]}},
    "notes": ["Edge temperatures are exact by construction (hard constraint).",
              "Exactly scale-equivariant: scaling the source amplitude and all edges by c scales the field by c.",
              "~3 ms per field per member on a laptop CPU; the 5-member ensemble ~15 ms."],
}
if DST.exists():
    shutil.rmtree(DST)
ens.save(DST)
(DST / "model_card.json").write_text(json.dumps(card, indent=2))
size = sum(f.stat().st_size for f in DST.iterdir()) / 1e6
print(f"exported {len(ens.members)} U-Nets to {DST} ({size:.1f} MB); q per bin "
      f"{[round(v, 3) for v in cal.q]}, certified up to OOD distance {card['max_calibrated_ood_distance']:.3f}")
