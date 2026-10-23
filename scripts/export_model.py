"""Package the trained ensemble into model/ for the CLI and the Docker image.

Usage:  python scripts/export_model.py      (after scripts/train_ensemble.py)
Writes: model/member_*.pt, model/ensemble.json, model/model_card.json
"""
import json
import shutil
import subprocess
from datetime import date
from pathlib import Path

import numpy as np

from thermal_surrogate import __version__
from thermal_surrogate.dataset import PARAM_NAMES, DatasetConfig, load
from thermal_surrogate.uq import Ensemble, impose_edges, per_sample_spread

ROOT = Path(__file__).resolve().parents[1]
SRC, DST = ROOT / "runs" / "ensemble", ROOT / "model"
TOL = 1.0      # default uncertainty tolerance (°C RMS) for "trust surrogate" vs. "run solver"

ens = Ensemble.load(SRC)
cfg = DatasetConfig.from_yaml(ROOT / "configs" / "default.yaml")
d = load(ROOT / "data" / "plate_dataset.npz")
uq = json.loads((ROOT / "results" / "ensemble.json").read_text())

deferred = {}
for split in ("test", "ood"):
    mu, sigma = impose_edges(*ens.predict(d[f"X_{split}"]), d[f"X_{split}"])
    deferred[split] = float((per_sample_spread(sigma) > TOL).mean())

try:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip() or "unknown"
except OSError:
    commit = "unknown"

card = {
    "model": "thermal-plate-surrogate: 5-member deep ensemble of POD + MLP surrogates",
    "version": __version__, "exported": str(date.today()), "git_commit": commit,
    "grid": cfg.n, "parameters": PARAM_NAMES,
    "training_box": {"low": cfg.lows.tolist(), "high": cfg.highs.tolist()},
    "n_members": len(ens.members), "calibration_scale": ens.scale,
    "default_tolerance_C": TOL,
    "accuracy_rel_l2": {s: uq["splits"][s]["ensemble_rel_l2"] for s in ("test", "ood")},
    "coverage_90_calibrated": {s: uq["splits"][s]["coverage_calibrated"]["0.9"] for s in ("test", "ood")},
    "fraction_deferred_at_default_tol": deferred,
    "notes": ["Edge temperatures are imposed exactly; uncertainty there is 0.",
              "Outside the training box predictions degrade; the uncertainty usually flags it."],
}
if DST.exists():
    shutil.rmtree(DST)
ens.save(DST)
(DST / "model_card.json").write_text(json.dumps(card, indent=2))
size = sum(f.stat().st_size for f in DST.iterdir()) / 1e6
print(f"exported {len(ens.members)} members to {DST} ({size:.1f} MB); "
      f"deferred at {TOL} °C: test {deferred['test']:.0%}, OOD {deferred['ood']:.0%}")
