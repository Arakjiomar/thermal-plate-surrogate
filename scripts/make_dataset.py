"""Generate the plate dataset.

Usage:
    python scripts/make_dataset.py                       # configs/default.yaml -> data/plate_dataset.npz
    python scripts/make_dataset.py --config my.yaml --out data/other.npz
"""
import argparse
import time
from pathlib import Path

from thermal_surrogate import __version__
from thermal_surrogate.dataset import DatasetConfig, fingerprint, generate, save

ROOT = Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=ROOT / "configs" / "default.yaml")
ap.add_argument("--out", default=ROOT / "data" / "plate_dataset.npz")
args = ap.parse_args()

cfg = DatasetConfig.from_yaml(args.config)
t0 = time.perf_counter()
data = generate(cfg, verbose=True)
elapsed = time.perf_counter() - t0
fp = fingerprint(data)

meta = {
    "config": str(args.config),
    "config_text": Path(args.config).read_text(),
    "package_version": __version__,
    "fingerprint": fp,
    "seconds": round(elapsed, 2),
}
save(data, args.out, meta)

n_solves = sum(len(data[f"X_{s}"]) for s in ("train", "val", "test", "ood"))
print(f"{n_solves} solves in {elapsed:.1f} s ({1e3 * elapsed / n_solves:.1f} ms each)")
for s in ("train", "val", "test", "ood"):
    T, d = data[f"T_{s}"], data[f"ood_dist_{s}"]
    print(f"  {s:5s}: X {data[f'X_{s}'].shape}  T {T.shape}  "
          f"T range [{T.min():7.1f}, {T.max():7.1f}]  max OOD distance {d.max():.2f}")
print(f"fingerprint {fp}  ->  {args.out}")
