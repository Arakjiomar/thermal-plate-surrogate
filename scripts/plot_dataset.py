"""Dataset overview figure -> figures/dataset_overview.png

Top row: three training fields. Bottom row: three OOD fields.
Right: the conductivity-amplitude projection of the samples, with the
in-distribution box drawn, so you can see where OOD points sit.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from thermal_surrogate.dataset import PARAM_NAMES, load

ROOT = Path(__file__).resolve().parents[1]
d = load(ROOT / "data" / "plate_dataset.npz")
IN_COLOR, OOD_COLOR = "#2a78d6", "#eb6834"   # categorical slots 1 and 2
ik, ia = PARAM_NAMES.index("k"), PARAM_NAMES.index("amp")

fig = plt.figure(figsize=(13, 6.2))
gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 0.12, 1.35], wspace=0.35, hspace=0.35)

def show(ax, X, T, label, color):
    im = ax.imshow(T, origin="lower", extent=[0, 1, 0, 1], cmap="inferno")
    ax.plot(X[PARAM_NAMES.index("x0")], X[PARAM_NAMES.index("y0")], "+", color="#00d0ff", ms=9, mew=1.5)
    ax.set_title(f"{label}  k={X[ik]:.2f}, A={X[ia]:.0f}", fontsize=9, color="0.2")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(color); s.set_linewidth(2)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=7)

rng = np.random.default_rng(0)
for c, i in enumerate(rng.choice(len(d["X_train"]), 3, replace=False)):
    show(fig.add_subplot(gs[0, c]), d["X_train"][i], d["T_train"][i], "train", IN_COLOR)
order = np.argsort(d["ood_dist_ood"])
for c, i in enumerate(order[[10, 50, 95]]):
    show(fig.add_subplot(gs[1, c]), d["X_ood"][i], d["T_ood"][i],
         f"OOD (dist {d['ood_dist_ood'][i]:.2f})", OOD_COLOR)

ax = fig.add_subplot(gs[:, 4])
lo, hi = d["box_lows"], d["box_highs"]
ax.add_patch(Rectangle((lo[ik], lo[ia]), hi[ik] - lo[ik], hi[ia] - lo[ia],
                       fill=False, ec="0.35", lw=1, ls="--"))
ax.scatter(d["X_train"][:, ik], d["X_train"][:, ia], s=10, c=IN_COLOR, alpha=0.55,
           lw=0, label="train (800)")
ax.scatter(d["X_ood"][:, ik], d["X_ood"][:, ia], s=22, c=OOD_COLOR, ec="white", lw=0.6,
           label="OOD (100)")
ax.set_xlabel("conductivity k"); ax.set_ylabel("source amplitude A")
ax.set_title("2 of 9 dimensions; dashed = training box", fontsize=10, color="0.2", pad=26)
ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=9)
ax.grid(alpha=0.2, lw=0.5)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.text(0, -0.13, "OOD points inside the dashed box leave it in another dimension",
        transform=ax.transAxes, fontsize=8, color="0.35")

out = ROOT / "figures" / "dataset_overview.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved {out}")
