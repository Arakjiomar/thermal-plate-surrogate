"""Solver verification figure: O(h^2) convergence + an example plate solution.

Usage:  python scripts/verify_solver.py   ->   figures/solver_verification.png
"""
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from thermal_surrogate.solver import PlateParams, make_grid, solve_plate, solve_poisson

K = 2.5
t_exact = lambda X, Y: np.sin(2 * np.pi * X) * np.cos(np.pi * Y) + X**2 + Y
q_exact = lambda X, Y: K * (5 * np.pi**2 * np.sin(2 * np.pi * X) * np.cos(np.pi * Y) - 2.0)

ns = [9, 17, 33, 65, 129, 257]
hs, errs = [], []
for n in ns:
    X, Y, h = make_grid(n)
    T = solve_poisson(n, K, q_exact, t_exact)
    hs.append(h)
    errs.append(np.max(np.abs(T - t_exact(X, Y))))
hs, errs = np.array(hs), np.array(errs)
slope = np.polyfit(np.log(hs), np.log(errs), 1)[0]

print(f"{'n':>5} {'h':>10} {'max error':>12} {'order':>7}")
for i, (n, h, e) in enumerate(zip(ns, hs, errs)):
    order = "" if i == 0 else f"{np.log(errs[i-1]/e)/np.log(hs[i-1]/h):7.3f}"
    print(f"{n:>5} {h:10.5f} {e:12.3e} {order:>7}")
print(f"fitted slope: {slope:.3f}")

p = PlateParams(k=1.0, amp=2500.0, x0=0.35, y0=0.6, sigma=0.08,
                t_left=20, t_right=80, t_bottom=40, t_top=10)
reps = 20
t0 = time.perf_counter()
for _ in range(reps):
    T = solve_plate(p, 64)
dt = (time.perf_counter() - t0) / reps
print(f"64x64 plate solve: {dt*1e3:.2f} ms per solve")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
a1.loglog(hs, errs, "o-", label=f"FD solver (slope {slope:.2f})")
a1.loglog(hs, errs[-1] * (hs / hs[-1]) ** 2, "k--", lw=1, label=r"$\propto h^2$")
a1.set_xlabel("grid spacing $h$")
a1.set_ylabel(r"$\max|T_h - T^*|$")
a1.set_title("Convergence vs. manufactured solution")
a1.legend()
a1.grid(True, which="both", alpha=0.3)

im = a2.imshow(T, origin="lower", extent=[0, 1, 0, 1], cmap="inferno")
a2.plot(p.x0, p.y0, "c+", ms=12, mew=2)
a2.set_title("Example plate field (64×64)")
a2.set_xlabel("x"); a2.set_ylabel("y")
fig.colorbar(im, ax=a2, label="T")
fig.tight_layout()
out = Path(__file__).resolve().parents[1] / "figures" / "solver_verification.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=150)
print(f"saved {out}")
