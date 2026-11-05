"""Field-to-field U-Net surrogate: input images of the plate -> temperature image.

    T = lift + mask * s * c_out * UNet( [source/s * c_src, lift/s * c_lift, x, y] )

    lift, mask    exact edges by construction (fields.py)
    s             per-plate scale (scale_norm=True) or 1 (scale_norm=False)
    c_*           fixed constants fitted on the training set

With scale_norm=True the model is exactly scale-equivariant: multiplying the
source amplitude and all edge temperatures by c > 0 multiplies the predicted
field by c, because every input channel that carries temperature is divided by
s and the output is multiplied by s, and s(cX) = c s(X). That is a property of
the true solution (the PDE is linear), so building it in can't hurt accuracy
and should help when A/k or the edge temperatures go beyond the training range.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .fields import encode, interior_mask, plate_scale


# ----------------------------------------------------------------------------
# Network
# ----------------------------------------------------------------------------
def _block(cin: int, cout: int, norm: str) -> nn.Sequential:
    """Two 3x3 'same' convolutions, each followed by normalization and SiLU."""
    layers = []
    for c_in in (cin, cout):
        layers.append(nn.Conv2d(c_in, cout, 3, padding=1))
        if norm == "group":
            layers.append(nn.GroupNorm(min(8, cout), cout))
        layers.append(nn.SiLU())
    return nn.Sequential(*layers)


class UNet(nn.Module):
    """Encoder-decoder CNN with skip connections.

    Encoder: at each level a conv block, then 2x2 average pooling (halves the
    grid, like multigrid restriction). Decoder: 2x2 transposed conv (doubles the
    grid, like prolongation), concatenate the encoder features of the same size
    (skip connection: restores the fine detail the coarse path lost), conv block.
    The grid size must be divisible by 2^(levels - 1).
    """

    def __init__(self, cin: int = 4, cout: int = 1, widths=(16, 32, 64, 128, 128),
                 norm: str = "group"):
        super().__init__()
        if norm not in ("group", "none"):
            raise ValueError("norm must be 'group' or 'none'")
        self.widths = tuple(widths)
        self.down = nn.ModuleList()
        prev = cin
        for w in widths:
            self.down.append(_block(prev, w, norm))
            prev = w
        self.pool = nn.AvgPool2d(2)
        rev = widths[::-1]
        self.up = nn.ModuleList(nn.ConvTranspose2d(rev[i], rev[i + 1], 2, stride=2)
                                for i in range(len(rev) - 1))
        self.dec = nn.ModuleList(_block(2 * rev[i + 1], rev[i + 1], norm)
                                 for i in range(len(rev) - 1))
        self.head = nn.Conv2d(widths[0], cout, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for i, block in enumerate(self.down):
            x = block(x)
            if i < len(self.down) - 1:
                skips.append(x)
                x = self.pool(x)
        for up, dec in zip(self.up, self.dec):
            x = up(x)
            x = dec(torch.cat([x, skips.pop()], dim=1))
        return self.head(x)


def receptive_field(levels: int, convs_per_level: int = 2, kernel: int = 3) -> int:
    """Width (in pixels) of the input region one output pixel can see.

    Walk the layers keeping the 'jump' (input pixels per feature pixel at the
    current resolution). A k x k conv adds (k - 1) * jump; a 2x2 pool adds
    jump and doubles it; a stride-2, kernel-2 transposed conv adds nothing
    (each output pixel reads exactly one input pixel) and halves it.
    """
    rf, jump = 1, 1
    for lvl in range(levels):
        rf += convs_per_level * (kernel - 1) * jump        # encoder block
        if lvl < levels - 1:
            rf += jump                                     # 2x2 average pool
            jump *= 2
    for _ in range(levels - 1):
        jump //= 2                                         # transposed conv
        rf += convs_per_level * (kernel - 1) * jump        # decoder block
    return rf


# ----------------------------------------------------------------------------
# Surrogate: encoding + scaling + network + hard boundary conditions
# ----------------------------------------------------------------------------
@dataclass
class UNetConfig:
    widths: tuple = (16, 32, 64, 128, 128)
    norm: str = "group"
    scale_norm: bool = True         # exact scale equivariance via the per-plate scale s
    lr: float = 2e-3
    weight_decay: float = 1e-4
    epochs: int = 150
    batch_size: int = 16
    eval_every: int = 5
    seed: int = 0


class FieldUNet:
    """Parameters (N, 9) -> temperature fields (N, n, n), via images on the grid."""

    def __init__(self, cfg: UNetConfig = UNetConfig(), n: int = 64):
        self.cfg, self.n = cfg, n
        self.mask = torch.as_tensor(interior_mask(n), dtype=torch.float32)

    # -- preprocessing --------------------------------------------------------
    def _scale(self, X: np.ndarray) -> np.ndarray:
        return plate_scale(X, self.n) if self.cfg.scale_norm else np.ones(len(X))

    def _encode(self, X: np.ndarray):
        """Network input (N, 4, n, n), lift (N, n, n), per-plate scale (N,)."""
        E, s = encode(X, self.n), self._scale(X)
        lift = E[:, 1].copy()
        E[:, 0] /= s[:, None, None]
        E[:, 1] /= s[:, None, None]
        return E, lift, s

    def _inputs(self, E: np.ndarray) -> torch.Tensor:
        E = E.copy()
        E[:, 0] *= self.c_src
        E[:, 1] *= self.c_lift
        return torch.as_tensor(E, dtype=torch.float32)

    # -- training -------------------------------------------------------------
    def fit(self, X, T, X_val, T_val, verbose=False):
        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        E, lift, s = self._encode(X)
        R = (T - lift) / s[:, None, None]                            # residual target, ring = 0
        inner = (slice(None), slice(1, -1), slice(1, -1))
        # Fixed constants: 1 / std on the training set. Constants (not per-plate
        # statistics) so the exact scale equivariance survives.
        self.c_src = 1.0 / float(E[:, 0].std())
        self.c_lift = 1.0 / float(E[:, 1].std())
        self.c_out = float(R[inner].std())

        Ev, liftv, sv = self._encode(X_val)
        xt, xv = self._inputs(E), self._inputs(Ev)
        yt = torch.as_tensor(R / self.c_out, dtype=torch.float32)[:, None]
        yv = torch.as_tensor((T_val - liftv) / sv[:, None, None] / self.c_out,
                             dtype=torch.float32)[:, None]

        self.net = UNet(4, 1, cfg.widths, cfg.norm)
        opt = torch.optim.AdamW(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
        m = self.mask

        def loss_fn(pred, y):                                       # MSE over interior pixels
            return ((pred - y) ** 2 * m).sum() / (m.sum() * len(y))

        best, self.history = (np.inf, None, 0), {"epoch": [], "train": [], "val": []}
        g = torch.Generator().manual_seed(cfg.seed)
        for epoch in range(1, cfg.epochs + 1):
            self.net.train()
            perm = torch.randperm(len(xt), generator=g)
            run, nb = 0.0, 0
            for i in range(0, len(xt), cfg.batch_size):
                idx = perm[i : i + cfg.batch_size]
                opt.zero_grad()
                loss = loss_fn(self.net(xt[idx]), yt[idx])
                loss.backward()
                opt.step()
                run, nb = run + loss.item(), nb + 1
            sched.step()
            if epoch % cfg.eval_every == 0 or epoch == cfg.epochs:
                va = self._val_loss(xv, yv, loss_fn)
                self.history["epoch"].append(epoch)
                self.history["train"].append(run / nb)
                self.history["val"].append(va)
                if va < best[0]:
                    best = (va, copy.deepcopy(self.net.state_dict()), epoch)
                if verbose:
                    print(f"  epoch {epoch:4d}  train {run / nb:.2e}  val {va:.2e}", flush=True)
        self.net.load_state_dict(best[1])
        self.best_epoch = best[2]
        return self

    @torch.no_grad()
    def _val_loss(self, xv, yv, loss_fn, chunk: int = 50) -> float:
        self.net.eval()
        tot = sum(loss_fn(self.net(xv[i:i + chunk]), yv[i:i + chunk]).item() * len(yv[i:i + chunk])
                  for i in range(0, len(xv), chunk))
        return tot / len(xv)

    # -- inference ------------------------------------------------------------
    @torch.no_grad()
    def predict(self, X: np.ndarray, chunk: int = 64) -> np.ndarray:
        """Fields (N, n, n). Edges are exact: the network only fills the interior."""
        self.net.eval()
        E, lift, s = self._encode(X)
        x = self._inputs(E)
        out = np.empty((len(X), self.n, self.n))
        for i in range(0, len(X), chunk):
            corr = (self.net(x[i:i + chunk])[:, 0] * self.mask).double().numpy()
            out[i:i + chunk] = lift[i:i + chunk] + s[i:i + chunk, None, None] * self.c_out * corr
        return out

    # -- persistence ----------------------------------------------------------
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"cfg": asdict(self.cfg), "n": self.n, "c_src": self.c_src,
                    "c_lift": self.c_lift, "c_out": self.c_out,
                    "best_epoch": self.best_epoch, "state": self.net.state_dict()}, path)

    @classmethod
    def load(cls, path: str | Path) -> "FieldUNet":
        d = torch.load(path, weights_only=False)
        cfg = UNetConfig(**{**d["cfg"], "widths": tuple(d["cfg"]["widths"])})
        m = cls(cfg, d["n"])
        m.c_src, m.c_lift, m.c_out, m.best_epoch = d["c_src"], d["c_lift"], d["c_out"], d["best_epoch"]
        m.net = UNet(4, 1, cfg.widths, cfg.norm)
        m.net.load_state_dict(d["state"])
        return m
