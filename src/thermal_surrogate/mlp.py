"""Neural field surrogates in PyTorch.

Two variants, same training loop:

* kind="pod"    - the MLP predicts the first R POD coefficients (replaces ridge).
* kind="direct" - the MLP predicts all n*n temperatures directly.

Targets are scaled by ONE global number (not per output), so the MSE loss is
proportional to the squared field error in both cases. For the POD variant this
works because the modes are orthonormal:  ||sum_r (a_r - a_hat_r) phi_r||^2 = ||a - a_hat||^2.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .pod import POD, Standardizer, physics_features


# ----------------------------------------------------------------------------
# Network
# ----------------------------------------------------------------------------
class MLP(nn.Module):
    """in_dim -> [Linear -> SiLU (-> Dropout)] x depth -> Linear -> out_dim."""

    def __init__(self, in_dim: int, out_dim: int, width: int = 256, depth: int = 3,
                 dropout: float = 0.0):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.SiLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = width
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class TrainConfig:
    width: int = 256
    depth: int = 3
    lr: float = 2e-3
    weight_decay: float = 0.0
    dropout: float = 0.0
    epochs: int = 1500
    batch_size: int = 64
    eval_every: int = 10
    seed: int = 0


# ----------------------------------------------------------------------------
# Surrogate: preprocessing + network + postprocessing
# ----------------------------------------------------------------------------
class FieldSurrogate:
    """Parameters (N, 9) -> temperature fields (N, n, n)."""

    def __init__(self, kind: str = "pod", physics: bool = True, n_modes: int = 17):
        if kind not in ("pod", "direct"):
            raise ValueError("kind must be 'pod' or 'direct'")
        self.kind, self.physics, self.n_modes = kind, physics, n_modes

    # -- preprocessing ------------------------------------------------------
    def _inputs(self, X: np.ndarray) -> torch.Tensor:
        Z = physics_features(X) if self.physics else X
        return torch.as_tensor(self.x_scaler(Z), dtype=torch.float32)

    def _targets(self, T: np.ndarray) -> torch.Tensor:
        if self.kind == "pod":
            Y = self.pod.project(T, self.n_modes)                     # (N, R)
        else:
            Y = T.reshape(len(T), -1) - self.mean                     # (N, n*n)
        return torch.as_tensor(Y / self.y_scale, dtype=torch.float32)

    def _fields(self, Y: np.ndarray) -> np.ndarray:
        Y = Y * self.y_scale
        if self.kind == "pod":
            return self.pod.reconstruct(Y)
        return (self.mean + Y).reshape(len(Y), *self.shape)

    # -- training -------------------------------------------------------------
    def fit(self, X, T, X_val, T_val, cfg: TrainConfig = TrainConfig(), verbose=False):
        torch.manual_seed(cfg.seed)
        self.cfg, self.shape = cfg, T.shape[1:]
        Z = physics_features(X) if self.physics else X
        self.x_scaler = Standardizer().fit(Z)
        if self.kind == "pod":
            self.pod = POD().fit(T)
            self.y_scale = float(self.pod.project(T, self.n_modes).std())
        else:
            self.mean = T.reshape(len(T), -1).mean(axis=0)
            self.y_scale = float((T.reshape(len(T), -1) - self.mean).std())

        xt, yt = self._inputs(X), self._targets(T)
        xv, yv = self._inputs(X_val), self._targets(T_val)
        self.net = MLP(xt.shape[1], yt.shape[1], cfg.width, cfg.depth, cfg.dropout)
        opt = torch.optim.AdamW(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
        loss_fn = nn.MSELoss()

        best, self.history = (np.inf, None, 0), {"epoch": [], "train": [], "val": []}
        g = torch.Generator().manual_seed(cfg.seed)
        for epoch in range(1, cfg.epochs + 1):
            self.net.train()
            perm = torch.randperm(len(xt), generator=g)
            for i in range(0, len(xt), cfg.batch_size):
                idx = perm[i : i + cfg.batch_size]
                opt.zero_grad()
                loss = loss_fn(self.net(xt[idx]), yt[idx])
                loss.backward()
                opt.step()
            sched.step()

            if epoch % cfg.eval_every == 0 or epoch == cfg.epochs:
                self.net.eval()
                with torch.no_grad():
                    tr = loss_fn(self.net(xt), yt).item()
                    va = loss_fn(self.net(xv), yv).item()
                self.history["epoch"].append(epoch)
                self.history["train"].append(tr)
                self.history["val"].append(va)
                if va < best[0]:                       # keep the best-on-validation weights
                    best = (va, copy.deepcopy(self.net.state_dict()), epoch)
                if verbose and epoch % (cfg.eval_every * 20) == 0:
                    print(f"  epoch {epoch:5d}  train {tr:.2e}  val {va:.2e}")
        self.net.load_state_dict(best[1])
        self.best_epoch = best[2]
        return self

    # -- inference ------------------------------------------------------------
    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        return self._fields(self.net(self._inputs(X)).numpy().astype(np.float64))

    # -- persistence ----------------------------------------------------------
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {"kind": self.kind, "physics": self.physics, "n_modes": self.n_modes,
                 "cfg": asdict(self.cfg), "shape": tuple(self.shape), "y_scale": self.y_scale,
                 "x_mu": self.x_scaler.mu, "x_sd": self.x_scaler.sd,
                 "net": self.net.state_dict()}
        if self.kind == "pod":
            state.update(pod_mean=self.pod.mean, pod_modes=self.pod.modes[:, : self.n_modes])
        else:
            state.update(mean=self.mean)
        torch.save(state, path)

    @classmethod
    def load(cls, path: str | Path) -> "FieldSurrogate":
        s = torch.load(path, weights_only=False)
        m = cls(s["kind"], s["physics"], s["n_modes"])
        m.cfg, m.shape, m.y_scale = TrainConfig(**s["cfg"]), s["shape"], s["y_scale"]
        m.x_scaler = Standardizer(); m.x_scaler.mu, m.x_scaler.sd = s["x_mu"], s["x_sd"]
        if m.kind == "pod":
            m.pod = POD(); m.pod.mean, m.pod.modes, m.pod.shape = s["pod_mean"], s["pod_modes"], s["shape"]
            out_dim = m.n_modes
        else:
            m.mean = s["mean"]; out_dim = int(np.prod(s["shape"]))
        in_dim = len(m.x_scaler.mu)
        m.net = MLP(in_dim, out_dim, m.cfg.width, m.cfg.depth, m.cfg.dropout)
        m.net.load_state_dict(s["net"])
        return m
