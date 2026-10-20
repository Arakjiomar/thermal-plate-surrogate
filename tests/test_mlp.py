"""Tests for the PyTorch field surrogates."""
import numpy as np
import pytest
import torch

from thermal_surrogate.metrics import rel_l2
from thermal_surrogate.mlp import MLP, FieldSurrogate, TrainConfig

rng = np.random.default_rng(0)
FAST = TrainConfig(width=32, depth=2, epochs=150, batch_size=16, eval_every=5)


def smooth_fields(N=60, n=8):
    """Toy 'physics': fields that depend smoothly (and nonlinearly) on 9 inputs."""
    X = rng.uniform(0.5, 1.5, size=(N, 9))
    xs = np.linspace(0, 1, n)
    gx, gy = np.meshgrid(xs, xs)
    T = (X[:, 0, None, None] * np.sin(np.pi * gx) + X[:, 1, None, None] ** 2 * gy
         + X[:, 2, None, None] * gx * gy)
    return X, T


def test_mlp_shapes_and_layer_count():
    net = MLP(11, 17, width=64, depth=3)
    assert net(torch.zeros(5, 11)).shape == (5, 17)
    assert sum(isinstance(m, torch.nn.Linear) for m in net.modules()) == 4   # 3 hidden + output


def test_rejects_unknown_kind():
    with pytest.raises(ValueError):
        FieldSurrogate(kind="cnn")


@pytest.mark.parametrize("kind", ["pod", "direct"])
def test_training_reduces_loss_and_fits(kind):
    X, T = smooth_fields()
    m = FieldSurrogate(kind, physics=False, n_modes=5).fit(X[:48], T[:48], X[48:], T[48:], FAST)
    h = m.history
    assert h["train"][-1] < 0.1 * h["train"][0]           # loss fell by more than 10x
    assert m.predict(X[48:]).shape == (12, 8, 8)
    assert rel_l2(m.predict(X[48:]), T[48:]).mean() < 0.1


def test_best_validation_epoch_is_restored():
    X, T = smooth_fields()
    m = FieldSurrogate("pod", physics=False, n_modes=5).fit(X[:48], T[:48], X[48:], T[48:], FAST)
    i = m.history["epoch"].index(m.best_epoch)
    assert m.history["val"][i] == min(m.history["val"])


def test_pod_target_loss_equals_field_error():
    # Orthonormal modes: squared coefficient error == squared field error (within the span).
    X, T = smooth_fields()
    m = FieldSurrogate("pod", physics=False, n_modes=5).fit(X[:48], T[:48], X[48:], T[48:], FAST)
    a_true = m.pod.project(T[:5], 5)
    a_pert = a_true + rng.normal(size=a_true.shape)
    field_err = ((m.pod.reconstruct(a_pert) - m.pod.reconstruct(a_true)) ** 2).sum()
    assert field_err == pytest.approx(((a_pert - a_true) ** 2).sum(), rel=1e-9)


@pytest.mark.parametrize("kind", ["pod", "direct"])
def test_save_load_roundtrip(tmp_path, kind):
    X, T = smooth_fields()
    m = FieldSurrogate(kind, physics=False, n_modes=5).fit(X[:48], T[:48], X[48:], T[48:], FAST)
    m.save(tmp_path / "m.pt")
    np.testing.assert_allclose(FieldSurrogate.load(tmp_path / "m.pt").predict(X), m.predict(X),
                               rtol=1e-6, atol=1e-6)


def test_same_seed_same_model():
    X, T = smooth_fields()
    a = FieldSurrogate("pod", physics=False, n_modes=5).fit(X[:48], T[:48], X[48:], T[48:], FAST)
    b = FieldSurrogate("pod", physics=False, n_modes=5).fit(X[:48], T[:48], X[48:], T[48:], FAST)
    np.testing.assert_allclose(a.predict(X), b.predict(X))
