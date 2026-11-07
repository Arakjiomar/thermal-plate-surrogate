"""Tests for the plate-surrogate command-line interface."""
import json

import numpy as np
import pytest

from thermal_surrogate.cli import main
from thermal_surrogate.dataset import PARAM_NAMES
from thermal_surrogate.mlp import TrainConfig
from thermal_surrogate.solver import PlateBatchSolver, PlateParams, edge_values, solve_plate
from thermal_surrogate.uq import Ensemble, impose_edges

N = 9                                                    # tiny grid keeps the tests fast
LOW = np.array([0.5, 500, 0.2, 0.2, 0.05, 0, 0, 0, 0], float)
HIGH = np.array([5.0, 5000, 0.8, 0.8, 0.20, 100, 100, 100, 100], float)
CASE = dict(k=2, amp=3000, x0=0.4, y0=0.6, sigma=0.1, t_left=20, t_right=80, t_bottom=40, t_top=10)
FLAGS = [f"--{k.replace('_', '-')}={v}" for k, v in CASE.items()]


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    rng = np.random.default_rng(0)
    X = LOW + rng.random((48, 9)) * (HIGH - LOW)
    T = PlateBatchSolver(N).solve(X)
    cfg = TrainConfig(width=16, depth=2, epochs=30, batch_size=16, eval_every=5)
    ens = Ensemble.train(X[:36], T[:36], X[36:], T[36:], cfg, seeds=[0, 1, 2])
    ens.calibrate(X[36:], T[36:])
    folder = tmp_path_factory.mktemp("model")
    ens.save(folder)
    card = {"parameters": PARAM_NAMES, "grid": N, "default_tolerance_C": 1.0,
            "training_box": {"low": LOW.tolist(), "high": HIGH.tolist()}}
    (folder / "model_card.json").write_text(json.dumps(card))
    return folder


def run(capsys, *argv):
    assert main(list(argv)) == 0
    return json.loads(capsys.readouterr().out)


def test_info(capsys, model_dir):
    assert run(capsys, "--model", str(model_dir), "info")["grid"] == N


def test_predict_from_flags(capsys, model_dir):
    (r,) = run(capsys, "--model", str(model_dir), "predict", *FLAGS)
    assert r["params"]["k"] == 2.0 and r["inside_training_range"]
    assert r["verdict"] in ("trust surrogate", "run solver")
    for key in ("uncertainty_rms_C", "peak_T", "interior_peak_T", "mean_T", "ms"):
        assert key in r


def test_tolerance_controls_verdict_and_fallback_runs_the_solver(capsys, model_dir, tmp_path):
    (r,) = run(capsys, "--model", str(model_dir), "predict", *FLAGS, "--tol", "1e9")
    assert r["verdict"] == "trust surrogate" and r["source"] == "surrogate"
    out = tmp_path / "f.npz"
    (r,) = run(capsys, "--model", str(model_dir), "predict", *FLAGS, "--tol", "0",
               "--fallback", "--save", str(out))
    assert r["verdict"] == "run solver" and r["source"] == "solver"
    np.testing.assert_allclose(np.load(out)["T"][0], solve_plate(PlateParams(**CASE), N))


def test_edges_are_imposed_exactly(capsys, model_dir, tmp_path):
    out = tmp_path / "f.npz"
    run(capsys, "--model", str(model_dir), "predict", *FLAGS, "--tol", "1e9", "--save", str(out))
    z = np.load(out)
    G = edge_values(PlateParams(**CASE), N)
    for edge in (np.s_[0, :], np.s_[-1, :], np.s_[:, 0], np.s_[:, -1]):
        np.testing.assert_array_equal(z["T"][0][edge], G[edge])
        np.testing.assert_array_equal(z["sigma"][0][edge], 0.0)


def test_impose_edges_does_not_modify_inputs():
    mu, sigma = np.ones((1, N, N)), np.ones((1, N, N))
    X = np.array([[CASE[k] for k in PARAM_NAMES]], float)
    m2, s2 = impose_edges(mu, sigma, X)
    assert mu[0, 0, 0] == 1.0 and m2[0, 0, 0] != 1.0          # copies, not in-place
    assert s2[0, 0, 0] == 0.0 and s2[0, 4, 4] == 1.0


def test_batch_input_file_and_ood_flag(capsys, model_dir, tmp_path):
    far = {**CASE, "k": 0.25, "x0": 0.05}
    f = tmp_path / "cases.json"
    f.write_text(json.dumps([CASE, far]))
    a, b = run(capsys, "--model", str(model_dir), "predict", "--input", str(f))
    assert a["inside_training_range"] and not b["inside_training_range"]
    assert b["ood_distance"] > 0


def test_solve_command(capsys, model_dir):
    (r,) = run(capsys, "solve", *FLAGS, "--n", "17")
    assert r["source"] == "solver"
    assert r["peak_T"] == pytest.approx(solve_plate(PlateParams(**CASE), 17).max(), abs=1e-3)


def test_bad_inputs_exit_with_a_message(model_dir):
    with pytest.raises(SystemExit, match="missing"):
        main(["--model", str(model_dir), "predict", "--k", "2"])
    with pytest.raises(SystemExit, match="positive"):
        main(["--model", str(model_dir), "predict", *FLAGS[1:], "--k=-1"])
    with pytest.raises(SystemExit, match="no model"):
        main(["--model", "/nonexistent", "info"])


# ---------------------------------------------------------------- U-Net model with a conformal gate
NU = 16                                                  # U-Net grid: divisible by 2^(levels - 1)


@pytest.fixture(scope="module")
def unet_dir(tmp_path_factory):
    from thermal_surrogate.conformal import ConformalCalibrator
    from thermal_surrogate.dataset import ood_distance
    from thermal_surrogate.unet import FieldUNet, UNetConfig
    rng = np.random.default_rng(1)
    X = LOW + rng.random((60, 9)) * (HIGH - LOW)
    T = PlateBatchSolver(NU).solve(X)
    cfg = UNetConfig(widths=(8, 16, 16), epochs=10, batch_size=16, eval_every=5)
    members = [FieldUNet(UNetConfig(**{**cfg.__dict__, "seed": s}), NU).fit(X[:30], T[:30], X[30:40], T[30:40])
               for s in (0, 1)]
    ens = Ensemble(members)
    mu, sigma = ens.predict(X[40:])
    dist = ood_distance(X[40:], LOW, HIGH)               # all 0: in-box calibration plates
    cal = ConformalCalibrator(0.1, "gate", bins=(0.0,)).fit(T[40:], mu, sigma, dist)
    folder = tmp_path_factory.mktemp("model_unet")
    ens.save(folder)
    card = {"parameters": PARAM_NAMES, "grid": NU, "default_tolerance_C": 1.0, "member_type": "unet",
            "training_box": {"low": LOW.tolist(), "high": HIGH.tolist()},
            "conformal": cal.to_dict(), "max_calibrated_ood_distance": 0.0}
    (folder / "model_card.json").write_text(json.dumps(card))
    return folder, cal


def test_unet_ensemble_round_trips_with_its_member_type(unet_dir):
    folder, _ = unet_dir
    assert json.loads((folder / "ensemble.json").read_text())["member_type"] == "unet"
    ens = Ensemble.load(folder)
    assert type(ens.members[0]).__name__ == "FieldUNet"


def test_model_names_resolve_to_the_shipped_folders(capsys, unet_dir, model_dir, monkeypatch):
    from thermal_surrogate import cli
    monkeypatch.setattr(cli, "DEFAULT_UNET_DIR", unet_dir[0])
    monkeypatch.setattr(cli, "DEFAULT_MODEL_DIR", model_dir)
    assert run(capsys, "--model", "unet", "info")["member_type"] == "unet"
    assert "conformal" not in run(capsys, "info")            # default is the POD + MLP model


def test_conformal_bound_is_q_times_spread_and_drives_the_verdict(capsys, unet_dir):
    folder, cal = unet_dir
    (r,) = run(capsys, "--model", str(folder), "predict", *FLAGS, "--tol", "1e6")
    assert r["certified"] and r["verdict"] == "trust surrogate"
    assert r["error_bound_90_C"] == pytest.approx(cal.q[0] * r["uncertainty_rms_C"], rel=1e-2, abs=2e-3)
    (r,) = run(capsys, "--model", str(folder), "predict", *FLAGS, "--tol", "1e-9", "--fallback")
    assert r["verdict"] == "run solver" and r["source"] == "solver" and "> tolerance" in r["reason"]


def test_queries_beyond_the_calibrated_range_are_never_trusted(capsys, unet_dir):
    folder, _ = unet_dir
    far = [f for f in FLAGS if not f.startswith("--amp")] + ["--amp=8000"]   # A outside the box
    (r,) = run(capsys, "--model", str(folder), "predict", *far, "--tol", "1e6")
    assert not r["certified"] and r["error_bound_90_C"] is None
    assert r["verdict"] == "run solver" and "not certified" in r["reason"]


def test_pod_model_output_is_unchanged_by_the_conformal_option(capsys, model_dir):
    (r,) = run(capsys, "--model", str(model_dir), "predict", *FLAGS)
    assert "error_bound_90_C" not in r and "certified" not in r
