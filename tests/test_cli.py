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
