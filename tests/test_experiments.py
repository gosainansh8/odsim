import json

import numpy as np
import pytest

from odsim import SimConfig, Site, compare_scenarios, expand_grid, run_replicates, run_sweep
from odsim.cli import main

CFG = SimConfig(
    width=15,
    height=15,
    n_agents=150,
    n_steps=40,
    p_overdose=0.02,
    sites=[Site(4, 4, "treatment"), Site(10, 10, "ssp")],
)


def test_expand_grid():
    assert expand_grid({}) == [{}]
    assert expand_grid({"a": [1, 2], "b": ["x"]}) == [{"a": 1, "b": "x"}, {"a": 2, "b": "x"}]


def test_parallel_equals_serial():
    serial = run_replicates(CFG, reps=4, seed=11, workers=1)
    parallel = run_replicates(CFG, reps=4, seed=11, workers=2)
    for s, p in zip(serial, parallel):
        np.testing.assert_array_equal(s.durations, p.durations)
        assert s.n_overdoses == p.n_overdoses


def test_replicates_are_independent():
    results = run_replicates(CFG, reps=3, seed=0)
    assert not np.array_equal(results[0].durations, results[1].durations)


def test_sweep_shape():
    df = run_sweep(CFG, {"p_enroll": [0.01, 0.2], "frac_housed": [0.3]}, reps=2)
    assert len(df) == 4
    assert {"p_enroll", "frac_housed", "rep", "mortality"} <= set(df.columns)


def test_compare_scenarios():
    scenarios = {
        "baseline": {},
        "full_naloxone": {"naloxone_radius": 100.0, "naloxone_effect": 1.0},
    }
    comparison = compare_scenarios(CFG, scenarios, reps=3, seed=2)
    table = comparison.table
    assert list(table["scenario"]) == ["baseline", "full_naloxone"]
    row = table.set_index("scenario").loc["full_naloxone"]
    assert row["mortality_mean"] == 0.0
    assert row["logrank_p"] < 0.05
    assert set(comparison.curves) == set(scenarios)


def test_scenario_hazard_ratios():
    scenarios = {"baseline": {}, "deadlier": {"p_fatal": 0.4}}
    table = compare_scenarios(CFG, scenarios, reps=3, seed=4).table.set_index("scenario")
    assert table.loc["baseline", "hazard_ratio"] == 1.0
    row = table.loc["deadlier"]
    assert row["hazard_ratio"] > 1.5
    assert row["hr_ci_low"] < row["hazard_ratio"] < row["hr_ci_high"]


def test_housing_cox_finds_higher_risk_when_unhoused():
    from odsim import Simulation, housing_cox

    cfg = CFG.with_overrides(n_agents=3000, n_steps=120, unhoused_od_mult=3.0)
    fit = housing_cox(Simulation(cfg, seed=1).run())
    assert fit.hazard_ratio[0] > 1.5
    assert fit.p_value[0] < 0.01


def test_compare_rejects_unknown_baseline():
    with pytest.raises(ValueError):
        compare_scenarios(CFG, {"a": {}}, reps=1, baseline="missing")


def test_cli_end_to_end(tmp_path):
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(CFG.to_dict()))
    scen_path = tmp_path / "scen.json"
    scen_path.write_text(json.dumps({"baseline": {}, "more_housing": {"frac_housed": 0.9}}))

    out = tmp_path / "run.csv"
    assert main(["run", str(cfg_path), "--reps", "2", "--workers", "1", "--out", str(out)]) == 0
    assert out.exists()

    out = tmp_path / "sweep.csv"
    args = ["sweep", str(cfg_path), "--param", "p_enroll=0.01,0.1", "--reps", "2"]
    assert main(args + ["--workers", "1", "--aggregate", "--out", str(out)]) == 0
    assert len(out.read_text().strip().splitlines()) == 3  # header + 2 rows

    out, curves = tmp_path / "cmp.csv", tmp_path / "km.csv"
    args = ["compare", str(cfg_path), str(scen_path), "--reps", "2", "--workers", "1"]
    assert main(args + ["--out", str(out), "--curves", str(curves)]) == 0
    assert curves.exists()

    report = tmp_path / "report.html"
    args = ["report", str(cfg_path), str(scen_path), "--reps", "2", "--workers", "1"]
    assert main(args + ["--html", str(report)]) == 0
    assert "data:image/png;base64" in report.read_text()

    heat = tmp_path / "heat.png"
    args = ["sweep", str(cfg_path), "--param", "p_enroll=0.01,0.1", "--param", "frac_housed=0.2,0.8"]
    assert main(args + ["--reps", "1", "--workers", "1", "--heatmap", str(heat)]) == 0
    assert heat.stat().st_size > 0

    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"priors": {"p_enroll": [0.01, 0.2]}, "targets": {"enrollment": 0.3}}))
    post = tmp_path / "post.csv"
    args = ["calibrate", str(cfg_path), str(spec), "--particles", "10", "--generations", "1"]
    assert main(args + ["--workers", "1", "--out", str(post)]) == 0
    assert len(post.read_text().strip().splitlines()) == 11

    args = ["run", str(cfg_path), "--reps", "1", "--workers", "1", "--backend", "numba"]
    assert main(args) == 0


def test_cli_reports_bad_config(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"n_agents": -1}))
    assert main(["run", str(bad), "--workers", "1"]) == 1
    assert "error" in capsys.readouterr().err
