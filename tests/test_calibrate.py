import numpy as np
import pytest

from odsim import SimConfig, Site, abc_smc
from odsim.calibrate import weighted_quantile
from odsim.model import Simulation

CFG = SimConfig(
    width=20,
    height=20,
    n_agents=300,
    n_steps=90,
    sites=[Site(5, 5, "treatment"), Site(14, 14, "treatment"), Site(10, 10, "ssp")],
)
TRUTH = {"p_overdose": 0.008, "p_enroll": 0.04}
PRIORS = {"p_overdose": (0.001, 0.03), "p_enroll": (0.005, 0.2)}
STATS = ("mortality", "enrollment", "overdoses_per_agent")


@pytest.fixture(scope="module")
def targets():
    truth_cfg = CFG.with_overrides(**TRUTH)
    sims = [Simulation(truth_cfg, seed=100 + i).run().summary() for i in range(10)]
    return {k: float(np.mean([s[k] for s in sims])) for k in STATS}


@pytest.fixture(scope="module")
def result(targets):
    return abc_smc(CFG, PRIORS, targets, n_particles=60, generations=3, reps=2, seed=1)


def test_recovers_known_parameters(result):
    """Parameter recovery: data simulated from known values must yield a
    posterior that contains them and is much narrower than the prior."""
    summary = result.summary().set_index("parameter")
    for name, true in TRUTH.items():
        row = summary.loc[name]
        assert row["ci90_low"] <= true <= row["ci90_high"], name
        prior_sd = (PRIORS[name][1] - PRIORS[name][0]) / np.sqrt(12)
        assert row["sd"] < 0.5 * prior_sd, name


def test_tolerance_shrinks_and_weights_normalized(result):
    eps = [h["epsilon"] for h in result.history[1:]]
    assert all(a > b for a, b in zip(eps, eps[1:]))
    assert result.weights.sum() == pytest.approx(1.0)
    assert len(result.particles) == 60


def test_particles_respect_priors(result):
    for j, name in enumerate(result.names):
        lo, hi = PRIORS[name]
        assert (result.particles[:, j] >= lo).all() and (result.particles[:, j] <= hi).all()


def test_calibrated_config(result):
    cfg = result.calibrated_config(CFG)
    assert cfg.p_overdose == pytest.approx(result.summary().set_index("parameter").loc["p_overdose", "mean"])


def test_reproducible_across_worker_counts(targets):
    kwargs = dict(n_particles=20, generations=1, reps=1, seed=5)
    a = abc_smc(CFG, PRIORS, targets, workers=1, **kwargs)
    b = abc_smc(CFG, PRIORS, targets, workers=2, **kwargs)
    np.testing.assert_array_equal(a.particles, b.particles)


@pytest.mark.parametrize(
    "priors,targets",
    [
        ({"n_agents": (1, 2)}, {"mortality": 0.1}),
        ({"p_enroll": (0.2, 0.1)}, {"mortality": 0.1}),
        ({"p_enroll": (0.1, 1.5)}, {"mortality": 0.1}),
        ({"p_enroll": (0.1, 0.2)}, {"not_a_stat": 0.1}),
        ({}, {"mortality": 0.1}),
    ],
)
def test_invalid_specs(priors, targets):
    with pytest.raises(ValueError):
        abc_smc(CFG, priors, targets, n_particles=10, generations=1)


def test_weighted_quantile():
    assert weighted_quantile([1, 2, 3], [1, 1, 1], [0.5])[0] == pytest.approx(2)
    assert weighted_quantile([1, 2, 3], [0, 0, 1], [0.5])[0] == pytest.approx(3)
