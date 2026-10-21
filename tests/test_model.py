import numpy as np
import pytest

from odsim import SimConfig, Simulation, Site
from odsim.model import StepDraws, neighbourhood_counts, reference_step

try:
    import numba  # noqa: F401

    BACKENDS = ["numpy", "numba"]
except ImportError:  # pragma: no cover
    BACKENDS = ["numpy"]

SITES = [Site(5, 5, "treatment"), Site(15, 12, "treatment"), Site(10, 10, "ssp")]


def small_config(**overrides):
    base = SimConfig(
        width=20,
        height=20,
        n_agents=300,
        n_steps=60,
        sites=SITES,
        p_overdose=0.02,  # high rates so every code path is exercised
        p_enroll=0.2,
        p_dropout=0.05,
        bystander_effect=0.5,
    )
    return base.with_overrides(**overrides)


@pytest.mark.parametrize("backend", BACKENDS)
def test_same_seed_same_result(backend):
    a = Simulation(small_config(), seed=7, backend=backend).run()
    b = Simulation(small_config(), seed=7, backend=backend).run()
    np.testing.assert_array_equal(a.durations, b.durations)
    np.testing.assert_array_equal(a.enrolled, b.enrolled)
    assert a.n_overdoses == b.n_overdoses


def test_different_seeds_differ():
    a = Simulation(small_config(), seed=1).run()
    b = Simulation(small_config(), seed=2).run()
    assert not np.array_equal(a.durations, b.durations)


def test_unknown_backend_rejected():
    with pytest.raises(ValueError):
        Simulation(small_config(), backend="gpu")


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("bystander_effect", [0.0, 0.7])
def test_backends_match_reference_loop(backend, seed, bystander_effect):
    """Every backend must agree with the per-agent loop, draw for draw, on
    every agent's full state after every step."""
    cfg = small_config(bystander_effect=bystander_effect)
    fast = Simulation(cfg, seed=seed, backend=backend)
    slow = Simulation(cfg, seed=seed)
    draw_rng = np.random.default_rng(1000 + seed)
    for t in range(cfg.n_steps):
        draws = StepDraws.sample(draw_rng, cfg.n_agents)
        fast.step(draws)
        slow.n_overdoses += reference_step(slow, slow.state, draws, t)
        slow.t += 1
        for name in ("x", "y", "alive", "in_treatment", "death_time", "enroll_time"):
            np.testing.assert_array_equal(
                getattr(fast.state, name), getattr(slow.state, name), err_msg=f"{name} at t={t}"
            )
    assert fast.n_overdoses == slow.n_overdoses
    assert fast.n_overdoses > 0  # the test must actually exercise overdoses


def test_neighbourhood_counts_brute_force():
    rng = np.random.default_rng(0)
    w, h, n = 7, 5, 60
    x, y = rng.integers(0, w, n), rng.integers(0, h, n)
    alive = rng.random(n) < 0.8
    window = neighbourhood_counts(x, y, alive, w, h)
    for cx in range(w):
        for cy in range(h):
            expected = sum(
                1 for i in range(n) if alive[i] and abs(x[i] - cx) <= 1 and abs(y[i] - cy) <= 1
            )
            assert window[cx, cy] == expected


def test_bystanders_reduce_deaths():
    crowded = dict(width=6, height=6, n_agents=2000, n_steps=60, sites=[], p_overdose=0.01)
    base = Simulation(small_config(**crowded, bystander_effect=0.0), seed=3).run()
    helped = Simulation(small_config(**crowded, bystander_effect=1.0), seed=3).run()
    assert base.events.sum() > 0
    assert helped.events.sum() == 0  # everyone has company on a crowded 6x6 grid


@pytest.mark.parametrize("backend", BACKENDS)
def test_agents_stay_on_grid(backend):
    sim = Simulation(small_config(width=5, height=5, sites=[]), seed=3, backend=backend)
    for _ in range(100):
        sim.step()
        assert sim.state.x.min() >= 0 and sim.state.x.max() < 5
        assert sim.state.y.min() >= 0 and sim.state.y.max() < 5


def test_dead_agents_do_not_move_or_revive():
    cfg = small_config(p_overdose=0.2, p_fatal=1.0, naloxone_effect=0.0, bystander_effect=0.0)
    sim = Simulation(cfg, seed=4)
    sim.step()
    dead = ~sim.state.alive
    assert dead.any()
    x, y = sim.state.x[dead].copy(), sim.state.y[dead].copy()
    for _ in range(10):
        sim.step()
    assert not sim.state.alive[dead].any()
    np.testing.assert_array_equal(sim.state.x[dead], x)
    np.testing.assert_array_equal(sim.state.y[dead], y)


def test_no_overdose_means_no_deaths():
    res = Simulation(small_config(p_overdose=0.0), seed=5).run()
    assert res.n_overdoses == 0
    assert not res.events.any()
    assert (res.durations == res.n_steps).all()


def test_full_naloxone_coverage_prevents_deaths():
    cfg = small_config(naloxone_radius=100.0, naloxone_effect=1.0)
    res = Simulation(cfg, seed=6).run()
    assert res.n_overdoses > 0
    assert not res.events.any()


def test_no_treatment_sites_means_no_enrollment():
    res = Simulation(small_config(sites=[Site(1, 1, "ssp")]), seed=8).run()
    assert not res.enrolled.any()


def test_death_durations_are_within_horizon():
    res = Simulation(small_config(), seed=9).run()
    assert res.events.any()
    assert res.durations[res.events].min() >= 1
    assert res.durations.max() <= res.n_steps


def test_summary_keys():
    summary = Simulation(small_config(), seed=10).run().summary()
    assert {"mortality", "enrollment", "overdoses", "overdoses_per_agent"} <= summary.keys()
    assert 0.0 <= summary["mortality"] <= 1.0
