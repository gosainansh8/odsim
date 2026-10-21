"""Running many simulations: replicates, parameter sweeps and scenario comparisons.

Replicates run in parallel across processes. Seeds come from
``numpy.random.SeedSequence.spawn``, so results are identical no matter how many
workers are used, and scenario comparisons reuse the same seeds across scenarios
(common random numbers), which reduces noise in the differences between them.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
import pandas as pd

from .config import SimConfig
from .model import SimResult, Simulation
from .cox import CoxResult, cox_ph
from .survival import KaplanMeier, kaplan_meier, logrank_test


def _run_one(args: tuple[SimConfig, np.random.SeedSequence, str]) -> SimResult:
    config, seed, backend = args
    return Simulation(config, seed=seed, backend=backend).run()


def run_replicates(
    config: SimConfig, reps: int, seed: int = 0, workers: int = 1, backend: str = "numpy"
) -> list[SimResult]:
    return run_many([config], reps, seed, workers, backend)[0]


def run_many(
    configs: list[SimConfig],
    reps: int,
    seed: int = 0,
    workers: int = 1,
    backend: str = "numpy",
) -> list[list[SimResult]]:
    """Run ``reps`` replicates of each config. Replicate i of every config uses
    the same seed. Returns one list of results per config, in order."""
    if reps <= 0:
        raise ValueError("reps must be positive")
    seeds = np.random.SeedSequence(seed).spawn(reps)
    jobs = [(cfg, s, backend) for cfg in configs for s in seeds]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            flat = list(pool.map(_run_one, jobs, chunksize=max(1, len(jobs) // (4 * workers))))
    else:
        flat = [_run_one(job) for job in jobs]
    return [flat[i * reps : (i + 1) * reps] for i in range(len(configs))]


def expand_grid(params: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Cartesian product of parameter values: {"a": [1, 2], "b": [3]} ->
    [{"a": 1, "b": 3}, {"a": 2, "b": 3}]."""
    if not params:
        return [{}]
    keys = list(params)
    return [dict(zip(keys, combo)) for combo in itertools.product(*params.values())]


def run_sweep(
    base: SimConfig,
    grid: dict[str, list[Any]],
    reps: int,
    seed: int = 0,
    workers: int = 1,
    backend: str = "numpy",
) -> pd.DataFrame:
    """One row per (parameter combination, replicate) with summary metrics."""
    combos = expand_grid(grid)
    configs = [base.with_overrides(**combo) for combo in combos]
    results = run_many(configs, reps, seed, workers, backend)
    rows = []
    for combo, reps_results in zip(combos, results):
        for rep, res in enumerate(reps_results):
            rows.append({**combo, "rep": rep, **res.summary()})
    return pd.DataFrame(rows)


def _pool(results: list[SimResult]) -> SimResult:
    return SimResult(
        durations=np.concatenate([r.durations for r in results]),
        events=np.concatenate([r.events for r in results]),
        housed=np.concatenate([r.housed for r in results]),
        enrolled=np.concatenate([r.enrolled for r in results]),
        n_overdoses=sum(r.n_overdoses for r in results),
        n_steps=results[0].n_steps,
    )


@dataclass
class Comparison:
    """Outcome of ``compare_scenarios``."""

    table: pd.DataFrame  # one row per scenario
    curves: dict[str, KaplanMeier]  # pooled Kaplan-Meier curve per scenario
    pooled: dict[str, SimResult]  # all replicates of a scenario concatenated
    baseline: str


def _scenario_hazard_ratio(p: SimResult, ref: SimResult) -> tuple[float, float, float]:
    """Hazard ratio (with 95% CI) of a scenario relative to the baseline."""
    durations = np.concatenate([p.durations, ref.durations])
    events = np.concatenate([p.events, ref.events])
    in_scenario = np.concatenate([np.ones(p.events.size), np.zeros(ref.events.size)])
    # With zero deaths in either group the hazard ratio is 0 or infinite and the
    # Cox model has no finite maximum; report it as not estimable.
    if not p.events.any() or not ref.events.any():
        return float("nan"), float("nan"), float("nan")
    fit = cox_ph(durations, events, in_scenario[:, None], ["scenario"])
    lo, hi = fit.confidence_interval()
    return float(fit.hazard_ratio[0]), float(lo[0]), float(hi[0])


def compare_scenarios(
    base: SimConfig,
    scenarios: dict[str, dict[str, Any]],
    reps: int,
    seed: int = 0,
    workers: int = 1,
    baseline: str | None = None,
    backend: str = "numpy",
) -> Comparison:
    """Run each scenario, pool its replicates, and compare its survival against
    the baseline with a log-rank test and a Cox hazard ratio."""
    if not scenarios:
        raise ValueError("at least one scenario is required")
    names = list(scenarios)
    baseline = baseline or names[0]
    if baseline not in scenarios:
        raise ValueError(f"baseline scenario {baseline!r} not found")

    configs = [base.with_overrides(**scenarios[name]) for name in names]
    results = dict(zip(names, run_many(configs, reps, seed, workers, backend)))
    pooled = {name: _pool(rs) for name, rs in results.items()}
    ref = pooled[baseline]

    rows, curves = [], {}
    nan = float("nan")
    for name in names:
        p = pooled[name]
        per_rep = pd.DataFrame([r.summary() for r in results[name]])
        km = kaplan_meier(p.durations, p.events)
        curves[name] = km
        if name == baseline:
            stat = pval = nan
            hr, lo, hi = 1.0, nan, nan
        else:
            lr = logrank_test(p.durations, p.events, ref.durations, ref.events)
            stat, pval = lr.statistic, lr.p_value
            hr, lo, hi = _scenario_hazard_ratio(p, ref)
        rows.append(
            {
                "scenario": name,
                "agents": int(p.events.size),
                "mortality_mean": per_rep["mortality"].mean(),
                "mortality_sd": per_rep["mortality"].std(ddof=1) if reps > 1 else 0.0,
                "enrollment_mean": per_rep["enrollment"].mean(),
                "overdoses_per_run": per_rep["overdoses"].mean(),
                "survival_end": km.survival_at(p.n_steps),
                "hazard_ratio": hr,
                "hr_ci_low": lo,
                "hr_ci_high": hi,
                "logrank_stat": stat,
                "logrank_p": pval,
            }
        )
    return Comparison(pd.DataFrame(rows), curves, pooled, baseline)


def housing_cox(result: SimResult) -> CoxResult:
    """Hazard ratio of death for unhoused vs housed agents within one scenario."""
    unhoused = (~result.housed).astype(float)[:, None]
    return cox_ph(result.durations, result.events, unhoused, ["unhoused"])
