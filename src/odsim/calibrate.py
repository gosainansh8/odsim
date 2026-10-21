"""Model calibration with Approximate Bayesian Computation (ABC-SMC).

Agent-based models have no tractable likelihood, so standard Bayesian inference
does not apply. ABC replaces the likelihood with simulation: draw parameters,
simulate, and keep the draws whose simulated summary statistics land close to
the observed targets. The kept draws approximate the posterior distribution.

Plain rejection ABC wastes most simulations on implausible parameters. ABC-SMC
(Toni et al. 2009; Beaumont et al. 2009) instead runs a sequence of generations
with a shrinking tolerance. Each generation proposes new parameters by
perturbing the previous generation's particles, and importance weights correct
for the fact that proposals no longer come from the prior.

Implementation notes:
* priors are independent uniforms over user-given ranges;
* the perturbation kernel is a multivariate normal with twice the weighted
  covariance of the previous population (Beaumont et al.);
* the tolerance for each generation is a quantile of the previous generation's
  distances, so it adapts automatically;
* distances are Euclidean after scaling each statistic by its spread under the
  prior, so statistics on different scales count equally;
* candidates are simulated in parallel batches, and every simulation gets its
  own child of one SeedSequence, so a run is reproducible for any worker count.
"""

from __future__ import annotations

from concurrent.futures import Executor, ProcessPoolExecutor
from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

from .config import SimConfig
from .model import Simulation


@dataclass
class CalibrationResult:
    names: list[str]
    particles: np.ndarray  # (n_particles, n_params)
    weights: np.ndarray  # normalized importance weights
    distances: np.ndarray
    history: list[dict[str, float]] = field(default_factory=list)
    priors: dict[str, tuple[float, float]] = field(default_factory=dict)
    targets: dict[str, float] = field(default_factory=dict)

    def summary(self, level: float = 0.9) -> pd.DataFrame:
        """Posterior mean, standard deviation and central credible interval."""
        lo_q, hi_q = (1 - level) / 2, (1 + level) / 2
        rows = []
        for j, name in enumerate(self.names):
            v = self.particles[:, j]
            mean = float(np.average(v, weights=self.weights))
            sd = float(np.sqrt(np.average((v - mean) ** 2, weights=self.weights)))
            lo, med, hi = weighted_quantile(v, self.weights, [lo_q, 0.5, hi_q])
            rows.append(
                {
                    "parameter": name,
                    "mean": mean,
                    "sd": sd,
                    "median": med,
                    f"ci{int(level * 100)}_low": lo,
                    f"ci{int(level * 100)}_high": hi,
                    "prior_low": self.priors[name][0],
                    "prior_high": self.priors[name][1],
                }
            )
        return pd.DataFrame(rows)

    def posterior_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self.particles, columns=self.names)
        df["weight"] = self.weights
        df["distance"] = self.distances
        return df

    def calibrated_config(self, base: SimConfig) -> SimConfig:
        """``base`` with every calibrated parameter set to its posterior mean."""
        means = self.summary().set_index("parameter")["mean"]
        return base.with_overrides(**{k: float(v) for k, v in means.items()})


def weighted_quantile(values, weights, qs) -> list[float]:
    order = np.argsort(values)
    v, w = np.asarray(values)[order], np.asarray(weights)[order]
    cdf = (np.cumsum(w) - 0.5 * w) / w.sum()
    return [float(np.interp(q, cdf, v)) for q in qs]


def _simulate(args: tuple[SimConfig, list[np.random.SeedSequence], str, list[str]]) -> np.ndarray:
    """Average summary statistics over replicates for one parameter draw."""
    config, seeds, backend, stats = args
    out = np.zeros(len(stats))
    for s in seeds:
        summary = Simulation(config, seed=s, backend=backend).run().summary()
        out += [summary[k] for k in stats]
    return out / len(seeds)


class _Evaluator:
    def __init__(self, base, names, stats, reps, seed, backend, pool: Executor | None):
        self.base, self.names, self.stats = base, names, stats
        self.reps, self.backend, self.pool = reps, backend, pool
        self.seeds = np.random.SeedSequence(seed)
        self.n_sims = 0

    def __call__(self, thetas: np.ndarray) -> np.ndarray:
        jobs = []
        for theta in thetas:
            cfg = self.base.with_overrides(**dict(zip(self.names, map(float, theta))))
            jobs.append((cfg, self.seeds.spawn(self.reps), self.backend, self.stats))
        self.n_sims += len(jobs) * self.reps
        if self.pool is None:
            return np.array([_simulate(j) for j in jobs])
        return np.array(list(self.pool.map(_simulate, jobs)))


def abc_smc(
    base: SimConfig,
    priors: dict[str, tuple[float, float]],
    targets: dict[str, float],
    n_particles: int = 200,
    generations: int = 5,
    reps: int = 1,
    quantile: float = 0.5,
    seed: int = 0,
    workers: int = 1,
    backend: str = "numpy",
    max_sims: int = 200_000,
    progress=None,
) -> CalibrationResult:
    """Infer the parameters named in ``priors`` from observed ``targets``.

    ``targets`` maps summary statistics (keys of ``SimResult.summary()``, such
    as ``mortality`` or ``enrollment``) to observed values.
    """
    _validate(base, priors, targets, n_particles, generations, quantile)
    names, stats = list(priors), list(targets)
    lo = np.array([priors[k][0] for k in names], dtype=float)
    hi = np.array([priors[k][1] for k in names], dtype=float)
    target = np.array([targets[k] for k in stats], dtype=float)
    rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])

    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        evaluate = _Evaluator(base, names, stats, reps, seed, backend, pool)

        # Generation 0: sample the prior. Its spread sets the distance scale.
        theta = rng.uniform(lo, hi, size=(n_particles, len(names)))
        sims = evaluate(theta)
        scale = sims.std(axis=0)
        scale[scale == 0] = 1.0

        def distance(s):
            return np.sqrt((((s - target) / scale) ** 2).sum(axis=-1))

        dist = distance(sims)
        weights = np.full(n_particles, 1.0 / n_particles)
        history = [_record(0, np.inf, 1.0, evaluate.n_sims, theta, weights, names)]
        if progress:
            progress(history[-1])

        for g in range(1, generations + 1):
            eps = float(np.quantile(dist, quantile))
            cov = 2.0 * np.atleast_2d(np.cov(theta.T, aweights=weights))
            cov += np.eye(len(names)) * 1e-12  # keep it positive definite
            kernel_norm = multivariate_normal(mean=np.zeros(len(names)), cov=cov)

            new_theta, new_dist, proposed = [], [], 0
            while len(new_theta) < n_particles:
                if evaluate.n_sims >= max_sims:
                    raise RuntimeError(
                        f"stopped after {evaluate.n_sims} simulations in generation {g}; "
                        "loosen the quantile, widen priors, or raise max_sims"
                    )
                need = n_particles - len(new_theta)
                rate = len(new_theta) / proposed if proposed else 0.3
                batch = int(min(max(need / max(rate, 0.02), 16), 20 * n_particles))
                parents = rng.choice(n_particles, size=batch, p=weights)
                cand = theta[parents] + rng.multivariate_normal(
                    np.zeros(len(names)), cov, size=batch
                )
                cand = cand[((cand >= lo) & (cand <= hi)).all(axis=1)]  # prior support
                if cand.size == 0:
                    continue
                proposed += len(cand)
                d = distance(evaluate(cand))
                keep = d <= eps
                new_theta.extend(cand[keep])
                new_dist.extend(d[keep])

            new_theta = np.array(new_theta[:n_particles])
            new_dist = np.array(new_dist[:n_particles])
            # Importance weights: uniform prior density over the proposal mixture.
            dens = np.array(
                [np.dot(weights, kernel_norm.pdf(t - theta)) for t in new_theta]
            )
            new_w = 1.0 / dens
            theta, dist, weights = new_theta, new_dist, new_w / new_w.sum()
            history.append(
                _record(g, eps, n_particles / proposed, evaluate.n_sims, theta, weights, names)
            )
            if progress:
                progress(history[-1])
    finally:
        if pool is not None:
            pool.shutdown()

    return CalibrationResult(names, theta, weights, dist, history, dict(priors), dict(targets))


def _record(g, eps, acceptance, n_sims, theta, weights, names) -> dict[str, float]:
    ess = 1.0 / np.sum(weights**2)  # effective sample size
    row = {"generation": g, "epsilon": eps, "acceptance": acceptance,
           "simulations": n_sims, "ess": float(ess)}  # fmt: skip
    for j, name in enumerate(names):
        row[f"mean_{name}"] = float(np.average(theta[:, j], weights=weights))
    return row


def _validate(base, priors, targets, n_particles, generations, quantile) -> None:
    numeric = {f.name for f in fields(SimConfig)} - {"sites", "width", "height", "n_agents", "n_steps"}
    if not priors:
        raise ValueError("at least one prior is required")
    for name, bounds in priors.items():
        if name not in numeric:
            raise ValueError(f"cannot calibrate {name!r}; choose from {sorted(numeric)}")
        low, high = bounds
        if not low < high:
            raise ValueError(f"prior for {name!r} needs low < high, got {bounds}")
        base.with_overrides(**{name: low}).with_overrides(**{name: high})  # range is legal
    if not targets:
        raise ValueError("at least one target statistic is required")
    allowed = set(Simulation(base.with_overrides(n_agents=1, n_steps=1)).run().summary())
    unknown = set(targets) - allowed
    if unknown:
        raise ValueError(f"unknown target statistics {sorted(unknown)}; choose from {sorted(allowed)}")
    if n_particles < 10:
        raise ValueError("use at least 10 particles")
    if generations < 1:
        raise ValueError("generations must be at least 1")
    if not 0 < quantile < 1:
        raise ValueError("quantile must be in (0, 1)")


def load_spec(spec: dict[str, Any]) -> tuple[dict[str, tuple[float, float]], dict[str, float]]:
    priors = {k: (float(v[0]), float(v[1])) for k, v in spec["priors"].items()}
    targets = {k: float(v) for k, v in spec["targets"].items()}
    return priors, targets
