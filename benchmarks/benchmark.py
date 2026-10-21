"""Benchmark the three implementations of one simulation step.

    python benchmarks/benchmark.py

Times are per step with agent-agent interactions on. "step" excludes drawing
random numbers (identical for every backend); "end to end" includes it.
The per-agent loop is only timed up to 10,000 agents.
"""

import os
import time

import numpy as np

from odsim import SimConfig, Simulation
from odsim.model import StepDraws, reference_step

STEPS = 10
HERE = os.path.dirname(os.path.abspath(__file__))


def time_step(config: SimConfig, impl: str) -> tuple[float, float]:
    backend = "numba" if impl == "numba" else "numpy"
    sim = Simulation(config, seed=0, backend=backend)
    rng = np.random.default_rng(1)
    if impl == "numba":  # compile outside the timed region
        warm = Simulation(config.with_overrides(n_agents=10), seed=0, backend="numba")
        warm.step()
    draws, draw_time = [], 0.0
    for _ in range(STEPS):
        t0 = time.perf_counter()
        draws.append(StepDraws.sample(rng, config.n_agents))
        draw_time += time.perf_counter() - t0
    start = time.perf_counter()
    for t, d in enumerate(draws):
        if impl == "loop":
            reference_step(sim, sim.state, d, t)
        else:
            sim.step(d)
    step = (time.perf_counter() - start) / STEPS
    return step, step + draw_time / STEPS


def main() -> None:
    base = SimConfig.from_json(os.path.join(HERE, "..", "examples", "config.json"))
    base = base.with_overrides(width=500, height=500, n_steps=STEPS)
    print(f"{'agents':>10} {'impl':>6} {'step ms':>10} {'end-to-end ms':>15} {'vs loop':>9}")
    for n in (1_000, 10_000, 100_000, 1_000_000):
        cfg = base.with_overrides(n_agents=n)
        loop = time_step(cfg, "loop")[0] if n <= 10_000 else None
        for impl in ("loop", "numpy", "numba"):
            if impl == "loop" and loop is None:
                continue
            step, e2e = (loop, None) if impl == "loop" else time_step(cfg, impl)
            speed = f"{loop / step:>8.0f}x" if loop else f"{'':>9}"
            e2e_s = f"{e2e * 1e3:>15.2f}" if e2e else f"{'':>15}"
            print(f"{n:>10,} {impl:>6} {step * 1e3:>10.2f} {e2e_s} {speed}")


if __name__ == "__main__":
    main()
