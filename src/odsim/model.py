"""The agent-based model.

Agent state is stored as a struct of arrays (one NumPy array per attribute, one
element per agent). A step runs in three phases:

1. move and update treatment status,
2. hash every living agent into its grid cell and count neighbours,
3. resolve overdoses, where a witnessed overdose (another agent in the same or
   an adjacent cell) is less likely to be fatal.

Three interchangeable implementations of the same rules exist:

* ``numpy``  vectorized array operations (default)
* ``numba``  JIT-compiled, multi-threaded kernels (``pip install odsim[fast]``)
* ``reference_step``  a plain per-agent Python loop, used only to prove the
  other two are correct and to benchmark them

All randomness for a step is drawn up front (``StepDraws``), so the three can be
compared draw for draw; the tests require identical results.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SimConfig

BACKENDS = ("numpy", "numba")


@dataclass
class AgentState:
    x: np.ndarray
    y: np.ndarray
    alive: np.ndarray
    housed: np.ndarray
    in_treatment: np.ndarray
    death_time: np.ndarray  # step of death, -1 if alive
    enroll_time: np.ndarray  # step of first enrollment, -1 if never

    def copy(self) -> AgentState:
        return AgentState(**{k: v.copy() for k, v in self.__dict__.items()})


@dataclass(frozen=True)
class StepDraws:
    """Every random number one step needs, drawn up front in a fixed order."""

    move_u: np.ndarray
    dx: np.ndarray
    dy: np.ndarray
    enroll_u: np.ndarray
    dropout_u: np.ndarray
    od_u: np.ndarray
    fatal_u: np.ndarray

    @classmethod
    def sample(cls, rng: np.random.Generator, n: int) -> StepDraws:
        return cls(
            move_u=rng.random(n),
            dx=rng.integers(-1, 2, n),
            dy=rng.integers(-1, 2, n),
            enroll_u=rng.random(n),
            dropout_u=rng.random(n),
            od_u=rng.random(n),
            fatal_u=rng.random(n),
        )


@dataclass
class SimResult:
    """Per-agent outcomes of one run, in a form ready for survival analysis."""

    durations: np.ndarray  # steps survived (event time, or n_steps if censored)
    events: np.ndarray  # True if the agent died
    housed: np.ndarray
    enrolled: np.ndarray  # ever enrolled in treatment
    n_overdoses: int
    n_steps: int

    def summary(self) -> dict[str, float]:
        def rate(mask: np.ndarray, values: np.ndarray) -> float:
            return float(values[mask].mean()) if mask.any() else float("nan")

        everyone = np.ones_like(self.events)
        return {
            "mortality": rate(everyone, self.events),
            "enrollment": rate(everyone, self.enrolled),
            "overdoses": float(self.n_overdoses),
            "overdoses_per_agent": self.n_overdoses / max(self.events.size, 1),
            "mortality_housed": rate(self.housed, self.events),
            "mortality_unhoused": rate(~self.housed, self.events),
            "enrollment_housed": rate(self.housed, self.enrolled),
            "enrollment_unhoused": rate(~self.housed, self.enrolled),
        }


class Simulation:
    def __init__(
        self,
        config: SimConfig,
        seed: int | np.random.SeedSequence | None = None,
        backend: str = "numpy",
    ):
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
        if backend == "numba":
            from . import _numba_kernels  # noqa: F401  (fails clearly if numba is missing)
        self.config = config
        self.backend = backend
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.n_overdoses = 0
        self._precompute_site_fields()
        self.state = self._init_agents()

    # ---- setup ----------------------------------------------------------------

    def _precompute_site_fields(self) -> None:
        """For every grid cell: distance to the nearest treatment site, distance
        to the nearest SSP, and coordinates of the nearest site of any kind.
        Agents then look these up in O(1) instead of searching the site list."""
        c = self.config
        gx, gy = np.meshgrid(np.arange(c.width), np.arange(c.height), indexing="ij")

        def nearest(sites):
            if not sites:
                zeros = np.zeros((c.width, c.height), dtype=np.int64)
                return np.full((c.width, c.height), np.inf), zeros, zeros.copy()
            sx = np.array([s.x for s in sites], dtype=np.int64)
            sy = np.array([s.y for s in sites], dtype=np.int64)
            d = np.hypot(gx[..., None] - sx, gy[..., None] - sy)  # (W, H, S)
            idx = d.argmin(axis=-1)
            return d.min(axis=-1), sx[idx], sy[idx]

        self.dist_treatment, _, _ = nearest(c.sites_of_kind("treatment"))
        self.dist_ssp, _, _ = nearest(c.sites_of_kind("ssp"))
        _, self.nearest_x, self.nearest_y = nearest(list(c.sites))
        self.has_sites = bool(c.sites)

    def _init_agents(self) -> AgentState:
        c, n = self.config, self.config.n_agents
        return AgentState(
            x=self.rng.integers(0, c.width, n),
            y=self.rng.integers(0, c.height, n),
            alive=np.ones(n, dtype=bool),
            housed=self.rng.random(n) < c.frac_housed,
            in_treatment=np.zeros(n, dtype=bool),
            death_time=np.full(n, -1, dtype=np.int64),
            enroll_time=np.full(n, -1, dtype=np.int64),
        )

    # ---- dynamics -------------------------------------------------------------

    def step(self, draws: StepDraws | None = None) -> None:
        if draws is None:
            draws = StepDraws.sample(self.rng, self.config.n_agents)
        if self.backend == "numba":
            from ._numba_kernels import numba_step

            self.n_overdoses += numba_step(self, self.state, draws, self.t)
        else:
            self.n_overdoses += numpy_step(self, self.state, draws, self.t)
        self.t += 1

    def run(self) -> SimResult:
        while self.t < self.config.n_steps:
            self.step()
        return self.result()

    def result(self) -> SimResult:
        s = self.state
        events = ~s.alive
        durations = np.where(events, s.death_time + 1, self.t)
        return SimResult(
            durations=durations,
            events=events,
            housed=s.housed.copy(),
            enrolled=s.enroll_time >= 0,
            n_overdoses=self.n_overdoses,
            n_steps=self.t,
        )


# ---------------------------------------------------------------------------
# Spatial hashing
# ---------------------------------------------------------------------------


def neighbourhood_counts(x: np.ndarray, y: np.ndarray, alive: np.ndarray, w: int, h: int):
    """Count living agents in each cell's 3x3 neighbourhood.

    Agents are hashed to cells with one ``bincount`` (O(N)), then the 3x3 window
    sums are nine shifted adds over the grid (O(cells)). Total cost is
    O(N + cells) instead of the O(N^2) of comparing every pair of agents.
    """
    cell = x[alive] * h + y[alive]
    counts = np.bincount(cell, minlength=w * h).reshape(w, h)
    padded = np.pad(counts, 1)
    window = np.zeros_like(counts)
    for ox in range(3):
        for oy in range(3):
            window += padded[ox : ox + w, oy : oy + h]
    return window


# ---------------------------------------------------------------------------
# NumPy backend
# ---------------------------------------------------------------------------


def numpy_step(sim: Simulation, s: AgentState, d: StepDraws, t: int) -> int:
    c = sim.config
    active = s.alive & ~s.in_treatment

    # Phase 1a: movement. Active agents head toward the nearest site or wander.
    if sim.has_sites:
        seek = d.move_u < c.p_seek
        dx = np.where(seek, np.sign(sim.nearest_x[s.x, s.y] - s.x), d.dx)
        dy = np.where(seek, np.sign(sim.nearest_y[s.x, s.y] - s.y), d.dy)
    else:
        dx, dy = d.dx, d.dy
    s.x = np.clip(s.x + np.where(active, dx, 0), 0, c.width - 1)
    s.y = np.clip(s.y + np.where(active, dy, 0), 0, c.height - 1)

    # Phase 1b: treatment enrollment and dropout.
    near_treatment = sim.dist_treatment[s.x, s.y] <= c.enroll_radius
    p_enroll = c.p_enroll * np.where(s.housed, 1.0, c.unhoused_enroll_mult)
    enroll = active & near_treatment & (d.enroll_u < p_enroll)
    dropout = s.alive & s.in_treatment & (d.dropout_u < c.p_dropout)
    s.in_treatment = (s.in_treatment & ~dropout) | enroll
    first = enroll & (s.enroll_time < 0)
    s.enroll_time[first] = t

    # Phase 2: who has company? (count includes the agent itself)
    if c.bystander_effect > 0:
        window = neighbourhood_counts(s.x, s.y, s.alive, c.width, c.height)
        witnessed = window[s.x, s.y] >= 2
    else:
        witnessed = np.zeros(c.n_agents, dtype=bool)

    # Phase 3: overdoses, and death unless naloxone or a bystander intervenes.
    p_od = (
        c.p_overdose
        * np.where(s.in_treatment, c.treatment_od_mult, 1.0)
        * np.where(s.housed, 1.0, c.unhoused_od_mult)
    )
    overdose = s.alive & (d.od_u < np.minimum(p_od, 1.0))
    covered = sim.dist_ssp[s.x, s.y] <= c.naloxone_radius
    p_fatal = (
        c.p_fatal
        * np.where(covered, 1.0 - c.naloxone_effect, 1.0)
        * np.where(witnessed, 1.0 - c.bystander_effect, 1.0)
    )
    dies = overdose & (d.fatal_u < p_fatal)
    s.alive = s.alive & ~dies
    s.death_time[dies] = t
    return int(overdose.sum())


# ---------------------------------------------------------------------------
# Reference implementation (correctness oracle)
# ---------------------------------------------------------------------------


def reference_step(sim: Simulation, s: AgentState, d: StepDraws, t: int) -> int:
    """The same rules, one agent at a time. Slow on purpose: it is the
    readable specification the fast backends are tested against."""
    c = sim.config
    n = c.n_agents

    # Phase 1: movement and treatment.
    for i in range(n):
        if not s.alive[i]:
            continue
        x, y = int(s.x[i]), int(s.y[i])
        if not s.in_treatment[i]:
            if sim.has_sites and d.move_u[i] < c.p_seek:
                dx = int(np.sign(sim.nearest_x[x, y] - x))
                dy = int(np.sign(sim.nearest_y[x, y] - y))
            else:
                dx, dy = int(d.dx[i]), int(d.dy[i])
            x = min(max(x + dx, 0), c.width - 1)
            y = min(max(y + dy, 0), c.height - 1)
            s.x[i], s.y[i] = x, y
            p_enroll = c.p_enroll * (1.0 if s.housed[i] else c.unhoused_enroll_mult)
            if sim.dist_treatment[x, y] <= c.enroll_radius and d.enroll_u[i] < p_enroll:
                s.in_treatment[i] = True
                if s.enroll_time[i] < 0:
                    s.enroll_time[i] = t
        elif d.dropout_u[i] < c.p_dropout:
            s.in_treatment[i] = False

    # Phase 2: occupancy of every cell.
    occupancy: dict[tuple[int, int], int] = {}
    for i in range(n):
        if s.alive[i]:
            key = (int(s.x[i]), int(s.y[i]))
            occupancy[key] = occupancy.get(key, 0) + 1

    # Phase 3: overdoses.
    overdoses = 0
    for i in range(n):
        if not s.alive[i]:
            continue
        x, y = int(s.x[i]), int(s.y[i])
        p_od = c.p_overdose
        p_od *= c.treatment_od_mult if s.in_treatment[i] else 1.0
        p_od *= 1.0 if s.housed[i] else c.unhoused_od_mult
        if d.od_u[i] < min(p_od, 1.0):
            overdoses += 1
            witnessed = False
            if c.bystander_effect > 0:
                company = sum(
                    occupancy.get((x + ox, y + oy), 0) for ox in (-1, 0, 1) for oy in (-1, 0, 1)
                )
                witnessed = company >= 2
            p_fatal = c.p_fatal
            p_fatal *= (1.0 - c.naloxone_effect) if sim.dist_ssp[x, y] <= c.naloxone_radius else 1.0
            p_fatal *= (1.0 - c.bystander_effect) if witnessed else 1.0
            if d.fatal_u[i] < p_fatal:
                s.alive[i] = False
                s.death_time[i] = t
    return overdoses
