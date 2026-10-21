"""Numba backend: the model's rules compiled to machine code and run across
CPU cores with ``prange``.

Each kernel only writes to the slot of the agent it is processing, so the
parallel loops are race-free and results are deterministic regardless of the
number of threads. Occupancy counting is a scatter (many agents can hit the same
cell), so it runs serially; it is a single cheap O(N) pass.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit, prange
except ImportError as exc:  # pragma: no cover - exercised only without numba
    raise ImportError(
        "the 'numba' backend needs numba: pip install 'odsim[fast]'"
    ) from exc


@njit(parallel=True, cache=True)
def _move_and_treat(
    x, y, alive, housed, in_treatment, enroll_time,
    move_u, rdx, rdy, enroll_u, dropout_u,
    nearest_x, nearest_y, dist_treatment, has_sites,
    p_seek, width, height, enroll_radius, p_enroll, unhoused_enroll_mult, p_dropout, t,
):  # fmt: skip
    for i in prange(x.shape[0]):
        if not alive[i]:
            continue
        if in_treatment[i]:
            if dropout_u[i] < p_dropout:
                in_treatment[i] = False
            continue
        cx, cy = x[i], y[i]
        if has_sites and move_u[i] < p_seek:
            dx = np.sign(nearest_x[cx, cy] - cx)
            dy = np.sign(nearest_y[cx, cy] - cy)
        else:
            dx, dy = rdx[i], rdy[i]
        cx = min(max(cx + dx, 0), width - 1)
        cy = min(max(cy + dy, 0), height - 1)
        x[i], y[i] = cx, cy
        p = p_enroll * (1.0 if housed[i] else unhoused_enroll_mult)
        if dist_treatment[cx, cy] <= enroll_radius and enroll_u[i] < p:
            in_treatment[i] = True
            if enroll_time[i] < 0:
                enroll_time[i] = t


@njit(cache=True)
def _occupancy(x, y, alive, width, height):
    counts = np.zeros((width, height), dtype=np.int64)
    for i in range(x.shape[0]):
        if alive[i]:
            counts[x[i], y[i]] += 1
    return counts


@njit(parallel=True, cache=True)
def _overdoses(
    x, y, alive, housed, in_treatment, death_time, od_u, fatal_u, counts, dist_ssp,
    use_bystanders, p_overdose, treatment_od_mult, unhoused_od_mult,
    p_fatal, naloxone_radius, naloxone_effect, bystander_effect, width, height, t,
):  # fmt: skip
    n = x.shape[0]
    overdosed = np.zeros(n, dtype=np.bool_)
    for i in prange(n):
        if not alive[i]:
            continue
        p_od = p_overdose
        p_od *= treatment_od_mult if in_treatment[i] else 1.0
        p_od *= 1.0 if housed[i] else unhoused_od_mult
        if od_u[i] >= min(p_od, 1.0):
            continue
        overdosed[i] = True
        cx, cy = x[i], y[i]
        witnessed = False
        if use_bystanders:
            company = 0
            for ox in range(max(cx - 1, 0), min(cx + 2, width)):
                for oy in range(max(cy - 1, 0), min(cy + 2, height)):
                    company += counts[ox, oy]
            witnessed = company >= 2
        pf = p_fatal
        pf *= (1.0 - naloxone_effect) if dist_ssp[cx, cy] <= naloxone_radius else 1.0
        pf *= (1.0 - bystander_effect) if witnessed else 1.0
        if fatal_u[i] < pf:
            alive[i] = False
            death_time[i] = t
    return overdosed.sum()


def numba_step(sim, s, d, t: int) -> int:
    c = sim.config
    _move_and_treat(
        s.x, s.y, s.alive, s.housed, s.in_treatment, s.enroll_time,
        d.move_u, d.dx, d.dy, d.enroll_u, d.dropout_u,
        sim.nearest_x, sim.nearest_y, sim.dist_treatment, sim.has_sites,
        c.p_seek, c.width, c.height, c.enroll_radius, c.p_enroll,
        c.unhoused_enroll_mult, c.p_dropout, t,
    )  # fmt: skip
    use_bystanders = c.bystander_effect > 0
    counts = (
        _occupancy(s.x, s.y, s.alive, c.width, c.height)
        if use_bystanders
        else np.zeros((1, 1), dtype=np.int64)
    )
    return int(
        _overdoses(
            s.x, s.y, s.alive, s.housed, s.in_treatment, s.death_time, d.od_u, d.fatal_u,
            counts, sim.dist_ssp, use_bystanders,
            c.p_overdose, c.treatment_od_mult, c.unhoused_od_mult,
            c.p_fatal, c.naloxone_radius, c.naloxone_effect, c.bystander_effect,
            c.width, c.height, t,
        )  # fmt: skip
    )
