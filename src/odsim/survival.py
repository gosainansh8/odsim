"""Survival analysis implemented from scratch: Kaplan-Meier and the log-rank test."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2


@dataclass(frozen=True)
class KaplanMeier:
    times: np.ndarray  # distinct event times
    survival: np.ndarray  # S(t) just after each event time
    at_risk: np.ndarray
    deaths: np.ndarray

    def survival_at(self, t: float) -> float:
        """Step-function value S(t)."""
        i = np.searchsorted(self.times, t, side="right")
        return 1.0 if i == 0 else float(self.survival[i - 1])

    def median(self) -> float:
        below = np.nonzero(self.survival <= 0.5)[0]
        return float(self.times[below[0]]) if below.size else float("inf")


def _validate(durations: np.ndarray, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events, dtype=bool)
    if durations.shape != events.shape or durations.ndim != 1:
        raise ValueError("durations and events must be 1-D arrays of equal length")
    if (durations < 0).any():
        raise ValueError("durations must be non-negative")
    return durations, events


def _risk_table(durations: np.ndarray, events: np.ndarray, times: np.ndarray):
    """Number at risk and number of deaths at each time in ``times``."""
    sorted_d = np.sort(durations)
    at_risk = durations.size - np.searchsorted(sorted_d, times, side="left")
    event_times = np.sort(durations[events])
    deaths = np.searchsorted(event_times, times, side="right") - np.searchsorted(
        event_times, times, side="left"
    )
    return at_risk, deaths


def kaplan_meier(durations, events) -> KaplanMeier:
    durations, events = _validate(durations, events)
    times = np.unique(durations[events])
    at_risk, deaths = _risk_table(durations, events, times)
    survival = np.cumprod(1.0 - deaths / at_risk)
    return KaplanMeier(times=times, survival=survival, at_risk=at_risk, deaths=deaths)


@dataclass(frozen=True)
class LogRankResult:
    statistic: float
    p_value: float
    observed_a: float
    expected_a: float


def logrank_test(durations_a, events_a, durations_b, events_b) -> LogRankResult:
    """Two-sample log-rank test (chi-squared, 1 degree of freedom)."""
    da, ea = _validate(durations_a, events_a)
    db, eb = _validate(durations_b, events_b)
    times = np.unique(np.concatenate([da[ea], db[eb]]))
    if times.size == 0:
        return LogRankResult(0.0, 1.0, 0.0, 0.0)

    n_a, d_a = _risk_table(da, ea, times)
    n_b, d_b = _risk_table(db, eb, times)
    n = n_a + n_b
    d = d_a + d_b

    expected_a = d * n_a / n
    with np.errstate(divide="ignore", invalid="ignore"):
        var = np.where(n > 1, d * (n_a / n) * (n_b / n) * (n - d) / (n - 1), 0.0)

    o, e, v = d_a.sum(), expected_a.sum(), var.sum()
    stat = float((o - e) ** 2 / v) if v > 0 else 0.0
    return LogRankResult(stat, float(chi2.sf(stat, df=1)), float(o), float(e))
