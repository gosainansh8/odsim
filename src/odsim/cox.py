"""Cox proportional hazards regression, implemented from scratch.

Fits log hazard ratios by maximizing the partial likelihood with Newton-Raphson.
Tied event times use Efron's approximation (the same default as lifelines,
R's ``survival::coxph`` and Stata), so results match those packages.

Risk-set sums are computed with reverse cumulative sums over agents sorted by
time, so each Newton iteration costs O(n p^2 + m d p^2) for n agents, p
covariates and m distinct event times with d tied deaths each, rather than
O(n m p^2) for a naive loop over risk sets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass(frozen=True)
class CoxResult:
    names: list[str]
    coef: np.ndarray  # log hazard ratios
    se: np.ndarray
    log_likelihood: float
    n: int
    n_events: int
    iterations: int

    @property
    def hazard_ratio(self) -> np.ndarray:
        return np.exp(self.coef)

    @property
    def z(self) -> np.ndarray:
        return self.coef / self.se

    @property
    def p_value(self) -> np.ndarray:
        return 2 * norm.sf(np.abs(self.z))

    def confidence_interval(self, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
        """Confidence interval for the hazard ratio."""
        q = norm.ppf(0.5 + level / 2)
        return np.exp(self.coef - q * self.se), np.exp(self.coef + q * self.se)

    def to_frame(self, level: float = 0.95) -> pd.DataFrame:
        lo, hi = self.confidence_interval(level)
        return pd.DataFrame(
            {
                "covariate": self.names,
                "coef": self.coef,
                "hazard_ratio": self.hazard_ratio,
                "ci_low": lo,
                "ci_high": hi,
                "se": self.se,
                "z": self.z,
                "p": self.p_value,
            }
        )


def _efron_terms(beta, X, durations, events, order, times, starts, ends, risk_index):
    """Log partial likelihood, gradient and Hessian at ``beta``."""
    p = X.shape[1]
    eta = X @ beta
    w = np.exp(eta)

    # Reverse cumulative sums over agents sorted by descending time give the
    # risk-set totals S0 = sum w, S1 = sum w x, S2 = sum w x x^T.
    Xs, ws = X[order], w[order]
    S0_all = np.cumsum(ws)
    S1_all = np.cumsum(ws[:, None] * Xs, axis=0)
    S2_all = np.cumsum(ws[:, None, None] * (Xs[:, :, None] * Xs[:, None, :]), axis=0)

    loglik, grad, hess = 0.0, np.zeros(p), np.zeros((p, p))
    dead_order = np.argsort(durations[events], kind="stable")
    Xd, wd, etad = X[events][dead_order], w[events][dead_order], eta[events][dead_order]

    for j in range(times.size):
        k = risk_index[j]  # last sorted position still at risk at times[j]
        S0, S1, S2 = S0_all[k], S1_all[k], S2_all[k]
        sl = slice(starts[j], ends[j])
        d = ends[j] - starts[j]
        Xj, wj = Xd[sl], wd[sl]
        D0, D1 = wj.sum(), wj @ Xj
        D2 = np.einsum("i,ij,ik->jk", wj, Xj, Xj)

        loglik += etad[sl].sum()
        grad += Xj.sum(axis=0)
        for l in range(d):
            frac = l / d
            phi = S0 - frac * D0
            mu = (S1 - frac * D1) / phi
            loglik -= np.log(phi)
            grad -= mu
            hess -= (S2 - frac * D2) / phi - np.outer(mu, mu)
    return loglik, grad, hess


def cox_ph(
    durations,
    events,
    covariates,
    names: list[str] | None = None,
    max_iter: int = 50,
    tol: float = 1e-9,
) -> CoxResult:
    """Fit a Cox model. ``covariates`` is an (n, p) array or a DataFrame."""
    if isinstance(covariates, pd.DataFrame):
        names = names or list(covariates.columns)
        covariates = covariates.to_numpy(dtype=float)
    X = np.asarray(covariates, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events, dtype=bool)
    n, p = X.shape
    if durations.shape != (n,) or events.shape != (n,):
        raise ValueError("durations, events and covariates must have the same length")
    if not events.any():
        raise ValueError("cannot fit a Cox model with no events")
    names = names or [f"x{i}" for i in range(p)]
    if (X.std(axis=0) == 0).any():
        raise ValueError("a covariate is constant; its effect is not identifiable")

    # Centering does not change the coefficients but keeps exp(X beta) well scaled.
    X = X - X.mean(axis=0)

    order = np.argsort(-durations, kind="stable")
    sorted_desc = durations[order]
    times, counts = np.unique(durations[events], return_counts=True)
    ends = np.cumsum(counts)
    starts = ends - counts
    # Agents at risk at time t are those with duration >= t: a prefix of the
    # descending order. risk_index[j] is the last index of that prefix.
    risk_index = np.searchsorted(-sorted_desc, -times, side="right") - 1

    beta = np.zeros(p)
    args = (X, durations, events, order, times, starts, ends, risk_index)
    loglik, grad, hess = _efron_terms(beta, *args)
    for it in range(1, max_iter + 1):
        step = np.linalg.solve(-hess, grad)
        # Step halving guards against overshooting on hard problems.
        for _ in range(30):
            new_beta = beta + step
            new_ll, new_grad, new_hess = _efron_terms(new_beta, *args)
            if np.isfinite(new_ll) and new_ll >= loglik - 1e-12:
                break
            step /= 2
        converged = abs(new_ll - loglik) < tol * (1 + abs(loglik))
        beta, loglik, grad, hess = new_beta, new_ll, new_grad, new_hess
        if converged:
            break

    se = np.sqrt(np.diag(np.linalg.inv(-hess)))
    return CoxResult(names, beta, se, float(loglik), n, int(events.sum()), it)
