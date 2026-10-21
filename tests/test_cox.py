import numpy as np
import pandas as pd
import pytest

from odsim import cox_ph


def simulated_data(n=2000, seed=0, tied=True):
    rng = np.random.default_rng(seed)
    X = np.column_stack([rng.random(n) < 0.5, rng.normal(size=n)]).astype(float)
    hazard = np.exp(X @ np.array([0.7, -0.4]))
    T = rng.exponential(1 / hazard) * 20
    if tied:
        T = np.ceil(T)  # integer days: many tied event times, like the simulation
    C = rng.uniform(5, 60, n)
    return np.minimum(T, C), T <= C, X


def test_recovers_true_coefficients():
    d, e, X = simulated_data(n=20_000, tied=False)
    fit = cox_ph(d, e, X, ["a", "b"])
    np.testing.assert_allclose(fit.coef, [0.7, -0.4], atol=0.05)
    lo, hi = fit.confidence_interval()
    assert (lo < np.exp([0.7, -0.4])).all() and (np.exp([0.7, -0.4]) < hi).all()


def test_dataframe_input_and_frame_output():
    d, e, X = simulated_data()
    fit = cox_ph(d, e, pd.DataFrame(X, columns=["housed", "risk"]))
    frame = fit.to_frame()
    assert list(frame["covariate"]) == ["housed", "risk"]
    assert {"hazard_ratio", "ci_low", "ci_high", "p"} <= set(frame.columns)


def test_null_effect_not_significant():
    rng = np.random.default_rng(3)
    n = 3000
    fit = cox_ph(rng.integers(1, 50, n), rng.random(n) < 0.6, rng.random(n) < 0.5)
    assert fit.p_value[0] > 0.01


def test_invalid_inputs():
    with pytest.raises(ValueError):
        cox_ph([1, 2, 3], [0, 0, 0], [[0], [1], [0]])  # no events
    with pytest.raises(ValueError):
        cox_ph([1, 2, 3], [1, 1, 1], [[1], [1], [1]])  # constant covariate
    with pytest.raises(ValueError):
        cox_ph([1, 2], [1, 1], [[0], [1], [0]])  # length mismatch


@pytest.mark.parametrize("tied", [True, False])
def test_matches_lifelines(tied):
    lifelines = pytest.importorskip("lifelines")
    d, e, X = simulated_data(tied=tied)
    ours = cox_ph(d, e, X, ["a", "b"])
    df = pd.DataFrame(X, columns=["a", "b"]).assign(T=d, E=e)
    ref = lifelines.CoxPHFitter().fit(df, "T", "E")
    np.testing.assert_allclose(ours.coef, ref.params_.to_numpy(), rtol=1e-4)
    np.testing.assert_allclose(ours.se, ref.standard_errors_.to_numpy(), rtol=1e-4)
    assert ours.log_likelihood == pytest.approx(ref.log_likelihood_, rel=1e-6)
