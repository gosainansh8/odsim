import numpy as np
import pytest

from odsim import kaplan_meier, logrank_test


def test_kaplan_meier_hand_computed():
    # t=1: 5 at risk, 1 death -> 0.8
    # t=2: 4 at risk, 1 death (+1 censored) -> 0.6
    # t=3: 2 at risk, 1 death -> 0.3
    km = kaplan_meier([1, 2, 2, 3, 4], [1, 1, 0, 1, 0])
    np.testing.assert_array_equal(km.times, [1, 2, 3])
    np.testing.assert_array_equal(km.at_risk, [5, 4, 2])
    np.testing.assert_allclose(km.survival, [0.8, 0.6, 0.3])
    assert km.survival_at(0.5) == 1.0
    assert km.survival_at(2.5) == pytest.approx(0.6)
    assert km.median() == 3


def test_kaplan_meier_no_events():
    km = kaplan_meier([3, 4, 5], [0, 0, 0])
    assert km.times.size == 0
    assert km.survival_at(10) == 1.0
    assert km.median() == float("inf")


def test_logrank_hand_computed():
    # Group A dies at 1 and 2; group B at 3 and 4.
    # O_A - E_A = 2 - 5/6 = 7/6, variance = 1/4 + 2/9 = 17/36 -> statistic 49/17.
    res = logrank_test([1, 2], [1, 1], [3, 4], [1, 1])
    assert res.statistic == pytest.approx(49 / 17)
    assert res.observed_a == 2
    assert res.expected_a == pytest.approx(5 / 6)


def test_logrank_identical_groups():
    d, e = [1, 2, 3, 4], [1, 0, 1, 1]
    res = logrank_test(d, e, d, e)
    assert res.statistic == pytest.approx(0.0)
    assert res.p_value == pytest.approx(1.0)


def test_logrank_detects_real_difference():
    rng = np.random.default_rng(0)
    a = rng.exponential(10, 400)
    b = rng.exponential(20, 400)
    res = logrank_test(a, np.ones(400), b, np.ones(400))
    assert res.p_value < 1e-6


def test_invalid_inputs():
    with pytest.raises(ValueError):
        kaplan_meier([1, 2], [1])
    with pytest.raises(ValueError):
        kaplan_meier([-1, 2], [1, 1])


def test_matches_lifelines():
    lifelines = pytest.importorskip("lifelines")
    from lifelines.statistics import logrank_test as ll_logrank

    rng = np.random.default_rng(1)
    da, db = rng.integers(1, 50, 200), rng.integers(1, 60, 200)
    ea, eb = rng.random(200) < 0.7, rng.random(200) < 0.6

    km = kaplan_meier(da, ea)
    ref = lifelines.KaplanMeierFitter().fit(da, ea)
    for t in km.times:
        assert km.survival_at(t) == pytest.approx(float(ref.predict(t)))

    ours = logrank_test(da, ea, db, eb)
    theirs = ll_logrank(da, db, event_observed_A=ea, event_observed_B=eb)
    assert ours.statistic == pytest.approx(theirs.test_statistic)
    assert ours.p_value == pytest.approx(theirs.p_value)
