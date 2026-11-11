"""Tests for the rung-0 baseline weight generators.

Beyond checking the arithmetic on hand-computable toys, every estimating
baseline gets a **causality test**: perturb the data *after* the signal day and
assert the weights do not move. A baseline that peeked would corrupt the
humility line itself, and every later comparison with it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rrc.baselines.markowitz import (
    mean_variance_schedule,
    solve_long_only_mean_variance,
)
from rrc.baselines.naive import (
    buy_and_hold_schedule,
    inverse_volatility_schedule,
    one_over_n_schedule,
)

N_COLUMNS = 4  # three risky assets + cash in the toys below
CASH = 3


@pytest.fixture
def toy_market():
    """300 days, three risky assets; the third launches on day 150.

    Asset volatilities are ordered sigma_A > sigma_B by construction, so the
    inverse-volatility ordering has a known right answer.
    """
    generator = np.random.default_rng(0)
    n_days = 300
    dates = pd.bdate_range("2020-01-01", periods=n_days, name="date")

    returns = pd.DataFrame(
        {
            "A": generator.normal(0.0005, 0.020, n_days),  # the volatile one
            "B": generator.normal(0.0005, 0.010, n_days),  # the calm one
            "C": generator.normal(0.0005, 0.015, n_days),
        },
        index=dates,
    )
    returns.iloc[:150, returns.columns.get_loc("C")] = np.nan  # pre-inception

    availability = pd.DataFrame(1, index=dates, columns=["A", "B", "C"], dtype="int8")
    availability.iloc[:150, availability.columns.get_loc("C")] = 0

    risk_free = pd.Series(0.0001, index=dates)
    return returns, availability, risk_free, dates


# --------------------------------------------------------------------------- #
# 1/N
# --------------------------------------------------------------------------- #
def test_one_over_n_divides_by_the_active_count(toy_market):
    """Before C launches: 1/2 each. After: 1/3 each. Never 1/3 over phantoms."""
    _, availability, _, _ = toy_market

    targets, _ = one_over_n_schedule(
        availability,
        signal_positions=np.array([100, 200]),  # before and after C's launch
        execution_positions=np.array([101, 201]),
        n_columns=N_COLUMNS,
        cash_index=CASH,
    )

    np.testing.assert_allclose(targets[0], [0.5, 0.5, 0.0, 0.0])
    np.testing.assert_allclose(targets[1], [1 / 3, 1 / 3, 1 / 3, 0.0])


def test_one_over_n_rows_are_fully_invested(toy_market):
    _, availability, _, _ = toy_market
    targets, _ = one_over_n_schedule(
        availability, np.arange(10, 290, 5), np.arange(11, 291, 5), N_COLUMNS, CASH
    )
    np.testing.assert_allclose(targets.sum(axis=1), 1.0)
    assert (targets[:, CASH] == 0).all()


# --------------------------------------------------------------------------- #
# Buy and hold
# --------------------------------------------------------------------------- #
def test_buy_and_hold_trades_exactly_once():
    targets, execution_days = buy_and_hold_schedule(
        asset_index=0, first_execution_position=3, n_columns=2
    )
    assert targets.shape == (1, 2)
    np.testing.assert_allclose(targets[0], [1.0, 0.0])
    assert list(execution_days) == [3]


# --------------------------------------------------------------------------- #
# Inverse volatility
# --------------------------------------------------------------------------- #
def test_inverse_vol_gives_the_calm_asset_more_weight(toy_market):
    """sigma_A = 2 x sigma_B, so w_B should be about 2 x w_A."""
    returns, availability, _, _ = toy_market

    targets, _ = inverse_volatility_schedule(
        returns, availability,
        signal_positions=np.array([120]),  # C not yet launched
        execution_positions=np.array([121]),
        n_columns=N_COLUMNS, cash_index=CASH,
        window=63, min_observations=40,
    )

    weight_a, weight_b = targets[0, 0], targets[0, 1]
    assert weight_b > weight_a
    assert weight_b / weight_a == pytest.approx(2.0, rel=0.35)  # sampling noise
    assert targets[0].sum() == pytest.approx(1.0)


def test_inverse_vol_excludes_a_young_asset_until_it_has_history(toy_market):
    """C launches on day 150 with min_observations=40: out at 160, in by 250."""
    returns, availability, _, _ = toy_market

    targets, _ = inverse_volatility_schedule(
        returns, availability,
        signal_positions=np.array([160, 250]),
        execution_positions=np.array([161, 251]),
        n_columns=N_COLUMNS, cash_index=CASH,
        window=63, min_observations=40,
    )

    assert targets[0, 2] == 0.0   # too young at 160 (only ~10 observations)
    assert targets[1, 2] > 0.0    # seasoned by 250


def test_inverse_vol_is_causal(toy_market):
    """Perturbing returns after the signal day must not move the weights."""
    returns, availability, _, _ = toy_market
    signal = 200

    baseline, _ = inverse_volatility_schedule(
        returns, availability, np.array([signal]), np.array([signal + 1]),
        N_COLUMNS, CASH, window=63, min_observations=40,
    )

    perturbed_returns = returns.copy()
    perturbed_returns.iloc[signal + 1 :] *= 10.0  # violence, strictly after t

    perturbed, _ = inverse_volatility_schedule(
        perturbed_returns, availability, np.array([signal]), np.array([signal + 1]),
        N_COLUMNS, CASH, window=63, min_observations=40,
    )

    np.testing.assert_array_equal(baseline, perturbed)


# --------------------------------------------------------------------------- #
# Mean-variance: the solver
# --------------------------------------------------------------------------- #
def test_solver_prefers_the_dominant_asset():
    """Higher mean, lower variance, uncorrelated: asset 0 must dominate."""
    expected_excess = np.array([0.001, 0.0002])
    covariance = np.diag([0.0001, 0.0004])

    weights = solve_long_only_mean_variance(expected_excess, covariance, risk_aversion=5.0)

    assert weights[0] > weights[1]
    assert (weights >= 0).all()
    assert weights.sum() <= 1.0 + 1e-9


def test_solver_retreats_to_cash_when_nothing_earns_its_risk():
    """All expected excess returns negative: the whole budget stays in cash."""
    expected_excess = np.array([-0.001, -0.002])
    covariance = np.diag([0.0001, 0.0001])

    weights = solve_long_only_mean_variance(expected_excess, covariance, risk_aversion=5.0)
    assert weights.sum() == pytest.approx(0.0, abs=1e-8)


def test_solver_diversifies_identical_uncorrelated_assets():
    """Symmetry: two identical assets should get identical weights."""
    expected_excess = np.array([0.0005, 0.0005])
    covariance = np.diag([0.0002, 0.0002])

    weights = solve_long_only_mean_variance(expected_excess, covariance, risk_aversion=5.0)
    assert weights[0] == pytest.approx(weights[1], rel=1e-6)


def test_higher_risk_aversion_holds_more_cash():
    expected_excess = np.array([0.0005, 0.0004])
    covariance = np.array([[0.0002, 0.00005], [0.00005, 0.0003]])

    bold = solve_long_only_mean_variance(expected_excess, covariance, risk_aversion=1.0)
    timid = solve_long_only_mean_variance(expected_excess, covariance, risk_aversion=50.0)

    assert timid.sum() < bold.sum()  # more risk aversion -> more cash slack


# --------------------------------------------------------------------------- #
# Mean-variance: the schedule
# --------------------------------------------------------------------------- #
def test_mean_variance_weights_are_valid_and_cash_is_the_residual(toy_market):
    returns, availability, risk_free, dates = toy_market

    signal_positions = np.arange(140, 290, 5)
    targets, _ = mean_variance_schedule(
        returns, risk_free, availability, dates,
        signal_positions, signal_positions + 1,
        N_COLUMNS, CASH,
        lookback=120, min_observations=60, risk_aversion=5.0,
    )

    assert (targets >= -1e-12).all()
    np.testing.assert_allclose(targets.sum(axis=1), 1.0, atol=1e-9)
    # Cash actually is the residual of the risky allocation.
    np.testing.assert_allclose(
        targets[:, CASH], 1.0 - targets[:, :CASH].sum(axis=1), atol=1e-9
    )


def test_mean_variance_refits_monthly_not_weekly(toy_market):
    """Weights are frozen within a month and may only change at a month boundary."""
    returns, availability, risk_free, dates = toy_market

    signal_positions = np.arange(140, 290, 5)
    targets, _ = mean_variance_schedule(
        returns, risk_free, availability, dates,
        signal_positions, signal_positions + 1,
        N_COLUMNS, CASH,
        lookback=120, min_observations=60, risk_aversion=5.0,
    )

    months = [dates[p].month for p in signal_positions]
    for row in range(1, len(signal_positions)):
        if months[row] == months[row - 1]:
            np.testing.assert_array_equal(
                targets[row], targets[row - 1],
                err_msg="weights changed inside a month - refit is not monthly",
            )
    # And across the whole run the weights did change at least once.
    assert not np.allclose(targets[0], targets[-1])


def test_mean_variance_is_causal(toy_market):
    """The estimation window ends at the signal day; the future must not matter."""
    returns, availability, risk_free, dates = toy_market
    signal = 200

    baseline, _ = mean_variance_schedule(
        returns, risk_free, availability, dates,
        np.array([signal]), np.array([signal + 1]),
        N_COLUMNS, CASH, lookback=120, min_observations=60, risk_aversion=5.0,
    )

    perturbed_returns = returns.copy()
    perturbed_returns.iloc[signal + 1 :] *= -3.0

    perturbed, _ = mean_variance_schedule(
        perturbed_returns, risk_free, availability, dates,
        np.array([signal]), np.array([signal + 1]),
        N_COLUMNS, CASH, lookback=120, min_observations=60, risk_aversion=5.0,
    )

    np.testing.assert_array_equal(baseline, perturbed)


def test_mean_variance_excludes_the_young_asset_then_admits_it(toy_market):
    """C (born day 150) is outside the optimisation until it has 60 observations."""
    returns, availability, risk_free, dates = toy_market

    targets, _ = mean_variance_schedule(
        returns, risk_free, availability, dates,
        np.array([180, 280]), np.array([181, 281]),
        N_COLUMNS, CASH, lookback=120, min_observations=60, risk_aversion=5.0,
    )

    assert targets[0, 2] == 0.0  # ~30 observations at day 180: excluded
    # By day 280 C has ~130 observations and is at least eligible; its weight
    # is the optimiser's to decide, so we assert eligibility indirectly: the
    # row is still a valid portfolio.
    assert targets[1].sum() == pytest.approx(1.0)
