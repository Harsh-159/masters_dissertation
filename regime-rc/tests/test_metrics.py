"""Tests for the Section 13 metrics and the A9 cost/turnover conventions.

Each metric is checked against a hand-computable case, so a failure points at
the definition rather than at a plausible-looking number. The cost tests exist
because the factor-of-two between "cost" and "turnover" is the single easiest
convention to get wrong, and getting it wrong would understate every strategy's
trading drag by half.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rrc.backtest.costs import (
    DEFAULT_COST_PER_UNIT,
    annualised_turnover,
    one_way_turnover,
    traded_notional,
    transaction_cost,
)
from rrc.backtest.metrics import (
    TRADING_DAYS_PER_YEAR,
    annualised_return,
    annualised_volatility,
    cvar,
    cvar_torch,
    effective_number_of_assets,
    excess_sharpe,
    excess_sharpe_torch,
    max_drawdown,
    realised_holdings_count,
    sortino,
    summarise,
)


# --------------------------------------------------------------------------- #
# Cost and turnover conventions (A9)
# --------------------------------------------------------------------------- #
def test_cost_charges_the_full_traded_notional():
    """Selling 10% of A to buy B trades 20% of notional and is costed on all of it."""
    executed = torch.tensor([0.4, 0.6, 0.0], dtype=torch.float64)
    drifted = torch.tensor([0.5, 0.5, 0.0], dtype=torch.float64)

    assert float(traded_notional(executed, drifted)) == pytest.approx(0.2)
    assert float(transaction_cost(executed, drifted, 5e-4)) == pytest.approx(0.2 * 5e-4)


def test_turnover_is_half_the_traded_notional():
    """One-way turnover counts the notional moved in one direction only."""
    executed = torch.tensor([0.4, 0.6, 0.0], dtype=torch.float64)
    drifted = torch.tensor([0.5, 0.5, 0.0], dtype=torch.float64)

    assert float(one_way_turnover(executed, drifted)) == pytest.approx(0.1)


def test_cost_is_exactly_twice_the_turnover_times_the_rate():
    """The factor of two between the two conventions, pinned as a test.

    Using the turnover figure to compute costs would halve them silently, which
    is why these are separate functions with separate names.
    """
    executed = torch.tensor([0.7, 0.3], dtype=torch.float64)
    drifted = torch.tensor([0.2, 0.8], dtype=torch.float64)

    cost = float(transaction_cost(executed, drifted, DEFAULT_COST_PER_UNIT))
    turnover = float(one_way_turnover(executed, drifted))
    assert cost == pytest.approx(2.0 * turnover * DEFAULT_COST_PER_UNIT)


def test_no_trade_costs_nothing():
    weights = torch.tensor([0.3, 0.7], dtype=torch.float64)
    assert float(transaction_cost(weights, weights, DEFAULT_COST_PER_UNIT)) == 0.0


def test_annualised_turnover_is_calendar_based():
    """Total turnover scaled by 252/span - cadence-agnostic."""
    per_rebalance = np.array([0.1, 0.2, 0.3])  # total 0.6 over one year
    assert annualised_turnover(per_rebalance, n_days=252) == pytest.approx(0.6)


def test_weekly_cadence_matches_the_old_52x_convention():
    """For ~52 weekly rebalances a year the two definitions coincide (A9)."""
    per_rebalance = np.full(52, 0.2)
    assert annualised_turnover(per_rebalance, n_days=252) == pytest.approx(52 * 0.2)


def test_buy_and_hold_turnover_is_about_one_not_fifty_two():
    """The bug the dashboard exposed: one entry trade != 52x a year."""
    single_entry = np.array([1.0])
    assert annualised_turnover(single_entry, n_days=252) == pytest.approx(1.0)


def test_annualised_turnover_of_nothing_is_zero():
    assert annualised_turnover(np.array([]), n_days=252) == 0.0


# --------------------------------------------------------------------------- #
# Return and risk metrics
# --------------------------------------------------------------------------- #
def test_annualised_return_is_geometric():
    """A constant 1% daily return compounds; it is not 252 x 1%."""
    returns = np.full(TRADING_DAYS_PER_YEAR, 0.01)
    expected = 1.01**TRADING_DAYS_PER_YEAR - 1.0

    assert annualised_return(returns) == pytest.approx(expected, rel=1e-10)
    assert annualised_return(returns) != pytest.approx(0.01 * TRADING_DAYS_PER_YEAR)


def test_annualised_return_of_a_flat_series_is_zero():
    assert annualised_return(np.zeros(500)) == pytest.approx(0.0)


def test_annualised_volatility_scales_by_root_time():
    returns = np.array([0.01, -0.01] * 200)
    daily_deviation = np.std(returns, ddof=1)

    assert annualised_volatility(returns) == pytest.approx(
        daily_deviation * np.sqrt(TRADING_DAYS_PER_YEAR)
    )


def test_excess_sharpe_on_a_hand_computable_series():
    """Constant 1 bp daily excess with a known spread."""
    excess_pattern = np.array([0.002, -0.001] * 150)
    risk_free = np.full(len(excess_pattern), 0.0001)
    net_returns = excess_pattern + risk_free

    expected = (
        np.mean(excess_pattern)
        / (np.std(excess_pattern, ddof=1) + 1e-6)
        * np.sqrt(TRADING_DAYS_PER_YEAR)
    )
    assert excess_sharpe(net_returns, risk_free) == pytest.approx(expected, rel=1e-9)


def test_excess_sharpe_is_zero_when_returns_only_match_the_risk_free_rate():
    """Earning exactly r_f is no achievement and must score zero."""
    risk_free = np.full(500, 0.0002)
    assert excess_sharpe(risk_free.copy(), risk_free) == pytest.approx(0.0, abs=1e-9)


def test_sortino_ignores_upside_volatility():
    """Two series with identical downside but different upside: same downside risk.

    Sortino must reward the one with more upside, unlike Sharpe which penalises
    any deviation.
    """
    risk_free = np.zeros(400)
    mild_upside = np.array([0.01, -0.01] * 200)
    wild_upside = np.array([0.05, -0.01] * 200)

    assert sortino(wild_upside, risk_free) > sortino(mild_upside, risk_free)


def test_max_drawdown_on_a_known_path():
    """Up 50%, then down 50%: the peak-to-trough fall is exactly -50%."""
    returns = np.array([0.5, -0.5])
    assert max_drawdown(returns) == pytest.approx(-0.5)


def test_max_drawdown_of_a_monotonic_riser_is_zero():
    assert max_drawdown(np.full(100, 0.001)) == pytest.approx(0.0)


def test_cvar_averages_the_worst_tail_and_is_positive():
    """CVaR at 95% over 100 days averages the worst 5 days, reported positive."""
    excess = np.concatenate([np.full(95, 0.001), np.array([-0.10, -0.08, -0.06, -0.04, -0.02])])
    risk_free = np.zeros(len(excess))

    expected = -np.mean([-0.10, -0.08, -0.06, -0.04, -0.02])
    assert cvar(excess, risk_free, level=0.95) == pytest.approx(expected)
    assert cvar(excess, risk_free, level=0.95) > 0


def test_cvar_is_worse_than_or_equal_to_the_average_loss():
    """A tail average can never be milder than the overall mean loss."""
    generator = np.random.default_rng(0)
    returns = generator.normal(0.0005, 0.01, 2000)
    risk_free = np.zeros(len(returns))

    assert cvar(returns, risk_free) > -np.mean(returns)


# --------------------------------------------------------------------------- #
# Holdings diagnostics
# --------------------------------------------------------------------------- #
def test_holdings_count_excludes_cash_and_ignores_dust():
    """Cash is not a "position", and a 1e-9 weight is not a holding."""
    held = np.array([[0.5, 0.5, 1e-9, 0.0]])  # column 3 is cash
    assert realised_holdings_count(held, cash_index=3) == pytest.approx(2.0)


def test_effective_n_penalises_concentration():
    """Five equal holdings give 5; one dominant holding gives barely above 1."""
    equal = np.array([[0.2, 0.2, 0.2, 0.2, 0.2, 0.0]])
    concentrated = np.array([[0.96, 0.01, 0.01, 0.01, 0.01, 0.0]])

    assert effective_number_of_assets(equal, cash_index=5) == pytest.approx(5.0)
    assert effective_number_of_assets(concentrated, cash_index=5) < 1.2


def test_effective_n_and_holdings_count_disagree_on_purpose():
    """Both "hold five names", but only one is genuinely diversified."""
    concentrated = np.array([[0.96, 0.01, 0.01, 0.01, 0.01, 0.0]])

    assert realised_holdings_count(concentrated, cash_index=5) == pytest.approx(5.0)
    assert effective_number_of_assets(concentrated, cash_index=5) < 2.0


# --------------------------------------------------------------------------- #
# Differentiable versions agree with the reporting versions
# --------------------------------------------------------------------------- #
def test_torch_and_numpy_sharpe_agree_up_to_annualisation():
    """The differentiable Sharpe omits the sqrt(252) factor by design."""
    generator = np.random.default_rng(1)
    net_returns = generator.normal(0.0005, 0.01, 1000)
    risk_free = np.full(1000, 0.0001)

    numpy_value = excess_sharpe(net_returns, risk_free)
    torch_value = float(
        excess_sharpe_torch(
            torch.as_tensor(net_returns, dtype=torch.float64),
            torch.as_tensor(risk_free, dtype=torch.float64),
        )
    )
    assert torch_value * np.sqrt(TRADING_DAYS_PER_YEAR) == pytest.approx(numpy_value, rel=1e-9)


def test_torch_and_numpy_cvar_agree():
    generator = np.random.default_rng(2)
    net_returns = generator.normal(0.0005, 0.01, 1000)
    risk_free = np.full(1000, 0.0001)

    numpy_value = cvar(net_returns, risk_free, level=0.95)
    torch_value = float(
        cvar_torch(
            torch.as_tensor(net_returns, dtype=torch.float64),
            torch.as_tensor(risk_free, dtype=torch.float64),
            level=0.95,
        )
    )
    assert torch_value == pytest.approx(numpy_value, rel=1e-9)


def test_cvar_torch_is_differentiable():
    """The tail selection is piecewise linear, so gradients must flow."""
    net_returns = torch.randn(200, dtype=torch.float64, requires_grad=True)
    risk_free = torch.zeros(200, dtype=torch.float64)

    cvar_torch(net_returns, risk_free).backward()
    assert net_returns.grad is not None
    assert float(net_returns.grad.abs().sum()) > 0


# --------------------------------------------------------------------------- #
# The summary bundle
# --------------------------------------------------------------------------- #
def test_summarise_returns_the_registry_metric_keys():
    """summarise() output must slot straight into a RegistryRow."""
    generator = np.random.default_rng(3)
    net_returns = generator.normal(0.0005, 0.01, 500)
    risk_free = np.full(500, 0.0001)
    held = generator.dirichlet(np.ones(4), size=500)

    summary = summarise(
        net_returns, risk_free,
        turnover=np.full(50, 0.1), held_weights=held, cash_index=3,
    )

    for key in (
        "ann_return", "ann_vol", "excess_sharpe", "sortino", "max_drawdown",
        "cvar_95", "ann_turnover", "holdings_count", "effective_n",
    ):
        assert key in summary
        assert np.isfinite(summary[key]), f"{key} was not finite"


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError, match="must align"):
        excess_sharpe(np.zeros(10), np.zeros(9))
