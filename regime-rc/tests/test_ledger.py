"""Engine property tests B1-B7 and causality test C7.

The plan calls these "load-bearing for the whole thesis", and the reason is
worth stating plainly: this engine is simultaneously the backtest **and** the
loss function's forward pass, and it is shared by the trained system and every
baseline. A bookkeeping error here would not produce an obviously broken
result - it would produce a plausible one, applied unevenly across the ladder.

===  ==================================================================
B1   an all-cash portfolio reproduces the compounded r_f path exactly
B2   zero-cost buy-and-hold equals the asset's own total return to 1e-10
B3   held weights stay on the simplex (non-negative, summing to 1)
B4   drifted weights match the hand-computed closed form on a toy
B5   total costs equal c x total traded notional
B6   execution lag: C7's perturbation run against the engine directly
B7   autograd matches finite differences to 1e-5
===  ==================================================================

C7   perturbing day t+1's return leaves the day-t decision untouched, and the
     day-(t+1) return accrues to the OLD weights
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rrc.backtest.costs import DEFAULT_COST_PER_UNIT, transaction_cost
from rrc.backtest.ledger import (
    LEDGER_DTYPE,
    run_ledger,
    run_ledger_numpy,
    weekly_execution_schedule,
)
from rrc.backtest.metrics import excess_sharpe, excess_sharpe_torch

CASH_INDEX = 2  # in the toy panels below, column 2 is cash


# --------------------------------------------------------------------------- #
# Toy fixtures - small enough to verify by hand
# --------------------------------------------------------------------------- #
@pytest.fixture
def toy_panel() -> tuple[torch.Tensor, torch.Tensor]:
    """A 3-asset, 5-day panel: two risky assets plus cash at a flat 0.01%/day."""
    simple_returns = torch.tensor(
        [
            [0.010, -0.005, 0.0001],
            [-0.020, 0.010, 0.0001],
            [0.005, 0.002, 0.0001],
            [0.030, -0.010, 0.0001],
            [-0.004, 0.006, 0.0001],
        ],
        dtype=LEDGER_DTYPE,
    )
    risk_free = simple_returns[:, CASH_INDEX].clone()
    return simple_returns, risk_free


@pytest.fixture
def random_panel() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    """A larger random panel with weekly rebalancing, for equivalence checks."""
    generator = torch.Generator().manual_seed(0)
    n_days, n_assets = 500, 4

    simple_returns = 0.01 * torch.randn(n_days, n_assets, generator=generator, dtype=LEDGER_DTYPE)
    simple_returns[:, CASH_INDEX] = 0.0001  # cash is riskless
    risk_free = simple_returns[:, CASH_INDEX].clone()

    exec_days = list(range(5, n_days - 5, 5))
    raw = torch.rand(len(exec_days), n_assets, generator=generator, dtype=LEDGER_DTYPE)
    targets = raw / raw.sum(dim=1, keepdim=True)
    return simple_returns, risk_free, targets, exec_days


def cash_only_targets(n_periods: int, n_assets: int) -> torch.Tensor:
    """Weight rows that are 100% cash."""
    targets = torch.zeros(n_periods, n_assets, dtype=LEDGER_DTYPE)
    targets[:, CASH_INDEX] = 1.0
    return targets


# --------------------------------------------------------------------------- #
# B1 - all-cash identity
# --------------------------------------------------------------------------- #
def test_b1_all_cash_reproduces_the_risk_free_path(toy_panel):
    """A permanently all-cash portfolio must earn exactly r_f every day."""
    simple_returns, risk_free = toy_panel
    exec_days = [0, 2]
    targets = cash_only_targets(len(exec_days), simple_returns.shape[1])

    result = run_ledger(targets, exec_days, simple_returns, CASH_INDEX)

    torch.testing.assert_close(result.net_returns, risk_free, atol=1e-15, rtol=0)
    assert float(result.costs.sum()) == 0.0  # cash -> cash trades nothing


def test_b1_all_cash_scores_zero_excess_sharpe(random_panel):
    """The load-bearing consequence: cash is a refuge, never an attractor.

    Regression guard for the degeneracy the plan flags as critical. An all-cash
    portfolio's excess returns are zero only in exact arithmetic; in float64 they
    are round-off noise of order 1e-19, and the ratio of two such numbers is an
    arbitrary O(1) value. Before the denominator was epsilon-stabilised this
    reported a Sharpe of 4.17 for a portfolio that never took any risk.
    """
    simple_returns, risk_free, _, exec_days = random_panel
    targets = cash_only_targets(len(exec_days), simple_returns.shape[1])

    result = run_ledger(targets, exec_days, simple_returns, CASH_INDEX)
    sharpe = excess_sharpe(result.net_returns.numpy(), risk_free.numpy())

    assert abs(sharpe) < 1e-6


def test_b1_reported_and_differentiable_sharpe_agree_on_cash(random_panel):
    """The quantity optimised must be the quantity reported, cash included."""
    simple_returns, risk_free, _, exec_days = random_panel
    targets = cash_only_targets(len(exec_days), simple_returns.shape[1])

    result = run_ledger(targets, exec_days, simple_returns, CASH_INDEX)
    differentiable = float(excess_sharpe_torch(result.net_returns, risk_free))

    assert abs(differentiable) < 1e-6


def test_b1_a_genuinely_risky_portfolio_still_gets_a_real_sharpe(random_panel):
    """The epsilon must not flatten real signal - only the degenerate case."""
    simple_returns, risk_free, targets, exec_days = random_panel

    result = run_ledger(targets, exec_days, simple_returns, CASH_INDEX)
    sharpe = excess_sharpe(result.net_returns.numpy(), risk_free.numpy())

    # A risky portfolio's daily volatility is ~1e-2, vastly larger than the
    # 1e-6 epsilon, so the stabilisation changes the value by ~0.01%.
    assert np.isfinite(sharpe)
    assert abs(sharpe) > 1e-3


# --------------------------------------------------------------------------- #
# B2 - buy-and-hold identity
# --------------------------------------------------------------------------- #
def test_b2_buy_and_hold_equals_the_assets_own_total_return(random_panel):
    """With zero costs and one trade into a single asset, the ledger IS that asset.

    This is the sharpest single check on the drift arithmetic: if the held
    weights drifted incorrectly, a one-asset portfolio would stop tracking its
    own asset, and the compounded difference over 500 days would be obvious.
    """
    simple_returns, _, _, _ = random_panel
    asset = 0

    targets = torch.zeros(1, simple_returns.shape[1], dtype=LEDGER_DTYPE)
    targets[0, asset] = 1.0

    result = run_ledger(targets, [0], simple_returns, CASH_INDEX, cost_per_unit=0.0)

    # From day 1 onward the ledger must equal the asset's own daily returns.
    torch.testing.assert_close(
        result.net_returns[1:], simple_returns[1:, asset], atol=1e-10, rtol=0
    )

    ledger_growth = float(torch.prod(1.0 + result.net_returns[1:]))
    asset_growth = float(torch.prod(1.0 + simple_returns[1:, asset]))
    assert ledger_growth == pytest.approx(asset_growth, rel=1e-10)


# --------------------------------------------------------------------------- #
# B3 - simplex invariant
# --------------------------------------------------------------------------- #
def test_b3_held_weights_stay_on_the_simplex(random_panel):
    """Held weights must be non-negative and sum to 1 on every single day.

    Without the explicit renormalisation, floating-point error in the cumulative
    product accumulates and the weights slowly drift off the simplex - silently,
    because nothing else in the pipeline checks.
    """
    simple_returns, _, targets, exec_days = random_panel

    result = run_ledger(
        targets, exec_days, simple_returns, CASH_INDEX, return_diagnostics=True
    )
    held = result.held_weights

    assert held is not None
    assert bool((held >= -1e-12).all()), "a held weight went negative"
    row_sums = held.sum(dim=1)
    torch.testing.assert_close(
        row_sums, torch.ones_like(row_sums), atol=1e-12, rtol=0
    )


# --------------------------------------------------------------------------- #
# B4 - drift closed form on a hand-computed toy
# --------------------------------------------------------------------------- #
def test_b4_drifted_weights_match_the_hand_computed_formula(toy_panel):
    """Verify the drift formula against arithmetic done by hand.

    Buy 50/50 into assets 0 and 1 at the close of day 0. Over days 1 and 2 the
    assets compound independently, so the held weights must become
    ``w_i * G_i / sum_j w_j * G_j``.
    """
    simple_returns, _ = toy_panel

    targets = torch.zeros(1, 3, dtype=LEDGER_DTYPE)
    targets[0, 0] = 0.5
    targets[0, 1] = 0.5

    result = run_ledger(
        targets, [0], simple_returns, CASH_INDEX, cost_per_unit=0.0,
        return_diagnostics=True,
    )

    # Hand computation for the close of day 2.
    growth_asset_0 = (1 + 0.010) * (1 + 0.005) * 1.0  # careful: see below
    # Days 1 and 2 are rows 1 and 2 of the panel.
    growth_asset_0 = (1 - 0.020) * (1 + 0.005)
    growth_asset_1 = (1 + 0.010) * (1 + 0.002)

    value = 0.5 * growth_asset_0 + 0.5 * growth_asset_1
    expected_weight_0 = 0.5 * growth_asset_0 / value
    expected_weight_1 = 0.5 * growth_asset_1 / value

    assert float(result.held_weights[2, 0]) == pytest.approx(expected_weight_0, rel=1e-12)
    assert float(result.held_weights[2, 1]) == pytest.approx(expected_weight_1, rel=1e-12)

    # And the day-2 portfolio return equals the drifted-weight-weighted return.
    weights_at_day_1 = result.held_weights[1]
    expected_day_2_return = float((weights_at_day_1 * simple_returns[2]).sum())
    assert float(result.net_returns[2]) == pytest.approx(expected_day_2_return, rel=1e-12)


# --------------------------------------------------------------------------- #
# B5 - cost accounting
# --------------------------------------------------------------------------- #
def test_b5_total_costs_equal_rate_times_traded_notional(random_panel):
    """Total charged cost must equal c x total notional traded, exactly."""
    simple_returns, _, targets, exec_days = random_panel
    cost_rate = DEFAULT_COST_PER_UNIT

    result = run_ledger(
        targets, exec_days, simple_returns, CASH_INDEX,
        cost_per_unit=cost_rate, return_diagnostics=True,
    )

    # One-way turnover is half the traded notional (A9), so recover the notional.
    total_notional = float(result.turnover.sum() * 2.0)
    assert float(result.costs.sum()) == pytest.approx(cost_rate * total_notional, rel=1e-12)


def test_b5_costs_land_only_on_execution_days(random_panel):
    """No cost may be charged on a day when nothing was traded."""
    simple_returns, _, targets, exec_days = random_panel

    result = run_ledger(targets, exec_days, simple_returns, CASH_INDEX)

    non_execution_days = np.setdiff1d(np.arange(simple_returns.shape[0]), np.array(exec_days))
    assert float(result.costs[non_execution_days].abs().sum()) == 0.0
    assert bool((result.costs[np.array(exec_days)] > 0).all())


def test_b5_first_trade_is_costed_against_cash(toy_panel):
    """The opening trade must pay to move out of cash, not be free."""
    simple_returns, _ = toy_panel

    targets = torch.zeros(1, 3, dtype=LEDGER_DTYPE)
    targets[0, 0] = 1.0

    result = run_ledger(targets, [0], simple_returns, CASH_INDEX)

    cash_vector = torch.zeros(3, dtype=LEDGER_DTYPE)
    cash_vector[CASH_INDEX] = 1.0
    expected = transaction_cost(targets[0], cash_vector, DEFAULT_COST_PER_UNIT)
    assert float(result.costs[0]) == pytest.approx(float(expected), rel=1e-12)


# --------------------------------------------------------------------------- #
# B6 / C7 - execution lag and no same-bar fill
# --------------------------------------------------------------------------- #
def test_b6_c7_perturbing_tomorrow_leaves_todays_ledger_untouched(random_panel):
    """C7: change day t+1's return by +50%; nothing at or before t may move.

    This kills an entire class of bug at once - any accidental same-bar fill,
    off-by-one in the execution index, or weight schedule that peeks at the
    return it is about to earn.
    """
    simple_returns, _, targets, exec_days = random_panel
    perturbation_day = 200

    baseline = run_ledger(targets, exec_days, simple_returns, CASH_INDEX)

    perturbed_returns = simple_returns.clone()
    perturbed_returns[perturbation_day, 0] += 0.50
    perturbed = run_ledger(targets, exec_days, perturbed_returns, CASH_INDEX)

    torch.testing.assert_close(
        baseline.net_returns[:perturbation_day],
        perturbed.net_returns[:perturbation_day],
        atol=1e-15, rtol=0,
    )
    # And it must actually have had an effect from that day on, or the test
    # would be vacuously true.
    assert not torch.allclose(
        baseline.net_returns[perturbation_day:], perturbed.net_returns[perturbation_day:]
    )


def test_c7_perturbed_day_accrues_to_the_old_weights(toy_panel):
    """The day-(t+1) return must be earned by the weights held *before* it."""
    simple_returns, _ = toy_panel

    # Trade into asset 0 at close of day 0; hold. Day 1's return is asset 0's.
    targets = torch.zeros(1, 3, dtype=LEDGER_DTYPE)
    targets[0, 0] = 1.0

    result = run_ledger(targets, [0], simple_returns, CASH_INDEX, cost_per_unit=0.0)
    assert float(result.net_returns[1]) == pytest.approx(float(simple_returns[1, 0]), rel=1e-12)


def test_c7_execution_day_return_uses_previous_weights_then_charges_cost(toy_panel):
    """On a rebalance day: earn the old portfolio's return, then pay to trade."""
    simple_returns, _ = toy_panel

    targets = torch.zeros(2, 3, dtype=LEDGER_DTYPE)
    targets[0, 0] = 1.0   # into asset 0 at close of day 0
    targets[1, 1] = 1.0   # switch to asset 1 at close of day 2

    result = run_ledger(
        targets, [0, 2], simple_returns, CASH_INDEX, cost_per_unit=DEFAULT_COST_PER_UNIT
    )

    # Day 2 is an execution day: its gross return is still asset 0's.
    assert float(result.gross_returns[2]) == pytest.approx(float(simple_returns[2, 0]), rel=1e-12)
    # Net return on that day is the gross return minus the trading cost.
    assert float(result.net_returns[2]) == pytest.approx(
        float(result.gross_returns[2] - result.costs[2]), rel=1e-12
    )
    # Day 3 belongs to the new portfolio.
    assert float(result.net_returns[3]) == pytest.approx(float(simple_returns[3, 1]), rel=1e-12)


# --------------------------------------------------------------------------- #
# B7 - gradient check
# --------------------------------------------------------------------------- #
def test_b7_autograd_matches_finite_differences(toy_panel):
    """Finite-difference the loss w.r.t. a target weight and compare to autograd.

    If this fails, every gradient the training loop takes is wrong, and nothing
    downstream means anything. It is the cheapest possible insurance.
    """
    simple_returns, risk_free = toy_panel

    def loss_from_weight(weight_on_asset_0: torch.Tensor) -> torch.Tensor:
        """Negative excess Sharpe of a 2-asset mix, as a function of one weight."""
        targets = torch.stack([
            torch.stack([
                weight_on_asset_0,
                1.0 - weight_on_asset_0,
                torch.zeros((), dtype=LEDGER_DTYPE),
            ])
        ])
        result = run_ledger(targets, [0], simple_returns, CASH_INDEX, cost_per_unit=0.0)
        return -excess_sharpe_torch(result.net_returns, risk_free)

    weight = torch.tensor(0.6, dtype=LEDGER_DTYPE, requires_grad=True)
    loss_from_weight(weight).backward()
    autograd_gradient = float(weight.grad)

    step = 1e-6
    with torch.no_grad():
        upper = loss_from_weight(torch.tensor(0.6 + step, dtype=LEDGER_DTYPE))
        lower = loss_from_weight(torch.tensor(0.6 - step, dtype=LEDGER_DTYPE))
    finite_difference_gradient = float((upper - lower) / (2 * step))

    assert autograd_gradient == pytest.approx(finite_difference_gradient, abs=1e-5)


def test_b7_gradients_reach_every_execution_period(random_panel):
    """Every rebalance must receive gradient, not just the first or last."""
    simple_returns, risk_free, targets, exec_days = random_panel

    weights = targets.clone().requires_grad_(True)
    result = run_ledger(weights, exec_days, simple_returns, CASH_INDEX)
    (-excess_sharpe_torch(result.net_returns, risk_free)).backward()

    assert weights.grad is not None
    per_period_gradient_norm = weights.grad.abs().sum(dim=1)
    assert bool((per_period_gradient_norm > 0).all()), "some rebalance received no gradient"


# --------------------------------------------------------------------------- #
# Reference vs fast equivalence, and the NumPy wrapper
# --------------------------------------------------------------------------- #
def test_fast_and_reference_implementations_agree(random_panel):
    """The vectorised path must reproduce the readable loop to 1e-12.

    This is what allows the fast path to be trusted: the reference mirrors the
    plan's pseudocode line for line, and the optimisation is proven, not assumed.
    """
    simple_returns, _, targets, exec_days = random_panel

    fast = run_ledger(
        targets, exec_days, simple_returns, CASH_INDEX,
        method="fast", return_diagnostics=True,
    )
    reference = run_ledger(
        targets, exec_days, simple_returns, CASH_INDEX,
        method="reference", return_diagnostics=True,
    )

    torch.testing.assert_close(fast.net_returns, reference.net_returns, atol=1e-12, rtol=0)
    torch.testing.assert_close(fast.costs, reference.costs, atol=1e-12, rtol=0)
    torch.testing.assert_close(fast.turnover, reference.turnover, atol=1e-12, rtol=0)
    torch.testing.assert_close(
        fast.held_weights, reference.held_weights, atol=1e-12, rtol=0
    )


def test_numpy_wrapper_matches_the_torch_engine(random_panel):
    """Baselines must be accounted for by exactly the same engine as the model."""
    simple_returns, _, targets, exec_days = random_panel

    torch_result = run_ledger(targets, exec_days, simple_returns, CASH_INDEX)
    numpy_result = run_ledger_numpy(
        targets.numpy(), exec_days, simple_returns.numpy(), CASH_INDEX
    )

    np.testing.assert_allclose(
        numpy_result["net_returns"], torch_result.net_returns.numpy(), atol=1e-14
    )


# --------------------------------------------------------------------------- #
# Execution timing schedule (E7)
# --------------------------------------------------------------------------- #
def test_execution_day_is_strictly_after_the_signal_day():
    """The core of E7: you can never trade on the close you computed from."""
    import pandas as pd

    calendar = pd.bdate_range("2020-01-01", periods=60)
    signal_days, execution_days = weekly_execution_schedule(calendar)

    assert len(signal_days) > 0
    assert bool(np.all(execution_days > signal_days))
    assert bool(np.all(execution_days == signal_days + 1))


def test_schedule_signals_on_the_last_trading_day_of_each_week():
    """A holiday-shortened week signals on its Thursday, not silently skipped."""
    import pandas as pd

    # A normal week, then one whose Friday is missing (a holiday).
    calendar = pd.DatetimeIndex([
        "2020-01-06", "2020-01-07", "2020-01-08", "2020-01-09", "2020-01-10",  # Mon-Fri
        "2020-01-13", "2020-01-14", "2020-01-15", "2020-01-16",                # Mon-Thu only
        "2020-01-21",                                                          # next Tuesday
    ])
    signal_days, _ = weekly_execution_schedule(calendar)

    signalled_on = calendar[signal_days]
    assert pd.Timestamp("2020-01-10") in signalled_on  # the full week's Friday
    assert pd.Timestamp("2020-01-16") in signalled_on  # the short week's Thursday


def test_schedule_drops_a_final_signal_with_no_day_to_execute_on():
    """A signal on the last day of the sample has no future close to trade at."""
    import pandas as pd

    calendar = pd.bdate_range("2020-01-01", periods=10)
    _, execution_days = weekly_execution_schedule(calendar)
    assert bool(np.all(execution_days < len(calendar)))


def test_schedule_of_a_trivial_calendar_is_empty():
    import pandas as pd

    signal_days, execution_days = weekly_execution_schedule(pd.DatetimeIndex(["2020-01-01"]))
    assert len(signal_days) == 0
    assert len(execution_days) == 0


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #
def test_weights_that_do_not_sum_to_one_are_rejected(toy_panel):
    simple_returns, _ = toy_panel
    targets = torch.full((1, 3), 0.2, dtype=LEDGER_DTYPE)  # sums to 0.6
    with pytest.raises(ValueError, match="fully invested"):
        run_ledger(targets, [0], simple_returns, CASH_INDEX)


def test_negative_weights_are_rejected(toy_panel):
    simple_returns, _ = toy_panel
    targets = torch.tensor([[1.5, -0.5, 0.0]], dtype=LEDGER_DTYPE)
    with pytest.raises(ValueError, match="long-only"):
        run_ledger(targets, [0], simple_returns, CASH_INDEX)


def test_non_increasing_execution_days_are_rejected(toy_panel):
    simple_returns, _ = toy_panel
    targets = cash_only_targets(2, 3)
    with pytest.raises(ValueError, match="strictly increasing"):
        run_ledger(targets, [2, 1], simple_returns, CASH_INDEX)


def test_mismatched_shapes_are_rejected(toy_panel):
    simple_returns, _ = toy_panel
    targets = cash_only_targets(3, 3)
    with pytest.raises(ValueError, match="execution days for"):
        run_ledger(targets, [0, 2], simple_returns, CASH_INDEX)
