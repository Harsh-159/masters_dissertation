"""The closed-form differentiable portfolio ledger.

What this module is
-------------------
Given a schedule of executed weights and a panel of daily asset returns, it
produces the daily net return stream of the portfolio. That is all it does. It
knows nothing about models, reservoirs, gating, or regimes - it consumes a
weight schedule and returns a ledger. Keeping it model-ignorant is what lets the
trained system and every baseline share **one** accounting implementation, so
their comparisons can never diverge because of bookkeeping.

Why it is written in torch
--------------------------
This engine *is* the forward pass of the loss function. The loss is the
risk-adjusted quality of the whole realised return stream, so gradients must
flow from every daily return back through the held weights to the executed
weights and into the model heads. Writing the ledger in torch means autograd
does that for free.

The closed-form trick
---------------------
Daily accounting with weekly trading has a pleasant consequence: **inside a
holding period the entire daily ledger is closed-form.** With weights ``w``
executed at the close of day ``e`` and held until the next execution, define the
elementwise cumulative growth ``G_d = prod_{s=e+1..d} (1 + rho_s)``. Then

    V_d = w . G_d          (portfolio value, with V_e = 1)
    r_d = V_d / V_{d-1} - 1
    w_held,d = (w * G_d) / V_d

so one cumulative product and one dot product replace a day-by-day Python loop.

Two implementations, and why both exist
---------------------------------------
``_reference_ledger`` follows the plan's pseudocode literally: a Python loop over
holding periods. It is slow but obviously correct, and it is the thing to read
when checking the maths.

``_fast_ledger`` removes the loop entirely. It uses the identity
``G_d = C_d / C_e`` where ``C`` is the cumulative growth from the start of the
sample, so a single global ``cumprod`` serves every holding period at once. With
~300 rebalances in a training fold and up to 500 epochs, the loop version costs
tens of seconds per run while the vectorised version costs well under a second -
the difference between fitting the plan's compute budget and missing it.

A test asserts the two agree to 1e-12 on random inputs. The reference keeps the
fast path honest; the fast path keeps the project inside its budget.

Timing convention (E7)
----------------------
Signals are computed through the close of day ``t``; the trade executes at the
close of day ``t+1``; the new portfolio earns from ``t+2``. **This module does
not apply that lag** - it receives ``exec_days`` already lagged by the caller.
Keeping the lag in one place, outside the engine, is what makes it testable
(causality test C7).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from rrc.backtest.costs import DEFAULT_COST_PER_UNIT, one_way_turnover, transaction_cost

# The ledger runs in float64. Over ~7,000 days a float32 cumulative product
# accumulates enough error to break the buy-and-hold identity test (B2), which
# is asserted to 1e-10.
LEDGER_DTYPE = torch.float64


@dataclass
class LedgerResult:
    """Output of a ledger run.

    Attributes
    ----------
    net_returns
        ``(D,)`` daily portfolio returns after costs. This is the series the
        loss is computed on.
    gross_returns
        ``(D,)`` daily returns before costs.
    costs
        ``(D,)`` cost charged on each day; nonzero only on execution days.
    turnover
        ``(J,)`` one-way turnover for each rebalance.
    held_weights
        ``(D, N)`` weights actually held at each day's close, or ``None`` when
        diagnostics were not requested. Used for holdings counts and effective
        N, not needed during training.
    """

    net_returns: torch.Tensor
    gross_returns: torch.Tensor
    costs: torch.Tensor
    turnover: torch.Tensor
    held_weights: torch.Tensor | None = None


def run_ledger(
    targets: torch.Tensor,
    exec_days: torch.Tensor | np.ndarray | list[int],
    simple_returns: torch.Tensor,
    cash_index: int,
    cost_per_unit: float = DEFAULT_COST_PER_UNIT,
    method: str = "fast",
    return_diagnostics: bool = False,
) -> LedgerResult:
    """Run the portfolio ledger over a return panel.

    Parameters
    ----------
    targets
        ``(J, N)`` weights executed at the close of each execution day. Each row
        must be non-negative and sum to 1 (long-only, fully invested, with cash
        as one of the N columns).
    exec_days
        ``(J,)`` day indices at which those weights are executed. Must be
        strictly increasing and already lagged for execution timing (E7).
    simple_returns
        ``(D, N)`` **simple** daily returns including the cash column. Simple,
        not log: a portfolio's return is the weighted sum of its assets' simple
        returns.
    cash_index
        Column index of the cash asset. Before the first trade the portfolio is
        held entirely in cash, so it earns exactly the risk-free rate.
    cost_per_unit
        Transaction cost per unit of notional traded (default 5 bp).
    method
        ``"fast"`` (vectorised) or ``"reference"`` (the readable loop). They are
        asserted equivalent in the test suite.
    return_diagnostics
        If True, also return the ``(D, N)`` held-weight path. Skipped during
        training because it is not needed for the loss.

    Returns
    -------
    LedgerResult
        Daily net returns and the accompanying accounting detail.
    """
    targets = targets.to(LEDGER_DTYPE)
    simple_returns = simple_returns.to(LEDGER_DTYPE)
    exec_days = _validate_inputs(targets, exec_days, simple_returns, cash_index)

    if method == "reference":
        return _reference_ledger(
            targets, exec_days, simple_returns, cash_index, cost_per_unit, return_diagnostics
        )
    if method == "fast":
        return _fast_ledger(
            targets, exec_days, simple_returns, cash_index, cost_per_unit, return_diagnostics
        )
    raise ValueError(f"method must be 'fast' or 'reference', got {method!r}.")


# --------------------------------------------------------------------------- #
# Reference implementation - readable, mirrors the plan's pseudocode
# --------------------------------------------------------------------------- #
def _reference_ledger(
    targets: torch.Tensor,
    exec_days: list[int],
    simple_returns: torch.Tensor,
    cash_index: int,
    cost_per_unit: float,
    return_diagnostics: bool,
) -> LedgerResult:
    """Loop over holding periods, exactly as the plan's Appendix C.1 states it."""
    n_days, n_assets = simple_returns.shape

    # Before the first trade everything sits in cash, so the portfolio earns the
    # cash column's return, which is r_f by construction.
    gross_returns = simple_returns[:, cash_index].clone()
    costs = torch.zeros(n_days, dtype=LEDGER_DTYPE)
    turnovers: list[torch.Tensor] = []
    held_weights = torch.zeros(n_days, n_assets, dtype=LEDGER_DTYPE)
    held_weights[:, cash_index] = 1.0

    # The portfolio held immediately before the first trade: all cash.
    weights_drifted = torch.zeros(n_assets, dtype=LEDGER_DTYPE)
    weights_drifted[cash_index] = 1.0

    for period, execution_day in enumerate(exec_days):
        target = targets[period]

        costs[execution_day] = transaction_cost(target, weights_drifted, cost_per_unit)
        turnovers.append(one_way_turnover(target, weights_drifted))

        # The holding period runs to the next execution day inclusive: on that
        # day the OLD weights still earn the day's return, and only then do we
        # trade at the close.
        period_end = exec_days[period + 1] if period + 1 < len(exec_days) else n_days - 1
        if period_end <= execution_day:
            weights_drifted = target
            continue

        window = simple_returns[execution_day + 1 : period_end + 1]     # (h, N)
        growth = torch.cumprod(1.0 + window, dim=0)                     # (h, N)
        value_path = growth @ target                                    # (h,)

        # V_e = 1 by definition, so prepend it to difference against.
        value_with_anchor = torch.cat([torch.ones(1, dtype=LEDGER_DTYPE), value_path])
        gross_returns[execution_day + 1 : period_end + 1] = (
            value_with_anchor[1:] / value_with_anchor[:-1] - 1.0
        )

        if return_diagnostics:
            held_weights[execution_day + 1 : period_end + 1] = (
                target.unsqueeze(0) * growth
            ) / value_path.unsqueeze(1)

        # Weights actually held at the close of the last day of the period -
        # these are what the next trade trades away from.
        weights_drifted = (target * growth[-1]) / value_path[-1]
        # Explicit renormalisation (B3): without it, floating-point error in the
        # cumulative product compounds across thousands of days and the weights
        # slowly stop summing to one.
        weights_drifted = weights_drifted / weights_drifted.sum()

    net_returns = gross_returns - costs
    return LedgerResult(
        net_returns=net_returns,
        gross_returns=gross_returns,
        costs=costs,
        turnover=torch.stack(turnovers) if turnovers else torch.zeros(0, dtype=LEDGER_DTYPE),
        held_weights=held_weights if return_diagnostics else None,
    )


# --------------------------------------------------------------------------- #
# Fast implementation - no Python loop over holding periods
# --------------------------------------------------------------------------- #
def _fast_ledger(
    targets: torch.Tensor,
    exec_days: list[int],
    simple_returns: torch.Tensor,
    cash_index: int,
    cost_per_unit: float,
    return_diagnostics: bool,
) -> LedgerResult:
    """Vectorised ledger using one global cumulative product.

    The identity that removes the loop: within the holding period beginning at
    ``e``, ``G_d = C_d / C_e`` where ``C_d = prod_{s<=d}(1 + rho_s)`` is the
    growth since the start of the sample. Defining the per-period "unit
    holdings" ``u = w / C_e``, the portfolio value is simply ``V_d = u . C_d``
    for every day in the period. Broadcasting ``u`` across the days it applies
    to turns the whole ledger into two elementwise products and two row sums.
    """
    n_days, n_assets = simple_returns.shape

    # C[d] = prod_{s=0..d} (1 + rho_s), with a leading 1 so that C_full[d] is
    # "growth up to and including day d-1", making C_{-1} = 1 addressable.
    cumulative_growth = torch.cumprod(1.0 + simple_returns, dim=0)
    growth_with_anchor = torch.cat(
        [torch.ones(1, n_assets, dtype=LEDGER_DTYPE), cumulative_growth], dim=0
    )  # (D+1, N); growth_with_anchor[d] == C_{d-1}

    # A virtual period covering the days before the first trade: all cash,
    # "executed" at day -1 where cumulative growth is 1. This makes the
    # pre-trade stretch earn the cash return with no special-casing.
    cash_weights = torch.zeros(1, n_assets, dtype=LEDGER_DTYPE)
    cash_weights[0, cash_index] = 1.0
    all_targets = torch.cat([cash_weights, targets], dim=0)              # (J+1, N)
    all_exec_days = np.concatenate([[-1], np.asarray(exec_days)])        # (J+1,)

    # Unit holdings u_j = w_j / C_{e_j}; index e_j + 1 hits C_{e_j} in the
    # anchored array (and index 0 gives C_{-1} = 1 for the virtual period).
    growth_at_execution = growth_with_anchor[torch.as_tensor(all_exec_days + 1)]
    unit_holdings = all_targets / growth_at_execution                    # (J+1, N)

    # Which period owns each day's return? Day d belongs to the period whose
    # execution day is the largest one strictly before d. On an execution day
    # the OLD weights still earn the return, which this assignment gives us.
    day_indices = np.arange(n_days)
    period_of_day = np.searchsorted(all_exec_days, day_indices, side="left") - 1
    period_of_day = np.clip(period_of_day, 0, len(all_exec_days) - 1)
    period_index = torch.as_tensor(period_of_day, dtype=torch.long)

    holdings_by_day = unit_holdings[period_index]                        # (D, N)
    value_today = (holdings_by_day * cumulative_growth).sum(dim=1)       # (D,)
    value_yesterday = (holdings_by_day * growth_with_anchor[:-1]).sum(dim=1)
    gross_returns = value_today / value_yesterday - 1.0

    # Held weights at each close, used for the next trade's cost and for the
    # holdings-count diagnostics.
    weights_held = (holdings_by_day * cumulative_growth) / value_today.unsqueeze(1)
    weights_held = weights_held / weights_held.sum(dim=1, keepdim=True)  # B3

    # Weights drifted into each execution day: the held weights at that close,
    # which belong to the previous period. If the first trade is on day 0 there
    # is no previous period, so it trades away from cash.
    execution_index = torch.as_tensor(np.asarray(exec_days), dtype=torch.long)
    weights_drifted = weights_held[execution_index]                      # (J, N)
    if len(exec_days) > 0 and exec_days[0] == 0:
        weights_drifted = weights_drifted.clone()
        weights_drifted[0] = cash_weights[0]

    per_rebalance_cost = transaction_cost(targets, weights_drifted, cost_per_unit)
    turnover = one_way_turnover(targets, weights_drifted)

    costs = torch.zeros(n_days, dtype=LEDGER_DTYPE)
    costs = costs.index_put((execution_index,), per_rebalance_cost)

    net_returns = gross_returns - costs
    return LedgerResult(
        net_returns=net_returns,
        gross_returns=gross_returns,
        costs=costs,
        turnover=turnover,
        held_weights=weights_held if return_diagnostics else None,
    )


# --------------------------------------------------------------------------- #
# Execution timing (E7)
# --------------------------------------------------------------------------- #
def weekly_execution_schedule(
    dates: np.ndarray | "pd.DatetimeIndex",
) -> tuple[np.ndarray, np.ndarray]:
    """Build the weekly signal/execution schedule under the E7 timing convention.

    The convention, stated once and implemented once:

        signal computed through the close of day t  ->  trade executes at the
        close of day t+1  ->  the new portfolio earns from t+2.

    Concretely: the **last trading day of each week** is the signal day, and the
    **next trading day** is the execution day. Using the last trading day of the
    week rather than "every Friday" means a holiday-shortened week signals on the
    Thursday instead of silently skipping.

    Keeping this lag here - outside the ledger, which receives execution days
    already lagged - is what makes it independently testable (C7). A model that
    computed its signal on day ``t`` and traded at the *same* close would be
    trading on information it could not have acted upon, and the whole result
    would be worthless.

    Parameters
    ----------
    dates
        The trading calendar, as a ``DatetimeIndex`` or array of datetimes.

    Returns
    -------
    signal_days, execution_days
        Positional indices into ``dates``. Each signal day is strictly earlier
        than its paired execution day. The final week is dropped when there is
        no following trading day to execute on.
    """
    import pandas as pd

    calendar = pd.DatetimeIndex(dates)
    if len(calendar) < 2:
        return np.array([], dtype=int), np.array([], dtype=int)

    positions = pd.Series(np.arange(len(calendar)), index=calendar)

    # Last trading day of each ISO week -> the signal day.
    week_keys = calendar.to_period("W")
    signal_positions = positions.groupby(week_keys).last().to_numpy()

    # Execute on the following trading day; drop any signal with no day after it.
    execution_positions = signal_positions + 1
    usable = execution_positions < len(calendar)

    return signal_positions[usable], execution_positions[usable]


# --------------------------------------------------------------------------- #
# NumPy wrapper for the baselines
# --------------------------------------------------------------------------- #
def run_ledger_numpy(
    targets: np.ndarray,
    exec_days: np.ndarray | list[int],
    simple_returns: np.ndarray,
    cash_index: int,
    cost_per_unit: float = DEFAULT_COST_PER_UNIT,
    return_diagnostics: bool = False,
) -> dict[str, np.ndarray]:
    """NumPy-facing wrapper around :func:`run_ledger`, for the baselines.

    The baselines are plain NumPy, but they must be accounted for by exactly the
    same engine as the trained system - otherwise a difference in the ladder
    could come from bookkeeping rather than from the method. This wrapper simply
    converts, calls the same code, and converts back.

    Returns
    -------
    dict
        ``net_returns``, ``gross_returns``, ``costs``, ``turnover``, and
        ``held_weights`` (the last only if requested), all as NumPy arrays.
    """
    result = run_ledger(
        targets=torch.as_tensor(np.asarray(targets), dtype=LEDGER_DTYPE),
        exec_days=exec_days,
        simple_returns=torch.as_tensor(np.asarray(simple_returns), dtype=LEDGER_DTYPE),
        cash_index=cash_index,
        cost_per_unit=cost_per_unit,
        return_diagnostics=return_diagnostics,
    )
    output = {
        "net_returns": result.net_returns.detach().numpy(),
        "gross_returns": result.gross_returns.detach().numpy(),
        "costs": result.costs.detach().numpy(),
        "turnover": result.turnover.detach().numpy(),
    }
    if return_diagnostics and result.held_weights is not None:
        output["held_weights"] = result.held_weights.detach().numpy()
    return output


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def _validate_inputs(
    targets: torch.Tensor,
    exec_days: torch.Tensor | np.ndarray | list[int],
    simple_returns: torch.Tensor,
    cash_index: int,
) -> list[int]:
    """Check shapes and the simplex constraint; return execution days as a list.

    Failing here is much cheaper than discovering a misaligned weight schedule
    in the middle of a training run, where it would simply look like a bad
    strategy rather than a bug.
    """
    if targets.ndim != 2:
        raise ValueError(f"targets must be (J, N), got shape {tuple(targets.shape)}.")
    if simple_returns.ndim != 2:
        raise ValueError(
            f"simple_returns must be (D, N), got shape {tuple(simple_returns.shape)}."
        )

    n_periods, n_assets = targets.shape
    n_days, n_return_assets = simple_returns.shape

    if n_assets != n_return_assets:
        raise ValueError(
            f"targets has {n_assets} assets but simple_returns has {n_return_assets}."
        )
    if not 0 <= cash_index < n_assets:
        raise ValueError(f"cash_index {cash_index} is outside [0, {n_assets}).")

    days = [int(day) for day in np.asarray(exec_days).ravel()]
    if len(days) != n_periods:
        raise ValueError(f"{len(days)} execution days for {n_periods} target rows.")
    if any(day < 0 or day >= n_days for day in days):
        raise ValueError(f"execution days must lie in [0, {n_days}).")
    if any(later <= earlier for earlier, later in zip(days, days[1:])):
        raise ValueError("execution days must be strictly increasing.")

    if torch.isnan(targets).any():
        raise ValueError("targets contain NaN.")
    if (targets < -1e-9).any():
        raise ValueError("targets contain negative weights; the portfolio is long-only.")
    row_sums = targets.sum(dim=1)
    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6):
        worst = (row_sums - 1.0).abs().max().item()
        raise ValueError(
            f"targets must be fully invested (rows summing to 1); worst deviation {worst:.2e}."
        )

    return days
