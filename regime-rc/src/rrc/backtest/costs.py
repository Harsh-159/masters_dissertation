"""Transaction-cost and turnover conventions, pinned once for the whole project.

Why this is its own module
--------------------------
Three defensible cost/turnover conventions coexist in the literature, and mixing
them corrupts comparisons in a way that is very hard to spot afterwards. The
project pins one set here, and **every** rung and **every** baseline is charged
by these same functions. If a baseline were charged differently from the model,
the ladder's comparisons would be meaningless.

The conventions (locked as A9)
------------------------------
Let ``w_exec`` be the weights we trade into and ``w_drift`` the weights we
currently hold (drifted by market moves since the last trade).

* **Transaction cost per rebalance** ``= c * sum_i |w_exec_i - w_drift_i|``
  with ``c = 5`` basis points. Note the full sum, not half: cost is charged on
  every unit of notional that changes hands, and a rebalance both sells and
  buys.

* **One-way turnover per rebalance** ``= 0.5 * sum_i |w_exec_i - w_drift_i|``.
  Here the half *is* correct, because "turnover" conventionally counts the
  notional traded in one direction only - selling 10% of the book to buy
  something else is 10% turnover, not 20%.

* **Annualised turnover** = total one-way turnover x (252 / span days) -
  calendar-based, which equals A9's "52 x the mean" for weekly cadence and
  stays correct for arms that trade less often (buy-and-hold trades once).

The factor-of-two difference between the cost formula and the turnover formula
is deliberate and is exactly the trap this module exists to prevent: using the
turnover number to compute costs would halve them silently.
"""

from __future__ import annotations

import numpy as np
import torch

# Five basis points per unit of notional traded (locked default, A9).
DEFAULT_COST_PER_UNIT = 5e-4

# Weekly rebalancing, so a year contains 52 rebalances (E7 / Appendix B).
REBALANCES_PER_YEAR = 52


def traded_notional(weights_executed: torch.Tensor, weights_drifted: torch.Tensor) -> torch.Tensor:
    """Total absolute weight change at a rebalance: ``sum_i |w_exec_i - w_drift_i|``.

    This is the quantity costs are charged on - every unit that changes hands,
    counting both the sells and the buys.

    Parameters
    ----------
    weights_executed
        Target weights actually traded into, shape ``(N,)`` or ``(J, N)``.
    weights_drifted
        Currently held weights, same shape.

    Returns
    -------
    torch.Tensor
        Scalar, or ``(J,)`` if a batch of rebalances was passed.
    """
    return (weights_executed - weights_drifted).abs().sum(dim=-1)


def transaction_cost(
    weights_executed: torch.Tensor,
    weights_drifted: torch.Tensor,
    cost_per_unit: float = DEFAULT_COST_PER_UNIT,
) -> torch.Tensor:
    """Cost of a rebalance as a fraction of portfolio value.

    Subtracted directly from the execution day's return, which is correct to
    first order: paying ``x`` of your value in costs reduces that day's return
    by ``x``.

    Parameters
    ----------
    weights_executed, weights_drifted
        Target and currently-held weights.
    cost_per_unit
        Cost ``c`` per unit of notional traded. Default 5 bp.

    Returns
    -------
    torch.Tensor
        The cost, as a return decrement.
    """
    return cost_per_unit * traded_notional(weights_executed, weights_drifted)


def one_way_turnover(
    weights_executed: torch.Tensor, weights_drifted: torch.Tensor
) -> torch.Tensor:
    """One-way turnover of a rebalance: ``0.5 * sum_i |w_exec_i - w_drift_i|``.

    Reported, never used to compute costs - see the module docstring for why
    conflating the two would halve the charged costs.
    """
    return 0.5 * traded_notional(weights_executed, weights_drifted)


def annualised_turnover(
    per_rebalance_turnover: np.ndarray | torch.Tensor,
    n_days: int,
    trading_days_per_year: int = 252,
) -> float:
    """Annualise one-way turnover by CALENDAR time, not by rebalance count.

    ``total one-way turnover over the span x (252 / span length in days)``.

    Why calendar-based rather than A9's literal "52 x the per-rebalance mean":
    the two are identical for any weekly-rebalancing strategy (~52 rebalances
    per 252 days), which is every model and every weekly baseline - but the
    52x form silently assumes the cadence. Buy-and-hold trades ONCE, and under
    the 52x convention its single entry (one-way turnover 1.0) was annualised
    to a nonsensical 52x a year; under the calendar form it is ~1.0x, which is
    what "you turned the book over once this year" should read.

    Parameters
    ----------
    per_rebalance_turnover
        One-way turnover of each rebalance in the span.
    n_days
        Trading days the span covers.
    trading_days_per_year
        Annualisation basis.

    Returns
    -------
    float
        Annualised one-way turnover; 1.0 means the whole book is replaced
        once a year.
    """
    values = (
        per_rebalance_turnover.detach().cpu().numpy()
        if isinstance(per_rebalance_turnover, torch.Tensor)
        else np.asarray(per_rebalance_turnover)
    )
    if values.size == 0 or n_days <= 0:
        return 0.0
    return float(np.nansum(values) * trading_days_per_year / n_days)
