"""The naive baselines: 1/N, buy-and-hold, and inverse-volatility risk parity.

What a baseline is in this project
----------------------------------
A baseline is a function that, at each weekly signal day, produces target
weights using **only information available at that day's close** - exactly the
information contract the trained models will operate under (E7: signal at the
close of day t, execute at the close of day t+1). The weights then go through
the *same* ledger, the *same* 5 bp cost function, and the *same* metrics as
every model. Nothing about a baseline is special-cased, which is the whole
point: rung 0 is the humility line, and it is only meaningful if the comparison
is exact.

Every function here returns the same thing: a ``(J, N)`` matrix of target
weights (cash column included) and the execution-day positions they trade at.

The three baselines
-------------------
* **1/N** - equal weight over the risky assets *active on the signal day*.
  "Active" matters: in the sector universe before October 2015 there are nine
  live sectors, not eleven, and dividing by eleven would hold phantom positions.
  DeMiguel, Garlappi and Uppal (2009) showed this embarrassingly simple rule is
  brutally hard to beat out of sample - which is exactly why it anchors rung 0.

* **Buy-and-hold** - one trade into a single asset (SPY, the market benchmark)
  at the start, then nothing. The only baseline that never rebalances, so it
  pays the cost function exactly once.

* **Inverse-volatility risk parity** - weight each active asset in proportion
  to one over its trailing realised volatility, so every position contributes
  roughly equal *risk* rather than equal money. Bonds get more weight than
  equities; the portfolio's realised volatility should come out below 1/N's,
  and the run script asserts that ordering as a sanity check.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def one_over_n_schedule(
    availability: pd.DataFrame,
    signal_positions: np.ndarray,
    execution_positions: np.ndarray,
    n_columns: int,
    cash_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Equal weight over the risky assets active at each signal day.

    Parameters
    ----------
    availability
        The 0/1 availability matrix on the universe's full calendar, risky
        assets only (no cash column).
    signal_positions
        Absolute row positions (into the full calendar) of the weekly signal
        days. Availability is read at these rows - the close of day t.
    execution_positions
        Positions of the execution days, relative to the segment being run.
        Returned unchanged; they are part of the schedule contract.
    n_columns
        Width of the target matrix (risky assets + cash).
    cash_index
        Column index of cash. 1/N is fully invested, so cash weight is 0.

    Returns
    -------
    targets, execution_positions
        ``(J, N)`` weights and the execution positions they pair with.
    """
    availability_rows = availability.to_numpy()
    targets = np.zeros((len(signal_positions), n_columns), dtype=np.float64)

    for row, signal_position in enumerate(signal_positions):
        active = availability_rows[signal_position] == 1
        n_active = int(active.sum())
        if n_active == 0:
            # No live asset at all (cannot happen in these universes, but the
            # honest fallback is cash rather than a crash).
            targets[row, cash_index] = 1.0
            continue
        targets[row, np.flatnonzero(active)] = 1.0 / n_active

    return targets, execution_positions


def buy_and_hold_schedule(
    asset_index: int,
    first_execution_position: int,
    n_columns: int,
) -> tuple[np.ndarray, np.ndarray]:
    """One trade into a single asset at the first execution day, then hold.

    Parameters
    ----------
    asset_index
        Column of the asset to buy (SPY in the benchmark's two-column panel).
    first_execution_position
        The segment-relative day the single trade executes on.
    n_columns
        Width of the target matrix.

    Returns
    -------
    targets, execution_positions
        A single-row schedule: buy once, never trade again. The ledger then
        lets the position drift for the rest of the segment, which for one
        asset means the portfolio simply *is* that asset (engine test B2).
    """
    targets = np.zeros((1, n_columns), dtype=np.float64)
    targets[0, asset_index] = 1.0
    return targets, np.array([first_execution_position])


def inverse_volatility_schedule(
    risky_returns: pd.DataFrame,
    availability: pd.DataFrame,
    signal_positions: np.ndarray,
    execution_positions: np.ndarray,
    n_columns: int,
    cash_index: int,
    window: int,
    min_observations: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Weights proportional to one over trailing realised volatility.

    At each signal day t, the volatility of each asset is estimated from the
    trailing ``window`` days of returns **up to and including t** - information
    available at the close of t, per the E7 timing contract. An asset qualifies
    only if it is active at t and has at least ``min_observations`` finite
    returns in the window; a recently launched asset (XLRE in late 2015, say)
    is therefore excluded until its history is long enough to estimate from,
    then joins automatically.

    Parameters
    ----------
    risky_returns
        Daily returns of the risky assets on the full calendar (NaN before an
        asset's inception). Used purely for volatility estimation.
    availability
        The 0/1 availability matrix, aligned with ``risky_returns``.
    signal_positions
        Absolute positions of the signal days in the full calendar.
    execution_positions
        Segment-relative execution days, returned unchanged.
    n_columns, cash_index
        Target-matrix width and the cash column (weight 0: fully invested).
    window
        Trailing estimation window in trading days (63 = one quarter).
    min_observations
        Minimum finite returns required inside the window.

    Returns
    -------
    targets, execution_positions
        ``(J, N)`` weights and their execution days.
    """
    returns_matrix = risky_returns.to_numpy()
    availability_rows = availability.to_numpy()
    targets = np.zeros((len(signal_positions), n_columns), dtype=np.float64)

    for row, signal_position in enumerate(signal_positions):
        window_start = max(0, signal_position - window + 1)
        trailing = returns_matrix[window_start : signal_position + 1]  # includes t

        active = availability_rows[signal_position] == 1
        enough_history = np.isfinite(trailing).sum(axis=0) >= min_observations
        usable = active & enough_history

        if not usable.any():
            # Early in the sample nothing has enough history yet; fall back to
            # equal weight over whatever is active, or cash if nothing is.
            if active.any():
                targets[row, np.flatnonzero(active)] = 1.0 / active.sum()
            else:
                targets[row, cash_index] = 1.0
            continue

        volatilities = np.nanstd(trailing[:, usable], axis=0, ddof=1)
        # A zero volatility would give an infinite weight; treat such an asset
        # (a flat price over the whole window) as unusable instead.
        positive = volatilities > 0
        if not positive.any():
            targets[row, np.flatnonzero(active)] = 1.0 / active.sum()
            continue

        inverse = np.zeros_like(volatilities)
        inverse[positive] = 1.0 / volatilities[positive]
        weights = inverse / inverse.sum()

        targets[row, np.flatnonzero(usable)] = weights

    return targets, execution_positions
