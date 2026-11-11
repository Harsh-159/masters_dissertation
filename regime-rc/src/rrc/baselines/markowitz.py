"""Rolling Ledoit-Wolf mean-variance - the "simple adaptivity" baseline.

Why this baseline exists
------------------------
The naive baselines never react to anything. This one does, in the most
classical way available: re-estimate expected returns and covariance from
trailing data once a month, and hold the portfolio a textbook mean-variance
optimiser would pick. If the trained system cannot beat *simple adaptivity*, its
machinery is not paying for itself - that is the question rung 0 exists to ask.

The three design rules, and why each is there
---------------------------------------------
1. **Ledoit-Wolf shrinkage for the covariance.** A sample covariance of ~10
   assets from 252 daily observations is noisy, and a mean-variance optimiser
   is an error maximiser: it leans hardest on exactly the entries the sample
   estimated worst. Ledoit and Wolf's estimator shrinks the sample matrix
   toward a scaled identity by an analytically optimal amount, taming those
   errors. The canonical scikit-learn implementation is used directly.

2. **Cash never enters the covariance (review fix E1).** Cash has zero
   variance, and a zero-variance column makes the covariance matrix singular.
   Instead the optimiser works over risky assets with the constraint
   ``sum(w) <= 1``, and whatever it does not allocate sits in cash earning
   ``r_f``. Because expected returns are measured in *excess* of that same
   ``r_f``, cash is automatically the zero-excess-return, zero-risk asset - the
   optimiser de-risks into it exactly when no risky asset looks worth its risk.

3. **Monthly refit, weekly execution.** Estimates are refreshed on the first
   signal day of each month ("simple adaptivity" - slow, deliberate). Between
   refits the portfolio still rebalances weekly back to the current target,
   like every other strategy in the project, so drift is controlled and the
   cost clock ticks identically for everyone.

The optimisation problem, stated once
-------------------------------------
    maximise    mu' w  -  (gamma / 2) w' Sigma w
    subject to  w >= 0,   sum(w) <= 1

with ``mu`` the trailing mean of daily excess simple returns, ``Sigma`` the
Ledoit-Wolf covariance of daily simple returns, and ``gamma`` a fixed risk
aversion (5.0, stated in the config and never tuned - a baseline does not get a
hyperparameter search). Solved with SciPy's SLSQP, which handles a 10-asset
quadratic programme in well under a millisecond.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import ledoit_wolf

logger = logging.getLogger(__name__)


def solve_long_only_mean_variance(
    expected_excess: np.ndarray,
    covariance: np.ndarray,
    risk_aversion: float,
) -> np.ndarray:
    """Solve the long-only mean-variance problem with a cash-slack budget.

    Parameters
    ----------
    expected_excess
        ``(n,)`` expected daily excess returns of the candidate assets.
    covariance
        ``(n, n)`` covariance of their daily returns (already shrunk).
    risk_aversion
        The gamma in ``mu'w - (gamma/2) w'Sigma w``. Larger = more cautious.

    Returns
    -------
    numpy.ndarray
        ``(n,)`` non-negative weights with ``sum(w) <= 1``. The shortfall from
        1 is the cash weight, assigned by the caller.
    """
    n_assets = len(expected_excess)

    def negative_utility(weights: np.ndarray) -> float:
        risk_penalty = 0.5 * risk_aversion * weights @ covariance @ weights
        return -(expected_excess @ weights - risk_penalty)

    def negative_utility_gradient(weights: np.ndarray) -> np.ndarray:
        return -(expected_excess - risk_aversion * covariance @ weights)

    budget_constraint = {
        "type": "ineq",  # SLSQP convention: fun(w) >= 0 means feasible
        "fun": lambda weights: 1.0 - weights.sum(),
        "jac": lambda weights: -np.ones(n_assets),
    }

    solution = minimize(
        negative_utility,
        x0=np.full(n_assets, 1.0 / n_assets),  # start at 1/N: feasible, neutral
        jac=negative_utility_gradient,
        bounds=[(0.0, 1.0)] * n_assets,
        constraints=[budget_constraint],
        method="SLSQP",
        options={"maxiter": 200, "ftol": 1e-10},
    )

    if not solution.success:
        # A failed solve falls back to 1/N rather than crashing a 20-year run;
        # it is logged so a systematic failure cannot hide.
        logger.warning("Mean-variance solve failed (%s); using 1/N.", solution.message)
        return np.full(n_assets, 1.0 / n_assets)

    # Numerical dust from the solver (weights like 1e-12) is clipped away.
    weights = np.clip(solution.x, 0.0, 1.0)
    weights[weights < 1e-10] = 0.0
    if weights.sum() > 1.0:  # tolerance slack from the inequality constraint
        weights = weights / weights.sum()
    return weights


def mean_variance_schedule(
    risky_returns: pd.DataFrame,
    risk_free: pd.Series,
    availability: pd.DataFrame,
    dates: pd.DatetimeIndex,
    signal_positions: np.ndarray,
    execution_positions: np.ndarray,
    n_columns: int,
    cash_index: int,
    lookback: int,
    min_observations: int,
    risk_aversion: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the mean-variance target schedule with monthly refits.

    At the first signal day of each calendar month, expected excess returns and
    the Ledoit-Wolf covariance are re-estimated from the trailing ``lookback``
    days up to and including the signal day, and the optimiser is re-run over
    the assets that are active and have at least ``min_observations`` finite
    returns in that window. Signal days later in the same month reuse the
    month's weights unchanged (the weekly execution then simply trades back the
    drift).

    Parameters
    ----------
    risky_returns
        Daily simple returns of the risky assets on the full calendar.
    risk_free
        Daily risk-free return, aligned with ``risky_returns`` - subtracted to
        make the expected-return estimate an *excess* estimate.
    availability
        The 0/1 availability matrix.
    dates
        The full calendar (needed to detect month boundaries at signal days).
    signal_positions
        Absolute positions of the signal days.
    execution_positions
        Segment-relative execution days, returned unchanged.
    n_columns, cash_index
        Target-matrix width and the cash column.
    lookback
        Estimation window in trading days (252 = one year).
    min_observations
        Minimum finite returns an asset needs inside the window to be eligible.
    risk_aversion
        Fixed gamma for the optimiser.

    Returns
    -------
    targets, execution_positions
        ``(J, N)`` weights (cash as the residual) and their execution days.
    """
    returns_matrix = risky_returns.to_numpy()
    excess_matrix = returns_matrix - risk_free.to_numpy()[:, None]
    availability_rows = availability.to_numpy()

    targets = np.zeros((len(signal_positions), n_columns), dtype=np.float64)
    current_weights: np.ndarray | None = None  # full-width row, reused within a month
    last_refit_month: tuple[int, int] | None = None

    for row, signal_position in enumerate(signal_positions):
        signal_day = dates[signal_position]
        month_key = (signal_day.year, signal_day.month)

        if month_key != last_refit_month or current_weights is None:
            current_weights = _refit(
                returns_matrix=returns_matrix,
                excess_matrix=excess_matrix,
                availability_row=availability_rows[signal_position],
                signal_position=signal_position,
                n_columns=n_columns,
                cash_index=cash_index,
                lookback=lookback,
                min_observations=min_observations,
                risk_aversion=risk_aversion,
            )
            last_refit_month = month_key

        targets[row] = current_weights

    return targets, execution_positions


def _refit(
    returns_matrix: np.ndarray,
    excess_matrix: np.ndarray,
    availability_row: np.ndarray,
    signal_position: int,
    n_columns: int,
    cash_index: int,
    lookback: int,
    min_observations: int,
    risk_aversion: float,
) -> np.ndarray:
    """Re-estimate and re-optimise at one signal day; return a full-width row."""
    weights_row = np.zeros(n_columns, dtype=np.float64)

    window_start = max(0, signal_position - lookback + 1)
    trailing_returns = returns_matrix[window_start : signal_position + 1]
    trailing_excess = excess_matrix[window_start : signal_position + 1]

    active = availability_row == 1
    enough_history = np.isfinite(trailing_returns).sum(axis=0) >= min_observations
    eligible = active & enough_history

    if eligible.sum() < 2:
        # Cannot estimate a covariance from one asset; park in cash. This only
        # occurs in degenerate configurations, never in the real universes.
        weights_row[cash_index] = 1.0
        return weights_row

    eligible_columns = np.flatnonzero(eligible)

    # Keep only rows where every eligible asset has a finite return, so the
    # covariance is estimated on a complete rectangle. After the min-history
    # filter this drops at most a young asset's pre-inception rows.
    block = trailing_returns[:, eligible_columns]
    complete_rows = np.isfinite(block).all(axis=1)
    block = block[complete_rows]
    excess_block = trailing_excess[:, eligible_columns][complete_rows]

    expected_excess = excess_block.mean(axis=0)
    shrunk_covariance, shrinkage = ledoit_wolf(block, assume_centered=False)
    logger.debug(
        "refit at position %d: %d assets, %d rows, shrinkage %.3f",
        signal_position, len(eligible_columns), block.shape[0], shrinkage,
    )

    risky_weights = solve_long_only_mean_variance(
        expected_excess, shrunk_covariance, risk_aversion
    )

    weights_row[eligible_columns] = risky_weights
    weights_row[cash_index] = 1.0 - risky_weights.sum()  # cash is the residual
    return weights_row
