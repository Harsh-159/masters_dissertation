"""The availability matrix: which assets were actually tradeable on each day.

The problem it solves
---------------------
XLRE begins in October 2015 and XLC in June 2018, yet the sector universe starts
in 1999. A model must not be allowed to hold an asset that did not exist, and it
must not be *fed* a fabricated price for one either. Both failures are silent:
the backtest simply reports a return that could never have been earned.

``A[t, i] in {0, 1}`` states, for each trading day and asset, whether the asset
was alive and tradeable. It is derived from the data itself - an asset is
available from its first real observation to its last - rather than from a
hand-maintained list of inception dates that can drift out of step with reality.

How it is consumed downstream
-----------------------------
* **Model heads** add ``-1e9`` to the logits of unavailable assets before the
  softmax, so their weight is exactly zero rather than merely small.
* **Baselines** allocate over the active set only, so that 1/N means "one over
  the number of assets that existed", not "one over eleven" in 1999.
* **Cardinality** counts only active risky holdings.

This is the mechanism that lets late entrants be tradeable from inception while
never driving the reservoir (locked decision R4).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# The additive mask applied to logits of unavailable assets. Large enough that
# exp(logit - 1e9) underflows to exactly 0.0, so unavailable assets receive
# precisely zero weight rather than a tiny nonzero one.
UNAVAILABLE_LOGIT_PENALTY = -1e9


def build_availability_matrix(observed_prices: pd.DataFrame) -> pd.DataFrame:
    """Derive ``A[t, i]`` from a panel of real (unfilled) observations.

    An asset is marked available from its first real observation through its
    last. Days inside that span with no observation - a suspended stock, a
    missing print - stay available, because the asset existed and would have
    been tradeable; the forward-filled price panel covers those days.

    Parameters
    ----------
    observed_prices
        Panel of *observed* prices, NaN wherever no real price was seen. Pass
        the unfilled panel here: using the forward-filled one would extend
        availability past an asset's delisting by the fill limit.

    Returns
    -------
    pandas.DataFrame
        Integer 0/1 matrix with the same index and columns as the input.
    """
    availability = pd.DataFrame(
        0, index=observed_prices.index, columns=observed_prices.columns, dtype="int8"
    )

    for ticker in observed_prices.columns:
        column = observed_prices[ticker]
        first_seen = column.first_valid_index()
        last_seen = column.last_valid_index()

        if first_seen is None:
            logger.warning("%s has no observations in the window; never available.", ticker)
            continue

        availability.loc[first_seen:last_seen, ticker] = 1

    availability.index.name = "date"
    return availability


def logit_mask(availability: pd.DataFrame) -> pd.DataFrame:
    """Convert the 0/1 availability matrix into an additive logit mask.

    Parameters
    ----------
    availability
        Matrix from :func:`build_availability_matrix`.

    Returns
    -------
    pandas.DataFrame
        ``0.0`` where available and ``-1e9`` where not, ready to be added to
        raw logits before a softmax.
    """
    return (1 - availability).astype("float64") * UNAVAILABLE_LOGIT_PENALTY


def active_assets_on(availability: pd.DataFrame, date: pd.Timestamp) -> list[str]:
    """List the assets tradeable on a given date.

    Used by the baselines, which must allocate over the active set: an equal
    weight baseline that divides by the full universe size in 1999 would be
    holding phantom positions in assets that did not exist.

    Parameters
    ----------
    availability
        Matrix from :func:`build_availability_matrix`.
    date
        The trading day to query.

    Returns
    -------
    list of str
        Tickers with ``A[t, i] == 1``, in column order.
    """
    row = availability.loc[date]
    return [ticker for ticker in availability.columns if row[ticker] == 1]


def availability_summary(availability: pd.DataFrame) -> pd.DataFrame:
    """Summarise each asset's availability, for the data-coverage report.

    Returns
    -------
    pandas.DataFrame
        One row per asset: first and last available date, number of available
        days, and the fraction of the window for which it existed.
    """
    n_days = len(availability.index)
    rows = []

    for ticker in availability.columns:
        active_days = availability.index[availability[ticker] == 1]
        rows.append(
            {
                "ticker": ticker,
                "first_available": active_days.min().date() if len(active_days) else None,
                "last_available": active_days.max().date() if len(active_days) else None,
                "days_available": int(len(active_days)),
                "fraction_of_window": round(len(active_days) / n_days, 4) if n_days else 0.0,
            }
        )

    return pd.DataFrame(rows).set_index("ticker")


def assert_no_prices_before_inception(
    filled_prices: pd.DataFrame, availability: pd.DataFrame
) -> None:
    """Assert that no price exists where the asset was unavailable.

    This is the executable form of "never invent pre-inception prices". It is
    called by the data build, so a regression in the alignment logic stops the
    build rather than quietly producing a backtest that trades XLC in 2008.

    Raises
    ------
    AssertionError
        If any price is present on a day the asset was not available.
    """
    prices_present = filled_prices.notna().to_numpy()
    asset_unavailable = (availability == 0).to_numpy()
    violations = np.logical_and(prices_present, asset_unavailable)

    if violations.any():
        row_indices, column_indices = np.nonzero(violations)
        examples = [
            f"{filled_prices.columns[column]} on "
            f"{filled_prices.index[row].date()}"
            for row, column in list(zip(row_indices, column_indices))[:5]
        ]
        raise AssertionError(
            f"{int(violations.sum())} price(s) exist where the asset was "
            f"unavailable. Examples: {examples}"
        )
