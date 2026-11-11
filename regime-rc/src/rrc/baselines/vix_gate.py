"""The VIX-tercile statistical gate - rung 4's second, simpler comparison.

The rule
--------
Each signal day is classed by where yesterday's VIX sits against the terciles
of its own trailing year: bottom third -> "low" regime, middle -> "mid",
top -> "high". The membership is one-hot over those three classes, and the
experts retrain under it exactly as under the HMM gate (R6).

This is the folk wisdom of regime investing ("VIX high = risk off") made into
a baseline. If the learned, decision-defined gate cannot beat routing on a
single, universally watched fear index, the extra machinery has not earned its
keep - which is precisely why this comparison exists.

Causality, spelled out
----------------------
Two shifts keep every ingredient strictly in the past (the plan's "quantiles
computed on data up to t-1 only"):

1. the VIX series itself is lagged one day, so the value classified on day t
   is the one published on day t-1 - the same publication-lag treatment every
   signal received at P1;
2. the tercile boundaries are computed on the trailing 252-day window of that
   lagged series and then lagged one further day, so the boundaries a value is
   compared against never include the value itself.

A unit test perturbs the future and asserts the past labels are bit-identical.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

N_TERCILES = 3
TERCILE_WINDOW = 252
TERCILE_MIN_PERIODS = 126


def vix_tercile_labels(
    raw_vix: pd.Series, calendar: pd.DatetimeIndex
) -> pd.Series:
    """Causal tercile labels (0 low / 1 mid / 2 high) on a trading calendar.

    Parameters
    ----------
    raw_vix
        The raw VIX level series (unshifted, un-z-scored - the gate defines
        its own causal treatment so it is self-contained and auditable).
    calendar
        The universe's trading calendar.

    Returns
    -------
    pandas.Series
        Integer labels on ``calendar``; NaN during the warm-up before a full
        half-window of history exists.
    """
    # Align to the trading calendar (carry the last published level forward
    # across holidays), then apply the one-day publication lag.
    aligned = raw_vix.reindex(calendar.union(raw_vix.index)).ffill().reindex(calendar)
    lagged = aligned.shift(1)

    rolling = lagged.rolling(TERCILE_WINDOW, min_periods=TERCILE_MIN_PERIODS)
    # The boundaries are lagged one further day so the value classified is
    # never inside the window that produced its own boundaries.
    lower_tercile = rolling.quantile(1 / 3).shift(1)
    upper_tercile = rolling.quantile(2 / 3).shift(1)

    labels = pd.Series(np.nan, index=calendar, dtype="float64")
    ready = lagged.notna() & lower_tercile.notna() & upper_tercile.notna()
    labels[ready & (lagged <= lower_tercile)] = 0.0
    labels[ready & (lagged > lower_tercile) & (lagged < upper_tercile)] = 1.0
    labels[ready & (lagged >= upper_tercile)] = 2.0
    return labels


def memberships_at(
    labels: pd.Series, signal_positions: np.ndarray
) -> np.ndarray:
    """One-hot memberships over the terciles at given signal days.

    Raises if any signal day falls inside the warm-up, for the same reason the
    HMM gate does: the fold schedule should make it impossible, and silence
    would hide an alignment bug.
    """
    at_signals = labels.iloc[signal_positions]
    if at_signals.isna().any():
        raise ValueError(
            f"{int(at_signals.isna().sum())} signal day(s) precede the VIX "
            "tercile warm-up."
        )

    memberships = np.zeros((len(signal_positions), N_TERCILES), dtype=np.float64)
    memberships[np.arange(len(signal_positions)), at_signals.astype(int).to_numpy()] = 1.0
    return memberships
