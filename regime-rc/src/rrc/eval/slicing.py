"""Walk-forward fold schedule, embargo, and the holdout firewall.

The evaluation protocol (Section 12 of the implementation plan)
---------------------------------------------------------------
Every experiment in this project - baselines and models alike - is evaluated
walk-forward: train on an expanding window, select on the following year
(validation), report on the year after that (test), then roll everything forward
by one year. Nothing is ever scored on data from before its own information set.

The two universes have fixed fold tables::

    etf10     F01: train 2006-02 .. 2013-12, val 2014, test 2015
              F02: train 2006-02 .. 2014-12, val 2015, test 2016
              ...                                                 (10 folds)
              F10: train 2006-02 .. 2022-12, val 2023, test 2024

    sector11  F01: train 1999-01 .. 2005-12, val 2006, test 2007
              ...                                                 (18 folds)
              F18: train 1999-01 .. 2022-12, val 2023, test 2024

Note the deliberate consequence for the sector universe: its first test year is
2007, so **2008 is genuinely out-of-sample** there - the property (A7) that
makes sector11 the headline regime-shift testbed.

The embargo
-----------
One rebalance period (a trading week) is removed from the **end of every
training window**. Without it, the last training day and the first validation
day are adjacent, and anything with memory - a trailing window, a reservoir
state - straddles the boundary, letting a sliver of training-fold information
leak into the segment that is supposed to select hyperparameters.

The holdout firewall
--------------------
Everything from 2025-01-01 onward is the final holdout: touched exactly once, in
Week 8, by a dedicated script. Until then no run may even *load* those rows.
:func:`assert_no_holdout` is the guard every evaluation script calls on its
data before doing anything else - the discipline is enforced by code, not by
remembering.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# The holdout boundary. Data from this date (inclusive) onward is untouchable
# until the single Week-8 holdout run.
HOLDOUT_START = pd.Timestamp("2025-01-01")
HOLDOUT_END = pd.Timestamp("2026-06-30")

# One rebalance period, in trading days, removed from the end of every
# training window (weekly rebalancing -> 5 trading days).
EMBARGO_TRADING_DAYS = 5

# First validation year and last test year per universe, from the plan's
# Section 12 tables. Everything else about a fold follows from these.
_FOLD_ANCHORS = {
    "etf10": {"first_val_year": 2014, "last_test_year": 2024, "train_start": "2006-02-01"},
    "sector11": {"first_val_year": 2006, "last_test_year": 2024, "train_start": "1999-01-01"},
}


@dataclass(frozen=True)
class Fold:
    """One walk-forward fold.

    Attributes
    ----------
    name
        ``"F01"``, ``"F02"``, ... - the label that appears in the registry.
    train_start
        First calendar day of the training window (the universe start).
    val_year
        The calendar year used for validation (early stopping, lambda
        selection). The training window ends at the close of the year before,
        minus the embargo.
    test_year
        The calendar year results are reported on. Always ``val_year + 1``.
    """

    name: str
    train_start: pd.Timestamp
    val_year: int
    test_year: int


class HoldoutViolationError(RuntimeError):
    """Raised when any evaluation touches data at or beyond the holdout start."""


def folds_for(universe: str) -> list[Fold]:
    """Return the fixed walk-forward folds for a universe.

    Parameters
    ----------
    universe
        ``"etf10"`` or ``"sector11"``.

    Returns
    -------
    list of Fold
        10 folds for etf10, 18 for sector11, in chronological order.
    """
    if universe not in _FOLD_ANCHORS:
        known = ", ".join(sorted(_FOLD_ANCHORS))
        raise ValueError(f"Unknown universe {universe!r}; known: {known}.")

    anchor = _FOLD_ANCHORS[universe]
    train_start = pd.Timestamp(anchor["train_start"])

    folds = []
    val_year = anchor["first_val_year"]
    fold_number = 1
    while val_year + 1 <= anchor["last_test_year"]:
        folds.append(
            Fold(
                name=f"F{fold_number:02d}",
                train_start=train_start,
                val_year=val_year,
                test_year=val_year + 1,
            )
        )
        val_year += 1
        fold_number += 1
    return folds


def segment_dates(calendar: pd.DatetimeIndex, year: int) -> pd.DatetimeIndex:
    """Trading days of one calendar year, on the universe's own calendar.

    Parameters
    ----------
    calendar
        The universe's full trading calendar.
    year
        The calendar year to slice out.

    Returns
    -------
    pandas.DatetimeIndex
        The trading days within ``[year-01-01, year-12-31]``.
    """
    year_start = pd.Timestamp(year=year, month=1, day=1)
    year_end = pd.Timestamp(year=year, month=12, day=31)
    return calendar[(calendar >= year_start) & (calendar <= year_end)]


def train_dates(calendar: pd.DatetimeIndex, fold: Fold) -> pd.DatetimeIndex:
    """Trading days of a fold's training window, with the embargo applied.

    The window runs from the fold's ``train_start`` to the last trading day of
    the year before validation, and then the final ``EMBARGO_TRADING_DAYS``
    rows are removed so that nothing with memory straddles the train/validation
    boundary.

    Parameters
    ----------
    calendar
        The universe's full trading calendar.
    fold
        The fold whose training window is wanted.

    Returns
    -------
    pandas.DatetimeIndex
        The embargoed training days.
    """
    train_end = pd.Timestamp(year=fold.val_year - 1, month=12, day=31)
    window = calendar[(calendar >= fold.train_start) & (calendar <= train_end)]

    if len(window) <= EMBARGO_TRADING_DAYS:
        raise ValueError(
            f"Training window of {fold.name} has only {len(window)} trading "
            f"days - shorter than the {EMBARGO_TRADING_DAYS}-day embargo."
        )
    return window[:-EMBARGO_TRADING_DAYS]


def assert_no_holdout(dates: pd.DatetimeIndex, context: str = "") -> None:
    """Fail loudly if any supplied date lies inside the holdout period.

    Call this on the dates of anything about to be evaluated. The holdout
    discipline - "touched exactly once, in Week 8, by a script" - is only
    credible if it is enforced mechanically, and this is the mechanism.

    Parameters
    ----------
    dates
        The dates about to be used.
    context
        Optional label for the error message, e.g. the script or arm name.

    Raises
    ------
    HoldoutViolationError
        If any date is at or beyond ``HOLDOUT_START``.
    """
    if len(dates) == 0:
        return
    latest = dates.max()
    if latest >= HOLDOUT_START:
        where = f" ({context})" if context else ""
        raise HoldoutViolationError(
            f"Attempted to use data up to {latest.date()}, which crosses the "
            f"holdout boundary {HOLDOUT_START.date()}{where}. The holdout is "
            f"evaluated exactly once, in Week 8, by the dedicated script."
        )


def truncate_before_holdout(frame: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Return the rows strictly before the holdout start.

    Evaluation scripts call this on every loaded artifact, so the holdout rows
    are not merely "not evaluated" - they are never even present in memory.
    """
    return frame.loc[frame.index < HOLDOUT_START]


def year_roles(folds: list[Fold]) -> dict[int, list[tuple[str, str]]]:
    """Map each calendar year to the (fold, split) roles it plays.

    Consecutive folds share years: fold k's test year is fold k+1's validation
    year. For deterministic strategies (the baselines) the daily returns of a
    year are identical in both roles, so a year is computed once and recorded
    under every role it has.

    Returns
    -------
    dict
        ``{year: [("F01", "test"), ("F02", "val"), ...], ...}``
    """
    roles: dict[int, list[tuple[str, str]]] = {}
    for fold in folds:
        roles.setdefault(fold.val_year, []).append((fold.name, "val"))
        roles.setdefault(fold.test_year, []).append((fold.name, "test"))
    return roles


def ordered_test_years(folds: list[Fold]) -> list[int]:
    """The test years of a fold list, in order - the out-of-sample era."""
    return [fold.test_year for fold in folds]
