"""Tests for the walk-forward fold schedule and the holdout firewall.

The fold tables are fixed by the plan (Section 12); these tests pin them so a
refactor cannot quietly shift a boundary. The holdout tests matter most: the
"touched exactly once" discipline is only as strong as the guard that enforces
it.
"""

from __future__ import annotations

import pandas as pd
import pytest

from rrc.eval.slicing import (
    EMBARGO_TRADING_DAYS,
    HOLDOUT_START,
    Fold,
    HoldoutViolationError,
    assert_no_holdout,
    folds_for,
    segment_dates,
    ordered_test_years,
    train_dates,
    truncate_before_holdout,
    year_roles,
)


# --------------------------------------------------------------------------- #
# The fold tables, exactly as the plan fixes them
# --------------------------------------------------------------------------- #
def test_etf10_has_ten_folds_with_the_plan_boundaries():
    folds = folds_for("etf10")

    assert len(folds) == 10
    first, last = folds[0], folds[-1]
    assert (first.name, first.val_year, first.test_year) == ("F01", 2014, 2015)
    assert (last.name, last.val_year, last.test_year) == ("F10", 2023, 2024)
    assert first.train_start == pd.Timestamp("2006-02-01")


def test_sector11_has_eighteen_folds_with_the_plan_boundaries():
    folds = folds_for("sector11")

    assert len(folds) == 18
    first, last = folds[0], folds[-1]
    assert (first.name, first.val_year, first.test_year) == ("F01", 2006, 2007)
    assert (last.name, last.val_year, last.test_year) == ("F18", 2023, 2024)
    assert first.train_start == pd.Timestamp("1999-01-01")


def test_2008_is_a_test_year_for_sector_but_not_for_etf():
    """The A7 property: the sector universe makes 2008 genuinely out-of-sample."""
    assert 2008 in ordered_test_years(folds_for("sector11"))
    assert 2008 not in ordered_test_years(folds_for("etf10"))


def test_no_fold_touches_the_holdout():
    """The last test year is 2024; the holdout begins 2025-01-01."""
    for universe in ("etf10", "sector11"):
        assert max(ordered_test_years(folds_for(universe))) == 2024
        assert pd.Timestamp("2024-12-31") < HOLDOUT_START


def test_unknown_universe_is_rejected():
    with pytest.raises(ValueError, match="Unknown universe"):
        folds_for("dow")


# --------------------------------------------------------------------------- #
# Slicing and the embargo
# --------------------------------------------------------------------------- #
def test_segment_dates_selects_exactly_one_calendar_year():
    calendar = pd.bdate_range("2013-06-01", "2016-06-30")
    segment = segment_dates(calendar, 2015)

    assert segment.min() >= pd.Timestamp("2015-01-01")
    assert segment.max() <= pd.Timestamp("2015-12-31")
    assert len(segment) > 250  # a full trading year


def test_train_window_ends_before_validation_with_the_embargo():
    """The embargo removes one rebalance period from the training tail.

    Anything with memory - a trailing window, a reservoir state - straddles an
    adjacent boundary; the gap is what stops training-fold information bleeding
    into the selection segment.
    """
    calendar = pd.bdate_range("2006-02-01", "2016-12-31")
    fold = folds_for("etf10")[0]  # train ends 2013-12, val 2014

    window = train_dates(calendar, fold)
    days_removed = calendar[calendar <= pd.Timestamp("2013-12-31")]

    assert len(window) == len(days_removed) - EMBARGO_TRADING_DAYS
    assert window.max() < pd.Timestamp("2013-12-31") - pd.Timedelta(days=3)
    assert window.min() == pd.Timestamp("2006-02-01")


def test_train_window_shorter_than_the_embargo_is_rejected():
    tiny_calendar = pd.bdate_range("2013-12-25", "2014-12-31")
    fold = Fold(name="FX", train_start=pd.Timestamp("2013-12-25"),
                val_year=2014, test_year=2015)
    with pytest.raises(ValueError, match="embargo"):
        train_dates(tiny_calendar, fold)


# --------------------------------------------------------------------------- #
# Year-role sharing between consecutive folds
# --------------------------------------------------------------------------- #
def test_consecutive_folds_share_years_in_different_roles():
    """Fold k's test year is fold k+1's validation year - computed once, used twice."""
    roles = year_roles(folds_for("etf10"))

    assert ("F01", "test") in roles[2015]
    assert ("F02", "val") in roles[2015]
    # The first validation year and the last test year appear in one role only.
    assert roles[2014] == [("F01", "val")]
    assert roles[2024] == [("F10", "test")]


# --------------------------------------------------------------------------- #
# The holdout firewall
# --------------------------------------------------------------------------- #
def test_dates_crossing_the_holdout_are_refused():
    crossing = pd.bdate_range("2024-11-01", "2025-02-01")
    with pytest.raises(HoldoutViolationError, match="holdout"):
        assert_no_holdout(crossing, context="unit test")


def test_dates_before_the_holdout_pass():
    assert_no_holdout(pd.bdate_range("2020-01-01", "2024-12-31"))  # must not raise


def test_truncate_before_holdout_removes_2025_onward():
    frame = pd.DataFrame(
        {"x": 1.0}, index=pd.bdate_range("2024-12-20", "2025-01-20")
    )
    truncated = truncate_before_holdout(frame)

    assert truncated.index.max() < HOLDOUT_START
    assert len(truncated) > 0
    assert_no_holdout(truncated.index)  # the truncated frame passes the guard
