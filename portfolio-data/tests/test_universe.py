"""point in time membership, including a ticker that left and came back."""

from __future__ import annotations

import datetime as dt

import pandas as pd

from src.config import Member, Universe
from src.universe import eligible_tickers, membership_spans
from tests.conftest import make_universe


def dow_like() -> Universe:
    return Universe(
        name="dow",
        description="",
        schedule=(
            Member("AAA", dt.date(2007, 1, 1), dt.date(2013, 1, 1)),
            Member("BBB", dt.date(2007, 1, 1), None),
            Member("HON", dt.date(2007, 1, 1), dt.date(2008, 2, 19)),
            Member("NEW", dt.date(2015, 1, 1), None),
            Member("HON", dt.date(2020, 8, 31), None),
        ),
    )


def test_membership_is_evaluated_on_the_date():
    dow = dow_like()
    assert eligible_tickers(dow, dt.date(2010, 1, 1)) == ["AAA", "BBB"]
    assert set(eligible_tickers(dow, dt.date(2016, 1, 1))) == {"BBB", "NEW"}
    assert set(eligible_tickers(dow, dt.date(2021, 1, 1))) == {"BBB", "HON", "NEW"}


def test_repeated_ticker_collapses_to_one_span():
    spans = membership_spans(dow_like())
    starts, ends = spans["HON"]

    assert starts == pd.Timestamp("2007-01-01")
    assert ends is None  # still a member beats the 2008 exit


def test_download_list_has_no_repeats():
    assert dow_like().download_tickers == ("AAA", "BBB", "HON", "NEW")


def test_list_universe_is_always_eligible():
    etf = make_universe(("SPY", "GLD"))
    assert eligible_tickers(etf, dt.date(1990, 1, 1)) == ["SPY", "GLD"]
