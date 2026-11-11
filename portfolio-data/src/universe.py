"""who was in the index, and when.

only the dow has a membership schedule. etf and sector hold the same assets the
whole time, so every ticker is always eligible for those.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from .config import Universe


def membership_spans(universe: Universe) -> dict[str, tuple[pd.Timestamp, pd.Timestamp | None]]:
    """one start and end date per ticker.

    a ticker can appear more than once, so we take its earliest start and its
    latest end. still being a member beats any end date.
    """
    spans: dict[str, tuple[pd.Timestamp, pd.Timestamp | None]] = {}
    for member in universe.schedule:
        starts = pd.Timestamp(member.member_from)
        ends = pd.Timestamp(member.member_to) if member.member_to else None
        if member.ticker not in spans:
            spans[member.ticker] = (starts, ends)
            continue
        old_start, old_end = spans[member.ticker]
        merged_end = None if (old_end is None or ends is None) else max(old_end, ends)
        spans[member.ticker] = (min(old_start, starts), merged_end)
    return spans


def eligible_tickers(universe: Universe, on_date: dt.date | pd.Timestamp) -> list[str]:
    """which tickers were actually in the universe on that date.

    this is what stops survivorship bias. backtesting today's dow members back to
    2007 would have you holding apple in 2010, four years before it was added,
    and never holding old gm, which went bankrupt.
    """
    if not universe.schedule:
        return list(universe.download_tickers)

    when = pd.Timestamp(on_date)
    eligible: dict[str, None] = {}
    for member in universe.schedule:
        starts = pd.Timestamp(member.member_from)
        ends = pd.Timestamp(member.member_to) if member.member_to else None
        if starts <= when and (ends is None or when <= ends):
            eligible.setdefault(member.ticker, None)
    return list(eligible)
