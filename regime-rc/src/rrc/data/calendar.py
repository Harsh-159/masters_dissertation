"""Trading-day calendar construction and alignment.

Why not a synthetic business-day range
--------------------------------------
It is tempting to build the calendar with ``pd.bdate_range``, but that invents
trading days: exchanges close for holidays that differ by year and by country,
and a synthetic range would place rows on days no asset traded. Every such row
then needs filling, and filling invented rows is indistinguishable from
fabricating data.

Instead the calendar is the **union of days on which at least one asset in the
universe actually traded**. It is observed, not assumed. A consequence worth
understanding: if *no* asset traded on some day, that day simply is not a row -
so a single asset never has an internal gap, and forward-filling only ever
matters when one asset is missing a day that another asset has.

Alignment rules
---------------
Each asset is reindexed onto the calendar and then forward-filled **within its
own active life only**, with a small limit. Forward-fill cannot touch leading
NaNs (there is no earlier value to carry), so an asset that starts mid-sample
keeps its leading NaNs automatically - which is exactly how differing inception
dates are handled without special-casing them.
"""

from __future__ import annotations

import pandas as pd


def normalise_index(series: pd.Series) -> pd.Series:
    """Return a copy with a tz-naive, midnight-normalised, unique, sorted index.

    Every series entering the pipeline passes through here, so that indices from
    different sources (yfinance, Stooq, FRED, an Excel file) are directly
    comparable rather than differing by a timezone or a time-of-day component.

    Parameters
    ----------
    series
        A series indexed by dates or timestamps.

    Returns
    -------
    pandas.Series
        The same data on a clean ``DatetimeIndex``. Duplicate timestamps keep
        their last occurrence, which is the convention for revised data.
    """
    cleaned = series.copy()
    index = pd.to_datetime(cleaned.index)

    if index.tz is not None:
        index = index.tz_localize(None)

    cleaned.index = index.normalize()
    cleaned = cleaned[~cleaned.index.duplicated(keep="last")]
    return cleaned.sort_index()


def build_trading_calendar(
    price_series: dict[str, pd.Series],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DatetimeIndex:
    """Build the trading calendar as the union of observed trading days.

    Parameters
    ----------
    price_series
        Mapping ``ticker -> price series`` (already normalised).
    start, end
        Inclusive window bounds.

    Returns
    -------
    pandas.DatetimeIndex
        Sorted, unique trading days within ``[start, end]``, named ``date``.

    Raises
    ------
    ValueError
        If no asset has any observation inside the window, which almost always
        means a misconfigured date range rather than a data problem.
    """
    calendar = pd.DatetimeIndex([])
    for series in price_series.values():
        in_window = series.loc[start:end]
        calendar = calendar.union(in_window.index)

    if len(calendar) == 0:
        raise ValueError(
            f"No trading days found between {start.date()} and {end.date()}. "
            "Check the configured window against the available price history."
        )

    calendar = calendar.sort_values()
    calendar.name = "date"
    return calendar


def align_to_calendar(
    series: pd.Series,
    calendar: pd.DatetimeIndex,
    ffill_limit: int,
) -> pd.Series:
    """Reindex one series onto the calendar and forward-fill within its life.

    Parameters
    ----------
    series
        The asset's observed values.
    calendar
        Target trading calendar.
    ffill_limit
        Maximum number of consecutive missing days to bridge. Gaps longer than
        this are left as NaN, because a long gap is genuinely absent data rather
        than a holiday to be smoothed over.

    Returns
    -------
    pandas.Series
        Aligned series. Cells before the asset's first real observation remain
        NaN; nothing is ever back-filled.
    """
    reindexed = series.reindex(calendar)
    return reindexed.ffill(limit=ffill_limit)


def align_panel(
    price_series: dict[str, pd.Series],
    calendar: pd.DatetimeIndex,
    columns: list[str],
    ffill_limit: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align many series into a panel, returning both filled and raw versions.

    Two panels are returned because later stages need to tell the difference
    between a price that was *observed* and one that was *carried forward*:
    the availability matrix and the causality tests are defined on real
    observations, while returns are computed on the filled panel.

    Parameters
    ----------
    price_series
        Mapping ``ticker -> observed series``.
    calendar
        Target trading calendar.
    columns
        Desired column order; tickers with no data still appear, as all-NaN
        columns, so that a missing asset is visible rather than silently absent.
    ffill_limit
        Forward-fill limit passed to :func:`align_to_calendar`.

    Returns
    -------
    filled, observed
        ``filled`` is the forward-filled panel used for returns; ``observed``
        contains real observations only, with NaN everywhere no price was seen.
    """
    observed = pd.DataFrame(index=calendar, columns=columns, dtype="float64")
    filled = pd.DataFrame(index=calendar, columns=columns, dtype="float64")

    for ticker in columns:
        series = price_series.get(ticker)
        if series is None or series.empty:
            continue  # leave the column all-NaN
        observed[ticker] = series.reindex(calendar)
        filled[ticker] = align_to_calendar(series, calendar, ffill_limit)

    observed.index.name = "date"
    filled.index.name = "date"
    return filled, observed
