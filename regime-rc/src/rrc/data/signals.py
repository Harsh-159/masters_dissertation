"""Market-level signal construction, with publication lags handled first.

What the signals are
--------------------
A small vector of **market-level** (never per-stock) conditioning variables that
only the gating network sees:

===============  ==========================  =====================================
signal           source                      publication lag applied
===============  ==========================  =====================================
vix              FRED VIXCLS                 1 trading day
sentiment        SF Fed Daily News Sentiment 5 business days (minimum)
term_slope       FRED DGS10 - DGS3MO         1 trading day
short_rate       FRED DGS3MO                 1 trading day
credit_spread    FRED BAA10Y                 1 trading day
===============  ==========================  =====================================

Keeping them market-level is what keeps regimes *decision-defined*: sentiment is
an input the gating **may** use, not the definition of a regime.

The causal order of operations
------------------------------
For every signal the order is fixed:

    align to the trading calendar  ->  SHIFT by the publication lag  ->  trailing z-score

**Shift before standardise, never the other way round.** For a purely backward
looking rolling window the two orderings happen to agree numerically, so this
looks like pedantry - but it is not. Ordering it this way means the rolling
mean and standard deviation are computed on a series that has *already* been
lagged, so the z-statistics themselves can never touch a value that was not yet
published. That guarantee survives someone later changing the window type
(expanding, centred, EWMA), whereas the numerical coincidence does not. It is
the difference between a property that holds and a property that happens to
hold today.

Standardisation uses a trailing 252-day window with ``min_periods=126``: no
signal is emitted until half a year of history exists behind it, so the earliest
z-scores are not divided by a standard deviation estimated from a handful of
points.
"""

from __future__ import annotations

import io
import logging
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from rrc.data.calendar import normalise_index

logger = logging.getLogger(__name__)

# Order is fixed so the signal matrix has a stable column layout across builds.
SIGNAL_NAMES = ("vix", "sentiment", "term_slope", "short_rate", "credit_spread")


class SentimentUnavailableError(RuntimeError):
    """Raised when the SF Fed sentiment file cannot be downloaded or parsed."""


# --------------------------------------------------------------------------- #
# The core causal transform
# --------------------------------------------------------------------------- #
def align_to_calendar_forward_only(
    series: pd.Series, calendar: pd.DatetimeIndex
) -> pd.Series:
    """Put a series onto the trading calendar, carrying values forward only.

    Macro series are published on their own schedules and skip holidays. A
    published value persists until the next publication, so forward-filling is
    the correct semantics - it says "the most recently published value". Back
    filling would say "tomorrow's value applied to today", which is look-ahead.

    Parameters
    ----------
    series
        Raw signal on its own index.
    calendar
        Trading calendar to align onto.

    Returns
    -------
    pandas.Series
        Values on ``calendar``, with leading cells NaN until the series' own
        first observation.
    """
    combined_index = calendar.union(series.index)
    return series.reindex(combined_index).ffill().reindex(calendar)


def shift_then_zscore(
    series: pd.Series,
    shift_periods: int,
    window: int,
    min_periods: int,
) -> pd.Series:
    """Apply the publication lag, then standardise on a trailing window.

    Parameters
    ----------
    series
        Signal already aligned to the trading calendar.
    shift_periods
        Publication lag, in rows of the trading calendar (i.e. trading days).
    window
        Length of the trailing standardisation window.
    min_periods
        Minimum observations required before any z-score is emitted.

    Returns
    -------
    pandas.Series
        The standardised signal. NaN until ``shift_periods + min_periods``
        observations exist, which is honest rather than convenient: an early
        z-score computed from a handful of points is noise.
    """
    if shift_periods < 0:
        raise ValueError(f"shift_periods must be >= 0, got {shift_periods}.")

    # 1. Publication lag FIRST. Everything downstream now sees only values that
    #    were knowable strictly before the row's own date.
    lagged = series.shift(shift_periods)

    # 2. Trailing statistics computed on the ALREADY-LAGGED series.
    rolling = lagged.rolling(window=window, min_periods=min_periods)
    standard_deviation = rolling.std(ddof=0)

    # A zero standard deviation means a flat window (e.g. a policy rate pinned
    # at zero for months). Dividing would give inf; NaN is the honest answer.
    # np.nan rather than pd.NA: this column stays a true float64, which pd.NA
    # would silently promote to a nullable dtype that later .astype(float) calls
    # cannot convert.
    standard_deviation = standard_deviation.replace(0.0, np.nan)

    return (lagged - rolling.mean()) / standard_deviation


# --------------------------------------------------------------------------- #
# The SF Fed sentiment index
# --------------------------------------------------------------------------- #
def fetch_news_sentiment(
    url: str,
    raw_dir: Path,
    force_refresh: bool = False,
    timeout_seconds: int = 60,
) -> pd.Series:
    """Download and parse the SF Fed Daily News Sentiment Index.

    The spreadsheet's column layout has changed before and will change again, so
    parsing is positional-with-checks rather than by hard-coded header name: the
    first datetime-like column is the date, and the first numeric column is the
    index. If neither can be identified the function raises - a silently
    mis-parsed sentiment series would be far worse than a failed build.

    Parameters
    ----------
    url
        Location of the ``.xlsx`` file.
    raw_dir
        Cache directory.
    force_refresh
        Ignore the cache.
    timeout_seconds
        Network timeout; the file is a few megabytes.

    Returns
    -------
    pandas.Series
        Daily sentiment index, named ``sentiment``.

    Raises
    ------
    SentimentUnavailableError
        If the file cannot be downloaded or no usable columns are found.
    """
    cache_path = Path(raw_dir) / "sentiment_sf_fed.parquet"

    if cache_path.is_file() and not force_refresh:
        cached = pd.read_parquet(cache_path)["sentiment"]
        logger.info("Sentiment: %d rows from cache.", len(cached))
        return cached

    try:
        request = urllib.request.Request(url, headers={"User-Agent": "regime-rc/0.1 (research)"})
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            workbook_bytes = response.read()
    except Exception as exc:  # noqa: BLE001
        raise SentimentUnavailableError(
            f"Could not download the SF Fed sentiment file from {url}: {exc}"
        ) from exc

    series = _parse_sentiment_workbook(workbook_bytes)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    series.to_frame(name="sentiment").to_parquet(cache_path)
    logger.info(
        "Sentiment: downloaded %d rows (%s to %s).",
        len(series), series.index.min().date(), series.index.max().date(),
    )
    return series


def _parse_sentiment_workbook(workbook_bytes: bytes) -> pd.Series:
    """Parse the sentiment workbook defensively across sheets and layouts."""
    try:
        sheets = pd.read_excel(io.BytesIO(workbook_bytes), sheet_name=None)
    except Exception as exc:  # noqa: BLE001
        raise SentimentUnavailableError(f"Could not open the sentiment workbook: {exc}") from exc

    for sheet_name, frame in sheets.items():
        if frame.empty or frame.shape[1] < 2:
            continue

        date_column = _first_datetime_column(frame)
        value_column = _first_numeric_column(frame, exclude=date_column)
        if date_column is None or value_column is None:
            continue

        series = pd.Series(
            pd.to_numeric(frame[value_column], errors="coerce").to_numpy(),
            index=pd.to_datetime(frame[date_column], errors="coerce"),
            name="sentiment",
        )
        series = series[series.index.notna()]
        series = normalise_index(series).dropna()

        if not series.empty:
            logger.info("Sentiment parsed from sheet %r using columns "
                        "(%r, %r).", sheet_name, date_column, value_column)
            return series

    raise SentimentUnavailableError(
        "No sheet in the sentiment workbook had an identifiable "
        "(date, numeric value) column pair. The layout has probably changed; "
        "inspect the file before trusting any sentiment result."
    )


def _first_datetime_column(frame: pd.DataFrame) -> str | None:
    """Return the first column that parses convincingly as dates."""
    for column in frame.columns:
        if pd.api.types.is_datetime64_any_dtype(frame[column]):
            return column
    # Fall back to attempting a parse; require most values to succeed.
    for column in frame.columns:
        parsed = pd.to_datetime(frame[column], errors="coerce")
        if parsed.notna().mean() > 0.9:
            return column
    return None


def _first_numeric_column(frame: pd.DataFrame, exclude: str | None) -> str | None:
    """Return the first mostly-numeric column, ignoring the date column."""
    for column in frame.columns:
        if column == exclude:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().mean() > 0.5:
            return column
    return None


# --------------------------------------------------------------------------- #
# Assembling the signal matrix
# --------------------------------------------------------------------------- #
def build_signal_matrix(
    calendar: pd.DatetimeIndex,
    raw_series: dict[str, pd.Series],
    shift_days: int,
    sentiment_shift_days: int,
    z_window: int,
    z_min_periods: int,
) -> pd.DataFrame:
    """Build the standardised signal matrix on a trading calendar.

    Parameters
    ----------
    calendar
        Trading calendar of the universe.
    raw_series
        Raw, unlagged inputs keyed by FRED-style name. Expected keys:
        ``vix``, ``term_slope``, ``short_rate``, ``credit_spread``, and
        optionally ``sentiment``. A missing key produces an all-NaN column
        rather than a silently absent one, so the gap is visible downstream.
    shift_days
        Publication lag for the market/macro series.
    sentiment_shift_days
        Publication lag for sentiment, which is larger because the index is
        released late and is itself built on trailing smoothing.
    z_window, z_min_periods
        Trailing standardisation parameters.

    Returns
    -------
    pandas.DataFrame
        Columns in the fixed order of :data:`SIGNAL_NAMES`, indexed by date.
    """
    matrix = pd.DataFrame(index=calendar, dtype="float64")
    matrix.index.name = "date"

    for name in SIGNAL_NAMES:
        lag = sentiment_shift_days if name == "sentiment" else shift_days
        raw = raw_series.get(name)

        if raw is None or raw.empty:
            # An unavailable source becomes an explicit all-NaN float column, so
            # the gap is visible downstream rather than the column vanishing.
            logger.warning("Signal %r is unavailable; emitting an all-NaN column.", name)
            matrix[name] = np.nan
            continue

        aligned = align_to_calendar_forward_only(normalise_index(raw), calendar)
        matrix[name] = shift_then_zscore(
            aligned, shift_periods=lag, window=z_window, min_periods=z_min_periods
        )

    return matrix.astype("float64")
