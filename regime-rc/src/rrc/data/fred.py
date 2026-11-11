"""FRED series ingestion and the risk-free rate conversion.

Series used
-----------
============  =========================================================
DGS3MO        3-month Treasury constant-maturity yield -> defines r_f
DGS10         10-year yield -> term slope is DGS10 - DGS3MO
VIXCLS        CBOE volatility index
BAA10Y        Moody's Baa corporate yield minus 10-year Treasury (credit)
============  =========================================================

All are pulled from FRED's no-key CSV endpoint and cached to parquet, so the
pipeline runs offline after the first pull.

The risk-free conversion, and why DGS3MO rather than DTB3
---------------------------------------------------------
``DTB3`` is quoted on a **discount basis**, not as a yield. Using it directly
misprices cash by a small but systematic amount - and because *every* excess
return in this project is measured against this same series, a systematic bias
in it biases every Sharpe ratio, every loss value, and the cash column that the
model is free to hold. ``DGS3MO`` is a constant-maturity **yield**, which is the
quantity we actually want.

The conversion from an annual percentage yield ``y`` to a daily simple return is
compounding-consistent:

    r_f,daily = (1 + y/100) ** (1/252) - 1

not the common shortcut ``y/100/252``, which ignores compounding. Then the
series is forward-filled (FRED omits holidays) and **shifted one day**, because
a rate published for day ``t`` is not actionable until the following day.
"""

from __future__ import annotations

import io
import logging
import urllib.request
from pathlib import Path

import pandas as pd

from rrc.data.calendar import normalise_index

logger = logging.getLogger(__name__)

FRED_CSV_ENDPOINT = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

# FRED writes a literal "." for missing observations (holidays, non-trading days).
FRED_MISSING_MARKER = "."

# Trading days per year, used for the annual -> daily compounding conversion.
TRADING_DAYS_PER_YEAR = 252


def fetch_fred_series(
    series_id: str,
    raw_dir: Path,
    force_refresh: bool = False,
    timeout_seconds: int = 30,
) -> pd.Series:
    """Fetch one FRED series, using the parquet cache when available.

    Parameters
    ----------
    series_id
        FRED identifier, e.g. ``'DGS3MO'``.
    raw_dir
        Directory holding the cache.
    force_refresh
        Ignore the cache and re-download.
    timeout_seconds
        Network timeout.

    Returns
    -------
    pandas.Series
        The series, named ``series_id``, on a clean date index, with FRED's
        missing markers dropped rather than converted to zero.

    Raises
    ------
    RuntimeError
        If the download fails and no cache exists. FRED outages should stop a
        data build loudly, not silently produce a dataset missing its signals.
    """
    cache_path = Path(raw_dir) / f"fred_{series_id}.parquet"

    if cache_path.is_file() and not force_refresh:
        cached = pd.read_parquet(cache_path)[series_id]
        logger.info("FRED %s: %d rows from cache.", series_id, len(cached))
        return cached

    url = FRED_CSV_ENDPOINT.format(series_id=series_id)
    try:
        with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
            csv_text = response.read().decode("utf-8")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Could not download FRED series {series_id} from {url}: {exc}") from exc

    series = _parse_fred_csv(csv_text, series_id)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    series.to_frame(name=series_id).to_parquet(cache_path)
    logger.info(
        "FRED %s: downloaded %d rows (%s to %s).",
        series_id, len(series), series.index.min().date(), series.index.max().date(),
    )
    return series


def _parse_fred_csv(csv_text: str, series_id: str) -> pd.Series:
    """Parse FRED's CSV defensively.

    FRED has changed its date column header over time (``DATE`` in older
    exports, ``observation_date`` in newer ones), so the first column is taken
    positionally rather than by name. Missing values are the literal ``"."``.
    """
    frame = pd.read_csv(io.StringIO(csv_text))

    if frame.shape[1] < 2:
        raise ValueError(
            f"FRED CSV for {series_id} has {frame.shape[1]} column(s); expected at least 2. "
            f"Header was: {list(frame.columns)}"
        )

    date_column = frame.columns[0]
    value_column = frame.columns[1]

    frame[date_column] = pd.to_datetime(frame[date_column])
    values = frame[value_column].replace(FRED_MISSING_MARKER, pd.NA)
    values = pd.to_numeric(values, errors="coerce")

    series = pd.Series(values.to_numpy(), index=frame[date_column], name=series_id)
    series = normalise_index(series).dropna()

    if series.empty:
        raise ValueError(f"FRED series {series_id} parsed to zero usable observations.")
    return series


def annual_yield_to_daily_return(annual_percent: pd.Series) -> pd.Series:
    """Convert an annual percentage yield to a daily simple return.

    Parameters
    ----------
    annual_percent
        Yield in percent (FRED's convention: ``4.87`` means 4.87% a year).

    Returns
    -------
    pandas.Series
        Daily simple return, compounding-consistent.

    Notes
    -----
    Uses ``(1 + y/100)**(1/252) - 1`` rather than ``y/100/252``. At 5% a year the
    two differ by a fraction of a basis point per day, but the error accumulates
    over 5,000 trading days and sits inside every excess return in the project.
    """
    return (1.0 + annual_percent / 100.0) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0


def build_risk_free_daily(
    short_rate_id: str,
    raw_dir: Path,
    calendar: pd.DatetimeIndex,
    shift_days: int = 1,
    force_refresh: bool = False,
) -> pd.Series:
    """Build the daily risk-free return series aligned to a trading calendar.

    The operation order is fixed and matters:

    1. fetch the annual constant-maturity yield;
    2. convert to a daily simple return (compounding-consistent);
    3. reindex onto the trading calendar and forward-fill (FRED omits holidays,
       and a rate persists until it is next published) - never back-fill;
    4. **shift by ``shift_days``**, so the rate used on day ``t`` is the one
       published on day ``t-1`` and therefore actually knowable at ``t``.

    Parameters
    ----------
    short_rate_id
        FRED id of the short rate, normally ``'DGS3MO'``.
    raw_dir
        Cache directory.
    calendar
        Trading calendar to align to.
    shift_days
        Publication lag in trading days.
    force_refresh
        Ignore the cache.

    Returns
    -------
    pandas.Series
        Daily risk-free simple return on the trading calendar, named ``r_f``.
    """
    annual_yield = fetch_fred_series(short_rate_id, raw_dir, force_refresh=force_refresh)
    daily_return = annual_yield_to_daily_return(annual_yield)

    aligned = daily_return.reindex(calendar.union(daily_return.index)).ffill()
    aligned = aligned.reindex(calendar)
    lagged = aligned.shift(shift_days)

    lagged.name = "r_f"
    lagged.index.name = "date"
    return lagged
